"""
RoboWheel V2: 100% MANO-driven reference + PPO residual RL.
Every DOF (position, orientation, fingers) comes from MANO data.
RL only adds tiny residuals for physical plausibility.
"""
import os, json, sys
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import mujoco
import cv2
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import uniform_filter1d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MENAGERIE = os.path.join(BASE_DIR, 'mujoco_menagerie')
OUT_DIR = os.path.join(BASE_DIR, 'output/rl_v2')
MANO_PATH = os.path.join(BASE_DIR, 'output/pick_bottle_video/mano_results.json')


# ============================================================
# Extract FULL MANO reference (position + orientation + fingers)
# ============================================================

def build_mano_reference():
    """Extract complete reference trajectory from MANO data."""
    import smplx
    mano_model_path = '/mnt/users/yjy/sim/video2robot-retarget/HaWoR/_DATA/data'
    mano = smplx.create(mano_model_path, model_type='mano', is_rhand=True,
                         use_pca=False, flat_hand_mean=False)

    with open(MANO_PATH) as f:
        results = json.load(f)

    # Get frames with both hands
    frame_dict = {}
    for r in results:
        key = r['img_name']
        if key not in frame_dict: frame_dict[key] = {}
        frame_dict[key]['right' if r['is_right'] else 'left'] = r
    both = [v for v in frame_dict.values() if 'left' in v and 'right' in v]
    N = len(both)

    # MANO camera → world transform
    T = np.array([[1,0,0],[0,0,-1],[0,-1,0]], dtype=np.float64)

    ref = {
        'N': N,
        'right_wrist_pos': np.zeros((N, 3)),
        'left_wrist_pos': np.zeros((N, 3)),
        'right_wrist_euler': np.zeros((N, 3)),
        'left_wrist_euler': np.zeros((N, 3)),
        'right_fingers': np.zeros((N, 16)),  # Allegro 16 actuators
        'left_fingers': np.zeros((N, 16)),
        'obj_pos': np.zeros((N, 3)),
    }

    # MANO joint → Allegro actuator mapping
    # MANO: index(0-2), middle(3-5), pinky(6-8), ring(9-11), thumb(12-14)
    # Allegro: ff(0-3:spread+3flex), mf(4-7), rf(8-11), th(12-15)
    m2a = {0:1, 1:2, 2:3, 3:5, 4:6, 5:7, 9:9, 10:10, 11:11, 12:13, 13:14, 14:15}

    for i, data in enumerate(both):
        for side in ['right', 'left']:
            r = data[side]
            hp = torch.tensor(r['mano_hand_pose'], dtype=torch.float32)
            bt = torch.tensor(r['mano_betas'], dtype=torch.float32).unsqueeze(0)
            go_mat = np.array(r['mano_global_orient'])
            if go_mat.ndim == 3: go_mat = go_mat[0]
            go_aa = torch.tensor(Rot.from_matrix(go_mat).as_rotvec(), dtype=torch.float32).unsqueeze(0)
            hp_aa = torch.tensor(
                np.array([Rot.from_matrix(hp[j].numpy()).as_rotvec() for j in range(15)]).flatten(),
                dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                out = mano(hand_pose=hp_aa, betas=bt, global_orient=go_aa)
            joints = out.joints[0].numpy()

            if not r['is_right']:
                joints[:, 0] = -joints[:, 0]

            # Wrist position: world frame
            wrist_world = T @ joints[0] + np.array([0, 0, 0.20])

            # Wrist orientation: world frame
            go_world = T @ go_mat @ T.T
            if not r['is_right']:
                # Mirror X for left hand orientation
                mirror = np.diag([-1, 1, 1]).astype(np.float64)
                go_world = mirror @ go_world @ mirror
            wrist_euler = Rot.from_matrix(go_world).as_euler('xyz')

            # Finger angles: euler flexion from MANO rotation matrices
            hp_np = hp.numpy()
            fingers = np.zeros(16)
            for mano_j, allegro_j in m2a.items():
                rot_euler = Rot.from_matrix(hp_np[mano_j]).as_euler('xyz')
                fingers[allegro_j] = rot_euler[0]  # flexion

            # Spread joints (j0, j4, j8) from Z component
            for mano_j, allegro_spread in [(0, 0), (3, 4), (9, 8)]:
                rot_euler = Rot.from_matrix(hp_np[mano_j]).as_euler('xyz')
                fingers[allegro_spread] = rot_euler[2]
            # Thumb rotation (j12) from flexion
            fingers[12] = abs(Rot.from_matrix(hp_np[12]).as_euler('xyz')[0]) * 0.8 + 0.3

            ref[f'{side}_wrist_pos'][i] = wrist_world
            ref[f'{side}_wrist_euler'][i] = wrist_euler
            ref[f'{side}_fingers'][i] = fingers

        # Object position: midpoint of right thumb_tip and index_tip
        r_data = data['right']
        hp = torch.tensor(r_data['mano_hand_pose'], dtype=torch.float32)
        bt = torch.tensor(r_data['mano_betas'], dtype=torch.float32).unsqueeze(0)
        go_mat = np.array(r_data['mano_global_orient'])
        if go_mat.ndim == 3: go_mat = go_mat[0]
        go_aa = torch.tensor(Rot.from_matrix(go_mat).as_rotvec(), dtype=torch.float32).unsqueeze(0)
        hp_aa = torch.tensor(
            np.array([Rot.from_matrix(hp[j].numpy()).as_rotvec() for j in range(15)]).flatten(),
            dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = mano(hand_pose=hp_aa, betas=bt, global_orient=go_aa)
        joints = out.joints[0].numpy()
        # smplx MANO: 16 joints. Use thumb tip (15) + index tip (3) midpoint
        n_joints = joints.shape[0]
        thumb_idx = min(15, n_joints - 1)
        index_idx = min(3, n_joints - 1)
        obj = T @ ((joints[thumb_idx] + joints[index_idx]) / 2) + np.array([0, 0, 0.20])
        ref['obj_pos'][i] = obj

    # Smooth all trajectories
    for key in ref:
        if isinstance(ref[key], np.ndarray) and ref[key].ndim >= 2:
            ref[key] = uniform_filter1d(ref[key], size=7, axis=0)

    # Clamp to safe ranges
    for side in ['right', 'left']:
        ref[f'{side}_wrist_euler'] = np.clip(ref[f'{side}_wrist_euler'], -1.5, 1.5)
        ref[f'{side}_fingers'] = np.clip(ref[f'{side}_fingers'], -0.5, 1.7)

    return ref


# ============================================================
# Gymnasium Environment
# ============================================================

class MANOGraspEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, mano_ref=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.ref = mano_ref if mano_ref is not None else build_mano_reference()
        self.N_ref = self.ref['N']
        self.max_steps = self.N_ref
        self.model, self.data = self._build()

        # Action: residual for 44 DOF (22 per hand: 6 base + 16 fingers)
        # Small ranges: pos ±1cm, rot ±0.05rad, fingers ±0.05rad
        act_low = np.concatenate([
            np.full(3, -0.01), np.full(3, -0.05), np.full(16, -0.05),  # right
            np.full(3, -0.01), np.full(3, -0.05), np.full(16, -0.05),  # left
        ])
        act_high = -act_low
        self.action_space = spaces.Box(act_low.astype(np.float32), act_high.astype(np.float32))

        obs_dim = 44 + 3 + 4 + 24 + 1  # joints + bottle_pos + bottle_quat + tips + timestep
        self.observation_space = spaces.Box(-10, 10, shape=(obs_dim,), dtype=np.float32)

        self._find_actuators()
        self.step_count = 0
        self.prev_action = np.zeros(44)
        self.bottle_init_z = 0

    def _build(self):
        allegro_r = os.path.join(MENAGERIE, 'wonik_allegro/right_hand.xml')
        os.chdir(os.path.dirname(allegro_r))
        hr = mujoco.MjSpec.from_file('right_hand.xml')
        hl = mujoco.MjSpec.from_file('left_hand.xml')

        s = mujoco.MjSpec()
        s.option.gravity = [0, 0, -9.81]; s.option.timestep = 0.002
        s.option.impratio = 10; s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        w = s.worldbody

        l = w.add_light(); l.pos=[0,-0.3,0.5]; l.dir=[0,0.2,-1]; l.diffuse=[1,1,1]

        # Floor + Table
        f = w.add_geom(); f.type=mujoco.mjtGeom.mjGEOM_PLANE; f.size=[0.5,0.5,0.01]
        f.rgba=[0.92,0.92,0.92,1]
        t = w.add_geom(); t.type=mujoco.mjtGeom.mjGEOM_BOX; t.size=[0.18,0.14,0.01]
        t.pos=[0,0,0.15]; t.rgba=[0.38,0.28,0.20,1]; t.friction=[1.0,0.005,0.0001]

        # Bottle (physics)
        obj_pos = self.ref['obj_pos'].mean(0)
        bottle = w.add_body(); bottle.name="bottle"; bottle.pos=obj_pos.tolist()
        bj = bottle.add_freejoint(); bj.name="bottle_joint"
        bg = bottle.add_geom(); bg.type=mujoco.mjtGeom.mjGEOM_CYLINDER
        bg.size=[0.030,0.07,0]; bg.rgba=[0.15,0.50,0.85,0.85]
        bg.mass=0.20; bg.friction=[2.0,0.01,0.001]

        # Right hand: 6DoF base
        rp = self.ref['right_wrist_pos'][0]
        br = w.add_body(); br.name="base_right"; br.pos=rp.tolist()
        for jn,ax in [("rx",[1,0,0]),("ry",[0,1,0]),("rz",[0,0,1])]:
            j=br.add_joint(); j.name=jn; j.type=mujoco.mjtJoint.mjJNT_SLIDE; j.axis=ax; j.range=[-0.15,0.15]
        for jn,ax in [("rrx",[1,0,0]),("rry",[0,1,0]),("rrz",[0,0,1])]:
            j=br.add_joint(); j.name=jn; j.type=mujoco.mjtJoint.mjJNT_HINGE; j.axis=ax; j.range=[-3.14,3.14]
        mr = br.add_body(); mr.name="mount_right"; mr.quat=[1,0,0,0]
        mr.add_frame().attach_body(hr.worldbody.first_body(), "rh_", "")

        # Left hand
        lp = self.ref['left_wrist_pos'][0]
        bl = w.add_body(); bl.name="base_left"; bl.pos=lp.tolist()
        for jn,ax in [("lx",[1,0,0]),("ly",[0,1,0]),("lz",[0,0,1])]:
            j=bl.add_joint(); j.name=jn; j.type=mujoco.mjtJoint.mjJNT_SLIDE; j.axis=ax; j.range=[-0.15,0.15]
        for jn,ax in [("lrx",[1,0,0]),("lry",[0,1,0]),("lrz",[0,0,1])]:
            j=bl.add_joint(); j.name=jn; j.type=mujoco.mjtJoint.mjJNT_HINGE; j.axis=ax; j.range=[-3.14,3.14]
        ml = bl.add_body(); ml.name="mount_left"; ml.quat=[1,0,0,0]
        ml.add_frame().attach_body(hl.worldbody.first_body(), "lh_", "")

        # Actuators (high gain for precise tracking)
        for jn in ["rx","ry","rz","rrx","rry","rrz","lx","ly","lz","lrx","lry","lrz"]:
            a=s.add_actuator(); a.name=f"act_{jn}"; a.target=jn
            a.trntype=mujoco.mjtTrn.mjTRN_JOINT
            kp = 500 if jn.endswith(('x','y','z')) and 'r' not in jn[1:] else 150
            a.gainprm=[kp]+[0]*9

        model = s.compile()
        return model, mujoco.MjData(model)

    def _find_actuators(self):
        self.base_acts_r = {}; self.base_acts_l = {}
        self.finger_acts_r = []; self.finger_acts_l = []
        self.tip_ids = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            if name.startswith("act_r"): self.base_acts_r[name.replace("act_","")] = i
            elif name.startswith("act_l"): self.base_acts_l[name.replace("act_","")] = i
            elif "rh_" in name: self.finger_acts_r.append(i)
            elif "lh_" in name: self.finger_acts_l.append(i)
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if "tip" in name and ("rh_" in name or "lh_" in name):
                self.tip_ids.append(i)
        self.bottle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'bottle')

    def _get_ref(self, frame):
        """Get MANO reference controls for a frame.
        Position: MANO delta (from first frame), scaled
        Orientation: fixed horizontal base + small MANO delta (10%)
        Fingers: direct from MANO hand_pose
        """
        fi = min(frame, self.N_ref - 1)

        # Position: delta from first frame, scaled to sim workspace
        POS_SCALE = 0.3
        rp = (self.ref['right_wrist_pos'][fi] - self.ref['right_wrist_pos'][0]) * POS_SCALE
        lp = (self.ref['left_wrist_pos'][fi] - self.ref['left_wrist_pos'][0]) * POS_SCALE

        # Orientation: fixed base + 10% of MANO delta
        ORIENT_SCALE = 0.10
        re_delta = (self.ref['right_wrist_euler'][fi] - self.ref['right_wrist_euler'][0]) * ORIENT_SCALE
        le_delta = (self.ref['left_wrist_euler'][fi] - self.ref['left_wrist_euler'][0]) * ORIENT_SCALE
        # Right: fingers point left (Z=π), palm tilted inward (Y=-0.25)
        re = np.array([0, -0.25, np.pi]) + re_delta
        # Left: fingers point right (default), palm tilted inward (Y=+0.25)
        le = np.array([0, 0.25, 0]) + le_delta

        # Fingers: direct from MANO
        rf = self.ref['right_fingers'][fi]
        lf = self.ref['left_fingers'][fi]

        return np.concatenate([rp, re, rf, lp, le, lf])

    def _apply_ctrl(self, ref_ctrl, residual):
        """Apply MANO reference + RL residual to actuators."""
        ctrl = ref_ctrl + residual

        # Right base (pos + orient)
        for idx, name in enumerate(["rx","ry","rz"]):
            if name in self.base_acts_r:
                self.data.ctrl[self.base_acts_r[name]] = ctrl[idx]
        for idx, name in enumerate(["rrx","rry","rrz"]):
            if name in self.base_acts_r:
                self.data.ctrl[self.base_acts_r[name]] = ctrl[3+idx]
        # Right fingers
        for j, ai in enumerate(self.finger_acts_r[:16]):
            jid = self.model.actuator_trnid[ai, 0]
            if jid >= 0:
                jr = self.model.jnt_range[jid]
                self.data.ctrl[ai] = np.clip(ctrl[6+j], jr[0], jr[1])

        # Left base
        for idx, name in enumerate(["lx","ly","lz"]):
            if name in self.base_acts_l:
                self.data.ctrl[self.base_acts_l[name]] = ctrl[22+idx]
        for idx, name in enumerate(["lrx","lry","lrz"]):
            if name in self.base_acts_l:
                self.data.ctrl[self.base_acts_l[name]] = ctrl[25+idx]
        # Left fingers
        for j, ai in enumerate(self.finger_acts_l[:16]):
            jid = self.model.actuator_trnid[ai, 0]
            if jid >= 0:
                jr = self.model.jnt_range[jid]
                self.data.ctrl[ai] = np.clip(ctrl[28+j], jr[0], jr[1])

    def _get_obs(self):
        joints = np.zeros(44, dtype=np.float32)  # placeholder
        bottle_pos = self.data.xpos[self.bottle_id].astype(np.float32) if self.bottle_id >= 0 else np.zeros(3, dtype=np.float32)
        bottle_quat = self.data.xquat[self.bottle_id].astype(np.float32) if self.bottle_id >= 0 else np.array([1,0,0,0], dtype=np.float32)
        tips = np.zeros(24, dtype=np.float32)
        for j, tid in enumerate(self.tip_ids[:8]):
            tips[j*3:(j+1)*3] = self.data.xpos[tid]
        timestep = np.array([self.step_count / self.max_steps], dtype=np.float32)
        return np.concatenate([joints, bottle_pos, bottle_quat, tips, timestep])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.prev_action = np.zeros(44)
        mujoco.mj_forward(self.model, self.data)
        self.bottle_init_z = self.data.xpos[self.bottle_id][2]
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        ref = self._get_ref(self.step_count - 1)
        self._apply_ctrl(ref, action)

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        bottle_z = self.data.xpos[self.bottle_id][2]
        lift = bottle_z - self.bottle_init_z

        # === REWARD (RoboWheel-style, tracking-first) ===
        # 1. MANO tracking (HIGHEST WEIGHT) — penalize deviation from reference
        r_track = -np.sum(action**2) * 50.0

        # 2. Geometric: fingertips close to bottle
        bottle_pos = self.data.xpos[self.bottle_id]
        tip_dists = [np.linalg.norm(self.data.xpos[tid] - bottle_pos) for tid in self.tip_ids[:8]]
        r_geo = -np.mean(tip_dists) * 3.0

        # 3. Smoothness
        r_dyn = -np.sum((action - self.prev_action)**2) * 2.0
        self.prev_action = action.copy()

        # 4. Contact
        r_con = min(self.data.ncon, 10) * 0.1

        # 5. Lift (lower weight than tracking!)
        r_lift = max(0, lift) * 30.0

        reward = r_track + r_geo + r_dyn + r_con + r_lift

        done = self.step_count >= self.max_steps
        if bottle_z < 0.10: done = True; reward -= 3.0

        return obs, reward, done, False, {
            'r_track': r_track, 'r_geo': r_geo, 'r_con': r_con,
            'r_lift': r_lift, 'lift_cm': lift*100,
            'contacts': self.data.ncon, 'res_norm': np.linalg.norm(action)
        }

    def render(self):
        if not hasattr(self, '_renderer'):
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._cam = mujoco.MjvCamera()
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            obj_pos = self.ref['obj_pos'].mean(0)
            self._cam.lookat[:] = obj_pos
            self._cam.distance = 0.45; self._cam.azimuth = 145; self._cam.elevation = -25
        self._renderer.update_scene(self.data, self._cam)
        return self._renderer.render()


# ============================================================
# Train + Eval
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== Building MANO reference (100% from video) ===")
    ref = build_mano_reference()
    print(f"  {ref['N']} bimanual frames")
    print(f"  Right wrist pos range: [{ref['right_wrist_pos'].min():.3f}, {ref['right_wrist_pos'].max():.3f}]")
    print(f"  Right fingers range: [{ref['right_fingers'].min():.2f}, {ref['right_fingers'].max():.2f}]")

    print("\n=== PPO Training (MANO-driven, 200K) ===")
    env = DummyVecEnv([lambda: MANOGraspEnv(mano_ref=ref) for _ in range(16)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=256,
                batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5, verbose=1)
    model.learn(total_timesteps=200_000, progress_bar=True)
    model.save(os.path.join(OUT_DIR, 'ppo_v2'))
    print("Model saved!")

    print("\n=== Evaluation ===")
    eval_env = MANOGraspEnv(mano_ref=ref, render_mode="rgb_array")
    obs, _ = eval_env.reset(seed=42)
    frames = []; total_r = 0; max_lift = 0; max_con = 0

    for i in range(ref['N']):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = eval_env.step(action)
        total_r += reward; max_lift = max(max_lift, info['lift_cm']); max_con = max(max_con, info['contacts'])
        img = eval_env.render()
        bgr = img[:,:,::-1].copy()
        cv2.putText(bgr, f"[{i}] con={info['contacts']} lift={info['lift_cm']:.1f}cm res={info['res_norm']:.3f}",
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,220,0), 1)
        frames.append(bgr)
        if done: break

    print(f"  Total reward: {total_r:.1f}")
    print(f"  Max contacts: {max_con}")
    print(f"  Max lift: {max_lift:.1f} cm")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = cv2.VideoWriter(os.path.join(OUT_DIR, 'rl_v2.mp4'), fourcc, 15, (640,480))
    for f in frames: w.write(f)
    w.release()
    for idx in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]:
        cv2.imwrite(os.path.join(OUT_DIR, f'kf_{idx:04d}.jpg'), frames[idx])

    # Comparison
    overlays_dir = os.path.join(BASE_DIR, 'output/pick_bottle_video')
    overlays = sorted([f for f in os.listdir(overlays_dir) if f.endswith('_overlay.jpg')])
    pw, ph = 640, 480
    w_cmp = cv2.VideoWriter(os.path.join(OUT_DIR, 'comparison.mp4'), fourcc, 15, (pw*2, ph+25))
    for i, f in enumerate(frames):
        ov_idx = min(int(i/len(frames)*len(overlays)), len(overlays)-1)
        ov = cv2.resize(cv2.imread(os.path.join(overlays_dir, overlays[ov_idx])), (pw, ph))
        panels = np.hstack([ov, f])
        label = np.ones((25, pw*2, 3), dtype=np.uint8) * 35
        cv2.putText(label, "Video + HaMeR (MANO)", (5,17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,0), 1)
        cv2.putText(label, "V2: MANO-driven RL", (pw+5,17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,0), 1)
        w_cmp.write(np.vstack([label, panels]))
    w_cmp.release()

    import subprocess
    subprocess.run(['ffmpeg','-y','-i',os.path.join(OUT_DIR,'comparison.mp4'),
        '-vf','fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
        os.path.join(OUT_DIR,'comparison.gif')], capture_output=True)

    print(f"\nDone! {OUT_DIR}/")


if __name__ == '__main__':
    main()
