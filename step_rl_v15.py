"""
RoboWheel V1.5: Hybrid approach
  - Base trajectory: staged approach/lift (V1, physically stable)
  - Finger curl: from MANO hand_pose (V2, visually correct)
  - Orientation: fixed horizontal + 10% MANO delta (stable)
  - RL residual: ±0.05rad (physics micro-adjustment)

This combines V1's physical stability with V2's MANO-driven finger poses.
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
OUT_DIR = os.path.join(BASE_DIR, 'output/rl_v15')
MANO_PATH = os.path.join(BASE_DIR, 'output/pick_bottle_video/mano_results.json')


def build_mano_fingers():
    """Extract ONLY finger angles from MANO (not position/orientation)."""
    import smplx
    mano = smplx.create('/mnt/users/yjy/sim/video2robot-retarget/HaWoR/_DATA/data',
                         model_type='mano', is_rhand=True, use_pca=False, flat_hand_mean=False)

    with open(MANO_PATH) as f:
        results = json.load(f)

    # Get bimanual frames
    frame_dict = {}
    for r in results:
        key = r['img_name']
        if key not in frame_dict: frame_dict[key] = {}
        frame_dict[key]['right' if r['is_right'] else 'left'] = r
    both = [v for v in frame_dict.values() if 'left' in v and 'right' in v]
    N = len(both)

    # MANO→Allegro finger mapping
    m2a = {0:1, 1:2, 2:3, 3:5, 4:6, 5:7, 9:9, 10:10, 11:11, 12:13, 13:14, 14:15}

    right_fingers = np.zeros((N, 16))
    left_fingers = np.zeros((N, 16))

    for i, data in enumerate(both):
        for side, arr in [('right', right_fingers), ('left', left_fingers)]:
            hp = np.array(data[side]['mano_hand_pose'])  # (15, 3, 3)
            fingers = np.zeros(16)
            for mano_j, allegro_j in m2a.items():
                euler = Rot.from_matrix(hp[mano_j]).as_euler('xyz')
                fingers[allegro_j] = euler[0]  # flexion
            # Spread joints
            for mano_j, allegro_s in [(0, 0), (3, 4), (9, 8)]:
                fingers[allegro_s] = Rot.from_matrix(hp[mano_j]).as_euler('xyz')[2]
            # Thumb rotation
            fingers[12] = abs(Rot.from_matrix(hp[12]).as_euler('xyz')[0]) * 0.8 + 0.3
            arr[i] = fingers

    # Smooth and clamp to safe Allegro ranges
    right_fingers = uniform_filter1d(right_fingers, size=7, axis=0)
    left_fingers = uniform_filter1d(left_fingers, size=7, axis=0)
    # Clamp: Allegro ranges are roughly [-0.47, 1.72]
    # But be more conservative to avoid instability
    right_fingers = np.clip(right_fingers, -0.3, 1.5)
    left_fingers = np.clip(left_fingers, -0.3, 1.5)

    print(f"  MANO fingers: {N} bimanual frames")
    print(f"  Right range: [{right_fingers.min():.2f}, {right_fingers.max():.2f}]")
    print(f"  Left range:  [{left_fingers.min():.2f}, {left_fingers.max():.2f}]")

    return N, right_fingers, left_fingers


class HybridGraspEnv(gym.Env):
    """V1.5: staged base (V1) + MANO fingers (V2) + small RL residual."""
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, mano_data=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        if mano_data is not None:
            self.N_mano, self.right_fingers, self.left_fingers = mano_data
        else:
            self.N_mano, self.right_fingers, self.left_fingers = build_mano_fingers()

        self.max_steps = 200
        self.model, self.data = self._build()

        # Action: small residual for 32 finger joints only
        # Base trajectory is NOT adjustable by RL (staged, physically stable)
        self.action_space = spaces.Box(-0.05, 0.05, shape=(32,), dtype=np.float32)

        obs_dim = 32 + 6 + 3 + 4 + 24 + 1  # 70
        self.observation_space = spaces.Box(-10, 10, shape=(obs_dim,), dtype=np.float32)

        self._find_actuators()
        self.step_count = 0
        self.prev_action = np.zeros(32)
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
        l2 = w.add_light(); l2.pos=[0.3,0.2,0.4]; l2.dir=[-0.2,-0.1,-0.5]; l2.diffuse=[0.4,0.4,0.4]

        f = w.add_geom(); f.type=mujoco.mjtGeom.mjGEOM_PLANE; f.size=[0.5,0.5,0.01]
        f.rgba=[0.92,0.92,0.92,1]
        t = w.add_geom(); t.type=mujoco.mjtGeom.mjGEOM_BOX; t.size=[0.18,0.14,0.01]
        t.pos=[0,0,0.15]; t.rgba=[0.38,0.28,0.20,1]; t.friction=[1.0,0.005,0.0001]

        # Bottle (physics)
        bottle = w.add_body(); bottle.name="bottle"; bottle.pos=[0,0,0.26]
        bj = bottle.add_freejoint(); bj.name="bottle_joint"
        bg = bottle.add_geom(); bg.type=mujoco.mjtGeom.mjGEOM_CYLINDER
        bg.size=[0.030,0.07,0]; bg.rgba=[0.15,0.50,0.85,0.85]
        bg.mass=0.20; bg.friction=[2.0,0.01,0.001]

        # Right hand (fingers point LEFT toward bottle)
        br = w.add_body(); br.name="base_right"; br.pos=[0.10,0,0.30]
        for jn,ax in [("rx",[1,0,0]),("ry",[0,1,0]),("rz",[0,0,1])]:
            j=br.add_joint(); j.name=jn; j.type=mujoco.mjtJoint.mjJNT_SLIDE
            j.axis=ax; j.range=[-0.12,0.12]
        mr = br.add_body(); mr.name="mount_right"
        # Fixed horizontal: fingers point left (Z=π), palm tilted inward
        rot_r = Rot.from_euler('zy', [180, -15], degrees=True)
        mr.quat = rot_r.as_quat(scalar_first=True).tolist()
        mr.add_frame().attach_body(hr.worldbody.first_body(), "rh_", "")

        # Left hand (fingers point RIGHT toward bottle)
        bl = w.add_body(); bl.name="base_left"; bl.pos=[-0.10,0,0.30]
        for jn,ax in [("lx",[1,0,0]),("ly",[0,1,0]),("lz",[0,0,1])]:
            j=bl.add_joint(); j.name=jn; j.type=mujoco.mjtJoint.mjJNT_SLIDE
            j.axis=ax; j.range=[-0.12,0.12]
        ml = bl.add_body(); ml.name="mount_left"
        rot_l = Rot.from_euler('y', 15, degrees=True)
        ml.quat = rot_l.as_quat(scalar_first=True).tolist()
        ml.add_frame().attach_body(hl.worldbody.first_body(), "lh_", "")

        # Base actuators (high gain for stable tracking)
        for jn in ["rx","ry","rz","lx","ly","lz"]:
            a=s.add_actuator(); a.name=f"act_{jn}"; a.target=jn
            a.trntype=mujoco.mjtTrn.mjTRN_JOINT; a.gainprm=[500]+[0]*9

        model = s.compile()
        return model, mujoco.MjData(model)

    def _find_actuators(self):
        self.base_acts = {}
        self.rh_acts = []; self.lh_acts = []
        self.tip_ids = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            if name.startswith("act_"): self.base_acts[name] = i
            elif "rh_" in name: self.rh_acts.append(i)
            elif "lh_" in name: self.lh_acts.append(i)
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if "tip" in name and ("rh_" in name or "lh_" in name):
                self.tip_ids.append(i)
        self.bottle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'bottle')

    def _get_staged_base(self, t):
        """V1-style staged base trajectory (physically stable)."""
        base = np.zeros(6)  # rx,ry,rz, lx,ly,lz
        if t < 0.25:
            p = t / 0.25
            base[0] = -0.06 * p   # right moves left toward bottle
            base[3] = 0.06 * p    # left moves right
        elif t < 0.65:
            base[0] = -0.06
            base[3] = 0.06
        else:
            lp = (t - 0.65) / 0.35
            base[0] = -0.06
            base[3] = 0.06
            base[2] = 0.06 * lp   # right lifts
            base[5] = 0.06 * lp   # left lifts
        return base

    def _get_mano_fingers(self, frame):
        """V2-style MANO finger angles."""
        fi = min(frame, self.N_mano - 1)
        return self.right_fingers[fi], self.left_fingers[fi]

    def _get_obs(self):
        joints = np.zeros(32, dtype=np.float32)
        for j, ai in enumerate(self.rh_acts[:16]):
            jid = self.model.actuator_trnid[ai, 0]
            if jid >= 0: joints[j] = self.data.qpos[self.model.jnt_qposadr[jid]]
        for j, ai in enumerate(self.lh_acts[:16]):
            jid = self.model.actuator_trnid[ai, 0]
            if jid >= 0: joints[16+j] = self.data.qpos[self.model.jnt_qposadr[jid]]

        base = np.zeros(6, dtype=np.float32)
        bottle_pos = self.data.xpos[self.bottle_id].astype(np.float32)
        bottle_quat = self.data.xquat[self.bottle_id].astype(np.float32)
        tips = np.zeros(24, dtype=np.float32)
        for j, tid in enumerate(self.tip_ids[:8]):
            tips[j*3:(j+1)*3] = self.data.xpos[tid]
        timestep = np.array([self.step_count / self.max_steps], dtype=np.float32)
        return np.concatenate([joints, base, bottle_pos, bottle_quat, tips, timestep])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.prev_action = np.zeros(32)
        if seed is not None:
            rng = np.random.RandomState(seed)
            self.data.qpos[0] += rng.uniform(-0.005, 0.005)
            self.data.qpos[1] += rng.uniform(-0.005, 0.005)
        mujoco.mj_forward(self.model, self.data)
        self.bottle_init_z = self.data.xpos[self.bottle_id][2]
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        t = (self.step_count - 1) / max(self.max_steps - 1, 1)

        # === BASE: V1 staged trajectory (NOT RL-adjustable) ===
        base = self._get_staged_base(t)
        for idx, name in enumerate(["act_rx","act_ry","act_rz","act_lx","act_ly","act_lz"]):
            if name in self.base_acts:
                self.data.ctrl[self.base_acts[name]] = base[idx]

        # === FINGERS: MANO reference + RL residual ===
        mano_frame = min(int(t * self.N_mano), self.N_mano - 1)
        r_ref, l_ref = self._get_mano_fingers(mano_frame)

        for j, ai in enumerate(self.rh_acts[:16]):
            jid = self.model.actuator_trnid[ai, 0]
            if jid >= 0:
                jr = self.model.jnt_range[jid]
                self.data.ctrl[ai] = np.clip(r_ref[j] + action[j], jr[0], jr[1])

        for j, ai in enumerate(self.lh_acts[:16]):
            jid = self.model.actuator_trnid[ai, 0]
            if jid >= 0:
                jr = self.model.jnt_range[jid]
                self.data.ctrl[ai] = np.clip(l_ref[j] + action[16+j], jr[0], jr[1])

        # Physics
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        bottle_z = self.data.xpos[self.bottle_id][2]
        lift = bottle_z - self.bottle_init_z
        bottle_pos = self.data.xpos[self.bottle_id]

        # === REWARD ===
        # 1. MANO tracking (highest priority)
        r_track = -np.sum(action**2) * 50.0

        # 2. Fingertips close to bottle
        tip_dists = [np.linalg.norm(self.data.xpos[tid] - bottle_pos) for tid in self.tip_ids[:8]]
        r_geo = -np.mean(tip_dists) * 3.0

        # 3. Smoothness
        r_dyn = -np.sum((action - self.prev_action)**2) * 2.0
        self.prev_action = action.copy()

        # 4. Contact
        r_con = min(self.data.ncon, 10) * 0.15

        # 5. Lift (moderate weight — don't sacrifice tracking for lifting)
        r_lift = max(0, lift) * 50.0

        # 6. Stability
        r_stable = -max(0, self.bottle_init_z - bottle_z) * 5.0

        reward = r_track + r_geo + r_dyn + r_con + r_lift + r_stable

        done = self.step_count >= self.max_steps
        if bottle_z < 0.10: done = True; reward -= 3.0

        return obs, reward, done, False, {
            'r_track': r_track, 'r_geo': r_geo, 'r_con': r_con,
            'r_lift': r_lift, 'lift_cm': lift*100,
            'contacts': self.data.ncon, 'res_norm': np.linalg.norm(action),
        }

    def render(self):
        if not hasattr(self, '_renderer'):
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._cam = mujoco.MjvCamera()
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._cam.lookat[:] = [0, 0, 0.28]
            self._cam.distance = 0.50; self._cam.azimuth = 145; self._cam.elevation = -25
        self._renderer.update_scene(self.data, self._cam)
        return self._renderer.render()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== V1.5: Hybrid (staged base + MANO fingers + RL residual) ===")
    print("Building MANO finger reference...")
    mano_data = build_mano_fingers()

    print("\nTraining PPO (200K, 16 envs, VecNormalize)...")
    env = DummyVecEnv([lambda: HybridGraspEnv(mano_data=mano_data) for _ in range(16)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=256,
                batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5, verbose=1)
    model.learn(total_timesteps=200_000, progress_bar=True)
    model.save(os.path.join(OUT_DIR, 'ppo_v15'))
    print("Model saved!")

    print("\n=== Evaluation ===")
    eval_env = HybridGraspEnv(mano_data=mano_data, render_mode="rgb_array")
    obs, _ = eval_env.reset(seed=42)
    frames = []; total_r = 0; max_lift = 0; max_con = 0

    for i in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = eval_env.step(action)
        total_r += reward; max_lift = max(max_lift, info['lift_cm']); max_con = max(max_con, info['contacts'])

        img = eval_env.render()
        bgr = img[:,:,::-1].copy()
        t = i / 199
        phase = "APPROACH" if t<0.25 else "CLOSE" if t<0.50 else "GRASP" if t<0.65 else "LIFT"
        cv2.putText(bgr, f"[{phase}] {i} con={info['contacts']} lift={info['lift_cm']:.1f}cm",
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,220,0), 1)
        cv2.putText(bgr, f"res={info['res_norm']:.3f} track={info['r_track']:.1f}",
                    (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,180,255), 1)
        frames.append(bgr)
        if done: break

    print(f"  Total reward: {total_r:.1f}")
    print(f"  Max contacts: {max_con}")
    print(f"  Max lift: {max_lift:.1f} cm")
    print(f"  Frames rendered: {len(frames)}")

    # Save videos
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = cv2.VideoWriter(os.path.join(OUT_DIR, 'rl_v15.mp4'), fourcc, 15, (640,480))
    for f in frames: w.write(f)
    w.release()

    for idx in [0, len(frames)//5, 2*len(frames)//5, 3*len(frames)//5, 4*len(frames)//5, len(frames)-1]:
        if idx < len(frames):
            cv2.imwrite(os.path.join(OUT_DIR, f'kf_{idx:04d}.jpg'), frames[idx])

    # Comparison with HaMeR
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
        cv2.putText(label, "V1.5: Staged Base + MANO Fingers + RL", (pw+5,17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,220,0), 1)
        w_cmp.write(np.vstack([label, panels]))
    w_cmp.release()

    import subprocess
    subprocess.run(['ffmpeg','-y','-i',os.path.join(OUT_DIR,'comparison.mp4'),
        '-vf','fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
        os.path.join(OUT_DIR,'comparison.gif')], capture_output=True)

    print(f"\nDone! {OUT_DIR}/")
    for fn in sorted(os.listdir(OUT_DIR)):
        print(f"  {fn} ({os.path.getsize(os.path.join(OUT_DIR, fn))/1024:.0f} KB)")


if __name__ == '__main__':
    main()
