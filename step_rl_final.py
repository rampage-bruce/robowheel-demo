"""
Final pipeline: Arti-MANO hand + SPIDER IK base trajectory + PPO residual.

- Base position/rotation: from SPIDER IK (world-space, physically valid)
- Finger angles: from MANO 1:1 direct mapping to Arti-MANO (visually correct)
- RL: ±0.05 rad residual on fingers only
- Physics: MuJoCo contact, bottle with freejoint
"""
import os, json, sys
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
import cv2
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import uniform_filter1d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_MANO = os.path.join(BASE_DIR, 'spider/spider/assets/robots/mano')
MENAGERIE = os.path.join(BASE_DIR, 'mujoco_menagerie')
OUT_DIR = os.path.join(BASE_DIR, 'output/rl_final')
MANO_PATH = os.path.join(BASE_DIR, 'output/pick_bottle_video/mano_results.json')
# Use Arti-MANO IK (not Allegro!) — coordinates are correct
SPIDER_IK = os.path.join(BASE_DIR, 'spider/example_datasets/processed/hamer_demo/mano/right/pick_bottle/0/trajectory_kinematic.npz')


def load_references():
    """Load SPIDER IK (base) + MANO (fingers)."""

    # SPIDER IK: base position + rotation (world-space)
    ik = np.load(SPIDER_IK)
    ik_qpos = ik['qpos']  # (147, 35): [base_pos(3), base_rot(3), fingers(22), object(7)]
    N_ik = ik_qpos.shape[0]

    # Arti-MANO IK: coordinates are already correct (no Allegro migration needed!)
    spider_hand = ik_qpos[:, :3]
    spider_obj = ik_qpos[:, 28:31]  # object pos at index 28 (not 22)
    spider_rot = ik_qpos[:, 3:6]
    spider_fingers = ik_qpos[:, 6:28]  # 22 Arti-MANO finger DOF

    # Our scene: bottle at [0, 0, 0.26]
    OUR_OBJ = np.array([0, 0, 0.26])

    # Transform hand position: keep hand-object relative offset, re-center on our bottle
    base_pos = (spider_hand - spider_obj) + OUR_OBJ
    base_rot = spider_rot

    # Smooth
    base_pos = uniform_filter1d(base_pos, size=5, axis=0)
    base_rot = uniform_filter1d(base_rot, size=5, axis=0)

    # Finger angles: directly from SPIDER IK (already solved for Arti-MANO!)
    # No need for manual MANO→Arti-MANO euler mapping — IK did it properly
    right_fingers = spider_fingers  # (N_ik, 22) — IK-solved finger joints
    left_fingers = spider_fingers.copy()  # mirror for left hand
    left_fingers = uniform_filter1d(left_fingers, size=5, axis=0)
    right_fingers = uniform_filter1d(right_fingers, size=5, axis=0)

    N = N_ik
    print(f"  IK frames: {N_ik}, using: {N}")
    print(f"  Base pos range: [{base_pos[:N].min():.3f}, {base_pos[:N].max():.3f}]")
    print(f"  Base rot range: [{base_rot[:N].min():.3f}, {base_rot[:N].max():.3f}]")
    print(f"  Right fingers range: [{right_fingers[:N].min():.2f}, {right_fingers[:N].max():.2f}]")

    return N, base_pos[:N], base_rot[:N], right_fingers[:N], left_fingers[:N]


class FinalGraspEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, ref_data=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        if ref_data:
            self.N, self.base_pos, self.base_rot, self.r_fingers, self.l_fingers = ref_data
        else:
            self.N, self.base_pos, self.base_rot, self.r_fingers, self.l_fingers = load_references()

        self.max_steps = self.N
        self.model, self.data = self._build()

        # RL residual: 44 finger DOF (22 per hand), small range
        self.action_space = spaces.Box(-0.08, 0.08, shape=(44,), dtype=np.float32)
        obs_dim = 44 + 6 + 7 + 30 + 1
        self.observation_space = spaces.Box(-10, 10, shape=(obs_dim,), dtype=np.float32)

        self._find_ids()
        self.step_count = 0
        self.prev_action = np.zeros(44)
        self.bottle_init_z = 0

    def _build(self):
        os.chdir(SPIDER_MANO)
        hr = mujoco.MjSpec.from_file('right.xml')
        hl = mujoco.MjSpec.from_file('left.xml')

        s = mujoco.MjSpec()
        s.option.gravity = [0, 0, -9.81]; s.option.timestep = 0.002
        s.option.impratio = 10; s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        w = s.worldbody

        l = w.add_light(); l.pos=[0,-0.3,0.5]; l.dir=[0,0.2,-1]; l.diffuse=[1,1,1]
        l2 = w.add_light(); l2.pos=[0.3,0.2,0.4]; l2.dir=[-0.2,-0.1,-0.5]; l2.diffuse=[0.4,0.4,0.4]

        f = w.add_geom(); f.type=mujoco.mjtGeom.mjGEOM_PLANE; f.size=[0.5,0.5,0.01]; f.rgba=[0.92,0.92,0.92,1]
        t = w.add_geom(); t.type=mujoco.mjtGeom.mjGEOM_BOX; t.size=[0.18,0.14,0.01]
        t.pos=[0,0,0.15]; t.rgba=[0.38,0.28,0.20,1]

        # Bottle
        bottle = w.add_body(); bottle.name="bottle"; bottle.pos=[0,0,0.26]
        bj = bottle.add_freejoint(); bj.name="bottle_joint"
        bg = bottle.add_geom(); bg.type=mujoco.mjtGeom.mjGEOM_CYLINDER
        bg.size=[0.030,0.07,0]; bg.rgba=[0.15,0.50,0.85,0.85]; bg.mass=0.20; bg.friction=[2.0,0.01,0.001]

        # Right hand: base from SPIDER IK (transformed to our scene)
        init_pos = self.base_pos[0].tolist()
        print(f"  Right hand init: {[round(x,3) for x in init_pos]}")
        br = w.add_body(); br.name="base_right"; br.pos=init_pos
        bm = br.add_geom(); bm.type=mujoco.mjtGeom.mjGEOM_SPHERE; bm.size=[0.001,0,0]
        bm.mass=0.1; bm.rgba=[0,0,0,0]; bm.contype=0; bm.conaffinity=0
        for jn,ax in [("rx",[1,0,0]),("ry",[0,1,0]),("rz",[0,0,1])]:
            j=br.add_joint(); j.name=jn; j.type=mujoco.mjtJoint.mjJNT_SLIDE; j.axis=ax; j.range=[-0.5,0.5]
        for jn,ax in [("rrx",[1,0,0]),("rry",[0,1,0]),("rrz",[0,0,1])]:
            j=br.add_joint(); j.name=jn; j.type=mujoco.mjtJoint.mjJNT_HINGE; j.axis=ax; j.range=[-3.14,3.14]
        mr = br.add_body(); mr.name="mount_right"; mr.quat=[1,0,0,0]
        mr.add_frame().attach_body(hr.worldbody.first_body(), "rh_", "")

        # Left hand: mirror position
        init_pos_l = [-init_pos[0], init_pos[1], init_pos[2]]
        bl = w.add_body(); bl.name="base_left"; bl.pos=init_pos_l
        bml = bl.add_geom(); bml.type=mujoco.mjtGeom.mjGEOM_SPHERE; bml.size=[0.001,0,0]
        bml.mass=0.1; bml.rgba=[0,0,0,0]; bml.contype=0; bml.conaffinity=0
        for jn,ax in [("lx",[1,0,0]),("ly",[0,1,0]),("lz",[0,0,1])]:
            j=bl.add_joint(); j.name=jn; j.type=mujoco.mjtJoint.mjJNT_SLIDE; j.axis=ax; j.range=[-0.5,0.5]
        for jn,ax in [("lrx",[1,0,0]),("lry",[0,1,0]),("lrz",[0,0,1])]:
            j=bl.add_joint(); j.name=jn; j.type=mujoco.mjtJoint.mjJNT_HINGE; j.axis=ax; j.range=[-3.14,3.14]
        ml = bl.add_body(); ml.name="mount_left"; ml.quat=[1,0,0,0]
        ml.add_frame().attach_body(hl.worldbody.first_body(), "lh_", "")

        # Base actuators (very high gain — must track SPIDER IK precisely)
        for jn in ["rx","ry","rz","rrx","rry","rrz","lx","ly","lz","lrx","lry","lrz"]:
            a=s.add_actuator(); a.name=f"act_{jn}"; a.target=jn
            a.trntype=mujoco.mjtTrn.mjTRN_JOINT; a.gainprm=[800]+[0]*9

        model = s.compile()
        return model, mujoco.MjData(model)

    def _find_ids(self):
        self.base_acts = {}; self.rh_acts = []; self.lh_acts = []; self.tip_ids = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            if name.startswith("act_"): self.base_acts[name] = i
            elif "rh_" in name: self.rh_acts.append(i)
            elif "lh_" in name: self.lh_acts.append(i)
        self.bottle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'bottle')
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if name.endswith("3") and any(f in name for f in ["index","middle","pinky","ring","thumb"]):
                self.tip_ids.append(i)

    def _get_obs(self):
        joints = np.zeros(44, dtype=np.float32)
        for j, ai in enumerate((self.rh_acts + self.lh_acts)[:44]):
            jid = self.model.actuator_trnid[ai, 0]
            if jid >= 0: joints[j] = self.data.qpos[self.model.jnt_qposadr[jid]]
        base = np.zeros(6, dtype=np.float32)
        bp = self.data.xpos[self.bottle_id].astype(np.float32)
        bq = self.data.xquat[self.bottle_id].astype(np.float32)
        tips = np.zeros(30, dtype=np.float32)
        for j, tid in enumerate(self.tip_ids[:10]):
            tips[j*3:(j+1)*3] = self.data.xpos[tid]
        ts = np.array([self.step_count / self.max_steps], dtype=np.float32)
        return np.concatenate([joints, base, bp, bq, tips, ts])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0; self.prev_action = np.zeros(44)
        mujoco.mj_forward(self.model, self.data)
        self.bottle_init_z = self.data.xpos[self.bottle_id][2]
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        fi = min(self.step_count - 1, self.N - 1)

        # === BASE: from SPIDER IK (world-space, accurate) ===
        # SPIDER IK base: delta from initial, scaled to move TOWARD bottle
        bp = self.base_pos[fi] - self.base_pos[0]
        br = self.base_rot[fi] - self.base_rot[0]
        bp = np.clip(bp, -0.08, 0.08)
        br = np.clip(br, -0.3, 0.3)

        # No staged approach needed — SPIDER IK already positions hand correctly
        for idx, name in enumerate(["act_rx","act_ry","act_rz"]):
            if name in self.base_acts:
                self.data.ctrl[self.base_acts[name]] = bp[idx]
        for idx, name in enumerate(["act_rrx","act_rry","act_rrz"]):
            if name in self.base_acts: self.data.ctrl[self.base_acts[name]] = br[idx]

        # Left: mirror X
        for idx, name in enumerate(["act_lx","act_ly","act_lz"]):
            if name in self.base_acts:
                val = -bp[idx] if idx == 0 else bp[idx]
                self.data.ctrl[self.base_acts[name]] = val
        for idx, name in enumerate(["act_lrx","act_lry","act_lrz"]):
            if name in self.base_acts: self.data.ctrl[self.base_acts[name]] = -br[idx]

        # No manual lift — SPIDER IK trajectory includes the correct motion

        # === FINGERS: MANO 1:1 + RL residual ===
        n_rh = min(len(self.rh_acts), 22)
        n_lh = min(len(self.lh_acts), 22)
        for j in range(n_rh):
            ai = self.rh_acts[j]
            jid = self.model.actuator_trnid[ai, 0]
            if jid >= 0:
                jr = self.model.jnt_range[jid]
                ref = self.r_fingers[fi, j] if j < 22 else 0
                res = action[j] if j < 44 else 0
                self.data.ctrl[ai] = np.clip(ref + res, jr[0], jr[1])
        for j in range(n_lh):
            ai = self.lh_acts[j]
            jid = self.model.actuator_trnid[ai, 0]
            if jid >= 0:
                jr = self.model.jnt_range[jid]
                ref = self.l_fingers[fi, j] if j < 22 else 0
                res = action[22+j] if 22+j < 44 else 0
                self.data.ctrl[ai] = np.clip(ref + res, jr[0], jr[1])

        for _ in range(10):
            try:
                mujoco.mj_step(self.model, self.data)
            except mujoco.FatalError:
                mujoco.mj_resetData(self.model, self.data)
                break

        obs = self._get_obs()
        bz = self.data.xpos[self.bottle_id][2]; lift = bz - self.bottle_init_z
        bpos = self.data.xpos[self.bottle_id]

        r_track = -np.sum(action**2) * 10.0
        td = [np.linalg.norm(self.data.xpos[tid]-bpos) for tid in self.tip_ids[:10]]
        r_geo = -np.mean(td) * 5.0 if td else 0
        r_dyn = -np.sum((action - self.prev_action)**2) * 2.0; self.prev_action = action.copy()
        r_con = min(self.data.ncon, 15) * 0.2
        r_lift = max(0, lift) * 100.0
        r_stable = -max(0, self.bottle_init_z - bz) * 5.0
        reward = r_track + r_geo + r_dyn + r_con + r_lift + r_stable

        done = self.step_count >= self.max_steps
        if bz < 0.10: done = True; reward -= 3.0

        return obs, reward, done, False, {
            'r_track': r_track, 'lift_cm': lift*100,
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
    print("=== Final: SPIDER IK base + Arti-MANO fingers + PPO ===")
    ref = load_references()

    print("\nTraining PPO (200K)...")
    env = DummyVecEnv([lambda: FinalGraspEnv(ref_data=ref) for _ in range(16)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=256, batch_size=256,
                n_epochs=10, gamma=0.99, clip_range=0.2, ent_coef=0.005, verbose=1)
    model.learn(total_timesteps=200_000, progress_bar=True)
    model.save(os.path.join(OUT_DIR, 'ppo_final'))

    print("\n=== Evaluation ===")
    ev = FinalGraspEnv(ref_data=ref, render_mode="rgb_array")
    obs, _ = ev.reset(seed=42)
    frames = []; total_r = 0; mx_lift = 0; mx_con = 0
    for i in range(ref[0]):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, info = ev.step(a)
        total_r += r; mx_lift = max(mx_lift, info['lift_cm']); mx_con = max(mx_con, info['contacts'])
        img = ev.render(); bgr = img[:,:,::-1].copy()
        t = i/(ref[0]-1); ph = "APPROACH" if t<0.25 else "CLOSE" if t<0.50 else "GRASP" if t<0.65 else "LIFT"
        cv2.putText(bgr, f"[{ph}] {i} con={info['contacts']} lift={info['lift_cm']:.1f}cm res={info['res_norm']:.3f}",
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0,220,0), 1)
        frames.append(bgr)
        if done: break

    print(f"  Reward: {total_r:.1f}, Contacts: {mx_con}, Lift: {mx_lift:.1f}cm, Frames: {len(frames)}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = cv2.VideoWriter(os.path.join(OUT_DIR, 'final_grasp.mp4'), fourcc, 15, (640,480))
    for f in frames: w.write(f)
    w.release()
    for idx in [0, len(frames)//5, 2*len(frames)//5, 3*len(frames)//5, 4*len(frames)//5, len(frames)-1]:
        if idx < len(frames):
            cv2.imwrite(os.path.join(OUT_DIR, f'kf_{idx:04d}.jpg'), frames[idx])

    # Comparison
    ov_dir = os.path.join(BASE_DIR, 'output/pick_bottle_video')
    ovs = sorted([f for f in os.listdir(ov_dir) if f.endswith('_overlay.jpg')])
    pw, ph = 640, 480
    wc = cv2.VideoWriter(os.path.join(OUT_DIR, 'comparison.mp4'), fourcc, 15, (pw*2, ph+25))
    for i, f in enumerate(frames):
        oi = min(int(i/len(frames)*len(ovs)), len(ovs)-1)
        ov = cv2.resize(cv2.imread(os.path.join(ov_dir, ovs[oi])), (pw, ph))
        panels = np.hstack([ov, f])
        lb = np.ones((25, pw*2, 3), dtype=np.uint8) * 35
        cv2.putText(lb, "Video + HaMeR", (5,17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,0), 1)
        cv2.putText(lb, "SPIDER IK + Arti-MANO + PPO", (pw+5,17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,220,0), 1)
        wc.write(np.vstack([lb, panels]))
    wc.release()

    import subprocess
    subprocess.run(['ffmpeg','-y','-i',os.path.join(OUT_DIR,'comparison.mp4'),
        '-vf','fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
        os.path.join(OUT_DIR,'comparison.gif')], capture_output=True)

    print(f"\nDone! {OUT_DIR}/")


if __name__ == '__main__':
    main()
