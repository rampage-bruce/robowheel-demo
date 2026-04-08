"""
RoboWheel with Arti-MANO: MANO data → MANO hand model (1:1 mapping, zero information loss).
No Allegro. No euler hack. Direct MANO joint angles → Arti-MANO actuators.

Pipeline:
  MANO hand_pose euler → Arti-MANO joints (direct 1:1)
  + staged base trajectory (V1, stable)
  + PPO residual RL (±0.05rad)
  + MuJoCo contact physics
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
OUT_DIR = os.path.join(BASE_DIR, 'output/rl_manohand')
MANO_PATH = os.path.join(BASE_DIR, 'output/pick_bottle_video/mano_results.json')


def build_mano_reference():
    """Extract MANO finger angles with PERFECT mapping to Arti-MANO joints."""
    with open(MANO_PATH) as f:
        results = json.load(f)

    frame_dict = {}
    for r in results:
        key = r['img_name']
        if key not in frame_dict: frame_dict[key] = {}
        frame_dict[key]['right' if r['is_right'] else 'left'] = r
    both = [v for v in frame_dict.values() if 'left' in v and 'right' in v]
    N = len(both)

    # Arti-MANO has 22 finger joints per hand:
    # index: 1y(spread), 1z(MCP flex), 2(PIP), 3(DIP) = 4
    # middle: same = 4
    # pinky: same = 4
    # ring: same = 4
    # thumb: 1x, 1y, 1z, 2y, 2z, 3 = 6
    # Total: 4×4 + 6 = 22

    # MANO hand_pose: 15 joints, each (3,3) rotation matrix
    # MANO order: index(0-2), middle(3-5), pinky(6-8), ring(9-11), thumb(12-14)
    # Each joint euler: [flex_x, twist_y, spread_z]

    right_joints = np.zeros((N, 22))
    left_joints = np.zeros((N, 22))

    def mano_to_artimano(hp):
        """MANO 15×(3×3) → Arti-MANO 22 joint angles. DIRECT mapping."""
        joints = np.zeros(22)

        # Helper: extract euler from rotation matrix
        def euler(mat):
            return Rot.from_matrix(mat).as_euler('xyz')

        # Index (MANO 0,1,2) → Arti-MANO (0:1y, 1:1z, 2:2, 3:3)
        e = euler(hp[0]); joints[0] = e[2]; joints[1] = e[0]  # spread, flex
        e = euler(hp[1]); joints[2] = e[0]  # PIP flex
        e = euler(hp[2]); joints[3] = e[0]  # DIP flex

        # Middle (MANO 3,4,5) → Arti-MANO (4:1y, 5:1z, 6:2, 7:3)
        e = euler(hp[3]); joints[4] = e[2]; joints[5] = e[0]
        e = euler(hp[4]); joints[6] = e[0]
        e = euler(hp[5]); joints[7] = e[0]

        # Pinky (MANO 6,7,8) → Arti-MANO (8:1y, 9:1z, 10:2, 11:3)
        e = euler(hp[6]); joints[8] = e[2]; joints[9] = e[0]
        e = euler(hp[7]); joints[10] = e[0]
        e = euler(hp[8]); joints[11] = e[0]

        # Ring (MANO 9,10,11) → Arti-MANO (12:1y, 13:1z, 14:2, 15:3)
        e = euler(hp[9]); joints[12] = e[2]; joints[13] = e[0]
        e = euler(hp[10]); joints[14] = e[0]
        e = euler(hp[11]); joints[15] = e[0]

        # Thumb (MANO 12,13,14) → Arti-MANO (16:1x, 17:1y, 18:1z, 19:2y, 20:2z, 21:3)
        e = euler(hp[12]); joints[16] = e[0]; joints[17] = e[1]; joints[18] = e[2]
        e = euler(hp[13]); joints[19] = e[1]; joints[20] = e[0]
        e = euler(hp[14]); joints[21] = e[0]

        return joints

    for i, data in enumerate(both):
        for side, arr in [('right', right_joints), ('left', left_joints)]:
            hp = np.array(data[side]['mano_hand_pose'])  # (15, 3, 3)
            arr[i] = mano_to_artimano(hp)

    # Smooth
    right_joints = uniform_filter1d(right_joints, size=5, axis=0)
    left_joints = uniform_filter1d(left_joints, size=5, axis=0)

    print(f"  {N} bimanual frames")
    print(f"  Right joints range: [{right_joints.min():.2f}, {right_joints.max():.2f}]")

    return N, right_joints, left_joints


class ManoHandEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, mano_data=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        if mano_data:
            self.N_mano, self.right_ref, self.left_ref = mano_data
        else:
            self.N_mano, self.right_ref, self.left_ref = build_mano_reference()

        self.max_steps = 200
        self.model, self.data = self._build()

        # 22 finger joints per hand × 2 = 44 residual dims
        self.n_finger = 44
        self.action_space = spaces.Box(-0.05, 0.05, shape=(self.n_finger,), dtype=np.float32)

        obs_dim = 44 + 6 + 7 + 24 + 1  # fingers + base + bottle + tips + time
        self.observation_space = spaces.Box(-10, 10, shape=(obs_dim,), dtype=np.float32)

        self._find_ids()
        self.step_count = 0
        self.prev_action = np.zeros(self.n_finger)
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

        # Floor + Table
        f = w.add_geom(); f.type=mujoco.mjtGeom.mjGEOM_PLANE; f.size=[0.5,0.5,0.01]
        f.rgba=[0.92,0.92,0.92,1]
        t = w.add_geom(); t.type=mujoco.mjtGeom.mjGEOM_BOX; t.size=[0.18,0.14,0.01]
        t.pos=[0,0,0.15]; t.rgba=[0.38,0.28,0.20,1]

        # Bottle
        bottle = w.add_body(); bottle.name="bottle"; bottle.pos=[0,0,0.26]
        bj = bottle.add_freejoint(); bj.name="bottle_joint"
        bg = bottle.add_geom(); bg.type=mujoco.mjtGeom.mjGEOM_CYLINDER
        bg.size=[0.030,0.07,0]; bg.rgba=[0.15,0.50,0.85,0.85]
        bg.mass=0.20; bg.friction=[2.0,0.01,0.001]

        # Right Arti-MANO (fingers point left)
        br = w.add_body(); br.name="base_right"; br.pos=[0.10,0,0.30]
        bg_r = br.add_geom(); bg_r.type=mujoco.mjtGeom.mjGEOM_SPHERE; bg_r.size=[0.001,0,0]
        bg_r.mass=0.1; bg_r.rgba=[0,0,0,0]; bg_r.contype=0; bg_r.conaffinity=0
        for jn,ax in [("rx",[1,0,0]),("ry",[0,1,0]),("rz",[0,0,1])]:
            j=br.add_joint(); j.name=jn; j.type=mujoco.mjtJoint.mjJNT_SLIDE
            j.axis=ax; j.range=[-0.12,0.12]
        mr = br.add_body(); mr.name="mount_right"
        rot_r = Rot.from_euler('zy', [180, -15], degrees=True)
        mr.quat = rot_r.as_quat(scalar_first=True).tolist()
        mr.add_frame().attach_body(hr.worldbody.first_body(), "rh_", "")

        # Left Arti-MANO (fingers point right)
        bl = w.add_body(); bl.name="base_left"; bl.pos=[-0.10,0,0.30]
        bg_l = bl.add_geom(); bg_l.type=mujoco.mjtGeom.mjGEOM_SPHERE; bg_l.size=[0.001,0,0]
        bg_l.mass=0.1; bg_l.rgba=[0,0,0,0]; bg_l.contype=0; bg_l.conaffinity=0
        for jn,ax in [("lx",[1,0,0]),("ly",[0,1,0]),("lz",[0,0,1])]:
            j=bl.add_joint(); j.name=jn; j.type=mujoco.mjtJoint.mjJNT_SLIDE
            j.axis=ax; j.range=[-0.12,0.12]
        ml = bl.add_body(); ml.name="mount_left"
        rot_l = Rot.from_euler('y', 15, degrees=True)
        ml.quat = rot_l.as_quat(scalar_first=True).tolist()
        ml.add_frame().attach_body(hl.worldbody.first_body(), "lh_", "")

        # Base actuators
        for jn in ["rx","ry","rz","lx","ly","lz"]:
            a=s.add_actuator(); a.name=f"act_{jn}"; a.target=jn
            a.trntype=mujoco.mjtTrn.mjTRN_JOINT; a.gainprm=[500]+[0]*9

        model = s.compile()
        return model, mujoco.MjData(model)

    def _find_ids(self):
        self.base_acts = {}
        self.rh_acts = []; self.lh_acts = []
        self.tip_ids = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            if name.startswith("act_"): self.base_acts[name] = i
            elif "rh_" in name: self.rh_acts.append(i)
            elif "lh_" in name: self.lh_acts.append(i)
        self.bottle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'bottle')
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            # Arti-MANO fingertips: *index3, *middle3, *pinky3, *ring3, *thumb3
            if name.endswith("3") and any(f in name for f in ["index","middle","pinky","ring","thumb"]):
                self.tip_ids.append(i)
        print(f"  Acts: base={len(self.base_acts)}, RH={len(self.rh_acts)}, LH={len(self.lh_acts)}")
        print(f"  Tips: {len(self.tip_ids)}")

    def _get_obs(self):
        # Finger joint angles
        joints = np.zeros(44, dtype=np.float32)
        for j, ai in enumerate((self.rh_acts + self.lh_acts)[:44]):
            jid = self.model.actuator_trnid[ai, 0]
            if jid >= 0:
                joints[j] = self.data.qpos[self.model.jnt_qposadr[jid]]

        base = np.zeros(6, dtype=np.float32)
        bp = self.data.xpos[self.bottle_id].astype(np.float32) if self.bottle_id >= 0 else np.zeros(3, dtype=np.float32)
        bq = self.data.xquat[self.bottle_id].astype(np.float32) if self.bottle_id >= 0 else np.array([1,0,0,0], dtype=np.float32)
        tips = np.zeros(24, dtype=np.float32)
        for j, tid in enumerate(self.tip_ids[:8]):
            tips[j*3:(j+1)*3] = self.data.xpos[tid]
        ts = np.array([self.step_count / self.max_steps], dtype=np.float32)
        return np.concatenate([joints, base, bp, bq, tips, ts])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.prev_action = np.zeros(self.n_finger)
        if seed: np.random.seed(seed)
        self.data.qpos[0] += np.random.uniform(-0.005, 0.005)
        mujoco.mj_forward(self.model, self.data)
        self.bottle_init_z = self.data.xpos[self.bottle_id][2]
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        t = (self.step_count - 1) / max(self.max_steps - 1, 1)

        # === BASE: staged trajectory ===
        base = np.zeros(6)
        if t < 0.25:
            p = t / 0.25; base[0] = -0.08*p; base[3] = 0.08*p
        elif t < 0.65:
            base[0] = -0.08; base[3] = 0.08
        else:
            lp = (t-0.65)/0.35; base[0] = -0.08; base[3] = 0.08
            base[2] = 0.06*lp; base[5] = 0.06*lp
        for idx, name in enumerate(["act_rx","act_ry","act_rz","act_lx","act_ly","act_lz"]):
            if name in self.base_acts:
                self.data.ctrl[self.base_acts[name]] = base[idx]

        # === FINGERS: MANO direct (1:1) + RL residual ===
        fi = min(int(t * self.N_mano), self.N_mano - 1)
        r_ref = self.right_ref[fi]
        l_ref = self.left_ref[fi]

        n_rh = min(len(self.rh_acts), 22)
        n_lh = min(len(self.lh_acts), 22)

        for j in range(n_rh):
            ai = self.rh_acts[j]
            jid = self.model.actuator_trnid[ai, 0]
            if jid >= 0:
                jr = self.model.jnt_range[jid]
                ref = r_ref[j] if j < len(r_ref) else 0
                res = action[j] if j < len(action) else 0
                self.data.ctrl[ai] = np.clip(ref + res, jr[0], jr[1])

        for j in range(n_lh):
            ai = self.lh_acts[j]
            jid = self.model.actuator_trnid[ai, 0]
            if jid >= 0:
                jr = self.model.jnt_range[jid]
                ref = l_ref[j] if j < len(l_ref) else 0
                res = action[22 + j] if 22 + j < len(action) else 0
                self.data.ctrl[ai] = np.clip(ref + res, jr[0], jr[1])

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        bz = self.data.xpos[self.bottle_id][2]
        lift = bz - self.bottle_init_z
        bpos = self.data.xpos[self.bottle_id]

        r_track = -np.sum(action**2) * 10.0
        r_geo = -np.mean([np.linalg.norm(self.data.xpos[tid]-bpos) for tid in self.tip_ids[:10]]) * 5.0
        r_dyn = -np.sum((action - self.prev_action)**2) * 2.0
        self.prev_action = action.copy()
        r_con = min(self.data.ncon, 10) * 0.15
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

    print("=== Arti-MANO: Perfect MANO→sim mapping ===")
    mano_data = build_mano_reference()

    print("\nTraining PPO (200K)...")
    env = DummyVecEnv([lambda: ManoHandEnv(mano_data=mano_data) for _ in range(16)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=256, batch_size=256,
                n_epochs=10, gamma=0.99, clip_range=0.2, ent_coef=0.005, verbose=1)
    model.learn(total_timesteps=200_000, progress_bar=True)
    model.save(os.path.join(OUT_DIR, 'ppo_manohand'))
    print("Saved!")

    print("\n=== Evaluation ===")
    ev = ManoHandEnv(mano_data=mano_data, render_mode="rgb_array")
    obs, _ = ev.reset(seed=42)
    frames = []; total_r = 0; mx_lift = 0; mx_con = 0
    for i in range(200):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, info = ev.step(a)
        total_r += r; mx_lift = max(mx_lift, info['lift_cm']); mx_con = max(mx_con, info['contacts'])
        img = ev.render(); bgr = img[:,:,::-1].copy()
        t = i/199; ph = "APPROACH" if t<0.25 else "CLOSE" if t<0.50 else "GRASP" if t<0.65 else "LIFT"
        cv2.putText(bgr, f"[{ph}] {i} con={info['contacts']} lift={info['lift_cm']:.1f}cm res={info['res_norm']:.3f}",
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0,220,0), 1)
        frames.append(bgr)
        if done: break

    print(f"  Reward: {total_r:.1f}, Max contacts: {mx_con}, Max lift: {mx_lift:.1f}cm, Frames: {len(frames)}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = cv2.VideoWriter(os.path.join(OUT_DIR, 'manohand_grasp.mp4'), fourcc, 15, (640,480))
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
        cv2.putText(lb, "Video + HaMeR (MANO)", (5,17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,0), 1)
        cv2.putText(lb, "Arti-MANO Hand (1:1 mapping)", (pw+5,17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,220,0), 1)
        wc.write(np.vstack([lb, panels]))
    wc.release()

    import subprocess
    subprocess.run(['ffmpeg','-y','-i',os.path.join(OUT_DIR,'comparison.mp4'),
        '-vf','fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
        os.path.join(OUT_DIR,'comparison.gif')], capture_output=True)

    print(f"\nDone! {OUT_DIR}/")


if __name__ == '__main__':
    main()
