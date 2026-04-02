"""
RoboWheel 3-layer approach:
  Layer 1: nvblox TSDF penetration removal (reference trajectory) — done previously
  Layer 2: PPO residual RL policy in MuJoCo (THIS FILE)
  Layer 3: Replay RL policy → physically valid grasp

Reward = λ_geo * tracking + λ_dyn * smoothness + λ_con * contact
"""
import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
import cv2
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MENAGERIE = os.path.join(BASE_DIR, 'mujoco_menagerie')
OUT_DIR = os.path.join(BASE_DIR, 'output/rl_grasp')


# ============================================================
# MuJoCo Grasp Environment
# ============================================================

class BimanualGraspEnv(gym.Env):
    """
    MuJoCo bimanual grasp environment for RL.

    Observation: [hand_joint_angles(32), hand_base_pos(6), bottle_pos(3),
                  bottle_quat(4), fingertip_positions(24), timestep(1)]
    Action: residual joint angle adjustments (32 finger joints)
    Reward: tracking + smoothness + contact consistency
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.model, self.data = self._build_scene()

        # Action: SMALL residual for 32 finger actuators (16 per hand)
        # ±0.05 rad max — RL can only micro-adjust, not override MANO reference
        self.n_finger_acts = 32
        self.action_space = spaces.Box(-0.05, 0.05, shape=(self.n_finger_acts,), dtype=np.float32)

        # Observation
        obs_dim = 32 + 6 + 3 + 4 + 24 + 1  # 70
        self.observation_space = spaces.Box(-10, 10, shape=(obs_dim,), dtype=np.float32)

        # Find actuator indices
        self.rh_finger_acts = []
        self.lh_finger_acts = []
        self.base_acts = {}
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            if "rh_" in name: self.rh_finger_acts.append(i)
            elif "lh_" in name: self.lh_finger_acts.append(i)
            elif "act_" in name: self.base_acts[name] = i

        # Find bodies
        self.bottle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'bottle')
        self.tip_ids = []
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if "tip" in name and ("rh_" in name or "lh_" in name):
                self.tip_ids.append(i)

        # Episode state
        self.step_count = 0
        self.max_steps = 200
        self.prev_action = np.zeros(self.n_finger_acts)
        self.bottle_init_z = 0

        # Reference trajectory (base trajectory — RL adds residuals)
        self._build_reference()

    def _build_scene(self):
        allegro_r = os.path.join(MENAGERIE, 'wonik_allegro/right_hand.xml')
        os.chdir(os.path.dirname(allegro_r))
        hand_r = mujoco.MjSpec.from_file('right_hand.xml')
        hand_l = mujoco.MjSpec.from_file('left_hand.xml')

        s = mujoco.MjSpec()
        s.option.gravity = [0, 0, -9.81]
        s.option.timestep = 0.002
        s.option.impratio = 10
        s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

        w = s.worldbody
        l = w.add_light(); l.pos = [0, -0.3, 0.5]; l.dir = [0, 0.2, -1]; l.diffuse = [1, 1, 1]

        # Floor + Table
        f = w.add_geom(); f.type = mujoco.mjtGeom.mjGEOM_PLANE
        f.size = [0.5, 0.5, 0.01]; f.rgba = [0.92, 0.92, 0.92, 1]
        t = w.add_geom(); t.type = mujoco.mjtGeom.mjGEOM_BOX
        t.size = [0.18, 0.14, 0.01]; t.pos = [0, 0, 0.15]
        t.rgba = [0.38, 0.28, 0.20, 1]; t.friction = [1.0, 0.005, 0.0001]

        # Bottle (physics)
        bottle = w.add_body(); bottle.name = "bottle"; bottle.pos = [0, 0, 0.26]
        bj = bottle.add_freejoint(); bj.name = "bottle_joint"
        bg = bottle.add_geom(); bg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        bg.size = [0.030, 0.07, 0]; bg.rgba = [0.15, 0.50, 0.85, 0.85]
        bg.mass = 0.20; bg.friction = [2.0, 0.01, 0.001]

        # Right hand
        br = w.add_body(); br.name = "base_right"; br.pos = [0.10, 0, 0.30]
        for jn, ax in [("rx",[1,0,0]),("ry",[0,1,0]),("rz",[0,0,1])]:
            j = br.add_joint(); j.name=jn; j.type=mujoco.mjtJoint.mjJNT_SLIDE
            j.axis=ax; j.range=[-0.12,0.12]
        mr = br.add_body(); mr.name = "mount_right"
        mr.quat = R.from_euler('z', 180, degrees=True).as_quat(scalar_first=True).tolist()
        mr.add_frame().attach_body(hand_r.worldbody.first_body(), "rh_", "")

        # Left hand
        bl = w.add_body(); bl.name = "base_left"; bl.pos = [-0.10, 0, 0.30]
        for jn, ax in [("lx",[1,0,0]),("ly",[0,1,0]),("lz",[0,0,1])]:
            j = bl.add_joint(); j.name=jn; j.type=mujoco.mjtJoint.mjJNT_SLIDE
            j.axis=ax; j.range=[-0.12,0.12]
        ml = bl.add_body(); ml.name = "mount_left"
        ml.quat = [1,0,0,0]
        ml.add_frame().attach_body(hand_l.worldbody.first_body(), "lh_", "")

        # Base actuators (high gain to resist contact forces)
        for jn in ["rx","ry","rz","lx","ly","lz"]:
            a = s.add_actuator(); a.name=f"act_{jn}"; a.target=jn
            a.trntype=mujoco.mjtTrn.mjTRN_JOINT; a.gainprm=[500]+[0]*9

        model = s.compile()
        data = mujoco.MjData(model)
        return model, data

    def _build_reference(self):
        """Build reference from MANO data (not hand-written curl)."""
        import json, smplx
        from scipy.spatial.transform import Rotation as Rot

        # Load MANO data
        mano_path = os.path.join(BASE_DIR, 'output/pick_bottle_video/mano_results.json')
        with open(mano_path) as f:
            results = json.load(f)

        # Get right hand data (use for both hands — mirror for left)
        right_frames = [r for r in results if r['is_right']]
        N_mano = len(right_frames)

        # Extract per-joint curl values from MANO rotation matrices
        # MANO hand_pose: 15 joints × (3×3) rotation matrix
        # Curl ≈ euler X component (flexion) of each joint
        mano_curls = np.zeros((N_mano, 15))
        for i, r in enumerate(right_frames):
            hp = np.array(r['mano_hand_pose'])  # (15, 3, 3)
            for j in range(15):
                euler = Rot.from_matrix(hp[j]).as_euler('xyz')
                mano_curls[i, j] = euler[0]  # flexion

        # Map MANO 15 joints → Allegro 16 actuators
        # MANO: index(0-2), middle(3-5), pinky(6-8), ring(9-11), thumb(12-14)
        # Allegro: ff(0-3), mf(4-7), rf(8-11), th(12-15)
        # j0=spread, j1=proximal, j2=medial, j3=distal
        mano_to_allegro = {
            0: 1, 1: 2, 2: 3,     # index → ff proximal/medial/distal
            3: 5, 4: 6, 5: 7,     # middle → mf
            9: 9, 10: 10, 11: 11, # ring → rf
            12: 13, 13: 14, 14: 15, # thumb → th
        }

        # Build reference: per-frame, per-actuator target
        self.ref_finger = np.zeros((self.max_steps, 16))  # 16 actuators per hand

        for i in range(self.max_steps):
            # Map to MANO frame index (resample from N_mano to max_steps)
            mi = min(int(i / self.max_steps * N_mano), N_mano - 1)
            for mano_j, allegro_j in mano_to_allegro.items():
                self.ref_finger[i, allegro_j] = mano_curls[mi, mano_j]

        # Smooth
        self.ref_finger = uniform_filter1d(self.ref_finger, size=7, axis=0)

        # Clip to Allegro joint ranges
        all_finger_acts = self.rh_finger_acts + self.lh_finger_acts
        for j in range(16):
            if j < len(self.rh_finger_acts):
                ai = self.rh_finger_acts[j]
                jid = self.model.actuator_trnid[ai, 0]
                if jid >= 0:
                    jr = self.model.jnt_range[jid]
                    self.ref_finger[:, j] = np.clip(self.ref_finger[:, j], jr[0], jr[1])

        print(f"  MANO reference: {N_mano} frames → {self.max_steps} steps, "
              f"curl range [{self.ref_finger.min():.2f}, {self.ref_finger.max():.2f}]")

        # Base trajectory: approach then lift (same as before)
        self.ref_base = np.zeros((self.max_steps, 6))
        for i in range(self.max_steps):
            t = i / (self.max_steps - 1)
            if t < 0.25:
                p = t / 0.25
                self.ref_base[i, 0] = -0.06 * p
                self.ref_base[i, 3] = 0.06 * p
            elif t < 0.65:
                self.ref_base[i, 0] = -0.06
                self.ref_base[i, 3] = 0.06
            else:
                lp = (t - 0.65) / 0.35
                self.ref_base[i, 0] = -0.06
                self.ref_base[i, 3] = 0.06
                self.ref_base[i, 2] = 0.06 * lp
                self.ref_base[i, 5] = 0.06 * lp

    def _get_obs(self):
        # Finger joint angles (32)
        finger_qpos = []
        for ai in self.rh_finger_acts + self.lh_finger_acts:
            jid = self.model.actuator_trnid[ai, 0]
            if jid >= 0:
                finger_qpos.append(self.data.qpos[self.model.jnt_qposadr[jid]])
        finger_qpos = np.array(finger_qpos[:32], dtype=np.float32)
        if len(finger_qpos) < 32:
            finger_qpos = np.pad(finger_qpos, (0, 32 - len(finger_qpos)))

        # Base positions (6)
        base_pos = np.zeros(6, dtype=np.float32)
        for idx, name in enumerate(["act_rx","act_ry","act_rz","act_lx","act_ly","act_lz"]):
            if name in self.base_acts:
                ai = self.base_acts[name]
                jid = self.model.actuator_trnid[ai, 0]
                if jid >= 0:
                    base_pos[idx] = self.data.qpos[self.model.jnt_qposadr[jid]]

        # Bottle state (7)
        bottle_pos = self.data.xpos[self.bottle_id].astype(np.float32) if self.bottle_id >= 0 else np.zeros(3, dtype=np.float32)
        bottle_quat = self.data.xquat[self.bottle_id].astype(np.float32) if self.bottle_id >= 0 else np.array([1,0,0,0], dtype=np.float32)

        # Fingertip positions (24 = 8 tips × 3)
        tips = np.zeros(24, dtype=np.float32)
        for j, tid in enumerate(self.tip_ids[:8]):
            tips[j*3:(j+1)*3] = self.data.xpos[tid]

        # Timestep (1)
        timestep = np.array([self.step_count / self.max_steps], dtype=np.float32)

        return np.concatenate([finger_qpos, base_pos, bottle_pos, bottle_quat, tips, timestep])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.prev_action = np.zeros(self.n_finger_acts)

        # Small random perturbation on bottle position
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        self.data.qpos[0] += rng.uniform(-0.01, 0.01)  # bottle x
        self.data.qpos[1] += rng.uniform(-0.01, 0.01)  # bottle y

        mujoco.mj_forward(self.model, self.data)
        self.bottle_init_z = self.data.xpos[self.bottle_id][2]
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        t_idx = min(self.step_count - 1, self.max_steps - 1)

        # Set base actuators from reference
        base_names = ["act_rx","act_ry","act_rz","act_lx","act_ly","act_lz"]
        for idx, name in enumerate(base_names):
            if name in self.base_acts:
                self.data.ctrl[self.base_acts[name]] = self.ref_base[t_idx, idx]

        # Set finger actuators: MANO reference + RL residual
        all_finger_acts = self.rh_finger_acts + self.lh_finger_acts
        self._current_ref = np.zeros(self.n_finger_acts)  # store for reward
        for j, ai in enumerate(all_finger_acts[:self.n_finger_acts]):
            jid = self.model.actuator_trnid[ai, 0]
            if jid >= 0:
                jr = self.model.jnt_range[jid]
                # MANO reference (same for both hands, mirrored)
                finger_idx = j % 16  # 0-15 for each hand
                base_ctrl = self.ref_finger[t_idx, finger_idx]
                self._current_ref[j] = base_ctrl
                residual = action[j] if j < len(action) else 0
                self.data.ctrl[ai] = np.clip(base_ctrl + residual, jr[0], jr[1])

        # Physics step
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # === REWARD (RoboWheel-style) ===
        obs = self._get_obs()
        bottle_pos = self.data.xpos[self.bottle_id]
        bottle_z = bottle_pos[2]

        # 1. Geometric tracking: fingertips close to bottle
        tip_dists = []
        for tid in self.tip_ids[:8]:
            d = np.linalg.norm(self.data.xpos[tid] - bottle_pos)
            tip_dists.append(d)
        r_geo = -np.mean(tip_dists) * 5.0  # closer = better

        # 2. MANO tracking: penalize deviation from MANO reference (KEY!)
        # This keeps the grasp looking like the human's grasp
        r_mano = -np.sum(action**2) * 20.0  # penalize large residuals

        # 3. Dynamic smoothness: penalize jerky actions
        r_dyn = -np.sum((action - self.prev_action)**2) * 2.0
        self.prev_action = action.copy()

        # 4. Contact consistency: reward contacts with bottle
        r_con = min(self.data.ncon, 10) * 0.15  # cap at 10 contacts

        # 5. Lift bonus
        lift = bottle_z - self.bottle_init_z
        r_lift = max(0, lift) * 100.0  # big reward for lifting

        # 6. Stability: penalize bottle falling
        r_stable = -max(0, self.bottle_init_z - bottle_z) * 10.0

        reward = r_geo + r_mano + r_dyn + r_con + r_lift + r_stable

        # Termination
        done = self.step_count >= self.max_steps
        truncated = False

        # Check if bottle fell off table
        if bottle_z < 0.10:
            done = True
            reward -= 5.0

        return obs, reward, done, truncated, {
            'r_geo': r_geo, 'r_mano': r_mano, 'r_dyn': r_dyn, 'r_con': r_con,
            'r_lift': r_lift, 'lift_cm': lift * 100,
            'contacts': self.data.ncon,
            'residual_norm': np.linalg.norm(action),
        }

    def render(self):
        if not hasattr(self, '_renderer'):
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._cam = mujoco.MjvCamera()
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._cam.lookat[:] = [0, 0, 0.28]
            self._cam.distance = 0.50
            self._cam.azimuth = 145
            self._cam.elevation = -25
        self._renderer.update_scene(self.data, self._cam)
        return self._renderer.render()


# ============================================================
# Training + Evaluation
# ============================================================

def train():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== Layer 2: PPO Residual RL Training ===")
    print("Creating environments...")

    env = DummyVecEnv([lambda: BimanualGraspEnv() for _ in range(4)])

    print("Training PPO...")
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        # tensorboard_log=os.path.join(OUT_DIR, 'tb_logs'),
    )

    # Train for 100K steps (~10 min on CPU)
    model.learn(total_timesteps=100_000, progress_bar=True)
    model.save(os.path.join(OUT_DIR, 'ppo_grasp'))
    print(f"Model saved to {OUT_DIR}/ppo_grasp.zip")

    return model


def evaluate_and_render(model):
    print("\n=== Layer 3: Replay RL Policy ===")
    env = BimanualGraspEnv(render_mode="rgb_array")

    obs, _ = env.reset(seed=42)
    frames = []
    total_reward = 0
    max_contacts = 0
    max_lift = 0

    for i in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        max_contacts = max(max_contacts, info['contacts'])
        max_lift = max(max_lift, info['lift_cm'])

        img = env.render()
        bgr = img[:, :, ::-1].copy()

        t = i / 199
        phase = "APPROACH" if t < 0.25 else "CLOSE" if t < 0.50 else "GRASP" if t < 0.65 else "LIFT"
        cv2.putText(bgr, f"[{phase}] {i} con={info['contacts']} lift={info['lift_cm']:.1f}cm",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)
        cv2.putText(bgr, f"r={reward:.2f} mano={info['r_mano']:.2f} res={info['residual_norm']:.3f}",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 255), 1)
        frames.append(bgr)

        if done:
            break

    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Max contacts: {max_contacts}")
    print(f"  Max lift: {max_lift:.1f} cm")

    # Save video
    fps = 15
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = cv2.VideoWriter(os.path.join(OUT_DIR, 'rl_grasp.mp4'), fourcc, fps, (640, 480))
    for f in frames:
        w.write(f)
    w.release()

    # Save keyframes
    for idx in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]:
        if idx < len(frames):
            cv2.imwrite(os.path.join(OUT_DIR, f'kf_{idx:04d}.jpg'), frames[idx])

    # Comparison with HaMeR
    overlays_dir = os.path.join(BASE_DIR, 'output/pick_bottle_video')
    overlays = sorted([f for f in os.listdir(overlays_dir) if f.endswith('_overlay.jpg')])

    pw, ph = 640, 480
    w_cmp = cv2.VideoWriter(os.path.join(OUT_DIR, 'comparison.mp4'), fourcc, fps, (pw*2, ph+25))
    for i, f in enumerate(frames):
        ov_idx = min(int(i / len(frames) * len(overlays)), len(overlays)-1)
        ov = cv2.resize(cv2.imread(os.path.join(overlays_dir, overlays[ov_idx])), (pw, ph))
        panels = np.hstack([ov, f])
        label = np.ones((25, pw*2, 3), dtype=np.uint8) * 35
        cv2.putText(label, "Video + HaMeR", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        cv2.putText(label, "RL Policy (PPO)", (pw+5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        w_cmp.write(np.vstack([label, panels]))
    w_cmp.release()

    # GIF
    import subprocess
    subprocess.run(['ffmpeg', '-y', '-i', os.path.join(OUT_DIR, 'comparison.mp4'),
                    '-vf', 'fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
                    os.path.join(OUT_DIR, 'comparison.gif')], capture_output=True)

    print(f"\nDone! {OUT_DIR}/")


if __name__ == '__main__':
    model = train()
    evaluate_and_render(model)
