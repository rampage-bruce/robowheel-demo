"""
ManipTrans-style 2-stage RL:
  Stage 1: Imitator — learn to track SPIDER IK reference in physics
  Stage 2: Residual — fine-tune for contact/grasp on top of imitator

Key: RL outputs FULL control signals. Reference only appears in reward.
The policy must figure out how to reach the reference within physics constraints.
"""
import os, json
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
OUT_DIR = os.path.join(BASE_DIR, 'output/maniptrans')
SPIDER_IK = os.path.join(BASE_DIR, 'spider/example_datasets/processed/hamer_demo/mano/right/pick_bottle/0/trajectory_kinematic.npz')


def load_ik_reference():
    """Load SPIDER IK Arti-MANO trajectory as tracking target."""
    ik = np.load(SPIDER_IK)
    qpos = ik['qpos']  # (147, 35): [base(6), fingers(22), object(7)]
    N = qpos.shape[0]

    # Normalize: object-centered coordinates
    obj_pos = qpos[:, 28:31]
    hand_pos = qpos[:, :3]
    hand_rot = qpos[:, 3:6]
    fingers = qpos[:, 6:28]

    # Re-center hand relative to our bottle
    OUR_BOTTLE = np.array([0, 0, 0.26])
    hand_rel = hand_pos - obj_pos  # hand-object offset in SPIDER frame
    hand_world = hand_rel + OUR_BOTTLE

    # Smooth
    hand_world = uniform_filter1d(hand_world, size=5, axis=0)
    hand_rot = uniform_filter1d(hand_rot, size=5, axis=0)
    fingers = uniform_filter1d(fingers, size=5, axis=0)

    # Full reference: [hand_pos(3), hand_rot(3), fingers(22)] = 28 per frame
    ref = np.concatenate([hand_world, hand_rot, fingers], axis=1)

    print(f"  Reference: {N} frames, 28 DOF per frame")
    print(f"  Hand pos range: [{hand_world.min():.3f}, {hand_world.max():.3f}]")
    print(f"  Fingers range: [{fingers.min():.3f}, {fingers.max():.3f}]")
    return N, ref


class ManipTransEnv(gym.Env):
    """
    ManipTrans-style: RL outputs FULL control, reference only in reward.

    Observation: [current_qpos(28), reference_target(28), bottle_state(7), contacts(1), time(1)] = 65
    Action: full actuator control (28 DOF: 6 base + 22 fingers)
    Reward: tracking + physics
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, ref_data=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        if ref_data:
            self.N_ref, self.ref = ref_data
        else:
            self.N_ref, self.ref = load_ik_reference()

        self.max_steps = self.N_ref
        self.model, self.data = self._build()

        # Action: FULL control for 28 DOF (6 base + 22 fingers)
        # Ranges match joint limits
        act_low = np.concatenate([np.full(3, -0.15), np.full(3, -1.5), np.full(22, -0.5)])
        act_high = np.concatenate([np.full(3, 0.15), np.full(3, 1.5), np.full(22, 1.6)])
        self.action_space = spaces.Box(act_low.astype(np.float32), act_high.astype(np.float32))

        # Observation: current_state(28) + ref_target(28) + bottle(7) + contacts(1) + time(1) = 65
        self.observation_space = spaces.Box(-10, 10, shape=(65,), dtype=np.float32)

        self._find_ids()
        self.step_count = 0
        self.prev_action = np.zeros(28)
        self.bottle_init_z = 0

    def _build(self):
        os.chdir(SPIDER_MANO)
        hr = mujoco.MjSpec.from_file('right.xml')

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

        # Right Arti-MANO hand with 6DoF base
        init_pos = self.ref[0, :3].tolist()
        br = w.add_body(); br.name="base_right"; br.pos=init_pos
        bm = br.add_geom(); bm.type=mujoco.mjtGeom.mjGEOM_SPHERE; bm.size=[0.001,0,0]
        bm.mass=0.1; bm.rgba=[0,0,0,0]; bm.contype=0; bm.conaffinity=0

        for jn,ax in [("rx",[1,0,0]),("ry",[0,1,0]),("rz",[0,0,1])]:
            j=br.add_joint(); j.name=jn; j.type=mujoco.mjtJoint.mjJNT_SLIDE; j.axis=ax; j.range=[-0.15,0.15]
        for jn,ax in [("rrx",[1,0,0]),("rry",[0,1,0]),("rrz",[0,0,1])]:
            j=br.add_joint(); j.name=jn; j.type=mujoco.mjtJoint.mjJNT_HINGE; j.axis=ax; j.range=[-1.5,1.5]
        mr = br.add_body(); mr.name="mount_right"; mr.quat=[1,0,0,0]
        mr.add_frame().attach_body(hr.worldbody.first_body(), "rh_", "")

        for jn in ["rx","ry","rz","rrx","rry","rrz"]:
            a=s.add_actuator(); a.name=f"act_{jn}"; a.target=jn
            a.trntype=mujoco.mjtTrn.mjTRN_JOINT
            kp = 300 if 'rr' not in jn else 100
            a.gainprm=[kp]+[0]*9

        model = s.compile()
        return model, mujoco.MjData(model)

    def _find_ids(self):
        self.base_act_ids = []  # 6 base actuators
        self.finger_act_ids = []  # 22 finger actuators
        self.tip_ids = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            if name.startswith("act_"): self.base_act_ids.append(i)
            elif "rh_" in name: self.finger_act_ids.append(i)
        self.bottle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'bottle')
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if name.endswith("3") and any(f in name for f in ["index","middle","pinky","ring","thumb"]):
                self.tip_ids.append(i)

    def _get_current_state(self):
        """Get current 28-DOF state: base(6) + fingers(22)."""
        state = np.zeros(28)
        # Base position (from body xpos delta)
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base_right')
        if base_id >= 0:
            state[:3] = self.data.xpos[base_id]
        # Base rotation (from qpos)
        for j in range(6):
            if j < len(self.base_act_ids):
                ai = self.base_act_ids[j]
                jid = self.model.actuator_trnid[ai, 0]
                if jid >= 0:
                    state[j] = self.data.qpos[self.model.jnt_qposadr[jid]]
        # Fingers
        for j in range(min(22, len(self.finger_act_ids))):
            ai = self.finger_act_ids[j]
            jid = self.model.actuator_trnid[ai, 0]
            if jid >= 0:
                state[6+j] = self.data.qpos[self.model.jnt_qposadr[jid]]
        return state.astype(np.float32)

    def _get_obs(self):
        current = self._get_current_state()  # 28
        fi = min(self.step_count, self.N_ref - 1)
        target = self.ref[fi].astype(np.float32)  # 28
        bp = self.data.xpos[self.bottle_id].astype(np.float32) if self.bottle_id >= 0 else np.zeros(3, dtype=np.float32)
        bq = self.data.xquat[self.bottle_id].astype(np.float32) if self.bottle_id >= 0 else np.array([1,0,0,0], dtype=np.float32)
        con = np.array([min(self.data.ncon, 20) / 20.0], dtype=np.float32)
        ts = np.array([self.step_count / self.max_steps], dtype=np.float32)
        return np.concatenate([current, target, bp, bq, con, ts])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0; self.prev_action = np.zeros(28)
        mujoco.mj_forward(self.model, self.data)
        self.bottle_init_z = self.data.xpos[self.bottle_id][2]
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # === RL outputs FULL control (not reference + residual) ===
        # Base actuators
        for j in range(min(6, len(self.base_act_ids))):
            self.data.ctrl[self.base_act_ids[j]] = action[j]
        # Finger actuators
        for j in range(min(22, len(self.finger_act_ids))):
            ai = self.finger_act_ids[j]
            jid = self.model.actuator_trnid[ai, 0]
            if jid >= 0:
                jr = self.model.jnt_range[jid]
                self.data.ctrl[ai] = np.clip(action[6+j], jr[0], jr[1])

        for _ in range(10):
            try:
                mujoco.mj_step(self.model, self.data)
            except mujoco.FatalError:
                mujoco.mj_resetData(self.model, self.data)
                break

        obs = self._get_obs()
        current = self._get_current_state()
        fi = min(self.step_count - 1, self.N_ref - 1)
        target = self.ref[fi]
        bz = self.data.xpos[self.bottle_id][2]
        lift = bz - self.bottle_init_z
        bpos = self.data.xpos[self.bottle_id]

        # === REWARD (ManipTrans-style) ===
        # 1. Tracking: how close current state is to reference
        pos_err = np.linalg.norm(current[:3] - target[:3])
        rot_err = np.linalg.norm(current[3:6] - target[3:6])
        finger_err = np.mean(np.abs(current[6:28] - target[6:28]))
        r_track = -(pos_err * 20.0 + rot_err * 5.0 + finger_err * 10.0)

        # 2. Fingertip proximity to bottle
        td = [np.linalg.norm(self.data.xpos[tid]-bpos) for tid in self.tip_ids[:5]]
        r_geo = -np.mean(td) * 3.0 if td else 0

        # 3. Smoothness
        r_smooth = -np.sum((action - self.prev_action)**2) * 0.5
        self.prev_action = action.copy()

        # 4. Contact
        r_con = min(self.data.ncon, 10) * 0.2

        # 5. Lift
        r_lift = max(0, lift) * 50.0

        reward = r_track + r_geo + r_smooth + r_con + r_lift

        done = self.step_count >= self.max_steps
        if bz < 0.10: done = True; reward -= 3.0

        return obs, reward, done, False, {
            'r_track': r_track, 'pos_err': pos_err, 'finger_err': finger_err,
            'lift_cm': lift*100, 'contacts': self.data.ncon,
        }

    def render(self):
        if not hasattr(self, '_renderer'):
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._cam = mujoco.MjvCamera()
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._cam.lookat[:] = [0, 0, 0.28]
            self._cam.distance = 0.45; self._cam.azimuth = 145; self._cam.elevation = -25
        self._renderer.update_scene(self.data, self._cam)
        return self._renderer.render()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== ManipTrans-style: RL outputs full control, reference in reward ===")
    ref_data = load_ik_reference()

    # Stage 1: Imitator — learn to track reference
    print("\n=== Stage 1: Imitator (300K) ===")
    env = DummyVecEnv([lambda: ManipTransEnv(ref_data=ref_data) for _ in range(16)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    imitator = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=256, batch_size=256,
                   n_epochs=10, gamma=0.99, clip_range=0.2, ent_coef=0.01, verbose=1)
    imitator.learn(total_timesteps=300_000, progress_bar=True)
    imitator.save(os.path.join(OUT_DIR, 'imitator'))
    print("Imitator saved!")

    # Evaluate imitator
    print("\n=== Evaluation ===")
    ev = ManipTransEnv(ref_data=ref_data, render_mode="rgb_array")
    obs, _ = ev.reset(seed=42)
    frames = []; total_r = 0; mx_lift = 0; mx_con = 0

    for i in range(ref_data[0]):
        a, _ = imitator.predict(obs, deterministic=True)
        obs, r, done, _, info = ev.step(a)
        total_r += r; mx_lift = max(mx_lift, info['lift_cm']); mx_con = max(mx_con, info['contacts'])
        img = ev.render(); bgr = img[:,:,::-1].copy()
        cv2.putText(bgr, f"[{i}] con={info['contacts']} lift={info['lift_cm']:.1f}cm pe={info['pos_err']:.3f} fe={info['finger_err']:.2f}",
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,220,0), 1)
        frames.append(bgr)
        if done: break

    print(f"  Reward: {total_r:.1f}, Contacts: {mx_con}, Lift: {mx_lift:.1f}cm, Frames: {len(frames)}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = cv2.VideoWriter(os.path.join(OUT_DIR, 'imitator.mp4'), fourcc, 15, (640,480))
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
        cv2.putText(lb, "ManipTrans Imitator (Arti-MANO)", (pw+5,17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,220,0), 1)
        wc.write(np.vstack([lb, panels]))
    wc.release()

    import subprocess
    subprocess.run(['ffmpeg','-y','-i',os.path.join(OUT_DIR,'comparison.mp4'),
        '-vf','fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
        os.path.join(OUT_DIR,'comparison.gif')], capture_output=True)

    print(f"\nDone! {OUT_DIR}/")


if __name__ == '__main__':
    main()
