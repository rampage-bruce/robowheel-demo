"""
Final unified pipeline:
  FoundationPose bottle 6DoF → hand world position
  + Arti-MANO 1:1 finger mapping
  + PPO residual RL
  = perfect hand shape + correct position + physical grasp
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
OUT_DIR = os.path.join(BASE_DIR, 'output/final_unified')
MANO_PATH = os.path.join(BASE_DIR, 'output/pick_bottle_video/mano_results.json')
FPOSE_PATH = os.path.join(BASE_DIR, 'output/bottle_6dof_poses.npz')


def load_all_references():
    """
    Combine FoundationPose bottle tracking + MANO hand data.
    Compute hand world position from: hand_cam - bottle_cam + bottle_world
    """
    # FoundationPose: bottle 6DoF in camera frame
    fpose = np.load(FPOSE_PATH)
    bottle_poses = fpose['poses']  # (151, 4, 4)
    bottle_pos_cam = bottle_poses[:, :3, 3]  # (151, 3) translation in camera frame
    bottle_rot_cam = bottle_poses[:, :3, :3]  # (151, 3, 3) rotation
    N_fp = len(bottle_poses)

    # MANO data
    with open(MANO_PATH) as f:
        results = json.load(f)

    # Group by frame, get bimanual
    frame_dict = {}
    for r in results:
        key = r['img_name']
        if key not in frame_dict: frame_dict[key] = {}
        frame_dict[key]['right' if r['is_right'] else 'left'] = r
    both = [v for v in frame_dict.values() if 'left' in v and 'right' in v]
    N_mano = len(both)
    N = min(N_fp, N_mano, 80)  # Only use grasp phase (frame 0-80), skip drinking

    # MANO cam_t for each hand (camera frame position)
    right_cam_t = np.array([both[i]['right']['cam_t_full'] for i in range(N)])
    left_cam_t = np.array([both[i]['left']['cam_t_full'] for i in range(N)])

    # === KEY COMPUTATION: hand world position ===
    # bottle_pos_cam: where FoundationPose says bottle is in camera frame
    # hand_cam_t: where MANO says hand is in camera frame
    # hand relative to bottle (in camera frame):
    #   hand_offset_cam = hand_cam_t_normalized - bottle_pos_cam
    # Then place in our sim world (bottle at [0, 0, 0.26]):

    OUR_BOTTLE = np.array([0, 0, 0.26])

    # Normalize MANO cam_t: cam_t is [x, y, depth]
    # x,y are in pixels*depth scale; normalize by depth for NDC-like coords
    def cam_t_to_cam_pos(ct, bottle_depth):
        """Convert MANO cam_t to camera-frame 3D position (meters).
        cam_t is NOT in meters — it's HaMeR internal normalized coords.
        NDC = cam_t[0:2] / cam_t[2], then scale by actual depth from FoundationPose.
        """
        x_ndc = ct[0] / ct[2]
        y_ndc = ct[1] / ct[2]
        pos = np.array([x_ndc * bottle_depth, y_ndc * bottle_depth, bottle_depth])
        return pos

    right_world = np.zeros((N, 3))
    left_world = np.zeros((N, 3))

    for i in range(N):
        bottle_cam = bottle_pos_cam[i]
        bottle_depth = bottle_cam[2]  # FoundationPose depth (meters)

        # Hand position in camera frame (meters, using FoundationPose depth as scale)
        rh_cam = cam_t_to_cam_pos(right_cam_t[i], bottle_depth)
        lh_cam = cam_t_to_cam_pos(left_cam_t[i], bottle_depth)

        # Hand-bottle offset in camera frame (meters)
        rh_offset = rh_cam - bottle_cam
        lh_offset = lh_cam - bottle_cam

        # Camera→World: X stays, cam Y(down)→world -Z(up), cam Z(forward)→world Y
        def cam_to_world_offset(offset):
            w = np.zeros(3)
            w[0] = offset[0]        # X: left-right (meters)
            w[1] = -offset[2] * 0.3 # Y: depth diff → forward/back (scaled down)
            w[2] = -offset[1]       # Z: cam Y(down) → world Z(up), meters
            return w

        # Add fixed lateral offset (wrist should be BESIDE bottle, not on top)
        # Right hand: +X (right side), Left hand: -X (left side)
        LATERAL_OFFSET_R = np.array([0.08, 0, 0.02])  # 8cm right, 2cm up
        LATERAL_OFFSET_L = np.array([-0.08, 0, 0.02])  # 8cm left, 2cm up

        if i == 0:
            right_world[i] = OUR_BOTTLE + cam_to_world_offset(rh_offset) + LATERAL_OFFSET_R
            left_world[i] = OUR_BOTTLE + cam_to_world_offset(lh_offset) + LATERAL_OFFSET_L
        else:
            # ALL FRAMES: use Frame 0 position (scene bottle is static)
            # Only fingers change per frame (from MANO hand_pose)
            right_world[i] = right_world[0]
            left_world[i] = left_world[0]

    # Smooth
    right_world = uniform_filter1d(right_world, size=7, axis=0)
    left_world = uniform_filter1d(left_world, size=7, axis=0)

    # MANO finger angles (direct 1:1 to Arti-MANO)
    def mano_to_artimano(hp):
        joints = np.zeros(22)
        def euler(mat): return Rot.from_matrix(mat).as_euler('xyz')
        e=euler(hp[0]); joints[0]=e[2]; joints[1]=e[0]
        e=euler(hp[1]); joints[2]=e[0]
        e=euler(hp[2]); joints[3]=e[0]
        e=euler(hp[3]); joints[4]=e[2]; joints[5]=e[0]
        e=euler(hp[4]); joints[6]=e[0]
        e=euler(hp[5]); joints[7]=e[0]
        e=euler(hp[6]); joints[8]=e[2]; joints[9]=e[0]
        e=euler(hp[7]); joints[10]=e[0]
        e=euler(hp[8]); joints[11]=e[0]
        e=euler(hp[9]); joints[12]=e[2]; joints[13]=e[0]
        e=euler(hp[10]); joints[14]=e[0]
        e=euler(hp[11]); joints[15]=e[0]
        e=euler(hp[12]); joints[16]=e[0]; joints[17]=e[1]; joints[18]=e[2]
        e=euler(hp[13]); joints[19]=e[1]; joints[20]=e[0]
        e=euler(hp[14]); joints[21]=e[0]
        return joints

    right_fingers = np.zeros((N, 22))
    left_fingers = np.zeros((N, 22))
    for i in range(N):
        right_fingers[i] = mano_to_artimano(np.array(both[i]['right']['mano_hand_pose']))
        left_fingers[i] = mano_to_artimano(np.array(both[i]['left']['mano_hand_pose']))

    right_fingers = uniform_filter1d(right_fingers, size=5, axis=0)
    left_fingers = uniform_filter1d(left_fingers, size=5, axis=0)

    print(f"  Frames: {N}")
    print(f"  Right hand world: [{right_world.min():.3f}, {right_world.max():.3f}]")
    print(f"  Left hand world:  [{left_world.min():.3f}, {left_world.max():.3f}]")
    print(f"  Bottle cam pos:   [{bottle_pos_cam[:N].min():.3f}, {bottle_pos_cam[:N].max():.3f}]")
    print(f"  Fingers range:    [{right_fingers.min():.2f}, {right_fingers.max():.2f}]")

    return {
        'N': N,
        'right_world': right_world,
        'left_world': left_world,
        'right_fingers': right_fingers,
        'left_fingers': left_fingers,
        'bottle_pos_cam': bottle_pos_cam[:N],
    }


class FinalEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, ref=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.ref = ref if ref else load_all_references()
        self.N = self.ref['N']
        self.max_steps = self.N

        self.model, self.data = self._build()

        # Action: residual on fingers only (22 per hand × 2 = 44)
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

        # Right hand
        rp = self.ref['right_world'][0].tolist()
        br = w.add_body(); br.name="base_right"; br.pos=rp
        bm=br.add_geom(); bm.type=mujoco.mjtGeom.mjGEOM_SPHERE; bm.size=[0.001,0,0]
        bm.mass=0.1; bm.rgba=[0,0,0,0]; bm.contype=0; bm.conaffinity=0
        for jn,ax in [("rx",[1,0,0]),("ry",[0,1,0]),("rz",[0,0,1])]:
            j=br.add_joint(); j.name=jn; j.type=mujoco.mjtJoint.mjJNT_SLIDE; j.axis=ax; j.range=[-0.05,0.05]
        mr=br.add_body(); mr.name="mount_right"
        rot_r = Rot.from_euler('zy', [180, -15], degrees=True)
        mr.quat = rot_r.as_quat(scalar_first=True).tolist()
        mr.add_frame().attach_body(hr.worldbody.first_body(), "rh_", "")

        # Left hand
        lp = self.ref['left_world'][0].tolist()
        bl = w.add_body(); bl.name="base_left"; bl.pos=lp
        bml=bl.add_geom(); bml.type=mujoco.mjtGeom.mjGEOM_SPHERE; bml.size=[0.001,0,0]
        bml.mass=0.1; bml.rgba=[0,0,0,0]; bml.contype=0; bml.conaffinity=0
        for jn,ax in [("lx",[1,0,0]),("ly",[0,1,0]),("lz",[0,0,1])]:
            j=bl.add_joint(); j.name=jn; j.type=mujoco.mjtJoint.mjJNT_SLIDE; j.axis=ax; j.range=[-0.05,0.05]
        ml=bl.add_body(); ml.name="mount_left"
        rot_l = Rot.from_euler('y', 15, degrees=True)
        ml.quat = rot_l.as_quat(scalar_first=True).tolist()
        ml.add_frame().attach_body(hl.worldbody.first_body(), "lh_", "")

        # Base actuators
        for jn in ["rx","ry","rz","lx","ly","lz"]:
            a=s.add_actuator(); a.name=f"act_{jn}"; a.target=jn
            a.trntype=mujoco.mjtTrn.mjTRN_JOINT; a.gainprm=[5000]+[0]*9

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
        for j, ai in enumerate((self.rh_acts+self.lh_acts)[:44]):
            jid = self.model.actuator_trnid[ai,0]
            if jid>=0: joints[j] = self.data.qpos[self.model.jnt_qposadr[jid]]
        base = np.zeros(6, dtype=np.float32)
        bp = self.data.xpos[self.bottle_id].astype(np.float32)
        bq = self.data.xquat[self.bottle_id].astype(np.float32)
        tips = np.zeros(30, dtype=np.float32)
        for j, tid in enumerate(self.tip_ids[:10]):
            tips[j*3:(j+1)*3] = self.data.xpos[tid]
        ts = np.array([self.step_count/self.max_steps], dtype=np.float32)
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
        fi = min(self.step_count-1, self.N-1)

        # BASE: from FoundationPose-derived world positions (delta from init)
        rp = self.ref['right_world'][fi] - self.ref['right_world'][0]
        lp = self.ref['left_world'][fi] - self.ref['left_world'][0]
        rp = np.clip(rp, -0.04, 0.04)
        lp = np.clip(lp, -0.04, 0.04)

        for idx, name in enumerate(["act_rx","act_ry","act_rz"]):
            if name in self.base_acts: self.data.ctrl[self.base_acts[name]] = rp[idx]
        for idx, name in enumerate(["act_lx","act_ly","act_lz"]):
            if name in self.base_acts: self.data.ctrl[self.base_acts[name]] = lp[idx]

        # FINGERS: MANO 1:1 + RL residual
        for j in range(min(22, len(self.rh_acts))):
            ai = self.rh_acts[j]
            jid = self.model.actuator_trnid[ai,0]
            if jid>=0:
                jr = self.model.jnt_range[jid]
                ref = self.ref['right_fingers'][fi,j] if j<22 else 0
                self.data.ctrl[ai] = np.clip(ref + action[j], jr[0], jr[1])
        for j in range(min(22, len(self.lh_acts))):
            ai = self.lh_acts[j]
            jid = self.model.actuator_trnid[ai,0]
            if jid>=0:
                jr = self.model.jnt_range[jid]
                ref = self.ref['left_fingers'][fi,j] if j<22 else 0
                self.data.ctrl[ai] = np.clip(ref + action[22+j], jr[0], jr[1])

        for _ in range(10):
            try: mujoco.mj_step(self.model, self.data)
            except: mujoco.mj_resetData(self.model, self.data); break

        obs = self._get_obs()
        bz = self.data.xpos[self.bottle_id][2]; lift = bz - self.bottle_init_z
        bpos = self.data.xpos[self.bottle_id]

        r_track = -np.sum(action**2) * 5.0  # Reduced: allow some residual
        td = [np.linalg.norm(self.data.xpos[tid]-bpos) for tid in self.tip_ids[:10]]
        r_geo = -np.mean(td) * 20.0 if td else 0  # Increased: fingertips must approach bottle
        r_dyn = -np.sum((action-self.prev_action)**2) * 1.0; self.prev_action = action.copy()
        r_con = min(self.data.ncon, 20) * 0.5  # Increased: more reward for contacts
        r_lift = max(0, lift) * 100.0
        r_stable = -max(0, self.bottle_init_z - bz) * 5.0
        # Penalize hands being far from bottle (world XY distance)
        hand_dists = []
        for prefix in ['base_right', 'base_left']:
            hid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, prefix)
            if hid >= 0:
                hand_dists.append(np.linalg.norm(self.data.xpos[hid][:2] - bpos[:2]))
        r_prox = -np.mean(hand_dists) * 10.0 if hand_dists else 0
        reward = r_track + r_geo + r_dyn + r_con + r_lift + r_stable + r_prox

        done = self.step_count >= self.max_steps
        if bz < 0.10: done = True; reward -= 3.0

        return obs, reward, done, False, {
            'lift_cm': lift*100, 'contacts': self.data.ncon, 'res_norm': np.linalg.norm(action),
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

    print("=== Final: FoundationPose + Arti-MANO + PPO ===")
    ref = load_all_references()

    print("\nTraining PPO (500K)...")
    env = DummyVecEnv([lambda: FinalEnv(ref=ref) for _ in range(16)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=512, batch_size=512,
                n_epochs=10, gamma=0.99, clip_range=0.2, ent_coef=0.005, verbose=1)
    model.learn(total_timesteps=500_000, progress_bar=True)
    model.save(os.path.join(OUT_DIR, 'ppo_final'))

    print("\n=== Evaluation ===")
    ev = FinalEnv(ref=ref, render_mode="rgb_array")
    obs, _ = ev.reset(seed=42)
    frames = []; total_r = 0; mx_lift = 0; mx_con = 0
    for i in range(ref['N']):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, info = ev.step(a)
        total_r += r; mx_lift = max(mx_lift, info['lift_cm']); mx_con = max(mx_con, info['contacts'])
        img = ev.render(); bgr = img[:,:,::-1].copy()
        cv2.putText(bgr, f"[{i}] con={info['contacts']} lift={info['lift_cm']:.1f}cm res={info['res_norm']:.3f}",
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0,220,0), 1)
        frames.append(bgr)
        if done: break

    print(f"  Reward: {total_r:.1f}, Contacts: {mx_con}, Lift: {mx_lift:.1f}cm, Frames: {len(frames)}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = cv2.VideoWriter(os.path.join(OUT_DIR, 'final.mp4'), fourcc, 15, (640,480))
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
        cv2.putText(lb, "FoundationPose + Arti-MANO + PPO", (pw+5,17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,220,0), 1)
        wc.write(np.vstack([lb, panels]))
    wc.release()

    import subprocess
    subprocess.run(['ffmpeg','-y','-i',os.path.join(OUT_DIR,'comparison.mp4'),
        '-vf','fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
        os.path.join(OUT_DIR,'comparison.gif')], capture_output=True)

    print(f"\nDone! {OUT_DIR}/")


if __name__ == '__main__':
    main()
