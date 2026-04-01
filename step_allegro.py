"""
MANO → Allegro Hand: compact dexterous hand with clear grasp motion.
Allegro has 4 fingers × 4 joints = 16 actuators, no forearm clutter.
Staged trajectory: reach → open → wrap → grip → lift
"""
import os, json, glob
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
import cv2
from scipy.spatial.transform import Rotation
from scipy.ndimage import uniform_filter1d
from collections import OrderedDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ALLEGRO_XML = os.path.join(BASE_DIR, 'mujoco_menagerie/wonik_allegro/right_hand.xml')
MANO_RESULTS = os.path.join(BASE_DIR, 'output/pick_bottle_video/mano_results.json')
if not os.path.exists(MANO_RESULTS):
    MANO_RESULTS = os.path.join(BASE_DIR, 'test_data/mano_results.json')
OUT_DIR = os.path.join(BASE_DIR, 'output/allegro_sim')


def build_model():
    """Allegro Hand on 6DoF base + table + bottle."""
    os.chdir(os.path.dirname(ALLEGRO_XML))
    hand_spec = mujoco.MjSpec.from_file(os.path.basename(ALLEGRO_XML))

    s = mujoco.MjSpec()
    s.option.gravity = [0, 0, -9.81]
    s.option.timestep = 0.002

    w = s.worldbody

    # Lights
    for pos, dir_, diff in [
        ([0, -0.2, 0.6], [0, 0.15, -1], [1, 1, 1]),
        ([0.2, -0.2, 0.5], [-0.1, 0.15, -0.5], [0.6, 0.6, 0.6]),
        ([-0.2, 0.2, 0.4], [0.1, -0.1, -0.5], [0.3, 0.3, 0.3]),
    ]:
        l = w.add_light(); l.pos = pos; l.dir = dir_; l.diffuse = diff

    # Floor
    f = w.add_geom(); f.type = mujoco.mjtGeom.mjGEOM_PLANE
    f.size = [0.5, 0.5, 0.01]; f.rgba = [0.93, 0.93, 0.93, 1]

    # Table (static geoms)
    tt = w.add_geom(); tt.type = mujoco.mjtGeom.mjGEOM_BOX
    tt.size = [0.15, 0.12, 0.008]; tt.pos = [0, 0, 0.20]
    tt.rgba = [0.38, 0.28, 0.20, 1]; tt.friction = [1.5, 0.01, 0.001]

    for lx, ly in [(-0.13, -0.10), (0.13, -0.10), (-0.13, 0.10), (0.13, 0.10)]:
        lg = w.add_geom(); lg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        lg.size = [0.008, 0.095, 0]; lg.pos = [lx, ly, 0.10]
        lg.rgba = [0.5, 0.4, 0.3, 1]

    # Bottle
    bottle = w.add_body(); bottle.name = "bottle"
    bottle.pos = [0, 0, 0.248]  # table top=0.208, + bottle half=0.04
    bj = bottle.add_freejoint(); bj.name = "bottle_free"
    bg = bottle.add_geom(); bg.name = "bottle_body"
    bg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    bg.size = [0.018, 0.035, 0]; bg.rgba = [0.15, 0.45, 0.85, 0.9]
    bg.mass = 0.12; bg.friction = [2.0, 0.01, 0.001]
    bn = bottle.add_geom(); bn.name = "bottle_neck"
    bn.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    bn.size = [0.009, 0.012, 0]; bn.pos = [0, 0, 0.047]
    bn.rgba = [0.1, 0.35, 0.7, 0.95]; bn.mass = 0.02

    # Hand base
    # Allegro palm is at origin, fingers extend ~11cm in +X direction
    # We mount it fingers-down: rotate 180° around Y so +X becomes -Z (down)
    # Then rotate to face the bottle
    base = w.add_body(); base.name = "hand_base"
    base.pos = [0, 0, 0.42]  # above bottle top (~0.29)

    for jn, ax, jt, jr in [
        ("bx", [1,0,0], mujoco.mjtJoint.mjJNT_SLIDE, [-0.12, 0.12]),
        ("by", [0,1,0], mujoco.mjtJoint.mjJNT_SLIDE, [-0.12, 0.12]),
        ("bz", [0,0,1], mujoco.mjtJoint.mjJNT_SLIDE, [-0.20, 0.10]),
        ("brx", [1,0,0], mujoco.mjtJoint.mjJNT_HINGE, [-1.5, 1.5]),
        ("bry", [0,1,0], mujoco.mjtJoint.mjJNT_HINGE, [-1.5, 1.5]),
        ("brz", [0,0,1], mujoco.mjtJoint.mjJNT_HINGE, [-1.5, 1.5]),
    ]:
        j = base.add_joint(); j.name = jn; j.type = jt; j.axis = ax; j.range = jr

    mount = base.add_body(); mount.name = "hand_mount"
    # Fingers point down (-Z): rotate +X → -Z → euler Y=+90° then flip palm
    rot = Rotation.from_euler('yzx', [90, 0, 180], degrees=True)
    mount.quat = rot.as_quat(scalar_first=True).tolist()

    # Attach Allegro
    hand_root = hand_spec.worldbody.first_body()
    frame = mount.add_frame()
    frame.attach_body(hand_root, "al_", "")

    # Base actuators (high gain for crisp motion)
    for jn, kp in [("bx", 300), ("by", 300), ("bz", 300),
                    ("brx", 100), ("bry", 100), ("brz", 100)]:
        a = s.add_actuator(); a.name = f"act_{jn}"
        a.target = jn; a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gainprm = [kp] + [0]*9

    return s.compile()


def mano_to_allegro(hand_pose_matrices):
    """
    MANO 15 joints → Allegro 16 actuators.
    Allegro: ff(0-3), mf(4-7), rf(8-11), th(12-15)
    Each finger: j0=spread, j1/j2/j3=flex (proximal/medial/distal)
    """
    ctrl = np.zeros(16)

    def flex_spread(rm):
        r = Rotation.from_matrix(rm)
        e = r.as_euler('xyz')
        return e[0], e[2]

    # Index (MANO 0,1,2) → Allegro ff (0,1,2,3)
    f0, s0 = flex_spread(hand_pose_matrices[0])
    f1, _ = flex_spread(hand_pose_matrices[1])
    f2, _ = flex_spread(hand_pose_matrices[2])
    ctrl[0] = np.clip(s0, -0.47, 0.47)
    ctrl[1] = np.clip(f0, -0.196, 1.61)
    ctrl[2] = np.clip(f1, -0.174, 1.71)
    ctrl[3] = np.clip(f2, -0.227, 1.62)

    # Middle (MANO 3,4,5) → Allegro mf (4,5,6,7)
    f0, s0 = flex_spread(hand_pose_matrices[3])
    f1, _ = flex_spread(hand_pose_matrices[4])
    f2, _ = flex_spread(hand_pose_matrices[5])
    ctrl[4] = np.clip(s0, -0.47, 0.47)
    ctrl[5] = np.clip(f0, -0.196, 1.61)
    ctrl[6] = np.clip(f1, -0.174, 1.71)
    ctrl[7] = np.clip(f2, -0.227, 1.62)

    # Ring (MANO 9,10,11) → Allegro rf (8,9,10,11)
    f0, s0 = flex_spread(hand_pose_matrices[9])
    f1, _ = flex_spread(hand_pose_matrices[10])
    f2, _ = flex_spread(hand_pose_matrices[11])
    ctrl[8] = np.clip(s0, -0.47, 0.47)
    ctrl[9] = np.clip(f0, -0.196, 1.61)
    ctrl[10] = np.clip(f1, -0.174, 1.71)
    ctrl[11] = np.clip(f2, -0.227, 1.62)

    # Thumb (MANO 12,13,14) → Allegro th (12,13,14,15)
    f0, s0 = flex_spread(hand_pose_matrices[12])
    f1, _ = flex_spread(hand_pose_matrices[13])
    f2, _ = flex_spread(hand_pose_matrices[14])
    ctrl[12] = np.clip(abs(f0) * 0.8 + 0.3, 0.263, 1.396)  # rotation
    ctrl[13] = np.clip(s0, -0.105, 1.163)
    ctrl[14] = np.clip(f1, -0.189, 1.644)
    ctrl[15] = np.clip(f2, -0.162, 1.719)

    return ctrl


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(MANO_RESULTS) as f:
        all_results = json.load(f)

    frame_data = OrderedDict()
    for r in all_results:
        if not r['is_right']: continue
        frame_data[r['img_name'].replace('.jpg', '')] = r

    names = list(frame_data.keys())
    n = len(names)
    print(f"Frames: {n}")

    # Extract MANO finger poses
    mano_ctrls = []
    for nm in names:
        hp = np.array(frame_data[nm]['mano_hand_pose'])
        mano_ctrls.append(mano_to_allegro(hp))
    mano_ctrls = np.array(mano_ctrls)
    mano_ctrls = uniform_filter1d(mano_ctrls, size=5, axis=0)

    # Design 5-phase trajectory
    # Phase 1 (0-15%):   Start above, fingers open
    # Phase 2 (15-35%):  Descend to bottle, fingers open wide
    # Phase 3 (35-55%):  Close fingers around bottle
    # Phase 4 (55-75%):  Hold grip tight
    # Phase 5 (75-100%): Lift up

    base_traj = np.zeros((n, 6))
    finger_traj = np.zeros((n, 16))

    OPEN_POSE = np.zeros(16)  # all joints at 0 = fingers straight

    GRASP_POSE = np.array([
        0.0, 1.2, 1.3, 1.0,   # ff: straight spread, strong curl
        0.0, 1.2, 1.3, 1.0,   # mf
        0.0, 1.2, 1.3, 1.0,   # rf
        1.0, 0.8, 1.2, 1.0,   # th: opposition + curl
    ])

    for i in range(n):
        t = i / (n - 1)

        if t < 0.15:
            # Phase 1: hover above
            p = t / 0.15
            base_traj[i] = [0, 0, 0, 0, 0, 0]
            finger_traj[i] = OPEN_POSE
        elif t < 0.35:
            # Phase 2: descend, fingers open
            p = (t - 0.15) / 0.20
            base_traj[i, 2] = -0.13 * p  # descend
            finger_traj[i] = OPEN_POSE
        elif t < 0.55:
            # Phase 3: at bottle height, close fingers
            p = (t - 0.35) / 0.20
            base_traj[i, 2] = -0.13
            finger_traj[i] = OPEN_POSE * (1 - p) + GRASP_POSE * p
        elif t < 0.75:
            # Phase 4: hold
            base_traj[i, 2] = -0.13
            finger_traj[i] = GRASP_POSE
        else:
            # Phase 5: lift
            p = (t - 0.75) / 0.25
            base_traj[i, 2] = -0.13 + 0.10 * p  # rise
            finger_traj[i] = GRASP_POSE

    base_traj = uniform_filter1d(base_traj, size=5, axis=0)
    finger_traj = uniform_filter1d(finger_traj, size=5, axis=0)

    # Build model
    model = build_model()
    data = mujoco.MjData(model)
    print(f"Model: joints={model.njnt}, actuators={model.nu}")

    # Print actuator order
    allegro_n = 16
    base_start = allegro_n  # base actuators after allegro's
    for i in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  act[{i:2d}]: {nm}")

    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'bottle')

    # Renderer
    renderer = mujoco.Renderer(model, height=480, width=640)

    cam1 = mujoco.MjvCamera()
    cam1.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam1.lookat[:] = [0, 0, 0.28]
    cam1.distance = 0.50
    cam1.azimuth = 150
    cam1.elevation = -20

    cam2 = mujoco.MjvCamera()
    cam2.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam2.lookat[:] = [0, 0, 0.28]
    cam2.distance = 0.45
    cam2.azimuth = 180
    cam2.elevation = -20

    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w1 = cv2.VideoWriter(os.path.join(OUT_DIR, 'allegro_side.mp4'), fourcc, fps, (640, 480))
    w2 = cv2.VideoWriter(os.path.join(OUT_DIR, 'allegro_front.mp4'), fourcc, fps, (640, 480))

    mujoco.mj_resetData(model, data)
    bz_init = None
    phases = ["HOVER", "DESCEND", "CLOSE", "HOLD", "LIFT"]

    for i in range(n):
        t = i / (n - 1)
        phase_idx = 0 if t < 0.15 else 1 if t < 0.35 else 2 if t < 0.55 else 3 if t < 0.75 else 4

        # Set finger actuators (0..15)
        for j in range(min(allegro_n, model.nu)):
            data.ctrl[j] = finger_traj[i, j]
        # Set base actuators
        for j in range(6):
            if base_start + j < model.nu:
                data.ctrl[base_start + j] = base_traj[i, j]

        for _ in range(12):
            mujoco.mj_step(model, data)

        bz = data.xpos[bottle_id][2] if bottle_id >= 0 else 0
        if bz_init is None: bz_init = bz

        renderer.update_scene(data, cam1)
        img1 = renderer.render()
        w1.write(img1[:, :, ::-1])

        renderer.update_scene(data, cam2)
        img2 = renderer.render()
        w2.write(img2[:, :, ::-1])

        if i % 20 == 0:
            print(f"  {i:3d}/{n} [{phases[phase_idx]:8s}] bz={bz:.3f} base_z={base_traj[i,2]:+.3f} "
                  f"finger[1]={finger_traj[i,1]:.2f}")

        if phase_idx != (0 if (i-1)/(n-1) < 0.15 else 1 if (i-1)/(n-1) < 0.35 else 2 if (i-1)/(n-1) < 0.55 else 3 if (i-1)/(n-1) < 0.75 else 4) or i == 0 or i == n-1:
            cv2.imwrite(os.path.join(OUT_DIR, f'kf_{i:04d}_{phases[phase_idx].lower()}.jpg'), img1[:, :, ::-1])

    w1.release(); w2.release()

    bz_final = data.xpos[bottle_id][2] if bottle_id >= 0 else 0
    lift = bz_final - bz_init
    print(f"\nBottle: {bz_init:.3f} → {bz_final:.3f} ({lift*100:+.1f}cm)")
    print(f"Grasp: {'SUCCESS' if lift > 0.02 else 'FAIL'}")

    # Combined video
    print("\nCombined video...")
    overlays = sorted(glob.glob(os.path.join(BASE_DIR, 'output/pick_bottle_video/*_overlay.jpg')))
    cs = cv2.VideoCapture(os.path.join(OUT_DIR, 'allegro_side.mp4'))
    cf = cv2.VideoCapture(os.path.join(OUT_DIR, 'allegro_front.mp4'))

    pw, ph = 427, 320
    wc = cv2.VideoWriter(os.path.join(OUT_DIR, 'mano_to_allegro.mp4'), fourcc, fps, (pw*3, ph+25))

    for i in range(min(len(overlays), n)):
        t = i/(n-1)
        pi = 0 if t<0.15 else 1 if t<0.35 else 2 if t<0.55 else 3 if t<0.75 else 4

        ov = cv2.resize(cv2.imread(overlays[i]), (pw, ph))
        _, s = cs.read(); _, f = cf.read()
        if s is None or f is None: break
        s = cv2.resize(s, (pw, ph)); f = cv2.resize(f, (pw, ph))

        panels = np.hstack([ov, s, f])
        label = np.ones((25, pw*3, 3), dtype=np.uint8) * 35
        cv2.putText(label, "1. Video + HaMeR", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 220, 0), 1)
        cv2.putText(label, f"2. Allegro [{phases[pi]}]", (pw+5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 220, 0), 1)
        cv2.putText(label, "3. Allegro (front)", (pw*2+5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 220, 0), 1)
        frame = np.vstack([label, panels])
        wc.write(frame)
        if i == n//3 or i == n//2 or i == 2*n//3:
            cv2.imwrite(os.path.join(OUT_DIR, f'combined_{phases[pi].lower()}.jpg'), frame)

    wc.release(); cs.release(); cf.release()

    import subprocess
    subprocess.run(['ffmpeg', '-y', '-i', os.path.join(OUT_DIR, 'mano_to_allegro.mp4'),
                    '-vf', 'fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
                    os.path.join(OUT_DIR, 'mano_to_allegro.gif')], capture_output=True)

    print(f"\nDone! {OUT_DIR}/")
    for fn in sorted(os.listdir(OUT_DIR)):
        print(f"  {fn} ({os.path.getsize(os.path.join(OUT_DIR, fn))/1024:.0f} KB)")


if __name__ == '__main__':
    main()
