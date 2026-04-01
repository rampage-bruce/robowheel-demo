"""
Shadow Hand v3: Fix collision + staged grasp trajectory + contact grasping.
Key fixes:
  1. Table as static geom (not body) for reliable collision
  2. 3-phase trajectory: approach → close fingers → lift
  3. Base Z trajectory designed to reach bottle then lift
  4. Finger curl increases during grasp phase
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

SHADOW_XML = os.path.join(BASE_DIR, 'mujoco_menagerie/shadow_hand/right_hand.xml')
MANO_RESULTS = os.path.join(BASE_DIR, 'output/pick_bottle_video/mano_results.json')
if not os.path.exists(MANO_RESULTS):
    MANO_RESULTS = os.path.join(BASE_DIR, 'test_data/mano_results.json')
OUT_DIR = os.path.join(BASE_DIR, 'output/dexterous_v3')


def build_model():
    """Build model with proper collision setup."""
    os.chdir(os.path.dirname(SHADOW_XML))
    hand_spec = mujoco.MjSpec.from_file(os.path.basename(SHADOW_XML))

    scene_spec = mujoco.MjSpec()
    scene_spec.option.gravity = [0, 0, -9.81]
    scene_spec.option.timestep = 0.002
    # Shadow Hand XML already sets angle=radian in its own compiler

    world = scene_spec.worldbody

    # Lights
    for pos, dir_, diff in [
        ([0, -0.3, 0.8], [0, 0.2, -1], [1, 1, 1]),
        ([0.4, -0.3, 0.6], [-0.2, 0.2, -0.5], [0.5, 0.5, 0.5]),
        ([-0.3, 0.3, 0.5], [0.2, -0.2, -0.5], [0.3, 0.3, 0.3]),
    ]:
        l = world.add_light()
        l.pos = pos; l.dir = dir_; l.diffuse = diff

    # Floor (static geom directly on worldbody = reliable collision)
    floor = world.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [1, 1, 0.01]
    floor.rgba = [0.92, 0.92, 0.92, 1]

    # Table top (static geom on worldbody, NOT a body with joints)
    table_top = world.add_geom()
    table_top.type = mujoco.mjtGeom.mjGEOM_BOX
    table_top.size = [0.20, 0.15, 0.012]
    table_top.pos = [0, 0, 0.30]
    table_top.rgba = [0.35, 0.25, 0.18, 1]
    table_top.friction = [1.5, 0.01, 0.001]

    # Table legs (static)
    for lx, ly in [(-0.17, -0.12), (0.17, -0.12), (-0.17, 0.12), (0.17, 0.12)]:
        leg = world.add_geom()
        leg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        leg.size = [0.012, 0.14, 0]
        leg.pos = [lx, ly, 0.15]
        leg.rgba = [0.5, 0.4, 0.3, 1]

    # Bottle on table (freejoint, with good collision)
    bottle = world.add_body()
    bottle.name = "bottle"
    bottle.pos = [0, 0, 0.365]  # table_top z(0.30) + table_half(0.012) + bottle_half(0.055) ≈ 0.367
    bj = bottle.add_freejoint()
    bj.name = "bottle_free"
    bg = bottle.add_geom()
    bg.name = "bottle_body"
    bg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    bg.size = [0.022, 0.05, 0]
    bg.rgba = [0.15, 0.45, 0.85, 0.9]
    bg.mass = 0.15
    bg.friction = [1.5, 0.01, 0.001]
    # Bottle neck
    bn = bottle.add_geom()
    bn.name = "bottle_neck"
    bn.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    bn.size = [0.011, 0.018, 0]
    bn.pos = [0, 0, 0.068]
    bn.rgba = [0.1, 0.35, 0.7, 0.95]
    bn.mass = 0.02

    # Hand base: positioned closer to bottle, fingers pointing down
    # Shadow Hand forearm is ~0.21m long, fingertips extend ~0.2m beyond wrist
    # Bottle top is at z≈0.43 (table 0.312 + bottle height 0.118)
    # So base at z=0.58, descend -0.20 → fingertips reach z≈0.38 (bottle mid)
    base = world.add_body()
    base.name = "hand_base"
    base.pos = [0, 0, 0.58]

    for jname, axis, jtype, jrange in [
        ("base_x", [1,0,0], mujoco.mjtJoint.mjJNT_SLIDE, [-0.15, 0.15]),
        ("base_y", [0,1,0], mujoco.mjtJoint.mjJNT_SLIDE, [-0.15, 0.15]),
        ("base_z", [0,0,1], mujoco.mjtJoint.mjJNT_SLIDE, [-0.25, 0.15]),
        ("base_rx", [1,0,0], mujoco.mjtJoint.mjJNT_HINGE, [-1.0, 1.0]),
        ("base_ry", [0,1,0], mujoco.mjtJoint.mjJNT_HINGE, [-1.0, 1.0]),
        ("base_rz", [0,0,1], mujoco.mjtJoint.mjJNT_HINGE, [-1.0, 1.0]),
    ]:
        j = base.add_joint()
        j.name = jname; j.type = jtype; j.axis = axis; j.range = jrange

    # Mount with rotation: Shadow Hand default is +X, rotate to fingers-down
    mount = base.add_body()
    mount.name = "hand_mount"
    rot = Rotation.from_euler('xyz', [0, 90, 180], degrees=True)
    mount.quat = rot.as_quat(scalar_first=True).tolist()

    # Attach Shadow Hand
    hand_root = hand_spec.worldbody.first_body()
    frame = mount.add_frame()
    frame.attach_body(hand_root, "sh_", "")

    # Base actuators
    for jname, kp in [("base_x", 200), ("base_y", 200), ("base_z", 200),
                       ("base_rx", 80), ("base_ry", 80), ("base_rz", 80)]:
        act = scene_spec.add_actuator()
        act.name = f"act_{jname}"
        act.target = jname
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gainprm = [kp] + [0]*9

    model = scene_spec.compile()
    return model


def mano_to_fingers(hand_pose_matrices):
    """MANO 15 joint rotations → Shadow 20 actuator values."""
    ctrl = np.zeros(20)

    def flex_spread(rm):
        r = Rotation.from_matrix(rm)
        e = r.as_euler('xyz')
        return e[0], e[2]

    # Wrist neutral (base handles global)
    ctrl[0] = 0.0; ctrl[1] = 0.0

    # Thumb
    f, s = flex_spread(hand_pose_matrices[12])
    ctrl[2] = np.clip(s, -1.05, 1.05)
    ctrl[3] = np.clip(abs(f) * 0.8, 0, 1.22)
    ctrl[4] = 0.0
    f1, _ = flex_spread(hand_pose_matrices[13])
    ctrl[5] = np.clip(f1, -0.70, 0.70)
    f2, _ = flex_spread(hand_pose_matrices[14])
    ctrl[6] = np.clip(f2, -0.26, 1.57)

    # Index
    f, s = flex_spread(hand_pose_matrices[0])
    ctrl[7] = np.clip(s, -0.35, 0.35)
    ctrl[8] = np.clip(f, -0.26, 1.57)
    f1, _ = flex_spread(hand_pose_matrices[1])
    f2, _ = flex_spread(hand_pose_matrices[2])
    ctrl[9] = np.clip((f1+f2)/2, 0, 1.57)

    # Middle
    f, s = flex_spread(hand_pose_matrices[3])
    ctrl[10] = np.clip(s, -0.35, 0.35)
    ctrl[11] = np.clip(f, -0.26, 1.57)
    f1, _ = flex_spread(hand_pose_matrices[4])
    f2, _ = flex_spread(hand_pose_matrices[5])
    ctrl[12] = np.clip((f1+f2)/2, 0, 1.57)

    # Ring
    f, s = flex_spread(hand_pose_matrices[9])
    ctrl[13] = np.clip(s, -0.35, 0.35)
    ctrl[14] = np.clip(f, -0.26, 1.57)
    f1, _ = flex_spread(hand_pose_matrices[10])
    f2, _ = flex_spread(hand_pose_matrices[11])
    ctrl[15] = np.clip((f1+f2)/2, 0, 1.57)

    # Pinky
    f, s = flex_spread(hand_pose_matrices[6])
    ctrl[16] = 0.2
    ctrl[17] = np.clip(s, -0.35, 0.35)
    ctrl[18] = np.clip(f, -0.26, 1.57)
    f1, _ = flex_spread(hand_pose_matrices[7])
    f2, _ = flex_spread(hand_pose_matrices[8])
    ctrl[19] = np.clip((f1+f2)/2, 0, 1.57)

    return ctrl


def design_base_trajectory(n_frames):
    """
    3-phase base trajectory:
      Phase 1 (0-40%):   Approach — descend from above to bottle height
      Phase 2 (40-70%):  Grasp — stay at bottle height (fingers close via MANO)
      Phase 3 (70-100%): Lift — rise up with bottle
    """
    base_traj = np.zeros((n_frames, 6))  # x, y, z, rx, ry, rz

    for i in range(n_frames):
        t = i / (n_frames - 1)  # 0 → 1

        if t < 0.35:
            # Phase 1: Approach - descend to wrap bottle
            phase_t = t / 0.35
            base_traj[i, 2] = -0.22 * phase_t  # z: 0 → -0.22 (deeper descent)
        elif t < 0.65:
            # Phase 2: Grasp - hold at bottle height, fingers close
            base_traj[i, 2] = -0.22
        else:
            # Phase 3: Lift - rise with bottle
            phase_t = (t - 0.65) / 0.35
            base_traj[i, 2] = -0.22 + 0.15 * phase_t  # rise to -0.07

    # Smooth
    base_traj = uniform_filter1d(base_traj, size=5, axis=0)
    return base_traj


def design_finger_trajectory(mano_fingers, n_frames):
    """
    Blend MANO finger poses with grasp phases:
      Phase 1: Open fingers (MANO pose scaled down)
      Phase 2: Close fingers (increase curl toward grasp pose)
      Phase 3: Hold tight grasp
    """
    finger_traj = np.zeros((n_frames, 20))

    for i in range(n_frames):
        t = i / (n_frames - 1)

        # Get MANO finger pose for this frame
        mano_ctrl = mano_fingers[i]

        if t < 0.30:
            # Phase 1: open fingers wide for approach
            open_fingers = np.zeros(20)
            open_fingers[2] = 0.8   # thumb spread wide
            open_fingers[7] = -0.2  # index spread out
            open_fingers[10] = -0.1 # middle spread
            finger_traj[i] = open_fingers * 0.6 + mano_ctrl * 0.4
        elif t < 0.65:
            # Phase 2: closing — blend from open to tight grasp
            phase_t = (t - 0.30) / 0.35
            # Target: strong curl for grasping
            grasp_pose = np.zeros(20)
            grasp_pose[2] = 0.6   # THJ5 thumb rotation
            grasp_pose[3] = 0.9   # THJ4 opposition
            grasp_pose[5] = 0.5   # THJ2 flex
            grasp_pose[6] = 0.8   # THJ1 flex
            grasp_pose[8] = 1.0   # FF MCP flex
            grasp_pose[9] = 1.2   # FF PIP+DIP
            grasp_pose[11] = 1.0  # MF MCP
            grasp_pose[12] = 1.2  # MF PIP+DIP
            grasp_pose[14] = 0.9  # RF MCP
            grasp_pose[15] = 1.1  # RF PIP+DIP
            grasp_pose[18] = 0.8  # LF MCP
            grasp_pose[19] = 1.0  # LF PIP+DIP
            finger_traj[i] = mano_ctrl * (1 - phase_t) + grasp_pose * phase_t
        else:
            # Phase 3: hold tight grasp (same as end of phase 2)
            grasp_pose = np.zeros(20)
            grasp_pose[2] = 0.6
            grasp_pose[3] = 0.9
            grasp_pose[5] = 0.5
            grasp_pose[6] = 0.8
            grasp_pose[8] = 1.0
            grasp_pose[9] = 1.2
            grasp_pose[11] = 1.0
            grasp_pose[12] = 1.2
            grasp_pose[14] = 0.9
            grasp_pose[15] = 1.1
            grasp_pose[18] = 0.8
            grasp_pose[19] = 1.0
            # Keep MANO influence for wrist
            finger_traj[i] = grasp_pose
            finger_traj[i, 0:2] = mano_ctrl[0:2]  # wrist from MANO

    finger_traj = uniform_filter1d(finger_traj, size=5, axis=0)
    return finger_traj


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(MANO_RESULTS) as f:
        all_results = json.load(f)

    frame_data = OrderedDict()
    for r in all_results:
        if not r['is_right']:
            continue
        key = r['img_name'].replace('.jpg', '')
        frame_data[key] = r

    frame_names = list(frame_data.keys())
    n = len(frame_names)
    print(f"Frames: {n}")

    # Extract MANO finger controls
    mano_fingers = []
    for fname in frame_names:
        r = frame_data[fname]
        hp = np.array(r['mano_hand_pose'])
        mano_fingers.append(mano_to_fingers(hp))
    mano_fingers = np.array(mano_fingers)
    mano_fingers = uniform_filter1d(mano_fingers, size=5, axis=0)

    # Design staged trajectories
    base_traj = design_base_trajectory(n)
    finger_traj = design_finger_trajectory(mano_fingers, n)

    print(f"Base traj range: z=[{base_traj[:,2].min():.3f}, {base_traj[:,2].max():.3f}]")
    print(f"Finger traj range: [{finger_traj.min():.2f}, {finger_traj.max():.2f}]")

    # Build model
    model = build_model()
    data = mujoco.MjData(model)

    print(f"Model: joints={model.njnt}, actuators={model.nu}")

    # Identify actuator layout
    shadow_n = 20  # shadow hand actuators (0-19)
    base_start = 20  # base actuators (20-25)

    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'bottle')

    # Renderer
    renderer = mujoco.Renderer(model, height=480, width=640)

    cam_side = mujoco.MjvCamera()
    cam_side.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_side.lookat[:] = [0.0, 0.0, 0.38]
    cam_side.distance = 0.7
    cam_side.azimuth = 150
    cam_side.elevation = -15

    cam_front = mujoco.MjvCamera()
    cam_front.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_front.lookat[:] = [0.0, 0.0, 0.38]
    cam_front.distance = 0.65
    cam_front.azimuth = 180
    cam_front.elevation = -15

    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w_side = cv2.VideoWriter(os.path.join(OUT_DIR, 'shadow_side.mp4'), fourcc, fps, (640, 480))
    w_front = cv2.VideoWriter(os.path.join(OUT_DIR, 'shadow_front.mp4'), fourcc, fps, (640, 480))

    mujoco.mj_resetData(model, data)
    bottle_z_init = None

    for i in range(n):
        # Finger actuators (0-19)
        for j in range(min(shadow_n, model.nu)):
            data.ctrl[j] = finger_traj[i, j]

        # Base actuators (20-25)
        for j in range(6):
            if base_start + j < model.nu:
                data.ctrl[base_start + j] = base_traj[i, j]

        for _ in range(10):
            mujoco.mj_step(model, data)

        bz = data.xpos[bottle_id][2] if bottle_id >= 0 else 0
        if bottle_z_init is None:
            bottle_z_init = bz

        renderer.update_scene(data, cam_side)
        img_s = renderer.render()
        w_side.write(img_s[:, :, ::-1])

        renderer.update_scene(data, cam_front)
        img_f = renderer.render()
        w_front.write(img_f[:, :, ::-1])

        t = i / (n - 1)
        phase = "APPROACH" if t < 0.4 else "GRASP" if t < 0.7 else "LIFT"

        if i % 25 == 0:
            print(f"  Frame {i:3d}/{n} [{phase:8s}] bottle_z={bz:.3f} base_z={base_traj[i,2]:+.3f}")

        if i in [0, n//5, 2*n//5, 3*n//5, 4*n//5, n-1]:
            cv2.imwrite(os.path.join(OUT_DIR, f'kf_{i:04d}_{phase.lower()}.jpg'), img_s[:, :, ::-1])

    w_side.release()
    w_front.release()

    bottle_z_final = data.xpos[bottle_id][2] if bottle_id >= 0 else 0
    lifted = bottle_z_final > bottle_z_init + 0.02
    print(f"\nBottle: z_init={bottle_z_init:.3f} → z_final={bottle_z_final:.3f}, lift={((bottle_z_final-bottle_z_init)*100):.1f}cm")
    print(f"Grasp success: {lifted}")

    # Combined video
    print("\nGenerating combined video...")
    overlay_dir = os.path.join(BASE_DIR, 'output/pick_bottle_video')
    overlays = sorted(glob.glob(os.path.join(overlay_dir, '*_overlay.jpg')))
    cap_s = cv2.VideoCapture(os.path.join(OUT_DIR, 'shadow_side.mp4'))
    cap_f = cv2.VideoCapture(os.path.join(OUT_DIR, 'shadow_front.mp4'))

    pw, ph = 427, 320
    w_c = cv2.VideoWriter(os.path.join(OUT_DIR, 'mano_to_shadow_v3.mp4'), fourcc, fps, (pw*3, ph+25))

    for i in range(min(len(overlays), n)):
        ov = cv2.resize(cv2.imread(overlays[i]), (pw, ph))
        _, s = cap_s.read(); _, f = cap_f.read()
        if s is None or f is None: break
        s = cv2.resize(s, (pw, ph))
        f = cv2.resize(f, (pw, ph))

        panels = np.hstack([ov, s, f])
        t = i / (n-1)
        phase = "APPROACH" if t < 0.4 else "GRASP" if t < 0.7 else "LIFT"
        label = np.ones((25, pw*3, 3), dtype=np.uint8) * 35
        cv2.putText(label, "1. Video + HaMeR MANO", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 220, 0), 1)
        cv2.putText(label, f"2. Shadow Hand [{phase}]", (pw+5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 220, 0), 1)
        cv2.putText(label, "3. Shadow Hand (front)", (pw*2+5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 220, 0), 1)
        frame = np.vstack([label, panels])
        w_c.write(frame)
        if i == n//2:
            cv2.imwrite(os.path.join(OUT_DIR, 'combined_preview.jpg'), frame)

    w_c.release(); cap_s.release(); cap_f.release()

    import subprocess
    subprocess.run(['ffmpeg', '-y', '-i', os.path.join(OUT_DIR, 'mano_to_shadow_v3.mp4'),
                    '-vf', 'fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
                    os.path.join(OUT_DIR, 'mano_to_shadow_v3.gif')], capture_output=True)

    print(f"\nDone! {OUT_DIR}/")
    for fn in sorted(os.listdir(OUT_DIR)):
        print(f"  {fn} ({os.path.getsize(os.path.join(OUT_DIR, fn))/1024:.0f} KB)")


if __name__ == '__main__':
    main()
