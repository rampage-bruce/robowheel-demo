"""
MANO → Shadow Hand v2: correct hand positioning + orientation + camera.
- Adds 6DoF base joint to position/orient the hand
- Maps MANO global_orient to hand base rotation per frame
- Maps MANO finger joints to Shadow actuators
- Positions hand around the bottle for grasping
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
OUT_DIR = os.path.join(BASE_DIR, 'output/dexterous_v2')


def build_model():
    """Build combined model: 6DoF base + Shadow Hand + table + bottle using MjSpec attach."""
    import mujoco

    # Load Shadow Hand spec
    os.chdir(os.path.dirname(SHADOW_XML))
    hand_spec = mujoco.MjSpec.from_file(os.path.basename(SHADOW_XML))

    # Create scene spec
    scene_spec = mujoco.MjSpec()
    scene_spec.option.gravity = [0, 0, -9.81]
    scene_spec.option.timestep = 0.002

    # Lights
    world = scene_spec.worldbody
    l1 = world.add_light()
    l1.pos = [0, 0, 0.8]; l1.dir = [0, 0, -1]; l1.diffuse = [1, 1, 1]
    l2 = world.add_light()
    l2.pos = [0.3, -0.4, 0.6]; l2.dir = [-0.2, 0.3, -0.5]; l2.diffuse = [0.5, 0.5, 0.5]

    # Floor
    floor = world.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [1, 1, 0.01]
    floor.rgba = [0.95, 0.95, 0.95, 1]

    # Table
    table_body = world.add_body()
    table_body.name = "table"
    table_body.pos = [0, 0, 0.25]
    tg = table_body.add_geom()
    tg.type = mujoco.mjtGeom.mjGEOM_BOX
    tg.size = [0.25, 0.20, 0.01]
    tg.rgba = [0.35, 0.25, 0.18, 1]

    # Table legs
    for lx, ly in [(-0.22, -0.17), (0.22, -0.17), (-0.22, 0.17), (0.22, 0.17)]:
        lg = table_body.add_geom()
        lg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        lg.size = [0.015, 0.12, 0]
        lg.pos = [lx, ly, -0.13]
        lg.rgba = [0.5, 0.4, 0.3, 1]

    # Bottle
    bottle_body = world.add_body()
    bottle_body.name = "bottle"
    bottle_body.pos = [0, 0, 0.31]
    bj = bottle_body.add_freejoint()
    bj.name = "bottle_free"
    bg = bottle_body.add_geom()
    bg.name = "bottle_body"
    bg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    bg.size = [0.022, 0.055, 0]
    bg.rgba = [0.2, 0.5, 0.85, 0.85]
    bg.mass = 0.25
    bn = bottle_body.add_geom()
    bn.name = "bottle_neck"
    bn.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    bn.size = [0.012, 0.02, 0]
    bn.pos = [0, 0, 0.075]
    bn.rgba = [0.15, 0.4, 0.75, 0.9]
    bn.mass = 0.03

    # Hand base with 6DoF joints
    base = world.add_body()
    base.name = "hand_base"
    base.pos = [0, 0, 0.55]

    for jname, axis, jtype in [
        ("base_x", [1,0,0], mujoco.mjtJoint.mjJNT_SLIDE),
        ("base_y", [0,1,0], mujoco.mjtJoint.mjJNT_SLIDE),
        ("base_z", [0,0,1], mujoco.mjtJoint.mjJNT_SLIDE),
        ("base_rx", [1,0,0], mujoco.mjtJoint.mjJNT_HINGE),
        ("base_ry", [0,1,0], mujoco.mjtJoint.mjJNT_HINGE),
        ("base_rz", [0,0,1], mujoco.mjtJoint.mjJNT_HINGE),
    ]:
        j = base.add_joint()
        j.name = jname
        j.type = jtype
        j.axis = axis
        j.range = [-0.3, 0.3] if jtype == mujoco.mjtJoint.mjJNT_SLIDE else [-3.14, 3.14]

    # Mount body with rotation (fingers point down)
    mount = base.add_body()
    mount.name = "hand_mount"
    # Rotate mount: fingers point down. euler → quaternion (wxyz)
    from scipy.spatial.transform import Rotation as R
    rot = R.from_euler('xyz', [0, 90, 180], degrees=True)
    mount.quat = rot.as_quat(scalar_first=True).tolist()

    # Attach Shadow Hand to mount
    # Find the root body of Shadow Hand (rh_forearm)
    hand_root = hand_spec.worldbody.first_body()
    attachment_frame = mount.add_frame()
    attachment_frame.attach_body(hand_root, "sh_", "")

    # Add base actuators
    for jname, kp in [("base_x", 100), ("base_y", 100), ("base_z", 100),
                       ("base_rx", 50), ("base_ry", 50), ("base_rz", 50)]:
        act = scene_spec.add_actuator()
        act.name = f"act_{jname}"
        act.target = jname
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gainprm = [kp] + [0]*9

    model = scene_spec.compile()
    return model


def mano_to_shadow_fingers(hand_pose_matrices):
    """Convert MANO 15 finger rotations → Shadow 20 finger actuator values."""
    ctrl = np.zeros(20)

    def get_flex_spread(rot_mat):
        rot = Rotation.from_matrix(rot_mat)
        euler = rot.as_euler('xyz')
        return euler[0], euler[2]  # flex, spread

    # Wrist (set to neutral, base joint handles global orientation)
    ctrl[0] = 0.0  # WRJ2
    ctrl[1] = 0.0  # WRJ1

    # Thumb (MANO joints 12, 13, 14)
    th0_flex, th0_spread = get_flex_spread(hand_pose_matrices[12])
    th1_flex, _ = get_flex_spread(hand_pose_matrices[13])
    th2_flex, _ = get_flex_spread(hand_pose_matrices[14])
    ctrl[2] = np.clip(th0_spread, -1.05, 1.05)
    ctrl[3] = np.clip(abs(th0_flex) * 0.8, 0, 1.22)
    ctrl[4] = 0.0
    ctrl[5] = np.clip(th1_flex, -0.70, 0.70)
    ctrl[6] = np.clip(th2_flex, -0.26, 1.57)

    # Index (MANO joints 0, 1, 2)
    ff0_flex, ff0_spread = get_flex_spread(hand_pose_matrices[0])
    ff1_flex, _ = get_flex_spread(hand_pose_matrices[1])
    ff2_flex, _ = get_flex_spread(hand_pose_matrices[2])
    ctrl[7] = np.clip(ff0_spread, -0.35, 0.35)
    ctrl[8] = np.clip(ff0_flex, -0.26, 1.57)
    ctrl[9] = np.clip((ff1_flex + ff2_flex) / 2, 0, 1.57)

    # Middle (MANO joints 3, 4, 5)
    mf0_flex, mf0_spread = get_flex_spread(hand_pose_matrices[3])
    mf1_flex, _ = get_flex_spread(hand_pose_matrices[4])
    mf2_flex, _ = get_flex_spread(hand_pose_matrices[5])
    ctrl[10] = np.clip(mf0_spread, -0.35, 0.35)
    ctrl[11] = np.clip(mf0_flex, -0.26, 1.57)
    ctrl[12] = np.clip((mf1_flex + mf2_flex) / 2, 0, 1.57)

    # Ring (MANO joints 9, 10, 11)
    rf0_flex, rf0_spread = get_flex_spread(hand_pose_matrices[9])
    rf1_flex, _ = get_flex_spread(hand_pose_matrices[10])
    rf2_flex, _ = get_flex_spread(hand_pose_matrices[11])
    ctrl[13] = np.clip(rf0_spread, -0.35, 0.35)
    ctrl[14] = np.clip(rf0_flex, -0.26, 1.57)
    ctrl[15] = np.clip((rf1_flex + rf2_flex) / 2, 0, 1.57)

    # Little/Pinky (MANO joints 6, 7, 8)
    lf0_flex, lf0_spread = get_flex_spread(hand_pose_matrices[6])
    lf1_flex, _ = get_flex_spread(hand_pose_matrices[7])
    lf2_flex, _ = get_flex_spread(hand_pose_matrices[8])
    ctrl[16] = 0.2
    ctrl[17] = np.clip(lf0_spread, -0.35, 0.35)
    ctrl[18] = np.clip(lf0_flex, -0.26, 1.57)
    ctrl[19] = np.clip((lf1_flex + lf2_flex) / 2, 0, 1.57)

    return ctrl


def mano_to_base_pose(global_orient, cam_t, is_grasping):
    """
    Convert MANO global orientation + camera translation → base 6DoF.
    Maps hand position relative to the bottle.
    """
    # Extract wrist rotation from MANO global_orient
    go = np.array(global_orient)
    if go.ndim == 3:
        go = go[0]
    rot = Rotation.from_matrix(go)
    euler = rot.as_euler('xyz')

    # Base rotation: map MANO wrist rotation to base hinge joints
    # Scale down to reasonable range
    base_rx = np.clip(euler[0] * 0.3, -0.5, 0.5)
    base_ry = np.clip(euler[1] * 0.3, -0.5, 0.5)
    base_rz = np.clip(euler[2] * 0.3, -0.8, 0.8)

    # Base position: derive from camera translation
    # cam_t[0]/cam_t[2] gives normalized X offset, cam_t[1]/cam_t[2] gives Y
    cx = cam_t[0] / cam_t[2] * 0.1  # scale to hand workspace
    cy = cam_t[1] / cam_t[2] * 0.1
    # Z: lower when grasping, higher when not
    cz = -0.15 if is_grasping else -0.05

    base_x = np.clip(cx, -0.15, 0.15)
    base_y = np.clip(cy, -0.15, 0.15)
    base_z = np.clip(cz, -0.25, 0.1)

    return np.array([base_x, base_y, base_z, base_rx, base_ry, base_rz])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(MANO_RESULTS) as f:
        all_results = json.load(f)

    # Right hand frames
    frame_data = OrderedDict()
    for r in all_results:
        if not r['is_right']:
            continue
        key = r['img_name'].replace('.jpg', '')
        frame_data[key] = r

    frame_names = list(frame_data.keys())
    print(f"Frames: {len(frame_names)}")

    # Convert all frames
    finger_controls = []
    base_controls = []

    for i, fname in enumerate(frame_names):
        r = frame_data[fname]
        hp = np.array(r['mano_hand_pose'])
        go = np.array(r['mano_global_orient'])
        cam_t = np.array(r['cam_t_full'])

        fc = mano_to_shadow_fingers(hp)
        finger_controls.append(fc)

        # Estimate if grasping: check thumb-index distance proxy from finger curl
        avg_curl = np.mean(np.abs(fc[6:]))  # average finger actuation
        is_grasping = avg_curl > 0.15

        bc = mano_to_base_pose(go, cam_t, is_grasping)
        base_controls.append(bc)

    finger_controls = np.array(finger_controls)
    base_controls = np.array(base_controls)

    # Smooth
    finger_smooth = uniform_filter1d(finger_controls, size=5, axis=0)
    base_smooth = uniform_filter1d(base_controls, size=7, axis=0)

    print(f"Finger ctrl range: [{finger_smooth.min():.2f}, {finger_smooth.max():.2f}]")
    print(f"Base ctrl range: [{base_smooth.min():.2f}, {base_smooth.max():.2f}]")

    # Build MuJoCo model using MjSpec attach
    model = build_model()
    data = mujoco.MjData(model)

    print(f"Model: joints={model.njnt}, actuators={model.nu}, bodies={model.nbody}")

    # Find our base actuators (first 6) and shadow actuators (remaining 20)
    print("Actuators:")
    for i in range(min(model.nu, 30)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i}: {name}")

    # Setup renderer
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Camera: front-side view showing hand descending on bottle
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.35]
    cam.distance = 0.65
    cam.azimuth = 160
    cam.elevation = -20

    # Camera 2: front view
    cam2 = mujoco.MjvCamera()
    cam2.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam2.lookat[:] = [0.0, 0.0, 0.35]
    cam2.distance = 0.60
    cam2.azimuth = 180
    cam2.elevation = -15

    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer1 = cv2.VideoWriter(os.path.join(OUT_DIR, 'shadow_side.mp4'), fourcc, fps, (640, 480))
    writer2 = cv2.VideoWriter(os.path.join(OUT_DIR, 'shadow_front.mp4'), fourcc, fps, (640, 480))

    mujoco.mj_resetData(model, data)
    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'bottle')

    for i in range(len(finger_smooth)):
        # Shadow Hand finger actuators are indices 0-19
        for j in range(20):
            if j < model.nu:
                data.ctrl[j] = finger_smooth[i, j]

        # Base actuators are indices 20-25
        for j in range(6):
            if 20 + j < model.nu:
                data.ctrl[20 + j] = base_smooth[i, j]

        # Step
        for _ in range(10):
            mujoco.mj_step(model, data)

        # Render
        renderer.update_scene(data, cam)
        img1 = renderer.render()
        writer1.write(img1[:, :, ::-1])

        renderer.update_scene(data, cam2)
        img2 = renderer.render()
        writer2.write(img2[:, :, ::-1])

        if i % 30 == 0:
            bz = data.xpos[bottle_id][2] if bottle_id >= 0 else 0
            print(f"  Frame {i}/{len(finger_smooth)}, bottle_z={bz:.3f}")

        if i in [0, 40, 75, 110, 150]:
            cv2.imwrite(os.path.join(OUT_DIR, f'keyframe_{i:04d}_side.jpg'), img1[:, :, ::-1])
            cv2.imwrite(os.path.join(OUT_DIR, f'keyframe_{i:04d}_front.jpg'), img2[:, :, ::-1])

    writer1.release()
    writer2.release()

    # Combined video: HaMeR overlay | Shadow side | Shadow front
    print("\nGenerating combined video...")
    overlay_dir = os.path.join(BASE_DIR, 'output/pick_bottle_video')
    overlays = sorted(glob.glob(os.path.join(overlay_dir, '*_overlay.jpg')))

    cap1 = cv2.VideoCapture(os.path.join(OUT_DIR, 'shadow_side.mp4'))
    cap2 = cv2.VideoCapture(os.path.join(OUT_DIR, 'shadow_front.mp4'))

    pw, ph = 427, 320
    writer_c = cv2.VideoWriter(os.path.join(OUT_DIR, 'mano_to_shadow_v2.mp4'),
                                fourcc, fps, (pw * 3, ph + 25))

    n = min(len(overlays), len(finger_smooth))
    for i in range(n):
        ov = cv2.resize(cv2.imread(overlays[i]), (pw, ph))
        ret1, s1 = cap1.read()
        ret2, s2 = cap2.read()
        if not ret1 or not ret2:
            break
        s1 = cv2.resize(s1, (pw, ph))
        s2 = cv2.resize(s2, (pw, ph))

        panels = np.hstack([ov, s1, s2])
        label = np.ones((25, pw * 3, 3), dtype=np.uint8) * 35
        cv2.putText(label, "1. Video + HaMeR MANO", (5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 220, 0), 1)
        cv2.putText(label, "2. Shadow Hand (side)", (pw + 5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 220, 0), 1)
        cv2.putText(label, "3. Shadow Hand (front)", (pw * 2 + 5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 220, 0), 1)

        frame = np.vstack([label, panels])
        writer_c.write(frame)

        if i == n // 2:
            cv2.imwrite(os.path.join(OUT_DIR, 'combined_preview.jpg'), frame)

    writer_c.release()
    cap1.release()
    cap2.release()

    # GIF
    import subprocess
    subprocess.run([
        'ffmpeg', '-y', '-i', os.path.join(OUT_DIR, 'mano_to_shadow_v2.mp4'),
        '-vf', 'fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
        os.path.join(OUT_DIR, 'mano_to_shadow_v2.gif')
    ], capture_output=True)

    print(f"\nDone! Output in {OUT_DIR}/")
    for fn in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, fn))
        print(f"  {fn} ({size/1024:.0f} KB)")


if __name__ == '__main__':
    main()
