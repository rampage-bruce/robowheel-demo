"""
MANO → Shadow Hand dexterous retargeting + MuJoCo simulation.
Maps MANO 15-joint hand pose to Shadow Hand 20 actuators.
Renders full trajectory video with bottle grasping.
"""
import os, json, glob
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
import cv2
from scipy.spatial.transform import Rotation
from collections import OrderedDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SHADOW_XML = os.path.join(BASE_DIR, 'mujoco_menagerie/shadow_hand/right_hand.xml')
MANO_RESULTS = os.path.join(BASE_DIR, 'output/pick_bottle_video/mano_results.json')
if not os.path.exists(MANO_RESULTS):
    MANO_RESULTS = os.path.join(BASE_DIR, 'test_data/mano_results.json')
OUT_DIR = os.path.join(BASE_DIR, 'output/dexterous_sim')

###############################################################################
# MANO → Shadow Hand joint mapping
###############################################################################

# MANO finger order: index(0-2), middle(3-5), pinky(6-8), ring(9-11), thumb(12-14)
# Shadow actuator order: WR(0-1), TH(2-6), FF(7-9), MF(10-12), RF(13-15), LF(16-19)
#
# Shadow actuators:
#   0: rh_A_WRJ2 (wrist deviation)    1: rh_A_WRJ1 (wrist flex)
#   2: rh_A_THJ5 (thumb rotation)     3: rh_A_THJ4 (thumb spread)
#   4: rh_A_THJ3 (thumb MCP twist)    5: rh_A_THJ2 (thumb MCP flex)
#   6: rh_A_THJ1 (thumb IP flex)
#   7: rh_A_FFJ4 (index spread)       8: rh_A_FFJ3 (index MCP flex)
#   9: rh_A_FFJ0 (index PIP+DIP coupled)
#  10: rh_A_MFJ4 (middle spread)     11: rh_A_MFJ3 (middle MCP flex)
#  12: rh_A_MFJ0 (middle PIP+DIP coupled)
#  13: rh_A_RFJ4 (ring spread)       14: rh_A_RFJ3 (ring MCP flex)
#  15: rh_A_RFJ0 (ring PIP+DIP coupled)
#  16: rh_A_LFJ5 (little metacarpal)  17: rh_A_LFJ4 (little spread)
#  18: rh_A_LFJ3 (little MCP flex)    19: rh_A_LFJ0 (little PIP+DIP coupled)

def mano_to_shadow(hand_pose_matrices, global_orient_matrix):
    """
    Convert MANO 15 rotation matrices → Shadow Hand 20 actuator values.
    hand_pose_matrices: (15, 3, 3) rotation matrices
    global_orient_matrix: (3, 3) wrist rotation
    Returns: (20,) actuator control values
    """
    ctrl = np.zeros(20)

    # Extract euler angles from each MANO joint
    def get_flex_spread(rot_mat):
        rot = Rotation.from_matrix(rot_mat)
        euler = rot.as_euler('xyz')  # radians
        flex = euler[0]    # X-axis = flexion/extension
        spread = euler[2]  # Z-axis = abduction/adduction
        return flex, spread

    # Wrist from global orient
    wrist_rot = Rotation.from_matrix(global_orient_matrix)
    wrist_euler = wrist_rot.as_euler('xyz')
    ctrl[0] = np.clip(wrist_euler[2], -0.52, 0.17)   # WRJ2: wrist deviation
    ctrl[1] = np.clip(wrist_euler[0], -0.70, 0.49)   # WRJ1: wrist flex

    # Thumb (MANO joints 12, 13, 14)
    th_mcp_flex, th_mcp_spread = get_flex_spread(hand_pose_matrices[12])
    th_pip_flex, _ = get_flex_spread(hand_pose_matrices[13])
    th_dip_flex, _ = get_flex_spread(hand_pose_matrices[14])
    ctrl[2] = np.clip(th_mcp_spread, -1.05, 1.05)     # THJ5: thumb rotation
    ctrl[3] = np.clip(abs(th_mcp_flex) * 0.8, 0, 1.22) # THJ4: thumb spread/opposition
    ctrl[4] = 0.0                                       # THJ3: twist (no MANO equiv)
    ctrl[5] = np.clip(th_pip_flex, -0.70, 0.70)        # THJ2: thumb MCP flex
    ctrl[6] = np.clip(th_dip_flex, -0.26, 1.57)        # THJ1: thumb IP flex

    # Index finger (MANO joints 0, 1, 2) → Shadow FF
    ff_mcp_flex, ff_spread = get_flex_spread(hand_pose_matrices[0])
    ff_pip_flex, _ = get_flex_spread(hand_pose_matrices[1])
    ff_dip_flex, _ = get_flex_spread(hand_pose_matrices[2])
    ctrl[7] = np.clip(ff_spread, -0.35, 0.35)          # FFJ4: spread
    ctrl[8] = np.clip(ff_mcp_flex, -0.26, 1.57)        # FFJ3: MCP flex
    ctrl[9] = np.clip((ff_pip_flex + ff_dip_flex) / 2, 0, 1.57)  # FFJ0: coupled PIP+DIP

    # Middle finger (MANO joints 3, 4, 5) → Shadow MF
    mf_mcp_flex, mf_spread = get_flex_spread(hand_pose_matrices[3])
    mf_pip_flex, _ = get_flex_spread(hand_pose_matrices[4])
    mf_dip_flex, _ = get_flex_spread(hand_pose_matrices[5])
    ctrl[10] = np.clip(mf_spread, -0.35, 0.35)
    ctrl[11] = np.clip(mf_mcp_flex, -0.26, 1.57)
    ctrl[12] = np.clip((mf_pip_flex + mf_dip_flex) / 2, 0, 1.57)

    # Ring finger (MANO joints 9, 10, 11) → Shadow RF
    rf_mcp_flex, rf_spread = get_flex_spread(hand_pose_matrices[9])
    rf_pip_flex, _ = get_flex_spread(hand_pose_matrices[10])
    rf_dip_flex, _ = get_flex_spread(hand_pose_matrices[11])
    ctrl[13] = np.clip(rf_spread, -0.35, 0.35)
    ctrl[14] = np.clip(rf_mcp_flex, -0.26, 1.57)
    ctrl[15] = np.clip((rf_pip_flex + rf_dip_flex) / 2, 0, 1.57)

    # Little/Pinky finger (MANO joints 6, 7, 8) → Shadow LF
    lf_mcp_flex, lf_spread = get_flex_spread(hand_pose_matrices[6])
    lf_pip_flex, _ = get_flex_spread(hand_pose_matrices[7])
    lf_dip_flex, _ = get_flex_spread(hand_pose_matrices[8])
    ctrl[16] = 0.2  # LFJ5: metacarpal (slight curl)
    ctrl[17] = np.clip(lf_spread, -0.35, 0.35)
    ctrl[18] = np.clip(lf_mcp_flex, -0.26, 1.57)
    ctrl[19] = np.clip((lf_pip_flex + lf_dip_flex) / 2, 0, 1.57)

    return ctrl


###############################################################################
# MuJoCo scene with Shadow Hand + bottle
###############################################################################

def create_scene_xml():
    return f"""
    <mujoco model="shadow_grasp">
      <include file="{SHADOW_XML}"/>
      <option gravity="0 0 -9.81" timestep="0.002"/>
      <worldbody>
        <light pos="0 0 0.5" dir="0 0 -1" diffuse="1 1 1"/>
        <light pos="0.3 -0.3 0.5" dir="-0.3 0.3 -0.5" diffuse="0.6 0.6 0.6"/>
        <!-- Table -->
        <body name="table" pos="0 0 -0.08">
          <geom type="box" size="0.25 0.25 0.02" rgba="0.35 0.25 0.18 1"/>
        </body>
        <!-- Bottle -->
        <body name="bottle" pos="0.04 0 0.0">
          <joint name="bottle_free" type="free"/>
          <geom name="bottle_body" type="cylinder" size="0.02 0.05"
                rgba="0.3 0.6 0.9 0.8" mass="0.2" friction="1 0.005 0.0001"/>
        </body>
      </worldbody>
    </mujoco>
    """


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load MANO results
    with open(MANO_RESULTS) as f:
        all_results = json.load(f)

    # Group by frame, right hand only
    frame_data = OrderedDict()
    for r in all_results:
        if not r['is_right']:
            continue
        key = r['img_name'].replace('.jpg', '')
        frame_data[key] = r

    frame_names = list(frame_data.keys())
    print(f"Right hand frames: {len(frame_names)}")

    # Convert all frames to Shadow Hand controls
    shadow_controls = []
    for fname in frame_names:
        r = frame_data[fname]
        hp = np.array(r['mano_hand_pose'])      # (15, 3, 3)
        go = np.array(r['mano_global_orient'])   # (1, 3, 3) or (3, 3)
        if go.ndim == 3:
            go = go[0]
        ctrl = mano_to_shadow(hp, go)
        shadow_controls.append(ctrl)

    shadow_controls = np.array(shadow_controls)
    print(f"Shadow controls shape: {shadow_controls.shape}")
    print(f"Control range: [{shadow_controls.min():.2f}, {shadow_controls.max():.2f}]")

    # Smooth controls temporally
    from scipy.ndimage import uniform_filter1d
    shadow_controls_smooth = uniform_filter1d(shadow_controls, size=5, axis=0)

    # Load MuJoCo model
    xml_str = create_scene_xml()
    orig_dir = os.getcwd()
    os.chdir(os.path.dirname(SHADOW_XML))
    model = mujoco.MjModel.from_xml_string(xml_str)
    os.chdir(orig_dir)
    data = mujoco.MjData(model)

    print(f"\nModel: joints={model.njnt}, actuators={model.nu}, bodies={model.nbody}")

    # Setup renderer
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Camera: looking at the hand from the front
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.05]
    cam.distance = 0.45
    cam.azimuth = 150
    cam.elevation = -20

    cam_top = mujoco.MjvCamera()
    cam_top.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_top.lookat[:] = [0.0, 0.0, 0.05]
    cam_top.distance = 0.40
    cam_top.azimuth = 180
    cam_top.elevation = -35

    # Video writers
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(os.path.join(OUT_DIR, 'shadow_hand_grasp.mp4'),
                             fourcc, fps, (640, 480))
    writer_top = cv2.VideoWriter(os.path.join(OUT_DIR, 'shadow_hand_top.mp4'),
                                 fourcc, fps, (640, 480))

    # Simulate
    mujoco.mj_resetData(model, data)
    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'bottle')

    for i, ctrl in enumerate(shadow_controls_smooth):
        # Set actuator controls
        n_act = min(len(ctrl), model.nu)
        data.ctrl[:n_act] = ctrl[:n_act]

        # Step physics
        for _ in range(10):
            mujoco.mj_step(model, data)

        # Render
        renderer.update_scene(data, cam)
        img = renderer.render()
        writer.write(img[:, :, ::-1])

        renderer.update_scene(data, cam_top)
        img_top = renderer.render()
        writer_top.write(img_top[:, :, ::-1])

        if i % 30 == 0:
            print(f"  Frame {i}/{len(shadow_controls_smooth)}")
            bottle_z = data.xpos[bottle_id][2]
            print(f"    Bottle z={bottle_z:.3f}, ctrl_sample={ctrl[:5].round(2)}")

        # Save keyframes
        if i in [0, len(shadow_controls_smooth)//4, len(shadow_controls_smooth)//2,
                 3*len(shadow_controls_smooth)//4, len(shadow_controls_smooth)-1]:
            cv2.imwrite(os.path.join(OUT_DIR, f'keyframe_{i:04d}.jpg'), img[:, :, ::-1])

    writer.release()
    writer_top.release()

    # Save retargeting data
    retarget_data = {
        "source": "MANO_HaMeR → Shadow Hand retarget",
        "n_frames": len(shadow_controls),
        "shadow_actuators": 20,
        "mano_joints": 15,
        "controls": shadow_controls.tolist(),
    }
    with open(os.path.join(OUT_DIR, 'shadow_retarget.json'), 'w') as f:
        json.dump(retarget_data, f)

    print(f"\nDone! Output in {OUT_DIR}/")
    for fn in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, fn))
        print(f"  {fn} ({size/1024:.0f} KB)")

    # Generate combined video: overlay | shadow_hand
    print("\nGenerating combined video...")
    overlay_dir = os.path.join(BASE_DIR, 'output/pick_bottle_video')
    overlays = sorted(glob.glob(os.path.join(overlay_dir, '*_overlay.jpg')))

    cap_sh = cv2.VideoCapture(os.path.join(OUT_DIR, 'shadow_hand_grasp.mp4'))
    cap_top = cv2.VideoCapture(os.path.join(OUT_DIR, 'shadow_hand_top.mp4'))

    pw, ph = 427, 360
    writer_combined = cv2.VideoWriter(
        os.path.join(OUT_DIR, 'mano_to_shadow_combined.mp4'),
        fourcc, fps, (pw * 3, ph + 25))

    n_out = min(len(overlays), len(shadow_controls_smooth))
    for i in range(n_out):
        ov = cv2.resize(cv2.imread(overlays[i]), (pw, ph))
        ret1, sh = cap_sh.read()
        ret2, top = cap_top.read()
        if not ret1 or not ret2:
            break
        sh = cv2.resize(sh, (pw, ph))
        top = cv2.resize(top, (pw, ph))

        panels = np.hstack([ov, sh, top])
        label = np.ones((25, pw * 3, 3), dtype=np.uint8) * 35
        cv2.putText(label, "1. HaMeR MANO Hand", (5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
        cv2.putText(label, "2. Shadow Hand (retargeted)", (pw + 5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
        cv2.putText(label, "3. Shadow Hand (top view)", (pw * 2 + 5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)

        frame = np.vstack([label, panels])
        writer_combined.write(frame)

        if i == n_out // 2:
            cv2.imwrite(os.path.join(OUT_DIR, 'combined_preview.jpg'), frame)

    writer_combined.release()
    cap_sh.release()
    cap_top.release()
    print(f"  Saved: mano_to_shadow_combined.mp4")


if __name__ == '__main__':
    main()
