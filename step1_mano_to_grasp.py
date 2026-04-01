"""
Step 1: MANO hand parameters → 6DoF Franka gripper grasp pose
Extracts grasp pose from thumb_tip + index_tip joint positions.
Output format compatible with SynGrasp/CuRobo pipeline.
"""
import os, sys, json, torch, numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    hamer_dir = os.path.join(BASE_DIR, 'hamer')
    sys.path.insert(0, hamer_dir)
    os.chdir(hamer_dir)
    from hamer.models import load_hamer, DEFAULT_CHECKPOINT

    print("Loading MANO model...")
    model, cfg = load_hamer(DEFAULT_CHECKPOINT)
    mano = model.mano

    results_path = os.path.join(BASE_DIR, 'output/pick_bottle_video/mano_results.json')
    with open(results_path) as f:
        all_results = json.load(f)

    # Find best grasping frame: two hands, smallest thumb-index distance = tightest grasp
    best_grasp = None
    best_width = float('inf')
    best_frame = None

    print("Scanning for best grasp frame...")
    for r in all_results:
        if not r['is_right']:
            continue  # Use right hand as primary grasper

        hp = torch.tensor(r['mano_hand_pose'], dtype=torch.float32).unsqueeze(0)
        bt = torch.tensor(r['mano_betas'], dtype=torch.float32).unsqueeze(0)
        go = torch.tensor(r['mano_global_orient'], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            out = mano(hand_pose=hp, betas=bt, global_orient=go)
        joints = out.joints[0].numpy()  # (21, 3)

        thumb_tip = joints[20]
        index_tip = joints[16]
        width = np.linalg.norm(thumb_tip - index_tip)

        # Want tight grasp (small width) but not zero (hand must be shaped)
        if 0.015 < width < best_width:
            best_width = width
            best_grasp = r
            best_frame = r['img_name']
            best_joints = joints

    if best_grasp is None:
        print("ERROR: No suitable grasp found")
        return

    print(f"Best grasp: {best_frame}, width={best_width*100:.1f} cm")

    # Extract grasp pose from MANO joints
    thumb_tip = best_joints[20]
    index_tip = best_joints[16]
    wrist = best_joints[0]
    middle_tip = best_joints[17]

    grasp_center = (thumb_tip + index_tip) / 2
    grasp_width = np.linalg.norm(thumb_tip - index_tip)

    # Build rotation matrix for Franka gripper
    # Gripper X-axis: along grasp axis (thumb → index)
    grasp_axis = (index_tip - thumb_tip)
    grasp_axis /= np.linalg.norm(grasp_axis)

    # Gripper Z-axis: approach direction (wrist → grasp center)
    approach = grasp_center - wrist
    approach /= np.linalg.norm(approach)

    # Gripper Y-axis: cross product
    y_axis = np.cross(approach, grasp_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Re-orthogonalize approach
    approach = np.cross(grasp_axis, y_axis)
    approach /= np.linalg.norm(approach)

    rotation_matrix = np.array([grasp_axis, y_axis, approach]).T  # 3x3

    print(f"\n=== Extracted Grasp Pose (MANO local frame) ===")
    print(f"Grasp center: {grasp_center.round(4)}")
    print(f"Grasp width:  {grasp_width*100:.2f} cm")
    print(f"Grasp axis:   {grasp_axis.round(4)}")
    print(f"Approach dir:  {approach.round(4)}")

    # Convert to Franka workspace coordinates
    # MANO local frame is ~10cm scale, Franka workspace is ~40-60cm from base
    # Place object at typical tabletop position
    obj_x, obj_y, obj_z = 0.4, 0.0, 0.55  # same as SynGrasp default

    # Franka gripper points DOWN for tabletop grasps
    # rotation_matrix for downward grasp:
    #   gripper Z-axis = [0, 0, -1] (pointing down)
    #   gripper X-axis = grasp_axis projected onto horizontal plane
    #   gripper Y-axis = cross(Z, X)

    # Project grasp_axis to horizontal (ignore Z component from MANO)
    horiz_axis = np.array([grasp_axis[0], grasp_axis[1], 0])
    if np.linalg.norm(horiz_axis) < 1e-6:
        horiz_axis = np.array([1, 0, 0])
    horiz_axis /= np.linalg.norm(horiz_axis)

    franka_z = np.array([0, 0, -1])  # pointing down
    franka_x = horiz_axis
    franka_y = np.cross(franka_z, franka_x)
    franka_y /= np.linalg.norm(franka_y)
    franka_x = np.cross(franka_y, franka_z)

    franka_rot = np.array([franka_x, franka_y, franka_z]).T

    # Convert rotation matrix to quaternion (wxyz for CuRobo)
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(franka_rot)
    quat_xyzw = r.as_quat()  # scipy returns xyzw
    quat_wxyz = [float(quat_xyzw[3]), float(quat_xyzw[0]),
                 float(quat_xyzw[1]), float(quat_xyzw[2])]

    # Clamp grasp width to Franka gripper range [0.01, 0.08]
    franka_width = np.clip(grasp_width, 0.01, 0.08)

    # EE offset: CuRobo panda_link8 vs fingertip = 0.173m
    EE_OFFSET = 0.173

    grasp_pose = {
        "source": "MANO_HaMeR",
        "source_frame": best_frame,
        "mano_grasp_width_m": float(grasp_width),
        "franka_grasp_width_m": float(franka_width),

        # SynGrasp compatible format (object local frame)
        "syngrasp_format": {
            "position": [0.0, 0.0, 0.0],
            "rotation_matrix": franka_rot.tolist(),
            "width": float(franka_width),
            "score": 1.0,
            "contact1": (-horiz_axis * franka_width / 2).tolist(),
            "contact2": (horiz_axis * franka_width / 2).tolist(),
        },

        # CuRobo trajectory targets (world frame, for plan_grasp_docker.py)
        "curobo_targets": {
            "object_pos": [obj_x, obj_y, obj_z],
            "ee_offset": EE_OFFSET,
            "approach": {
                "position": [obj_x, obj_y, obj_z + EE_OFFSET + 0.08],
                "quaternion_wxyz": quat_wxyz,
            },
            "grasp": {
                "position": [obj_x, obj_y, obj_z + EE_OFFSET],
                "quaternion_wxyz": quat_wxyz,
            },
            "lift": {
                "position": [obj_x, obj_y, obj_z + EE_OFFSET + 0.15],
                "quaternion_wxyz": quat_wxyz,
            },
        },

        # Raw MANO data for reference
        "mano_joints": {
            "thumb_tip": thumb_tip.tolist(),
            "index_tip": index_tip.tolist(),
            "wrist": wrist.tolist(),
            "grasp_center": grasp_center.tolist(),
        },
    }

    out_path = os.path.join(BASE_DIR, 'output/grasp_poses.json')
    with open(out_path, 'w') as f:
        json.dump(grasp_pose, f, indent=2)

    print(f"\n=== Franka Grasp Pose (world frame) ===")
    print(f"Object position: [{obj_x}, {obj_y}, {obj_z}]")
    print(f"Gripper width:   {franka_width*100:.1f} cm")
    print(f"Gripper quat:    {[round(q,4) for q in quat_wxyz]}")
    print(f"Approach pos:    {grasp_pose['curobo_targets']['approach']['position']}")
    print(f"Grasp pos:       {grasp_pose['curobo_targets']['grasp']['position']}")
    print(f"Lift pos:        {grasp_pose['curobo_targets']['lift']['position']}")
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
