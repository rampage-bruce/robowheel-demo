"""
Convert HaMeR MANO output → SPIDER trajectory_keypoints.npz format.

HaMeR output: mano_results.json with per-frame MANO parameters
SPIDER expects: trajectory_keypoints.npz with wrist pose, fingertip poses, object pose

Author: auto-generated for RoboWheel demo
"""
import os, sys, json, shutil
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.ndimage import uniform_filter1d
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HAMER_DIR = os.path.join(BASE_DIR, 'hamer')
SPIDER_DIR = os.path.join(BASE_DIR, 'spider')
MANO_RESULTS = os.path.join(BASE_DIR, 'output/pick_bottle_video/mano_results.json')
if not os.path.exists(MANO_RESULTS):
    MANO_RESULTS = os.path.join(BASE_DIR, 'test_data/mano_results.json')


def load_mano_model():
    sys.path.insert(0, HAMER_DIR)
    os.chdir(HAMER_DIR)
    from hamer.models import load_hamer, DEFAULT_CHECKPOINT
    model, cfg = load_hamer(DEFAULT_CHECKPOINT)
    return model.mano


def main():
    print("Loading MANO model...")
    mano = load_mano_model()

    with open(MANO_RESULTS) as f:
        all_results = json.load(f)

    # Separate right hand frames
    right_frames = [r for r in all_results if r['is_right']]
    print(f"Right hand frames: {len(right_frames)}")

    N = len(right_frames)

    # Arrays for SPIDER format
    qpos_wrist = np.zeros((N, 7))     # pos(3) + quat(4, wxyz)
    qpos_finger = np.zeros((N, 5, 7)) # 5 fingertips × (pos(3) + quat(4))
    qpos_obj = np.zeros((N, 7))       # object pose

    # MANO joint indices for fingertips
    # MANO 21 joints: 0=wrist, 16=index_tip, 17=middle_tip, 18=pinky_tip, 19=ring_tip, 20=thumb_tip
    TIP_INDICES = [16, 17, 19, 18, 20]  # index, middle, ring, pinky, thumb (SPIDER finger order)

    # Coordinate transform: MANO camera space → SPIDER world space
    # MANO: X-right, Y-down, Z-forward (camera convention)
    # SPIDER: X-right, Y-forward, Z-up (world convention)
    # Transform: swap Y↔Z and flip Y
    def mano_to_world(pts):
        """Convert MANO camera-space points to SPIDER world-space."""
        w = np.zeros_like(pts)
        w[..., 0] = pts[..., 0]       # X stays
        w[..., 1] = -pts[..., 2]      # Z_cam → -Y_world
        w[..., 2] = -pts[..., 1]      # -Y_cam → Z_world (up)
        return w

    # Reference data center: wrist ≈ [-0.12, 0.14, 0.19], obj ≈ [0.01, 0.20, 0.12]
    # We want our data to be in a similar range (wrist above table, ~0.1-0.2m height)
    WORLD_OFFSET = np.array([0.0, 0.0, 0.15])  # lift everything 15cm up (above table)

    for i, r in enumerate(right_frames):
        hp = torch.tensor(r['mano_hand_pose'], dtype=torch.float32).unsqueeze(0)
        bt = torch.tensor(r['mano_betas'], dtype=torch.float32).unsqueeze(0)
        go = torch.tensor(r['mano_global_orient'], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            out = mano(hand_pose=hp, betas=bt, global_orient=go)

        joints = out.joints[0].numpy()  # (21, 3) in MANO camera space

        # Transform all joints to world space
        joints_w = mano_to_world(joints) + WORLD_OFFSET

        # Wrist pose
        wrist_pos = joints_w[0]

        # Global orient: rotate MANO rotation to world frame
        go_mat = np.array(r['mano_global_orient'])
        if go_mat.ndim == 3:
            go_mat = go_mat[0]
        # Apply coordinate transform to rotation: R_world = T @ R_mano @ T^-1
        # where T swaps Y↔Z and flips
        T = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float64)
        go_world = T @ go_mat @ T.T
        rot = Rotation.from_matrix(go_world)
        quat_xyzw = rot.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        qpos_wrist[i, :3] = wrist_pos
        qpos_wrist[i, 3:] = quat_wxyz

        # Fingertip poses (in world space)
        for j, tip_idx in enumerate(TIP_INDICES):
            tip_pos = joints_w[tip_idx]
            qpos_finger[i, j, :3] = tip_pos
            qpos_finger[i, j, 3] = 1.0
            qpos_finger[i, j, 4:] = 0.0

        # Object pose: between thumb and index tip
        thumb_tip = joints_w[20]
        index_tip = joints_w[16]
        obj_pos = (thumb_tip + index_tip) / 2
        qpos_obj[i, :3] = obj_pos
        qpos_obj[i, 3] = 1.0
        qpos_obj[i, 4:] = 0.0

    # Smooth trajectories
    qpos_wrist[:, :3] = uniform_filter1d(qpos_wrist[:, :3], size=5, axis=0)
    for j in range(5):
        qpos_finger[:, j, :3] = uniform_filter1d(qpos_finger[:, j, :3], size=5, axis=0)
    qpos_obj[:, :3] = uniform_filter1d(qpos_obj[:, :3], size=5, axis=0)

    # Estimate contact: fingertip close to object = contact
    contact = np.zeros((N, 5))
    contact_pos = np.zeros((5, 3))
    for i in range(N):
        for j in range(5):
            dist = np.linalg.norm(qpos_finger[i, j, :3] - qpos_obj[i, :3])
            contact[i, j] = 1.0 if dist < 0.05 else 0.0

    # Average contact positions (relative to object)
    for j in range(5):
        contact_frames = contact[:, j] > 0.5
        if contact_frames.any():
            contact_pos[j] = np.mean(
                qpos_finger[contact_frames, j, :3] - qpos_obj[contact_frames, :3], axis=0)

    print(f"\nData summary:")
    print(f"  Frames: {N}")
    print(f"  Wrist pos range: [{qpos_wrist[:,:3].min():.4f}, {qpos_wrist[:,:3].max():.4f}]")
    print(f"  Object pos range: [{qpos_obj[:,:3].min():.4f}, {qpos_obj[:,:3].max():.4f}]")
    print(f"  Contact frames: {contact.sum(axis=0)}")

    # Create output directory matching SPIDER structure
    task_name = "pick_bottle"
    dataset_name = "hamer_demo"
    embodiment = "right"

    out_base = os.path.join(SPIDER_DIR, "example_datasets", "processed", dataset_name)

    # MANO keypoints
    mano_dir = os.path.join(out_base, "mano", embodiment, task_name, "0")
    os.makedirs(mano_dir, exist_ok=True)

    np.savez(os.path.join(mano_dir, "trajectory_keypoints.npz"),
             qpos_wrist_right=qpos_wrist,
             qpos_finger_right=qpos_finger,
             qpos_obj_right=qpos_obj,
             contact_right=contact,
             contact_pos_right=contact_pos)

    # Task info
    task_info = {
        "task": task_name,
        "dataset_name": dataset_name,
        "robot_type": "mano",
        "embodiment_type": embodiment,
        "data_id": 0,
        "right_object_mesh_dir": None,
        "left_object_mesh_dir": None,
        "ref_dt": 0.02,
    }
    with open(os.path.join(mano_dir, "task_info.json"), 'w') as f:
        json.dump(task_info, f, indent=2)

    print(f"\nSaved to {mano_dir}/")
    print(f"  trajectory_keypoints.npz")
    print(f"  task_info.json")

    # Copy a simple bottle mesh as object asset
    assets_dir = os.path.join(out_base, "assets", "objects")
    os.makedirs(assets_dir, exist_ok=True)

    # Create a simple bottle OBJ
    bottle_obj = os.path.join(assets_dir, "bottle.obj")
    import trimesh
    bottle = trimesh.creation.cylinder(radius=0.022, height=0.11, sections=16)
    bottle.export(bottle_obj)
    print(f"  Bottle mesh: {bottle_obj}")

    # Copy robot assets (symlink to existing)
    robot_assets_src = os.path.join(SPIDER_DIR, "example_datasets", "processed", "fair_fre", "assets", "robots")
    robot_assets_dst = os.path.join(out_base, "assets", "robots")
    if not os.path.exists(robot_assets_dst):
        os.symlink(robot_assets_src, robot_assets_dst)
        print(f"  Linked robot assets: {robot_assets_dst}")

    print("\nDone! Next steps:")
    print(f"  1. Generate scene: python spider/preprocess/generate_xml.py --task={task_name} --dataset-name={dataset_name} --data-id=0 --embodiment-type={embodiment} --robot-type=allegro")
    print(f"  2. IK: python spider/preprocess/ik_fast.py --task={task_name} --dataset-name={dataset_name} --data-id=0 --embodiment-type={embodiment} --robot-type=allegro --no-show-viewer")
    print(f"  3. MJWP: python examples/run_mjwp.py dataset_name={dataset_name} robot_type=allegro embodiment_type={embodiment} task={task_name} data_id=0 show_viewer=false save_video=true")


if __name__ == '__main__':
    main()
