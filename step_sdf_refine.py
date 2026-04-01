"""
Lightweight refinement: SPIDER IK trajectory + SDF penetration removal + MuJoCo replay.

Steps:
1. Load IK trajectory from SPIDER
2. For each frame, check finger-object penetration using trimesh SDF
3. Adjust finger joints to remove penetration
4. Replay refined trajectory in MuJoCo and render MP4
"""
import os, json
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
import cv2
import trimesh
from scipy.optimize import minimize
from scipy.ndimage import uniform_filter1d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


SPIDER_DIR = os.path.join(BASE_DIR, 'spider')
IK_PATH = f'{SPIDER_DIR}/example_datasets/processed/hamer_demo/allegro/right/pick_bottle/0/trajectory_kinematic.npz'
SCENE_PATH = f'{SPIDER_DIR}/example_datasets/processed/hamer_demo/allegro/right/pick_bottle/scene.xml'
BOTTLE_OBJ = f'{SPIDER_DIR}/example_datasets/processed/hamer_demo/assets/objects/bottle/visual.obj'
OUT_DIR = os.path.join(BASE_DIR, 'output/sdf_refined')


def load_model_and_trajectory():
    """Load MuJoCo model and IK trajectory."""
    ik_data = np.load(IK_PATH)
    qpos_traj = ik_data['qpos']  # (N, 29)

    orig_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(SCENE_PATH)))
    model = mujoco.MjModel.from_xml_path(os.path.basename(SCENE_PATH))
    os.chdir(orig_dir)

    return model, qpos_traj


def get_fingertip_positions(model, data, qpos):
    """Get fingertip positions for given qpos."""
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    tip_names = ['right_ff_tip', 'right_mf_tip', 'right_rf_tip', 'right_th_tip']
    tips = {}
    for name in tip_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            tips[name] = data.xpos[bid].copy()
    return tips


def get_object_position(model, data, qpos):
    """Get object position."""
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_object')
    return data.xpos[obj_id].copy() if obj_id >= 0 else np.zeros(3)


def sdf_penetration_check(point, obj_mesh):
    """Check if a point penetrates the object mesh. Returns signed distance (negative = inside)."""
    try:
        closest, distance, face_id = obj_mesh.nearest.on_surface([point])
        # Check if inside using ray casting
        contains = obj_mesh.contains([point])
        sd = -distance[0] if contains[0] else distance[0]
        return sd
    except:
        return 1.0  # far away, no penetration


def refine_frame(model, data, qpos, obj_mesh, obj_pos):
    """
    Refine a single frame's finger joints to remove penetration.
    Only adjusts finger joints (indices 6-21), keeps base pose fixed.
    """
    # Joint indices: 0-5 = base 6DoF, 6-21 = finger joints, 22-28 = object freejoint
    FINGER_START = 6
    FINGER_END = 22

    # Get current fingertip positions
    tips = get_fingertip_positions(model, data, qpos)

    # Check penetration for each tip
    penetrations = {}
    for name, pos in tips.items():
        # Transform tip to object-local frame
        local_pos = pos - obj_pos
        sd = sdf_penetration_check(local_pos, obj_mesh)
        if sd < -0.002:  # penetrating more than 2mm
            penetrations[name] = sd

    if not penetrations:
        return qpos, 0  # no penetration

    # Optimize: adjust finger joints to minimize penetration
    finger_joints = qpos[FINGER_START:FINGER_END].copy()
    joint_ranges = np.array([model.jnt_range[i] for i in range(FINGER_START, FINGER_END)])

    def objective(delta):
        test_qpos = qpos.copy()
        test_qpos[FINGER_START:FINGER_END] = finger_joints + delta

        # Clip to joint limits
        for j in range(len(delta)):
            test_qpos[FINGER_START + j] = np.clip(
                test_qpos[FINGER_START + j],
                joint_ranges[j, 0], joint_ranges[j, 1])

        # Forward kinematics
        data.qpos[:] = test_qpos
        mujoco.mj_forward(model, data)

        # Compute penetration cost
        pen_cost = 0.0
        for name in tips:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                tip_pos = data.xpos[bid]
                local_pos = tip_pos - obj_pos
                sd = sdf_penetration_check(local_pos, obj_mesh)
                if sd < 0:
                    pen_cost += sd ** 2 * 1000  # strong penalty for penetration

        # Regularization: don't change too much
        reg_cost = np.sum(delta ** 2) * 10.0

        return pen_cost + reg_cost

    result = minimize(objective, np.zeros(FINGER_END - FINGER_START),
                      method='L-BFGS-B', options={'maxiter': 20, 'ftol': 1e-6})

    refined_qpos = qpos.copy()
    refined_qpos[FINGER_START:FINGER_END] = finger_joints + result.x
    # Clip to limits
    for j in range(FINGER_END - FINGER_START):
        refined_qpos[FINGER_START + j] = np.clip(
            refined_qpos[FINGER_START + j],
            joint_ranges[j, 0], joint_ranges[j, 1])

    return refined_qpos, len(penetrations)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading model and trajectory...")
    model, qpos_traj = load_model_and_trajectory()
    data = mujoco.MjData(model)
    N = qpos_traj.shape[0]
    print(f"Trajectory: {N} frames, {model.nq} qpos dims")

    print("Loading bottle mesh for SDF...")
    obj_mesh = trimesh.load(BOTTLE_OBJ)
    # Scale mesh to match the model (bottle might need adjustment)
    print(f"Bottle mesh: {len(obj_mesh.vertices)} verts, extents={obj_mesh.extents.round(4)}")

    # Get object position from first frame
    obj_pos = get_object_position(model, data, qpos_traj[0])
    print(f"Object position: {obj_pos.round(4)}")

    # Phase 1: Refine trajectory with SDF
    print(f"\n=== Phase 1: SDF Penetration Removal ===")
    refined_traj = np.zeros_like(qpos_traj)
    total_pen = 0

    for i in range(N):
        # Update object position each frame
        obj_pos_i = get_object_position(model, data, qpos_traj[i])
        refined_traj[i], n_pen = refine_frame(model, data, qpos_traj[i], obj_mesh, obj_pos_i)
        total_pen += n_pen

        if i % 30 == 0:
            tips = get_fingertip_positions(model, data, refined_traj[i])
            tip_dists = [np.linalg.norm(p - obj_pos_i) for p in tips.values()]
            print(f"  Frame {i:3d}/{N}: penetrations={n_pen}, "
                  f"min_tip_dist={min(tip_dists):.4f}")

    print(f"Total penetrations fixed: {total_pen}")

    # Smooth refined trajectory
    refined_traj[:, 6:22] = uniform_filter1d(refined_traj[:, 6:22], size=3, axis=0)

    # Phase 2: Render comparison video
    print(f"\n=== Phase 2: Rendering ===")
    renderer = mujoco.Renderer(model, height=480, width=640)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.13, 0.02, 0.24]
    cam.distance = 0.30
    cam.azimuth = 150
    cam.elevation = -25

    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Render original IK
    w_orig = cv2.VideoWriter(os.path.join(OUT_DIR, 'ik_original.mp4'), fourcc, fps, (640, 480))
    for i in range(N):
        data.qpos[:] = qpos_traj[i]
        mujoco.mj_step(model, data)
        renderer.update_scene(data, cam)
        img = renderer.render()
        w_orig.write(img[:, :, ::-1])
    w_orig.release()
    print("  Saved: ik_original.mp4")

    # Render refined
    w_ref = cv2.VideoWriter(os.path.join(OUT_DIR, 'sdf_refined.mp4'), fourcc, fps, (640, 480))
    for i in range(N):
        data.qpos[:] = refined_traj[i]
        mujoco.mj_step(model, data)
        renderer.update_scene(data, cam)
        img = renderer.render()
        w_ref.write(img[:, :, ::-1])

        if i in [0, N//4, N//2, 3*N//4, N-1]:
            cv2.imwrite(os.path.join(OUT_DIR, f'kf_{i:04d}.jpg'), img[:, :, ::-1])
    w_ref.release()
    print("  Saved: sdf_refined.mp4")

    # Side-by-side comparison
    print("  Generating side-by-side...")
    cap_o = cv2.VideoCapture(os.path.join(OUT_DIR, 'ik_original.mp4'))
    cap_r = cv2.VideoCapture(os.path.join(OUT_DIR, 'sdf_refined.mp4'))
    overlays = sorted([f for f in os.listdir(os.path.join(BASE_DIR, 'output/pick_bottle_video/'))
                       if f.endswith('_overlay.jpg')])

    pw, ph = 427, 320
    w_cmp = cv2.VideoWriter(os.path.join(OUT_DIR, 'comparison.mp4'), fourcc, fps, (pw*3, ph+25))

    for i in range(N):
        ret_o, f_o = cap_o.read()
        ret_r, f_r = cap_r.read()
        if not ret_o or not ret_r:
            break

        # Original video overlay (sample from 151 to N frames)
        ov_idx = min(int(i / N * len(overlays)), len(overlays) - 1)
        ov_path = os.path.join(BASE_DIR, 'output/pick_bottle_video', overlays[ov_idx])
        f_ov = cv2.imread(ov_path)

        f_ov = cv2.resize(f_ov, (pw, ph))
        f_o = cv2.resize(f_o, (pw, ph))
        f_r = cv2.resize(f_r, (pw, ph))

        panels = np.hstack([f_ov, f_o, f_r])
        label = np.ones((25, pw*3, 3), dtype=np.uint8) * 35
        cv2.putText(label, "1. Video + HaMeR", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 0), 1)
        cv2.putText(label, "2. SPIDER IK (before)", (pw+5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 0), 1)
        cv2.putText(label, "3. SDF Refined (after)", (pw*2+5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 0), 1)

        frame = np.vstack([label, panels])
        w_cmp.write(frame)

        if i == N // 2:
            cv2.imwrite(os.path.join(OUT_DIR, 'comparison_preview.jpg'), frame)

    w_cmp.release()
    cap_o.release()
    cap_r.release()
    print("  Saved: comparison.mp4")

    # Save refined trajectory
    np.savez(os.path.join(OUT_DIR, 'trajectory_refined.npz'),
             qpos=refined_traj, qpos_original=qpos_traj)
    print(f"  Saved: trajectory_refined.npz")

    # Stats
    diff = np.abs(refined_traj[:, 6:22] - qpos_traj[:, 6:22])
    print(f"\n=== Refinement Stats ===")
    print(f"  Total frames: {N}")
    print(f"  Penetrations fixed: {total_pen}")
    print(f"  Max joint change: {diff.max():.4f} rad")
    print(f"  Mean joint change: {diff.mean():.4f} rad")
    print(f"\nOutput: {OUT_DIR}/")


if __name__ == '__main__':
    main()
