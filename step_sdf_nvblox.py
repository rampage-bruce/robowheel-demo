"""
SDF refinement using NVIDIA nvblox (GPU-accelerated TSDF).
Replaces trimesh CPU SDF with nvblox GPU batch query.

Pipeline:
  1. Load SPIDER IK trajectory + bottle mesh
  2. Build nvblox TSDF from bottle depth rendering
  3. For each frame, GPU batch query all fingertip SDF values
  4. Optimize penetrating joints
  5. Render comparison video
"""
import os, json
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
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
OUT_DIR = os.path.join(BASE_DIR, 'output/sdf_nvblox')


def build_nvblox_tsdf(bottle_mesh_path, voxel_size=0.003):
    """Build nvblox TSDF map from bottle mesh by rendering synthetic depth images."""
    from nvblox_torch.mapper import Mapper, QueryType
    from nvblox_torch.sensor import Sensor

    mapper = Mapper(voxel_sizes_m=[voxel_size])

    # Load bottle mesh
    mesh = trimesh.load(bottle_mesh_path)
    print(f"Bottle mesh: {len(mesh.vertices)} verts, extents={mesh.extents.round(4)}m")

    # Render depth images from multiple viewpoints around the bottle
    # Using trimesh's built-in renderer
    W, H = 320, 240
    fx, fy = 250.0, 250.0
    cx, cy = W / 2, H / 2
    sensor = Sensor.from_camera(fu=fx, fv=fy, cu=cx, cv=cy, width=W, height=H)

    scene = trimesh.Scene(mesh)
    camera_dist = 0.3  # 30cm from object center

    n_views = 12
    for i in range(n_views):
        angle = 2 * np.pi * i / n_views

        # Camera position on a circle around the object
        cam_pos = np.array([
            camera_dist * np.cos(angle),
            camera_dist * np.sin(angle),
            0.05  # slightly above center
        ])

        # Look-at matrix
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        # Camera extrinsics (world-to-camera)
        R = np.stack([right, -up, forward], axis=1)  # camera axes
        t = cam_pos

        T_w_c = np.eye(4)
        T_w_c[:3, :3] = R
        T_w_c[:3, 3] = t

        # Render depth using trimesh
        scene.camera_transform = T_w_c
        try:
            # Simple depth from raycasting
            origins = np.tile(cam_pos, (H * W, 1))
            # Generate ray directions
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            u = u.flatten().astype(float)
            v = v.flatten().astype(float)
            dirs_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=1)
            dirs_world = (R @ dirs_cam.T).T
            dirs_world /= np.linalg.norm(dirs_world, axis=1, keepdims=True)

            locations, index_ray, _ = mesh.ray.intersects_location(origins, dirs_world)

            depth_map = np.zeros(H * W, dtype=np.float32)
            if len(locations) > 0:
                depths = np.linalg.norm(locations - origins[index_ray], axis=1)
                # Keep closest intersection per ray
                for ray_idx, d in zip(index_ray, depths):
                    if depth_map[ray_idx] == 0 or d < depth_map[ray_idx]:
                        depth_map[ray_idx] = d

            depth_map = depth_map.reshape(H, W)
            valid = (depth_map > 0).sum()

            if valid > 100:
                depth_gpu = torch.from_numpy(depth_map).cuda()
                T_cpu = torch.from_numpy(T_w_c.astype(np.float32))
                mapper.add_depth_frame(depth_gpu, T_cpu, sensor)

                if i % 4 == 0:
                    print(f"  View {i}/{n_views}: {valid} valid pixels")
        except Exception as e:
            print(f"  View {i} failed: {e}")

    return mapper, sensor


def nvblox_sdf_query(mapper, points_np):
    """Query SDF for multiple points using nvblox GPU."""
    from nvblox_torch.mapper import QueryType

    pts_gpu = torch.from_numpy(points_np.astype(np.float32)).cuda()
    result = mapper.query_layer(QueryType.TSDF, pts_gpu)
    # result: (N, 2) = [distance, weight]
    distances = result[:, 0].cpu().numpy()
    weights = result[:, 1].cpu().numpy()
    return distances, weights


def load_model_and_trajectory():
    ik_data = np.load(IK_PATH)
    qpos_traj = ik_data['qpos']

    orig_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(SCENE_PATH)))
    model = mujoco.MjModel.from_xml_path(os.path.basename(SCENE_PATH))
    os.chdir(orig_dir)
    return model, qpos_traj


def get_fingertip_positions(model, data, qpos):
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
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_object')
    return data.xpos[obj_id].copy() if obj_id >= 0 else np.zeros(3)


def refine_frame_nvblox(model, data, qpos, mapper, obj_pos):
    """Refine finger joints using nvblox GPU SDF."""
    FINGER_START = 6
    FINGER_END = 22

    tips = get_fingertip_positions(model, data, qpos)

    # Batch query all fingertips
    tip_positions = np.array(list(tips.values()))
    local_positions = tip_positions - obj_pos

    distances, weights = nvblox_sdf_query(mapper, local_positions)

    # Check penetration (distance < -0.5mm AND weight > 0)
    penetrating = (distances < -0.0005) & (weights > 0)
    n_pen = penetrating.sum()

    if n_pen == 0:
        pen_loss = 0.0
        return qpos, 0, pen_loss, 0.0

    # Optimize
    finger_joints = qpos[FINGER_START:FINGER_END].copy()
    joint_ranges = np.array([model.jnt_range[i] for i in range(FINGER_START, FINGER_END)])

    _last_pen = [0.0]
    _last_reg = [0.0]

    def objective(delta):
        test_qpos = qpos.copy()
        test_qpos[FINGER_START:FINGER_END] = np.clip(
            finger_joints + delta, joint_ranges[:, 0], joint_ranges[:, 1])

        data.qpos[:] = test_qpos
        mujoco.mj_forward(model, data)

        # Get updated tip positions
        tip_pos = []
        for name in tips:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                tip_pos.append(data.xpos[bid].copy())
        tip_pos = np.array(tip_pos) - obj_pos

        # nvblox GPU query
        dists, ws = nvblox_sdf_query(mapper, tip_pos)

        pen_cost = 0.0
        for d, w in zip(dists, ws):
            if d < 0 and w > 0:
                pen_cost += d ** 2 * 1000

        reg_cost = np.sum(delta ** 2) * 10.0

        _last_pen[0] = pen_cost
        _last_reg[0] = reg_cost
        return pen_cost + reg_cost

    result = minimize(objective, np.zeros(FINGER_END - FINGER_START),
                      method='L-BFGS-B', options={'maxiter': 20, 'ftol': 1e-6})

    refined_qpos = qpos.copy()
    refined_qpos[FINGER_START:FINGER_END] = np.clip(
        finger_joints + result.x, joint_ranges[:, 0], joint_ranges[:, 1])

    return refined_qpos, int(n_pen), _last_pen[0], _last_reg[0]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading model and trajectory...")
    model, qpos_traj = load_model_and_trajectory()
    data = mujoco.MjData(model)
    N = qpos_traj.shape[0]
    print(f"Trajectory: {N} frames")

    print("\nBuilding nvblox TSDF from bottle mesh...")
    mapper, sensor = build_nvblox_tsdf(BOTTLE_OBJ)
    print("TSDF built!")

    # Phase 1: Refine
    print(f"\n=== Phase 1: nvblox GPU SDF Refinement ===")
    refined_traj = np.zeros_like(qpos_traj)
    total_pen = 0
    total_pen_loss = 0.0
    total_reg_loss = 0.0

    for i in range(N):
        obj_pos = get_object_position(model, data, qpos_traj[i])
        refined_traj[i], n_pen, pen_loss, reg_loss = refine_frame_nvblox(
            model, data, qpos_traj[i], mapper, obj_pos)
        total_pen += n_pen
        total_pen_loss += pen_loss
        total_reg_loss += reg_loss

        if i % 15 == 0:
            tips = get_fingertip_positions(model, data, refined_traj[i])
            tip_dists = [np.linalg.norm(p - obj_pos) for p in tips.values()]
            print(f"  Frame {i:3d}/{N}: pen={n_pen}, "
                  f"pen_loss={pen_loss:.6f}, reg_loss={reg_loss:.6f}, "
                  f"min_dist={min(tip_dists)*1000:.1f}mm")

    print(f"\n=== nvblox SDF Refinement Summary ===")
    print(f"  Total penetrations: {total_pen}")
    print(f"  Total pen_loss:  {total_pen_loss:.6f}")
    print(f"  Total reg_loss:  {total_reg_loss:.6f}")

    # Smooth
    refined_traj[:, 6:22] = uniform_filter1d(refined_traj[:, 6:22], size=3, axis=0)

    # Phase 2: Render
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

    w_ref = cv2.VideoWriter(os.path.join(OUT_DIR, 'nvblox_refined.mp4'), fourcc, fps, (640, 480))
    for i in range(N):
        data.qpos[:] = refined_traj[i]
        mujoco.mj_step(model, data)
        renderer.update_scene(data, cam)
        img = renderer.render()
        w_ref.write(img[:, :, ::-1])
        if i in [0, N//4, N//2, 3*N//4, N-1]:
            cv2.imwrite(os.path.join(OUT_DIR, f'kf_{i:04d}.jpg'), img[:, :, ::-1])
    w_ref.release()
    print("  Saved: nvblox_refined.mp4")

    # Comparison with original
    w_orig = cv2.VideoWriter(os.path.join(OUT_DIR, 'ik_original.mp4'), fourcc, fps, (640, 480))
    for i in range(N):
        data.qpos[:] = qpos_traj[i]
        mujoco.mj_step(model, data)
        renderer.update_scene(data, cam)
        img = renderer.render()
        w_orig.write(img[:, :, ::-1])
    w_orig.release()

    # Three-panel comparison
    overlays_dir = os.path.join(BASE_DIR, 'output/pick_bottle_video')
    overlays = sorted([f for f in os.listdir(overlays_dir) if f.endswith('_overlay.jpg')])

    cap_o = cv2.VideoCapture(os.path.join(OUT_DIR, 'ik_original.mp4'))
    cap_r = cv2.VideoCapture(os.path.join(OUT_DIR, 'nvblox_refined.mp4'))

    pw, ph = 427, 320
    w_cmp = cv2.VideoWriter(os.path.join(OUT_DIR, 'comparison.mp4'), fourcc, fps, (pw*3, ph+25))

    for i in range(N):
        ov_idx = min(int(i / N * len(overlays)), len(overlays) - 1)
        ov = cv2.resize(cv2.imread(os.path.join(overlays_dir, overlays[ov_idx])), (pw, ph))
        ret_o, f_o = cap_o.read()
        ret_r, f_r = cap_r.read()
        if not ret_o or not ret_r:
            break
        f_o = cv2.resize(f_o, (pw, ph))
        f_r = cv2.resize(f_r, (pw, ph))

        panels = np.hstack([ov, f_o, f_r])
        label = np.ones((25, pw*3, 3), dtype=np.uint8) * 35
        cv2.putText(label, "1. Video + HaMeR", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 0), 1)
        cv2.putText(label, "2. SPIDER IK (before)", (pw+5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 0), 1)
        cv2.putText(label, "3. nvblox SDF Refined", (pw*2+5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 0), 1)

        frame = np.vstack([label, panels])
        w_cmp.write(frame)
        if i == N // 2:
            cv2.imwrite(os.path.join(OUT_DIR, 'comparison_preview.jpg'), frame)

    w_cmp.release()
    cap_o.release()
    cap_r.release()

    # GIF
    import subprocess
    subprocess.run(['ffmpeg', '-y', '-i', os.path.join(OUT_DIR, 'comparison.mp4'),
                    '-vf', 'fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
                    os.path.join(OUT_DIR, 'comparison.gif')], capture_output=True)

    np.savez(os.path.join(OUT_DIR, 'trajectory_nvblox.npz'),
             qpos=refined_traj, qpos_original=qpos_traj)

    print(f"\nDone! Output: {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f} ({os.path.getsize(os.path.join(OUT_DIR, f))/1024:.0f} KB)")


if __name__ == '__main__':
    main()
