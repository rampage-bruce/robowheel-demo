"""
Unified grasp pipeline: staged trajectory + nvblox GPU SDF + correct rendering.

Combines:
  - step_approach_refine: 4-phase trajectory (approach→close→grasp→lift)
  - step_sdf_nvblox: GPU-accelerated penetration detection
  - Direct push-out instead of scipy optimization (fixes convergence issue)
  - Correct camera (0.45m distance, full hand+bottle visible)
"""
import os, json
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import mujoco
import cv2
import trimesh
from scipy.ndimage import uniform_filter1d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_DIR = os.path.join(BASE_DIR, 'spider')
IK_PATH = f'{SPIDER_DIR}/example_datasets/processed/hamer_demo/allegro/right/pick_bottle/0/trajectory_kinematic.npz'
SCENE_PATH = f'{SPIDER_DIR}/example_datasets/processed/hamer_demo/allegro/right/pick_bottle/scene.xml'
BOTTLE_OBJ = f'{SPIDER_DIR}/example_datasets/processed/hamer_demo/assets/objects/bottle/visual.obj'
OUT_DIR = os.path.join(BASE_DIR, 'output/unified_grasp')


def build_nvblox_tsdf(bottle_mesh_path):
    """Build nvblox TSDF from bottle mesh."""
    from nvblox_torch.mapper import Mapper
    from nvblox_torch.sensor import Sensor

    mapper = Mapper(voxel_sizes_m=[0.003])
    mesh = trimesh.load(bottle_mesh_path)
    print(f"Bottle: {len(mesh.vertices)} verts, extents={mesh.extents.round(4)}m")

    W, H = 320, 240
    fx, fy = 250.0, 250.0
    sensor = Sensor.from_camera(fu=fx, fv=fy, cu=W/2, cv=H/2, width=W, height=H)

    for i in range(16):  # more views for better coverage
        angle = 2 * np.pi * i / 16
        for elev in [-0.1, 0.05, 0.15]:  # 3 elevations
            cam_pos = np.array([0.25 * np.cos(angle), 0.25 * np.sin(angle), elev])
            forward = -cam_pos / np.linalg.norm(cam_pos)
            up = np.array([0, 0, 1])
            right = np.cross(forward, up)
            if np.linalg.norm(right) < 1e-6:
                right = np.array([1, 0, 0])
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)

            R = np.stack([right, -up, forward], axis=1)
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cam_pos

            origins = np.tile(cam_pos, (H * W, 1))
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            dirs_cam = np.stack([(u.flatten() - W/2) / fx, (v.flatten() - H/2) / fy, np.ones(H*W)], axis=1)
            dirs_world = (R @ dirs_cam.T).T
            dirs_world /= np.linalg.norm(dirs_world, axis=1, keepdims=True)

            try:
                locs, idx_ray, _ = mesh.ray.intersects_location(origins, dirs_world)
                depth = np.zeros(H * W, dtype=np.float32)
                if len(locs) > 0:
                    for ray_idx, loc in zip(idx_ray, locs):
                        d = np.linalg.norm(loc - origins[ray_idx])
                        if depth[ray_idx] == 0 or d < depth[ray_idx]:
                            depth[ray_idx] = d
                depth = depth.reshape(H, W)
                if (depth > 0).sum() > 50:
                    mapper.add_depth_frame(torch.from_numpy(depth).cuda(),
                                           torch.from_numpy(T.astype(np.float32)), sensor)
            except:
                pass

    return mapper


def nvblox_query(mapper, points):
    """GPU batch SDF query. Returns (distances, weights)."""
    from nvblox_torch.mapper import QueryType
    pts = torch.from_numpy(points.astype(np.float32)).cuda()
    r = mapper.query_layer(QueryType.TSDF, pts)
    return r[:, 0].cpu().numpy(), r[:, 1].cpu().numpy()


def get_tips(model, data, qpos):
    data.qpos[:] = qpos; mujoco.mj_forward(model, data)
    names = ['right_ff_tip', 'right_mf_tip', 'right_rf_tip', 'right_th_tip']
    return {n: data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)].copy()
            for n in names if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) >= 0}


def get_obj_pos(model, data, qpos):
    data.qpos[:] = qpos; mujoco.mj_forward(model, data)
    oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_object')
    return data.xpos[oid].copy() if oid >= 0 else np.zeros(3)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load model + trajectory
    ik_data = np.load(IK_PATH)
    qpos_orig = ik_data['qpos'].copy()
    N = qpos_orig.shape[0]

    orig_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(SCENE_PATH)))
    model = mujoco.MjModel.from_xml_path(os.path.basename(SCENE_PATH))
    os.chdir(orig_dir)
    data = mujoco.MjData(model)
    print(f"Trajectory: {N} frames, nq={model.nq}")

    # Build nvblox TSDF
    print("Building nvblox TSDF (48 views)...")
    mapper = build_nvblox_tsdf(BOTTLE_OBJ)
    print("TSDF built!")

    # Finger joint indices: 6-21 (16 joints for Allegro)
    # Proximal/medial/distal joints for curl:
    #   ff: 7,8,9  mf: 11,12,13  rf: 15,16,17  th: 19,20,21
    CURL_JOINTS = [7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21]
    FINGER_JOINTS_MAP = {
        'right_ff_tip': [7, 8, 9],
        'right_mf_tip': [11, 12, 13],
        'right_rf_tip': [15, 16, 17],
        'right_th_tip': [19, 20, 21],
    }

    # === Phase 1: Design 5-stage trajectory with REACH ===
    print("\n=== Phase 1: 5-Stage Trajectory (REACH→APPROACH→CLOSE→GRASP→LIFT) ===")

    # Compute reference object position (average across trajectory)
    obj_positions = np.array([get_obj_pos(model, data, qpos_orig[i]) for i in range(N)])
    obj_center = obj_positions.mean(0)
    print(f"  Object center: {obj_center.round(4)}")

    # Compute start position for reach: offset from object, above and to the side
    reach_start_offset = np.array([-0.12, -0.08, 0.10])  # 12cm back, 8cm left, 10cm up
    reach_start = obj_center + reach_start_offset
    print(f"  Reach start: {reach_start.round(4)}")

    # Expand trajectory: add REACH frames before IK data
    REACH_FRAMES = 40
    TOTAL = REACH_FRAMES + N
    qpos_new = np.zeros((TOTAL, model.nq))

    # Fill REACH frames (hand starts far away, fingers open)
    first_ik = qpos_orig[0].copy()
    first_obj = get_obj_pos(model, data, first_ik)

    for i in range(REACH_FRAMES):
        qpos_new[i] = first_ik.copy()
        p = i / (REACH_FRAMES - 1)  # 0→1

        # Smooth cubic interpolation for reach
        smooth_p = 3 * p**2 - 2 * p**3  # ease in-out

        # Base position: interpolate from reach_start to first IK position
        reach_pos = reach_start * (1 - smooth_p) + first_ik[:3] * smooth_p
        qpos_new[i, :3] = reach_pos

        # Base rotation: interpolate from upright to first IK rotation
        qpos_new[i, 3:6] = first_ik[3:6] * smooth_p

        # Fingers: fully open during reach (all joints at minimum)
        for jid in range(6, 22):
            jr = model.jnt_range[jid]
            qpos_new[i, jid] = jr[0]  # minimum = open

    # Fill remaining frames with staged grasp on top of IK data
    for idx in range(N):
        i = REACH_FRAMES + idx
        t = idx / (N - 1)
        qpos_new[i] = qpos_orig[idx].copy()

        obj_pos = obj_positions[idx]
        base_pos = qpos_orig[idx, :3]
        to_obj = obj_pos - base_pos
        to_obj_dir = to_obj / max(np.linalg.norm(to_obj), 1e-6)

        tips = get_tips(model, data, qpos_orig[idx])
        mean_gap = np.mean([np.linalg.norm(p - obj_pos) for p in tips.values()])

        if t < 0.25:
            # APPROACH: arc toward object
            p = t / 0.25
            offset = to_obj_dir * mean_gap * 0.5 * p
            lateral = np.cross(to_obj_dir, [0, 0, 1])
            if np.linalg.norm(lateral) > 1e-6:
                lateral /= np.linalg.norm(lateral)
            offset += lateral * 0.015 * np.sin(p * np.pi)
            qpos_new[i, :3] = qpos_orig[idx, :3] + offset

        elif t < 0.50:
            # CLOSE: close gap + curl
            p = (t - 0.25) / 0.25
            offset = to_obj_dir * mean_gap * (0.5 + 0.4 * p)
            qpos_new[i, :3] = qpos_orig[idx, :3] + offset
            curl = p * 0.5
            for jid in CURL_JOINTS:
                jr = model.jnt_range[jid]
                qpos_new[i, jid] = np.clip(qpos_orig[idx, jid] + curl, jr[0], jr[1])

        elif t < 0.70:
            # GRASP: tight
            offset = to_obj_dir * mean_gap * 0.9
            qpos_new[i, :3] = qpos_orig[idx, :3] + offset
            curl = 0.5 + ((t - 0.50) / 0.20) * 0.3
            for jid in CURL_JOINTS:
                jr = model.jnt_range[jid]
                qpos_new[i, jid] = np.clip(qpos_orig[idx, jid] + curl, jr[0], jr[1])

        else:
            # LIFT
            p = (t - 0.70) / 0.30
            offset = to_obj_dir * mean_gap * 0.9
            lift = np.array([0, 0, 0.05 * p])
            qpos_new[i, :3] = qpos_orig[idx, :3] + offset + lift
            curl = 0.8
            for jid in CURL_JOINTS:
                jr = model.jnt_range[jid]
                qpos_new[i, jid] = np.clip(qpos_orig[idx, jid] + curl, jr[0], jr[1])

    N = TOTAL  # update frame count

    # Smooth
    qpos_new[:, :3] = uniform_filter1d(qpos_new[:, :3], size=5, axis=0)
    qpos_new[:, 6:22] = uniform_filter1d(qpos_new[:, 6:22], size=3, axis=0)

    # === Phase 2: nvblox SDF direct push-out ===
    print("\n=== Phase 2: nvblox SDF Push-out ===")
    total_pen = 0
    total_pushed = 0

    for i in range(N):
        obj_pos = get_obj_pos(model, data, qpos_new[i])
        tips = get_tips(model, data, qpos_new[i])

        tip_arr = np.array(list(tips.values()))
        local = tip_arr - obj_pos
        dists, weights = nvblox_query(mapper, local)

        for j, (name, dist, w) in enumerate(zip(tips.keys(), dists, weights)):
            if dist < -0.0005 and w > 0:
                total_pen += 1
                # Direct push-out: reduce curl on this finger's joints
                for jid in FINGER_JOINTS_MAP.get(name, []):
                    qpos_new[i, jid] -= 0.08  # push back 0.08 rad
                    qpos_new[i, jid] = max(qpos_new[i, jid], model.jnt_range[jid][0])
                total_pushed += 1

        if i % 20 == 0:
            t = i / (N - 1)
            phase = "APPROACH" if t < 0.30 else "CLOSE" if t < 0.55 else "GRASP" if t < 0.75 else "LIFT"
            min_d = min(np.linalg.norm(p - obj_pos) for p in tips.values())
            print(f"  Frame {i:3d}/{N} [{phase:8s}]: pen={sum(1 for d,w in zip(dists,weights) if d<-0.0005 and w>0)}, "
                  f"min_dist={min_d*1000:.1f}mm")

    # Final smooth
    qpos_new[:, 6:22] = uniform_filter1d(qpos_new[:, 6:22], size=3, axis=0)

    print(f"\n  Total penetrations detected: {total_pen}")
    print(f"  Total push-outs applied: {total_pushed}")

    # === Phase 3: Render ===
    print("\n=== Phase 3: Rendering ===")
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Correct camera: see both hand and bottle
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.14, 0.03, 0.22]
    cam.distance = 0.45
    cam.azimuth = 155
    cam.elevation = -20

    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def get_phase(frame_idx):
        if frame_idx < REACH_FRAMES:
            return 'REACH'
        t = (frame_idx - REACH_FRAMES) / max(N - REACH_FRAMES - 1, 1)
        if t < 0.25: return 'APPROACH'
        if t < 0.50: return 'CLOSE'
        if t < 0.70: return 'GRASP'
        return 'LIFT'

    # Render refined
    w_ref = cv2.VideoWriter(os.path.join(OUT_DIR, 'unified_grasp.mp4'), fourcc, fps, (640, 480))
    for i in range(N):
        data.qpos[:] = qpos_new[i]
        mujoco.mj_step(model, data)
        renderer.update_scene(data, cam)
        img = renderer.render()
        bgr = img[:, :, ::-1].copy()

        phase = get_phase(i)
        cv2.putText(bgr, f"[{phase}] {i}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        w_ref.write(bgr)

        if i in [0, REACH_FRAMES//2, REACH_FRAMES-1, REACH_FRAMES + (N-REACH_FRAMES)//4,
                 REACH_FRAMES + (N-REACH_FRAMES)//2, REACH_FRAMES + 3*(N-REACH_FRAMES)//4, N-1]:
            cv2.imwrite(os.path.join(OUT_DIR, f'kf_{i:04d}_{phase.lower()}.jpg'), bgr)
    w_ref.release()

    # Render original IK (pad with first frame for REACH portion)
    N_ik = len(qpos_orig)
    w_orig = cv2.VideoWriter(os.path.join(OUT_DIR, 'ik_original.mp4'), fourcc, fps, (640, 480))
    for i in range(N):
        ik_idx = max(0, i - REACH_FRAMES)
        ik_idx = min(ik_idx, N_ik - 1)
        data.qpos[:] = qpos_orig[ik_idx]
        mujoco.mj_step(model, data)
        renderer.update_scene(data, cam)
        w_orig.write(renderer.render()[:, :, ::-1])
    w_orig.release()

    # Three-panel comparison
    overlays_dir = os.path.join(BASE_DIR, 'output/pick_bottle_video')
    overlays = sorted([f for f in os.listdir(overlays_dir) if f.endswith('_overlay.jpg')])
    cap_o = cv2.VideoCapture(os.path.join(OUT_DIR, 'ik_original.mp4'))
    cap_r = cv2.VideoCapture(os.path.join(OUT_DIR, 'unified_grasp.mp4'))

    pw, ph = 427, 320
    w_cmp = cv2.VideoWriter(os.path.join(OUT_DIR, 'comparison.mp4'), fourcc, fps, (pw*3, ph+25))

    for i in range(N):
        ov_idx = min(int(i / N * len(overlays)), len(overlays) - 1)
        ov = cv2.resize(cv2.imread(os.path.join(overlays_dir, overlays[ov_idx])), (pw, ph))
        _, f_o = cap_o.read(); _, f_r = cap_r.read()
        if f_o is None or f_r is None: break
        f_o = cv2.resize(f_o, (pw, ph)); f_r = cv2.resize(f_r, (pw, ph))

        panels = np.hstack([ov, f_o, f_r])
        phase = get_phase(i)
        label = np.ones((25, pw*3, 3), dtype=np.uint8) * 35
        cv2.putText(label, "1. Video + HaMeR", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 0), 1)
        cv2.putText(label, "2. SPIDER IK (original)", (pw+5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 0), 1)
        cv2.putText(label, f"3. Unified [{phase}]", (pw*2+5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 0), 1)

        frame = np.vstack([label, panels])
        w_cmp.write(frame)
        if i == N // 2:
            cv2.imwrite(os.path.join(OUT_DIR, 'comparison_preview.jpg'), frame)

    w_cmp.release(); cap_o.release(); cap_r.release()

    # GIF
    import subprocess
    subprocess.run(['ffmpeg', '-y', '-i', os.path.join(OUT_DIR, 'comparison.mp4'),
                    '-vf', 'fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
                    os.path.join(OUT_DIR, 'comparison.gif')], capture_output=True)

    np.savez(os.path.join(OUT_DIR, 'trajectory_unified.npz'), qpos=qpos_new, qpos_original=qpos_orig)

    print(f"\nDone! {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f} ({os.path.getsize(os.path.join(OUT_DIR, f))/1024:.0f} KB)")


if __name__ == '__main__':
    main()
