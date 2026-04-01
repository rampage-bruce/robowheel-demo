"""
Improved refinement: side approach + finger curl + SDF check.
Takes SPIDER IK trajectory and adds:
  1. Translate hand toward object (close the 3-4cm gap)
  2. Arc approach from the side (not top-down)
  3. Progressive finger curl when close to object
  4. SDF penetration check
"""
import os, json
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
import cv2
import trimesh
from scipy.ndimage import uniform_filter1d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SPIDER_DIR = os.path.join(BASE_DIR, 'spider')
IK_PATH = f'{SPIDER_DIR}/example_datasets/processed/hamer_demo/allegro/right/pick_bottle/0/trajectory_kinematic.npz'
SCENE_PATH = f'{SPIDER_DIR}/example_datasets/processed/hamer_demo/allegro/right/pick_bottle/scene.xml'
BOTTLE_OBJ = f'{SPIDER_DIR}/example_datasets/processed/hamer_demo/assets/objects/bottle/visual.obj'
OUT_DIR = os.path.join(BASE_DIR, 'output/approach_refined')


def get_body_pos(model, data, qpos, body_name):
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[bid].copy() if bid >= 0 else np.zeros(3)


def sdf_check(point, obj_mesh):
    try:
        contains = obj_mesh.contains([point])
        if contains[0]:
            _, dist, _ = obj_mesh.nearest.on_surface([point])
            return -dist[0]
        return 1.0
    except:
        return 1.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load
    ik_data = np.load(IK_PATH)
    qpos_orig = ik_data['qpos'].copy()
    N = qpos_orig.shape[0]

    orig_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(SCENE_PATH)))
    model = mujoco.MjModel.from_xml_path(os.path.basename(SCENE_PATH))
    os.chdir(orig_dir)
    data = mujoco.MjData(model)

    obj_mesh = trimesh.load(BOTTLE_OBJ)
    print(f"Trajectory: {N} frames")

    # Joint layout: 0-2=base_pos, 3-5=base_rot, 6-21=fingers, 22-28=object
    BASE_POS = slice(0, 3)
    BASE_ROT = slice(3, 6)
    FINGERS = slice(6, 22)

    # Phase 1: Analyze object and fingertip positions per frame
    print("Analyzing trajectory...")
    obj_positions = np.zeros((N, 3))
    thumb_positions = np.zeros((N, 3))
    finger_dists = np.zeros(N)

    for i in range(N):
        obj_positions[i] = get_body_pos(model, data, qpos_orig[i], 'right_object')
        thumb_positions[i] = get_body_pos(model, data, qpos_orig[i], 'right_th_tip')
        finger_dists[i] = np.linalg.norm(thumb_positions[i] - obj_positions[i])

    mean_gap = finger_dists.mean()
    print(f"  Mean thumb-object gap: {mean_gap*100:.1f} cm")
    print(f"  Object center: {obj_positions.mean(0).round(3)}")

    # Phase 2: Design improved trajectory
    print("Generating improved trajectory...")
    qpos_new = qpos_orig.copy()

    for i in range(N):
        t = i / (N - 1)  # 0→1

        # Get current positions
        obj_pos = obj_positions[i]
        base_pos = qpos_orig[i, :3]

        # Direction from BASE to object (this moves the wrist toward object)
        to_obj = obj_pos - base_pos
        to_obj_dist = np.linalg.norm(to_obj)
        if to_obj_dist > 1e-6:
            to_obj_dir = to_obj / to_obj_dist
        else:
            to_obj_dir = np.array([0, 0, 0])

        # Scale: close the 3.7cm gap measured at fingertip level
        # Base needs to move ~gap distance toward object
        close_dist = mean_gap * 0.9  # close 90% of the gap

        # --- Base position adjustment: push hand toward object ---
        if t < 0.3:
            # Phase 1 (0-30%): Approach from the side
            # Move base toward object, but offset to the side
            approach_t = t / 0.3
            # Start from current position, gradually close gap
            offset = to_obj_dir * close_dist * 0.6 * approach_t
            # Add slight arc (come from the side, not straight)
            lateral = np.cross(to_obj_dir, [0, 0, 1])
            if np.linalg.norm(lateral) > 1e-6:
                lateral /= np.linalg.norm(lateral)
            arc_offset = lateral * 0.02 * np.sin(approach_t * np.pi)  # 2cm arc
            qpos_new[i, BASE_POS] = qpos_orig[i, BASE_POS] + offset + arc_offset

        elif t < 0.6:
            # Phase 2 (30-60%): Close the gap, fingers start curling
            close_t = (t - 0.3) / 0.3
            offset = to_obj_dir * close_dist * (0.6 + 0.35 * close_t)
            qpos_new[i, BASE_POS] = qpos_orig[i, BASE_POS] + offset

            # Increase finger curl: add to proximal/medial/distal joints
            # Allegro fingers: ff(6-9), mf(10-13), rf(14-17), th(18-21)
            curl_amount = close_t * 0.4  # up to 0.4 rad extra curl
            for finger_start in [7, 8, 9, 11, 12, 13, 15, 16, 17]:  # proximal/medial/distal
                jid = finger_start
                jrange = model.jnt_range[jid]
                qpos_new[i, jid] = np.clip(
                    qpos_orig[i, jid] + curl_amount,
                    jrange[0], jrange[1])
            # Thumb: increase opposition
            for jid in [19, 20, 21]:
                jrange = model.jnt_range[jid]
                qpos_new[i, jid] = np.clip(
                    qpos_orig[i, jid] + curl_amount * 0.8,
                    jrange[0], jrange[1])

        elif t < 0.8:
            # Phase 3 (60-80%): Full grasp - maximum curl, tight position
            grasp_t = (t - 0.6) / 0.2
            offset = to_obj_dir * close_dist
            qpos_new[i, BASE_POS] = qpos_orig[i, BASE_POS] + offset

            curl_amount = 0.4 + grasp_t * 0.3  # 0.4 → 0.7 rad
            for finger_start in [7, 8, 9, 11, 12, 13, 15, 16, 17]:
                jrange = model.jnt_range[finger_start]
                qpos_new[i, finger_start] = np.clip(
                    qpos_orig[i, finger_start] + curl_amount,
                    jrange[0], jrange[1])
            for jid in [19, 20, 21]:
                jrange = model.jnt_range[jid]
                qpos_new[i, jid] = np.clip(
                    qpos_orig[i, jid] + curl_amount * 0.8,
                    jrange[0], jrange[1])

        else:
            # Phase 4 (80-100%): Lift - maintain grasp + move up
            lift_t = (t - 0.8) / 0.2
            offset = to_obj_dir * close_dist
            lift_offset = np.array([0, 0, 0.05 * lift_t])  # lift 5cm
            qpos_new[i, BASE_POS] = qpos_orig[i, BASE_POS] + offset + lift_offset

            curl_amount = 0.7
            for finger_start in [7, 8, 9, 11, 12, 13, 15, 16, 17]:
                jrange = model.jnt_range[finger_start]
                qpos_new[i, finger_start] = np.clip(
                    qpos_orig[i, finger_start] + curl_amount,
                    jrange[0], jrange[1])
            for jid in [19, 20, 21]:
                jrange = model.jnt_range[jid]
                qpos_new[i, jid] = np.clip(
                    qpos_orig[i, jid] + curl_amount * 0.8,
                    jrange[0], jrange[1])

    # Smooth
    qpos_new[:, BASE_POS] = uniform_filter1d(qpos_new[:, BASE_POS], size=5, axis=0)
    qpos_new[:, FINGERS] = uniform_filter1d(qpos_new[:, FINGERS], size=3, axis=0)

    # Phase 3: SDF penetration check
    print("SDF penetration check...")
    pen_count = 0
    tip_names = ['right_ff_tip', 'right_mf_tip', 'right_rf_tip', 'right_th_tip']
    for i in range(N):
        obj_pos = get_body_pos(model, data, qpos_new[i], 'right_object')
        for tip_name in tip_names:
            tip_pos = get_body_pos(model, data, qpos_new[i], tip_name)
            local = tip_pos - obj_pos
            sd = sdf_check(local, obj_mesh)
            if sd < -0.001:
                pen_count += 1
                # Push finger back slightly
                # Reduce the curl that caused penetration
                finger_idx = {'right_ff_tip': [7,8,9], 'right_mf_tip': [11,12,13],
                              'right_rf_tip': [15,16,17], 'right_th_tip': [19,20,21]}
                for jid in finger_idx.get(tip_name, []):
                    qpos_new[i, jid] -= 0.05  # reduce curl by 0.05 rad
                    qpos_new[i, jid] = max(qpos_new[i, jid], model.jnt_range[jid][0])

    print(f"  Penetrations fixed: {pen_count}")

    # Verify improvement
    print("Verifying...")
    new_dists = []
    for i in range(N):
        obj_pos = get_body_pos(model, data, qpos_new[i], 'right_object')
        thumb_pos = get_body_pos(model, data, qpos_new[i], 'right_th_tip')
        new_dists.append(np.linalg.norm(thumb_pos - obj_pos))
    new_dists = np.array(new_dists)
    print(f"  Original gap: {finger_dists.mean()*100:.1f} cm")
    print(f"  New gap:      {new_dists.mean()*100:.1f} cm")
    print(f"  Min gap:      {new_dists.min()*100:.1f} cm")

    # Phase 4: Render
    print("\nRendering...")
    renderer = mujoco.Renderer(model, height=480, width=640)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.13, 0.03, 0.22]
    cam.distance = 0.28
    cam.azimuth = 160
    cam.elevation = -20

    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Render improved trajectory
    w = cv2.VideoWriter(os.path.join(OUT_DIR, 'approach_refined.mp4'), fourcc, fps, (640, 480))
    phases = ['APPROACH', 'CLOSE', 'GRASP', 'LIFT']
    for i in range(N):
        t = i / (N - 1)
        phase = phases[min(int(t / 0.25), 3)]

        data.qpos[:] = qpos_new[i]
        mujoco.mj_step(model, data)
        renderer.update_scene(data, cam)
        img = renderer.render()

        # Add phase label
        img_bgr = img[:, :, ::-1].copy()
        cv2.putText(img_bgr, f"[{phase}] frame {i}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        w.write(img_bgr)

        if i in [0, N//4, N//2, 3*N//4, N-1]:
            cv2.imwrite(os.path.join(OUT_DIR, f'kf_{i:04d}_{phase.lower()}.jpg'), img_bgr)
    w.release()

    # Three-panel comparison
    print("Generating comparison...")
    overlays_dir = os.path.join(BASE_DIR, 'output/pick_bottle_video')
    overlays = sorted([f for f in os.listdir(overlays_dir) if f.endswith('_overlay.jpg')])

    cap_orig = cv2.VideoCapture(os.path.join(os.path.join(BASE_DIR, 'output/sdf_refined'), 'ik_original.mp4'))

    pw, ph = 427, 320
    w_cmp = cv2.VideoWriter(os.path.join(OUT_DIR, 'comparison.mp4'), fourcc, fps, (pw*3, ph+25))

    cap_new = cv2.VideoCapture(os.path.join(OUT_DIR, 'approach_refined.mp4'))

    for i in range(N):
        ov_idx = min(int(i / N * len(overlays)), len(overlays) - 1)
        ov = cv2.resize(cv2.imread(os.path.join(overlays_dir, overlays[ov_idx])), (pw, ph))
        ret_o, f_o = cap_orig.read()
        ret_n, f_n = cap_new.read()
        if not ret_o or not ret_n:
            break
        f_o = cv2.resize(f_o, (pw, ph))
        f_n = cv2.resize(f_n, (pw, ph))

        panels = np.hstack([ov, f_o, f_n])
        t = i / (N - 1)
        phase = phases[min(int(t / 0.25), 3)]
        label = np.ones((25, pw*3, 3), dtype=np.uint8) * 35
        cv2.putText(label, "1. Video + HaMeR", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 0), 1)
        cv2.putText(label, "2. IK Original", (pw+5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 0), 1)
        cv2.putText(label, f"3. Approach Refined [{phase}]", (pw*2+5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 0), 1)
        frame = np.vstack([label, panels])
        w_cmp.write(frame)
        if i == N//2:
            cv2.imwrite(os.path.join(OUT_DIR, 'comparison_preview.jpg'), frame)

    w_cmp.release()
    cap_orig.release()
    cap_new.release()

    # GIF
    import subprocess
    subprocess.run(['ffmpeg', '-y', '-i', os.path.join(OUT_DIR, 'comparison.mp4'),
                    '-vf', 'fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
                    os.path.join(OUT_DIR, 'comparison.gif')], capture_output=True)

    # Save trajectory
    np.savez(os.path.join(OUT_DIR, 'trajectory_approach.npz'),
             qpos=qpos_new, qpos_original=qpos_orig)

    print(f"\nDone! Output: {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f} ({os.path.getsize(os.path.join(OUT_DIR, f))/1024:.0f} KB)")


if __name__ == '__main__':
    main()
