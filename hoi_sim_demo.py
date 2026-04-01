"""
RoboWheel-style HOI Reconstruction Demo:
  1. Load HaMeR hand MANO parameters (already computed)
  2. Estimate object (bottle) position from hand positions
  3. Render full HOI scene: hands + object + table in 3D
  4. Generate side-by-side MP4: original | overlay | 3D sim
"""
import os, sys, json, glob
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import cv2
import trimesh
import pyrender
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

###############################################################################
# Step 1: Load MANO model and reconstruct hand meshes
###############################################################################

def load_mano(hamer_dir):
    sys.path.insert(0, hamer_dir)
    from hamer.models import load_hamer, DEFAULT_CHECKPOINT
    model, cfg = load_hamer(DEFAULT_CHECKPOINT)
    return model.mano, np.array(model.mano.faces)

def reconstruct_hand(mano_model, r):
    """Reconstruct hand vertices + joints from MANO params."""
    hp = torch.tensor(r['mano_hand_pose'], dtype=torch.float32).unsqueeze(0)
    bt = torch.tensor(r['mano_betas'], dtype=torch.float32).unsqueeze(0)
    go = torch.tensor(r['mano_global_orient'], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = mano_model(hand_pose=hp, betas=bt, global_orient=go)
    verts = out.vertices[0].numpy()
    joints = out.joints[0].numpy()  # 21 joints
    if not r['is_right']:
        verts[:, 0] = -verts[:, 0]
        joints[:, 0] = -joints[:, 0]
    return verts, joints

def cam_to_world(pts, cam_t):
    """Camera space → scene world space."""
    w = np.zeros_like(pts)
    w[:, 0] = pts[:, 0]
    w[:, 1] = -pts[:, 2]
    w[:, 2] = -pts[:, 1]
    w[:, 0] += cam_t[0] / cam_t[2] * 0.5
    w[:, 2] += -cam_t[1] / cam_t[2] * 0.5 + 0.15
    return w

###############################################################################
# Step 2: Estimate object position from hand joints
###############################################################################

def estimate_object_pose(hands_data, mano_model):
    """
    Estimate bottle position from hand joint positions.
    When grasping, the object center ≈ midpoint of grasp contacts.
    """
    all_joints_world = []
    for r in hands_data:
        _, joints = reconstruct_hand(mano_model, r)
        cam_t = np.array(r['cam_t_full'])
        joints_w = cam_to_world(joints, cam_t)
        all_joints_world.append((r['is_right'], joints_w))

    if len(all_joints_world) == 0:
        return None, 0

    if len(all_joints_world) >= 2:
        # Two hands: object is between them
        centers = [j.mean(0) for _, j in all_joints_world]
        obj_pos = np.mean(centers, axis=0)
        # Grasp width from hand separation
        hand_dist = np.linalg.norm(centers[0] - centers[1])
    else:
        # One hand: object is near fingertips
        is_right, joints_w = all_joints_world[0]
        # Thumb tip (20) and index tip (16) midpoint
        thumb_tip = joints_w[20] if len(joints_w) > 20 else joints_w[-1]
        index_tip = joints_w[16] if len(joints_w) > 16 else joints_w[-5]
        obj_pos = (thumb_tip + index_tip) / 2
        hand_dist = np.linalg.norm(thumb_tip - index_tip)

    return obj_pos, hand_dist

###############################################################################
# Step 3: Render HOI scene
###############################################################################

def render_hoi_frame(mano_model, faces, frame_results, renderer,
                     bottle_mesh, prev_obj_pos=None):
    """Render one frame of HOI scene with hands + bottle + table."""

    scene = pyrender.Scene(bg_color=[0.88, 0.90, 0.95, 1.0],
                           ambient_light=[0.35, 0.35, 0.35])

    # Lights
    dl = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.5)
    dlp = np.eye(4)
    dlp[:3, :3] = trimesh.transformations.euler_matrix(0.6, 0.3, 0)[:3, :3]
    scene.add(dl, pose=dlp)

    pl = pyrender.PointLight(color=[1, 0.95, 0.9], intensity=6.0)
    plp = np.eye(4); plp[:3, 3] = [0.2, -0.3, 0.6]
    scene.add(pl, pose=plp)

    # Table
    table = trimesh.creation.box(extents=[0.7, 0.5, 0.025])
    table_mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.32, 0.22, 0.15, 1.0], metallicFactor=0.05, roughnessFactor=0.8)
    scene.add(pyrender.Mesh.from_trimesh(table, material=table_mat, smooth=False))

    # Hands
    hand_color_r = [0.87, 0.73, 0.63, 1.0]  # right hand
    hand_color_l = [0.85, 0.70, 0.60, 1.0]  # left hand slightly different

    all_joints_w = []
    for r in frame_results:
        verts, joints = reconstruct_hand(mano_model, r)
        cam_t = np.array(r['cam_t_full'])
        verts_w = cam_to_world(verts, cam_t)
        joints_w = cam_to_world(joints, cam_t)
        all_joints_w.append((r['is_right'], joints_w))

        hand_mesh = trimesh.Trimesh(vertices=verts_w, faces=faces)
        color = hand_color_r if r['is_right'] else hand_color_l
        hmat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color, metallicFactor=0.0, roughnessFactor=0.6)
        scene.add(pyrender.Mesh.from_trimesh(hand_mesh, material=hmat, smooth=True))

    # Estimate object position and add bottle
    obj_pos, _ = estimate_object_pose(frame_results, mano_model)

    if obj_pos is not None:
        # Smooth object position with previous frame
        if prev_obj_pos is not None:
            alpha = 0.4  # smoothing factor
            obj_pos = alpha * obj_pos + (1 - alpha) * prev_obj_pos

        # Create bottle (cylinder + cap)
        bottle_pose = np.eye(4)
        bottle_pose[:3, 3] = obj_pos
        # Tilt bottle slightly based on hand positions
        if len(all_joints_w) >= 2:
            h1_center = all_joints_w[0][1].mean(0)
            h2_center = all_joints_w[1][1].mean(0)
            up = h1_center - h2_center
            up_norm = np.linalg.norm(up)
            if up_norm > 0.01:
                up = up / up_norm
                # Create rotation to align bottle axis with hand axis
                z_axis = np.array([0, 0, 1])
                v = np.cross(z_axis, up)
                s = np.linalg.norm(v)
                c = np.dot(z_axis, up)
                if s > 1e-6:
                    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                    R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)
                    bottle_pose[:3, :3] = R

        bottle_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.7, 0.85, 0.95, 0.7],  # translucent blue
            metallicFactor=0.1, roughnessFactor=0.3, alphaMode='BLEND')
        scene.add(pyrender.Mesh.from_trimesh(bottle_mesh, material=bottle_mat, smooth=True),
                  pose=bottle_pose)
    else:
        # No hands → bottle on table
        if prev_obj_pos is not None:
            obj_pos = prev_obj_pos
            bottle_pose = np.eye(4)
            bottle_pose[:3, 3] = obj_pos
            bottle_mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.7, 0.85, 0.95, 0.5], alphaMode='BLEND')
            scene.add(pyrender.Mesh.from_trimesh(bottle_mesh, material=bottle_mat, smooth=True),
                      pose=bottle_pose)

    # Camera: slightly elevated front view
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cp = np.eye(4)
    cam_pos = np.array([0.0, -0.38, 0.28])
    look_at = np.array([0.0, 0.05, 0.08])
    cp[:3, 3] = cam_pos
    fwd = look_at - cam_pos; fwd /= np.linalg.norm(fwd)
    up = np.array([0, 0, 1])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    cp[:3, 0] = right; cp[:3, 1] = -up; cp[:3, 2] = -fwd
    scene.add(camera, pose=cp)

    color_img, _ = renderer.render(scene)
    return color_img, obj_pos


###############################################################################
# Main
###############################################################################

def main():
    hamer_dir = os.path.join(BASE_DIR, 'hamer')
    data_dir = os.path.join(BASE_DIR, 'output/pick_bottle_video')
    orig_dir = os.path.join(BASE_DIR, 'test_videos/pick_bottle_all')
    out_dir = os.path.join(BASE_DIR, 'output/hoi_demo')
    os.makedirs(out_dir, exist_ok=True)

    print("Loading MANO model...")
    mano_model, faces = load_mano(hamer_dir)

    # Load all results
    with open(os.path.join(data_dir, 'mano_results.json')) as f:
        all_results = json.load(f)

    # Group by frame
    from collections import OrderedDict
    frame_results = OrderedDict()
    for r in all_results:
        key = r['img_name'].replace('.jpg', '')
        if key not in frame_results:
            frame_results[key] = []
        frame_results[key].append(r)

    frame_names = list(frame_results.keys())
    print(f"Frames: {len(frame_names)}")

    # Create bottle mesh (cylinder approximation)
    bottle_body = trimesh.creation.cylinder(radius=0.025, height=0.12, sections=16)
    bottle_cap = trimesh.creation.cylinder(radius=0.012, height=0.03, sections=16)
    bottle_cap.apply_translation([0, 0, 0.075])
    bottle_mesh = trimesh.util.concatenate([bottle_body, bottle_cap])

    # Setup renderer
    renderer = pyrender.OffscreenRenderer(480, 360)
    fps = 10

    # Get original frames and overlay frames
    orig_frames = sorted(glob.glob(os.path.join(orig_dir, '*.jpg')))
    overlay_frames = sorted(glob.glob(os.path.join(data_dir, '*_overlay.jpg')))

    # Output video: original | overlay | 3D HOI sim
    W_orig, H_orig = 480, 360  # resize target
    W_sim = 480
    total_w = W_orig + W_orig + W_sim

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        os.path.join(out_dir, 'robowheel_hoi_demo.mp4'),
        fourcc, fps, (total_w, H_orig))

    # Also save 3D sim only
    writer_sim = cv2.VideoWriter(
        os.path.join(out_dir, 'hoi_3d_scene.mp4'),
        fourcc, fps, (W_sim, 360))

    prev_obj_pos = None
    n_frames = min(len(frame_names), len(orig_frames), len(overlay_frames))

    for i in range(n_frames):
        fname = frame_names[i]

        # Render 3D HOI scene
        sim_img, obj_pos = render_hoi_frame(
            mano_model, faces, frame_results[fname], renderer,
            bottle_mesh, prev_obj_pos)
        prev_obj_pos = obj_pos

        sim_bgr = sim_img[:, :, ::-1]
        writer_sim.write(sim_bgr)

        # Load original and overlay
        orig = cv2.imread(orig_frames[i])
        overlay = cv2.imread(overlay_frames[i])

        orig_r = cv2.resize(orig, (W_orig, H_orig))
        overlay_r = cv2.resize(overlay, (W_orig, H_orig))
        sim_r = cv2.resize(sim_bgr, (W_sim, H_orig))

        # Combine
        combined = np.hstack([orig_r, overlay_r, sim_r])

        # Labels
        cv2.putText(combined, "Original Video", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        cv2.putText(combined, "HaMeR 3D Hands", (W_orig + 10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        cv2.putText(combined, "HOI Sim (Hand+Object)", (W_orig * 2 + 10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)

        writer.write(combined)

        if i % 30 == 0:
            print(f"  Frame {i}/{n_frames}")
            if i == 60:
                cv2.imwrite(os.path.join(out_dir, 'hoi_preview.jpg'), combined)

    writer.release()
    writer_sim.release()
    renderer.delete()

    # Also generate GIF
    print("Generating GIF...")
    import subprocess
    subprocess.run([
        'ffmpeg', '-y', '-i', os.path.join(out_dir, 'robowheel_hoi_demo.mp4'),
        '-vf', 'fps=8,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
        os.path.join(out_dir, 'robowheel_hoi_demo.gif')
    ], capture_output=True)

    print(f"\nDone! Output files in {out_dir}/")
    for f in sorted(os.listdir(out_dir)):
        size = os.path.getsize(os.path.join(out_dir, f))
        print(f"  {f}  ({size/1024:.0f} KB)")


if __name__ == '__main__':
    main()
