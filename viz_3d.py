"""
Reconstruct MANO meshes from saved parameters and render in 3D scene with pyrender.
"""
import os, json, glob
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import cv2
import trimesh
import pyrender
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_mano_model(hamer_dir):
    """Load MANO model from HaMeR."""
    import sys
    sys.path.insert(0, hamer_dir)
    from hamer.models import load_hamer, DEFAULT_CHECKPOINT
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    return model.mano, model.mano.faces

def reconstruct_mesh(mano_model, hand_pose, betas, global_orient, is_right):
    """Reconstruct vertices from MANO parameters."""
    hand_pose_t = torch.tensor(hand_pose, dtype=torch.float32).unsqueeze(0)
    betas_t = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
    global_orient_t = torch.tensor(global_orient, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        out = mano_model(
            hand_pose=hand_pose_t,
            betas=betas_t,
            global_orient=global_orient_t,
        )
    verts = out.vertices[0].numpy()
    if not is_right:
        verts[:, 0] = -verts[:, 0]
    return verts

def main():
    hamer_dir = os.path.join(BASE_DIR, 'hamer')
    mesh_dir = os.path.join(BASE_DIR, 'output/pick_bottle_video')
    out_dir = os.path.join(BASE_DIR, 'output')

    print("Loading MANO model...")
    mano_model, faces = load_mano_model(hamer_dir)

    with open(os.path.join(mesh_dir, 'mano_results.json')) as f:
        all_results = json.load(f)

    # Group by frame
    from collections import defaultdict, OrderedDict
    frame_results = OrderedDict()
    for r in all_results:
        key = r['img_name'].replace('.jpg', '')
        if key not in frame_results:
            frame_results[key] = []
        frame_results[key].append(r)

    frame_names = list(frame_results.keys())
    print(f"Total frames: {len(frame_names)}")

    # Setup renderer
    renderer = pyrender.OffscreenRenderer(640, 480)
    hand_color = [0.85, 0.72, 0.62, 1.0]
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(os.path.join(out_dir, 'hamer_3d_sim.mp4'), fourcc, fps, (640, 480))

    for i, fname in enumerate(frame_names):
        scene = pyrender.Scene(bg_color=[0.92, 0.92, 0.95, 1.0], ambient_light=[0.4, 0.4, 0.4])

        # Lights
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        lp = np.eye(4); lp[:3, :3] = trimesh.transformations.euler_matrix(0.5, 0.3, 0)[:3, :3]
        scene.add(light, pose=lp)

        point_light = pyrender.PointLight(color=[1.0, 0.95, 0.9], intensity=8.0)
        plp = np.eye(4); plp[:3, 3] = [0.3, -0.5, 0.8]
        scene.add(point_light, pose=plp)

        # Table
        table = trimesh.creation.box(extents=[0.6, 0.4, 0.03])
        table_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.35, 0.25, 0.18, 1.0], metallicFactor=0.1)
        tp = np.eye(4); tp[2, 3] = 0.0
        scene.add(pyrender.Mesh.from_trimesh(table, material=table_mat, smooth=False), pose=tp)

        # Hands
        for j, r in enumerate(frame_results[fname]):
            verts = reconstruct_mesh(
                mano_model,
                r['mano_hand_pose'], r['mano_betas'], r['mano_global_orient'],
                r['is_right']
            )
            cam_t = np.array(r['cam_t_full'])

            # MANO verts are ~0.1m scale centered near wrist
            # cam_t = [x_offset, y_offset, depth] in camera frame
            # We DON'T add cam_t to verts. Instead use cam_t for positioning only.

            # Camera → Scene: X-right, Y-down, Z-forward → X-right, Y-forward, Z-up
            w = np.zeros_like(verts)
            w[:, 0] = verts[:, 0]
            w[:, 1] = -verts[:, 2]  # cam Z → scene -Y (forward)
            w[:, 2] = -verts[:, 1]  # cam -Y → scene Z (up)

            # Use cam_t to space hands apart horizontally + vertically
            # Normalize by depth to get NDC-like offset
            spacing_x = cam_t[0] / cam_t[2]
            spacing_z = -cam_t[1] / cam_t[2]

            w[:, 0] += spacing_x * 0.5
            w[:, 2] += spacing_z * 0.5 + 0.15  # lift above table

            hand_mesh = trimesh.Trimesh(vertices=w, faces=faces)
            hmat = pyrender.MetallicRoughnessMaterial(baseColorFactor=hand_color, metallicFactor=0.0, roughnessFactor=0.7)
            scene.add(pyrender.Mesh.from_trimesh(hand_mesh, material=hmat, smooth=True))

        # Camera - look at hands from front
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        cp = np.eye(4)
        cp[:3, 3] = [0, -0.4, 0.2]
        fwd = np.array([0, 0.4, -0.05]); fwd /= np.linalg.norm(fwd)
        up = np.array([0, 0, 1])
        right = np.cross(fwd, up); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        cp[:3, 0] = right; cp[:3, 1] = -up; cp[:3, 2] = -fwd
        scene.add(camera, pose=cp)

        color, _ = renderer.render(scene)
        writer.write(color[:, :, ::-1])

        if i % 30 == 0:
            print(f"  Frame {i}/{len(frame_names)}")
            if i == 60:
                cv2.imwrite(os.path.join(out_dir, '3d_sim_preview.jpg'), color[:, :, ::-1])

    writer.release()
    renderer.delete()
    print(f"Saved: {out_dir}/hamer_3d_sim.mp4")

    # Create combined video
    print("Creating combined video...")
    overlay_dir = mesh_dir
    overlays = sorted(glob.glob(os.path.join(overlay_dir, '*_overlay.jpg')))
    cap = cv2.VideoCapture(os.path.join(out_dir, 'hamer_3d_sim.mp4'))

    writer2 = cv2.VideoWriter(os.path.join(out_dir, 'hamer_combined.mp4'), fourcc, fps, (1280, 480))
    for ov_path in overlays:
        ov = cv2.imread(ov_path)
        ret, sim = cap.read()
        if not ret:
            break
        ov_r = cv2.resize(ov, (640, 480))
        combined = np.hstack([ov_r, sim])
        cv2.putText(combined, "Video + HaMeR 3D Mesh", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        cv2.putText(combined, "Pyrender 3D Scene", (650, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        writer2.write(combined)
    writer2.release()
    cap.release()
    print(f"Saved: {out_dir}/hamer_combined.mp4")


if __name__ == '__main__':
    main()
