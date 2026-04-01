"""
Visualize HaMeR reconstructed hand meshes in MuJoCo.
Loads MANO parameters from mano_results.json, reconstructs meshes via SMPLX/MANO,
and renders them as a sequence in MuJoCo with a table + object scene.
"""
import os, sys, json, glob, shutil
import numpy as np

os.environ["MUJOCO_GL"] = "egl"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def mano_params_to_mesh(results_json, hamer_dir):
    """Use HaMeR's MANO model to convert parameters back to meshes."""
    sys.path.insert(0, hamer_dir)
    import torch
    from hamer.models import load_hamer, DEFAULT_CHECKPOINT

    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    mano_right = model.mano

    with open(results_json) as f:
        all_results = json.load(f)

    # Group by frame
    from collections import defaultdict
    frames = defaultdict(list)
    for r in all_results:
        frames[r['img_name']].append(r)

    return frames, mano_right


def create_mujoco_scene(mesh_dir, output_video, fps=10):
    """Create a MuJoCo scene with hand meshes and render video."""
    import mujoco
    import cv2

    # Get sorted OBJ files (overlay frames order)
    overlay_files = sorted(glob.glob(os.path.join(mesh_dir, '*_overlay.jpg')))
    frame_names = [os.path.basename(f).replace('_overlay.jpg', '') for f in overlay_files]

    print(f"Found {len(frame_names)} frames")

    # For each frame, collect hand meshes
    frame_meshes = []
    for fname in frame_names:
        hand_objs = sorted(glob.glob(os.path.join(mesh_dir, f'{fname}_hand*.obj')))
        frame_meshes.append(hand_objs)

    # Load MANO results for camera info
    with open(os.path.join(mesh_dir, 'mano_results.json')) as f:
        all_results = json.load(f)

    # Build per-frame result lookup
    from collections import defaultdict
    frame_results = defaultdict(list)
    for r in all_results:
        frame_results[r['img_name'].replace('.jpg', '')].append(r)

    # Create MuJoCo XML with table
    xml = """
    <mujoco model="hand_viz">
      <option gravity="0 0 -9.81"/>
      <visual>
        <global offwidth="640" offheight="480"/>
      </visual>
      <asset>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                 rgb1=".9 .9 .9" rgb2=".7 .7 .7"/>
        <material name="table_mat" texture="grid" texrepeat="4 4" reflectance="0.1"/>
        <material name="hand_mat" rgba="0.85 0.75 0.65 0.9"/>
      </asset>
      <worldbody>
        <light pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>
        <light pos="1 1 2" dir="-0.5 -0.5 -1" diffuse="0.5 0.5 0.5"/>
        <geom type="plane" size="2 2 0.01" rgba=".95 .95 .95 1" pos="0 0 0"/>
        <body name="table" pos="0 0 0.35">
          <geom type="box" size="0.4 0.3 0.02" material="table_mat"/>
        </body>
      </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Setup renderer
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Camera setup
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 0.4]
    cam.distance = 1.5
    cam.azimuth = 180
    cam.elevation = -25

    # Render each frame
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (640, 480))

    scene_opt = mujoco.MjvOption()

    for i, (fname, objs) in enumerate(zip(frame_names, frame_meshes)):
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, cam, scene_opt)

        # Add hand meshes as visual geoms
        for obj_path in objs:
            import trimesh
            mesh = trimesh.load(obj_path)
            verts = mesh.vertices.astype(np.float32)
            faces = mesh.faces.astype(np.int32)

            # Transform from camera space to world space
            # Flip Y and Z for MuJoCo coordinate system, scale and position on table
            verts_world = verts.copy()
            verts_world[:, 1] = -verts[:, 1]  # flip Y
            verts_world[:, 2] = -verts[:, 2]  # flip Z
            # Scale up and position above table
            verts_world *= 3.0
            verts_world[:, 2] += 0.55  # above table

            # Add mesh triangles to scene as small geoms
            # (MuJoCo doesn't natively support dynamic mesh insertion in renderer,
            #  so we render the mesh overlay separately)

        # Render base scene
        img = renderer.render()
        writer.write(img[:, :, ::-1])  # RGB to BGR

        if i % 20 == 0:
            print(f"  Rendered frame {i}/{len(frame_names)}")

    writer.release()
    print(f"Saved MuJoCo scene: {output_video}")


def create_trimesh_visualization(mesh_dir, output_video, fps=10):
    """
    Create a 3D visualization of hand meshes over time using trimesh + pyrender.
    This is more flexible than MuJoCo for visualizing arbitrary meshes.
    """
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    import trimesh
    import pyrender
    import cv2

    overlay_files = sorted(glob.glob(os.path.join(mesh_dir, '*_overlay.jpg')))
    frame_names = [os.path.basename(f).replace('_overlay.jpg', '') for f in overlay_files]

    with open(os.path.join(mesh_dir, 'mano_results.json')) as f:
        all_results = json.load(f)

    from collections import defaultdict
    frame_results = defaultdict(list)
    for r in all_results:
        frame_results[r['img_name'].replace('.jpg', '')].append(r)

    print(f"Rendering {len(frame_names)} frames in 3D viewer...")

    # Setup pyrender
    r = pyrender.OffscreenRenderer(640, 480)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (640, 480))

    hand_color = [0.85, 0.72, 0.62, 1.0]  # skin tone
    table_color = [0.4, 0.3, 0.2, 1.0]  # dark wood

    for i, fname in enumerate(frame_names):
        scene = pyrender.Scene(bg_color=[0.95, 0.95, 0.95, 1.0],
                                ambient_light=[0.3, 0.3, 0.3])

        # Add lights
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=np.eye(4))
        light2 = pyrender.PointLight(color=[1.0, 0.95, 0.9], intensity=5.0)
        light2_pose = np.eye(4)
        light2_pose[:3, 3] = [0.5, -0.5, 1.0]
        scene.add(light2, pose=light2_pose)

        # Add table
        table_mesh = trimesh.creation.box(extents=[0.8, 0.6, 0.04])
        table_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=table_color)
        table_node = pyrender.Mesh.from_trimesh(table_mesh, material=table_mat, smooth=False)
        table_pose = np.eye(4)
        table_pose[2, 3] = 0.0  # table at z=0
        scene.add(table_node, pose=table_pose)

        # Add floor
        floor = trimesh.creation.box(extents=[3, 3, 0.01])
        floor_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.9, 0.9, 0.9, 1.0])
        floor_node = pyrender.Mesh.from_trimesh(floor, material=floor_mat, smooth=False)
        floor_pose = np.eye(4)
        floor_pose[2, 3] = -0.3
        scene.add(floor_node, pose=floor_pose)

        # Add hand meshes for this frame
        hand_objs = sorted(glob.glob(os.path.join(mesh_dir, f'{fname}_hand*.obj')))
        results = frame_results.get(fname, [])

        for j, obj_path in enumerate(hand_objs):
            mesh = trimesh.load(obj_path)
            verts = mesh.vertices.copy()

            # Get camera translation for this hand
            if j < len(results):
                cam_t = np.array(results[j]['cam_t_full'])
            else:
                cam_t = np.array([0, 0, 1.0])

            # Transform: camera space → world space
            # Camera convention: X-right, Y-down, Z-forward
            # World convention: X-right, Y-forward, Z-up
            verts_world = np.zeros_like(verts)
            verts_world[:, 0] = verts[:, 0]     # X stays
            verts_world[:, 1] = -verts[:, 2]    # Z_cam → -Y_world
            verts_world[:, 2] = -verts[:, 1]    # Y_cam → -Z_world (flip)

            # Scale to reasonable size and position above table
            verts_world *= 2.5
            verts_world[:, 2] += 0.25  # above table

            new_mesh = trimesh.Trimesh(vertices=verts_world, faces=mesh.faces)
            hand_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=hand_color)
            hand_node = pyrender.Mesh.from_trimesh(new_mesh, material=hand_mat, smooth=True)
            scene.add(hand_node)

        # Camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.5)
        cam_pose = np.eye(4)
        cam_pose[:3, 3] = [0, -0.8, 0.6]
        # Look at table center
        forward = np.array([0, 0.8, -0.2])
        forward /= np.linalg.norm(forward)
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = -up
        cam_pose[:3, 2] = -forward
        scene.add(camera, pose=cam_pose)

        # Render
        color, _ = r.render(scene)
        writer.write(color[:, :, ::-1])

        if i % 30 == 0:
            print(f"  Frame {i}/{len(frame_names)}")

    writer.release()
    r.delete()
    print(f"Saved 3D visualization: {output_video}")


def create_combined_video(overlay_dir, sim_video, output_video, fps=10):
    """Combine original overlay video with 3D sim visualization side by side."""
    import cv2

    overlays = sorted(glob.glob(os.path.join(overlay_dir, '*_overlay.jpg')))

    cap = cv2.VideoCapture(sim_video)
    sample = cv2.imread(overlays[0])
    h1, w1 = sample.shape[:2]

    ret, sim_frame = cap.read()
    h2, w2 = sim_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Normalize heights
    target_h = min(h1, h2)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (w1 + w2, target_h))

    for ov_path in overlays:
        ov = cv2.imread(ov_path)
        ret, sim = cap.read()
        if not ret:
            break

        ov_resized = cv2.resize(ov, (w1, target_h))
        sim_resized = cv2.resize(sim, (w2, target_h))
        combined = np.hstack([ov_resized, sim_resized])

        # Add labels
        cv2.putText(combined, "HaMeR Overlay", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(combined, "3D Sim View", (w1 + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        writer.write(combined)

    writer.release()
    cap.release()
    print(f"Saved combined: {output_video}")


if __name__ == '__main__':
    mesh_dir = os.path.join(BASE_DIR, 'output/pick_bottle_video')
    out_dir = os.path.join(BASE_DIR, 'output')

    # 1. 3D scene visualization with pyrender
    sim_video = os.path.join(out_dir, 'hamer_3d_sim.mp4')
    create_trimesh_visualization(mesh_dir, sim_video, fps=10)

    # 2. Combined video (overlay | 3D sim)
    combined_video = os.path.join(out_dir, 'hamer_combined.mp4')
    create_combined_video(mesh_dir, sim_video, combined_video, fps=10)

    print("\nDone! Output files:")
    print(f"  3D sim:   {sim_video}")
    print(f"  Combined: {combined_video}")
