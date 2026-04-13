from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import trimesh
import numpy as np
from PIL import Image
import pyrender
import cv2
import os

# Ensure correct platform for headless offscreen rendering, e.g. EGL:
os.environ["PYOPENGL_PLATFORM"] = "egl"


def remove_black_background(image_path):
    import cv2
    import numpy as np

    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold: separate black background
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Clean mask (important for edges)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Convert to RGBA
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Set alpha channel
    rgba[:, :, 3] = mask

    # Save temporary cleaned image
    output_path = image_path.replace(".png", "_nobg.png")
    cv2.imwrite(output_path, rgba)

    return output_path

if __name__ == "__main__":

    # path of the object images
    clean_path = remove_black_background('assets/images/00442_obj0.png')
    # path to store rendered video to demonstrate the reconstructed object
    video_output_path = "dual_spin_fixed_bottle.mp4"
    # path to store the 3D object mesh
    mesh_output_path = "./results/reconstruction_results/"
    os.makedirs(mesh_output_path, exist_ok=True)

    
    
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
    mesh = pipeline(image=clean_path)[0]


    print("reconstructed mesh:",mesh)
    print("Vertices:", mesh.vertices.shape)
    print("Faces:", mesh.faces.shape)
    print("Is empty:", mesh.is_empty)

    # save the reconstructed mesh
    mesh.export(os.path.join(mesh_output_path,"bottle.glb"))
    print(f"mesh saved to: {mesh_output_path}")
    
    # Convert mesh
    tri_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    
    # Normalize mesh
    tri_mesh.apply_translation(-tri_mesh.centroid)
    tri_mesh.apply_scale(1.0 / tri_mesh.scale)
    
    # Create scene
    scene = pyrender.Scene()
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)
    mesh_node = scene.add(render_mesh, name="mesh")
    
    # Fixed camera (looking at origin)
    camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 3.0],  # camera 3 units back
        [0, 0, 0, 1]
    ])
    scene.add(camera, pose=camera_pose, name="camera")
    
    # Light (placed with camera)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
    scene.add(light, pose=camera_pose, name="light")
    
    # Video parameters
    width, height = 640, 640
    fps = 30
    total_frames = 180  # 6 seconds at 30 FPS
    half_frames = total_frames // 2
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    assert video_writer.isOpened(), "Failed to open video writer"
    
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    
    Y_AXIS = np.array([0.0, 1.0, 0.0])
    X_AXIS = np.array([1.0, 0.0, 0.0])
    
    for i in range(total_frames):
        if i < half_frames:
            # Horizontal spin (rotate around vertical axis)
            angle = 2 * np.pi * (i / half_frames)
            axis = Y_AXIS
        else:
            # Vertical spin (rotate around horizontal axis)
            angle = 2 * np.pi * ((i - half_frames) / half_frames)
            axis = X_AXIS
    
        # Compute rotation matrix about the axis, around mesh center
        transform = trimesh.transformations.rotation_matrix(angle, axis, tri_mesh.centroid)
    
        # Update existing mesh node pose
        scene.set_pose(mesh_node, pose=transform)
    
        # Render frame
        color, _ = renderer.render(scene)
        frame_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    
    # Clean up
    renderer.delete()
    video_writer.release()
    
    print(f"Saved video: {video_output_path}")
    
    
    # image to show the object
    
    # Center mesh
    tri_mesh.apply_translation(-tri_mesh.centroid)
    tri_mesh.apply_scale(1.0 / tri_mesh.scale)
    
    scene = pyrender.Scene()
    
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)
    scene.add(render_mesh)
    
    # Camera (this is KEY)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 2.5],  # move back
        [0, 0, 0, 1]
    ])
    scene.add(camera, pose=camera_pose)
    
    # Light (REQUIRED)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)
    
    # Render
    r = pyrender.OffscreenRenderer(1024, 1024)
    color, depth = r.render(scene)
    
    Image.fromarray(color).save("render_bottle.png")

