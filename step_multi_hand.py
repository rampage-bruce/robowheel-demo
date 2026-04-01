"""
Test multiple dexterous hands with the same MANO trajectory.
For each hand: SPIDER IK → approach trajectory → MuJoCo render.
Generates side-by-side comparison of all hands.
"""
import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
import cv2
from scipy.ndimage import uniform_filter1d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SPIDER_DIR = os.path.join(BASE_DIR, 'spider')
MENAGERIE = os.path.join(BASE_DIR, 'mujoco_menagerie')
OUT_DIR = os.path.join(BASE_DIR, 'output/multi_hand')

# Hands to test (name, xml_path, palm_forward_euler)
HANDS = [
    {
        'name': 'Allegro',
        'xml': f'{MENAGERIE}/wonik_allegro/right_hand.xml',
        'color': [0.4, 0.4, 0.4, 1.0],
    },
    {
        'name': 'LEAP',
        'xml': f'{MENAGERIE}/leap_hand/right_hand.xml',
        'color': [0.35, 0.35, 0.45, 1.0],
    },
    {
        'name': 'Shadow_DEXee',
        'xml': f'{MENAGERIE}/shadow_dexee/shadow_dexee.xml',
        'color': [0.3, 0.3, 0.3, 1.0],
    },
]


def build_scene(hand_xml):
    """Build MuJoCo scene: hand on 6DoF base + bottle."""
    from scipy.spatial.transform import Rotation

    os.chdir(os.path.dirname(os.path.abspath(hand_xml)))
    hand_spec = mujoco.MjSpec.from_file(os.path.basename(hand_xml))

    s = mujoco.MjSpec()
    s.option.gravity = [0, 0, -9.81]
    s.option.timestep = 0.002

    w = s.worldbody
    # Light
    l = w.add_light(); l.pos = [0, -0.2, 0.5]; l.dir = [0, 0.1, -1]; l.diffuse = [1, 1, 1]
    l2 = w.add_light(); l2.pos = [0.2, -0.1, 0.4]; l2.dir = [-0.1, 0.1, -0.5]; l2.diffuse = [0.5, 0.5, 0.5]

    # Floor
    f = w.add_geom(); f.type = mujoco.mjtGeom.mjGEOM_PLANE
    f.size = [0.3, 0.3, 0.01]; f.rgba = [0.92, 0.92, 0.92, 1]

    # Table
    t = w.add_geom(); t.type = mujoco.mjtGeom.mjGEOM_BOX
    t.size = [0.12, 0.10, 0.008]; t.pos = [0, 0, 0.15]
    t.rgba = [0.38, 0.28, 0.20, 1]; t.friction = [1.5, 0.01, 0.001]

    # Bottle
    bottle = w.add_body(); bottle.name = "bottle"
    bottle.pos = [0, 0, 0.195]
    bj = bottle.add_freejoint(); bj.name = "bottle_free"
    bg = bottle.add_geom(); bg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    bg.size = [0.018, 0.035, 0]; bg.rgba = [0.15, 0.45, 0.85, 0.9]
    bg.mass = 0.12; bg.friction = [2.0, 0.01, 0.001]

    # Hand base
    base = w.add_body(); base.name = "hand_base"
    base.pos = [0, 0, 0.35]
    for jn, ax, jt in [
        ("bx", [1,0,0], mujoco.mjtJoint.mjJNT_SLIDE),
        ("by", [0,1,0], mujoco.mjtJoint.mjJNT_SLIDE),
        ("bz", [0,0,1], mujoco.mjtJoint.mjJNT_SLIDE),
    ]:
        j = base.add_joint(); j.name = jn; j.type = jt; j.axis = ax
        j.range = [-0.2, 0.2]

    # Mount (fingers down)
    mount = base.add_body(); mount.name = "hand_mount"
    rot = Rotation.from_euler('yzx', [90, 0, 180], degrees=True)
    mount.quat = rot.as_quat(scalar_first=True).tolist()

    # Attach hand
    hand_root = hand_spec.worldbody.first_body()
    frame = mount.add_frame()
    frame.attach_body(hand_root, "h_", "")

    # Base actuators
    for jn in ["bx", "by", "bz"]:
        a = s.add_actuator(); a.name = f"act_{jn}"
        a.target = jn; a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gainprm = [200] + [0]*9

    return s.compile()


def generate_trajectory(model, n_frames=120):
    """Generate a simple approach→curl→lift trajectory for any hand."""
    data = mujoco.MjData(model)
    nu = model.nu  # total actuators
    # First 3 are base actuators, rest are hand joints
    hand_nu = nu - 3
    base_start = 0  # depends on actuator order

    # Find which actuators are base vs hand
    base_acts = []
    hand_acts = []
    for i in range(nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        if "act_b" in name:
            base_acts.append(i)
        else:
            hand_acts.append(i)

    ctrl_traj = np.zeros((n_frames, nu))

    for i in range(n_frames):
        t = i / (n_frames - 1)

        # Base: descend then lift
        if t < 0.4:
            z = -0.12 * (t / 0.4)
        elif t < 0.7:
            z = -0.12
        else:
            z = -0.12 + 0.06 * ((t - 0.7) / 0.3)

        for ai in base_acts:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or ""
            if "bz" in name:
                ctrl_traj[i, ai] = z

        # Fingers: progressive curl
        if t < 0.3:
            curl = 0.0
        elif t < 0.6:
            curl = ((t - 0.3) / 0.3) * 0.8
        else:
            curl = 0.8

        for ai in hand_acts:
            # Get joint range
            # Actuator → joint mapping
            jid = model.actuator_trnid[ai, 0]
            if jid >= 0:
                jrange = model.jnt_range[jid]
                target = jrange[0] + curl * (jrange[1] - jrange[0])
                ctrl_traj[i, ai] = target

    # Smooth
    ctrl_traj = uniform_filter1d(ctrl_traj, size=5, axis=0)
    return ctrl_traj


def render_hand(model, ctrl_traj, out_path, cam_lookat, cam_dist=0.25):
    """Render trajectory to MP4."""
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=360, width=480)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = cam_lookat
    cam.distance = cam_dist
    cam.azimuth = 160
    cam.elevation = -25

    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (480, 360))

    frames = []
    mujoco.mj_resetData(model, data)

    for i in range(len(ctrl_traj)):
        data.ctrl[:] = ctrl_traj[i]
        for _ in range(10):
            mujoco.mj_step(model, data)
        renderer.update_scene(data, cam)
        img = renderer.render()
        img_bgr = img[:, :, ::-1]
        writer.write(img_bgr)
        if i in [0, len(ctrl_traj)//4, len(ctrl_traj)//2, 3*len(ctrl_traj)//4, len(ctrl_traj)-1]:
            frames.append(img_bgr.copy())

    writer.release()
    return frames


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_frames = {}
    N_FRAMES = 120

    for hand in HANDS:
        name = hand['name']
        print(f"\n=== {name} ===")

        try:
            model = build_scene(hand['xml'])
            print(f"  joints={model.njnt}, actuators={model.nu}")

            ctrl = generate_trajectory(model, N_FRAMES)
            out_mp4 = os.path.join(OUT_DIR, f'{name.lower()}.mp4')
            frames = render_hand(model, ctrl, out_mp4, [0, 0, 0.22])
            all_frames[name] = frames
            print(f"  Saved: {out_mp4}")

            # Save keyframes
            for j, frame in enumerate(frames):
                cv2.imwrite(os.path.join(OUT_DIR, f'{name.lower()}_kf{j}.jpg'), frame)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    # Generate comparison grid
    if len(all_frames) >= 2:
        print("\nGenerating comparison...")
        names = list(all_frames.keys())
        n_cols = len(names)
        pw, ph = 320, 240

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        w = cv2.VideoWriter(os.path.join(OUT_DIR, 'multi_hand_comparison.mp4'),
                            fourcc, 10, (pw * n_cols, ph + 20))

        for frame_idx in range(5):  # 5 keyframes
            panels = []
            for name in names:
                f = all_frames[name][frame_idx] if frame_idx < len(all_frames[name]) else np.zeros((ph, pw, 3), dtype=np.uint8)
                panels.append(cv2.resize(f, (pw, ph)))

            row = np.hstack(panels)
            label = np.ones((20, pw * n_cols, 3), dtype=np.uint8) * 35
            for j, name in enumerate(names):
                cv2.putText(label, name, (j * pw + 5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
            frame = np.vstack([label, row])

            # Write multiple times for longer display
            for _ in range(20):
                w.write(frame)

            if frame_idx == 2:
                cv2.imwrite(os.path.join(OUT_DIR, 'comparison_preview.jpg'), frame)

        w.release()

        # GIF
        import subprocess
        subprocess.run(['ffmpeg', '-y', '-i', os.path.join(OUT_DIR, 'multi_hand_comparison.mp4'),
                        '-vf', 'fps=2,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
                        os.path.join(OUT_DIR, 'multi_hand_comparison.gif')], capture_output=True)

    print(f"\nDone! {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f} ({os.path.getsize(os.path.join(OUT_DIR, f))/1024:.0f} KB)")


if __name__ == '__main__':
    main()
