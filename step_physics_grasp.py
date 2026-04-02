"""
Physics-based bimanual grasp: MuJoCo native contact + friction.
No nvblox SDF hack — let the physics engine handle penetration and contact forces.

Key difference from previous scripts:
  - Bottle has freejoint + collision geoms + mass + friction
  - Hand-bottle contact handled by MuJoCo solver (not external SDF)
  - If grip force > gravity → bottle lifts with hands
"""
import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MENAGERIE = os.path.join(BASE_DIR, 'mujoco_menagerie')
OUT_DIR = os.path.join(BASE_DIR, 'output/physics_grasp')


def build_physics_scene():
    """Scene with proper physics: collision, friction, mass."""

    allegro_r = os.path.join(MENAGERIE, 'wonik_allegro/right_hand.xml')
    allegro_l = os.path.join(MENAGERIE, 'wonik_allegro/left_hand.xml')
    os.chdir(os.path.dirname(allegro_r))

    hand_r = mujoco.MjSpec.from_file('right_hand.xml')
    hand_l = mujoco.MjSpec.from_file('left_hand.xml')

    s = mujoco.MjSpec()
    s.option.gravity = [0, 0, -9.81]
    s.option.timestep = 0.002
    # Good contact parameters
    s.option.impratio = 10  # better friction cone
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

    w = s.worldbody

    # Lights
    for pos, dir_, diff in [
        ([0, -0.3, 0.5], [0, 0.2, -1], [1, 1, 1]),
        ([0.3, 0.2, 0.4], [-0.2, -0.1, -0.5], [0.4, 0.4, 0.4]),
    ]:
        l = w.add_light(); l.pos = pos; l.dir = dir_; l.diffuse = diff

    # Floor
    f = w.add_geom(); f.type = mujoco.mjtGeom.mjGEOM_PLANE
    f.size = [0.5, 0.5, 0.01]; f.rgba = [0.92, 0.92, 0.92, 1]
    f.friction = [1.0, 0.005, 0.0001]

    # Table (static)
    t = w.add_geom(); t.type = mujoco.mjtGeom.mjGEOM_BOX
    t.size = [0.18, 0.14, 0.01]; t.pos = [0, 0, 0.15]
    t.rgba = [0.38, 0.28, 0.20, 1]; t.friction = [1.0, 0.005, 0.0001]

    # === BOTTLE: physics-enabled (freejoint + collision + mass + friction) ===
    bottle = w.add_body(); bottle.name = "bottle"
    bottle.pos = [0, 0, 0.26]  # on table
    bj = bottle.add_freejoint(); bj.name = "bottle_joint"

    # Main body
    bg = bottle.add_geom(); bg.name = "bottle_body"
    bg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    bg.size = [0.030, 0.08, 0]  # r=3cm, half-height=8cm
    bg.rgba = [0.15, 0.50, 0.85, 0.85]
    bg.mass = 0.25
    bg.friction = [1.5, 0.01, 0.001]  # high friction for gripping
    bg.solref = [0.02, 1.0]  # contact stiffness
    bg.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]  # contact impedance (5 params)

    # Bottle neck
    bn = bottle.add_geom(); bn.name = "bottle_neck"
    bn.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    bn.size = [0.014, 0.02, 0]; bn.pos = [0, 0, 0.10]
    bn.rgba = [0.1, 0.4, 0.75, 0.9]; bn.mass = 0.03
    bn.friction = [1.5, 0.01, 0.001]

    # === RIGHT HAND: 6DoF base + Allegro ===
    base_r = w.add_body(); base_r.name = "base_right"
    base_r.pos = [0.12, 0, 0.30]  # to the right, slightly above bottle center
    for jn, ax, jt, jr in [
        ("rx", [1,0,0], mujoco.mjtJoint.mjJNT_SLIDE, [-0.15, 0.15]),
        ("ry", [0,1,0], mujoco.mjtJoint.mjJNT_SLIDE, [-0.15, 0.15]),
        ("rz", [0,0,1], mujoco.mjtJoint.mjJNT_SLIDE, [-0.15, 0.15]),
    ]:
        j = base_r.add_joint(); j.name = jn; j.type = jt; j.axis = ax; j.range = jr

    mount_r = base_r.add_body(); mount_r.name = "mount_right"
    # Fingers point LEFT (-X), palm faces bottle
    rot_r = R.from_euler('z', 180, degrees=True)
    mount_r.quat = rot_r.as_quat(scalar_first=True).tolist()
    frame_r = mount_r.add_frame()
    frame_r.attach_body(hand_r.worldbody.first_body(), "rh_", "")

    # === LEFT HAND: 6DoF base + Allegro ===
    base_l = w.add_body(); base_l.name = "base_left"
    base_l.pos = [-0.12, 0, 0.30]  # to the left, slightly above
    for jn, ax, jt, jr in [
        ("lx", [1,0,0], mujoco.mjtJoint.mjJNT_SLIDE, [-0.15, 0.15]),
        ("ly", [0,1,0], mujoco.mjtJoint.mjJNT_SLIDE, [-0.15, 0.15]),
        ("lz", [0,0,1], mujoco.mjtJoint.mjJNT_SLIDE, [-0.15, 0.15]),
    ]:
        j = base_l.add_joint(); j.name = jn; j.type = jt; j.axis = ax; j.range = jr

    mount_l = base_l.add_body(); mount_l.name = "mount_left"
    # Fingers point RIGHT (+X), default orientation
    mount_l.quat = [1, 0, 0, 0]  # identity
    frame_l = mount_l.add_frame()
    frame_l.attach_body(hand_l.worldbody.first_body(), "lh_", "")

    # Base actuators
    for jn, kp in [("rx", 300), ("ry", 300), ("rz", 300),
                    ("lx", 300), ("ly", 300), ("lz", 300)]:
        a = s.add_actuator(); a.name = f"act_{jn}"
        a.target = jn; a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gainprm = [kp] + [0]*9

    model = s.compile()
    return model


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Building physics scene...")
    model = build_physics_scene()
    data = mujoco.MjData(model)
    print(f"  joints={model.njnt}, actuators={model.nu}, bodies={model.nbody}")
    print(f"  ncon_max={model.nconmax}")

    # Actuator map
    act_map = {}
    rh_acts = []; lh_acts = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        act_map[name] = i
        if "rh_" in name: rh_acts.append((i, name))
        elif "lh_" in name: lh_acts.append((i, name))
    print(f"  Base acts: {sum(1 for n in act_map if 'act_r' in n or 'act_l' in n)}, "
          f"RH: {len(rh_acts)}, LH: {len(lh_acts)}")

    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'bottle')

    # === 5-stage trajectory ===
    REACH = 60
    GRASP_FRAMES = 200
    TOTAL = REACH + GRASP_FRAMES
    ctrl = np.zeros((TOTAL, model.nu))

    print(f"\nGenerating {TOTAL}-frame trajectory...")

    for i in range(TOTAL):
        if i < REACH:
            # REACH: hands come from sides, fingers open
            p = 3 * (i / REACH)**2 - 2 * (i / REACH)**3  # ease

            # Right hand: starts far right, moves left toward bottle
            if 'act_rx' in act_map: ctrl[i, act_map['act_rx']] = 0.06 * (1 - p)
            if 'act_rz' in act_map: ctrl[i, act_map['act_rz']] = 0.03 * (1 - p)
            # Left hand: starts far left, moves right toward bottle
            if 'act_lx' in act_map: ctrl[i, act_map['act_lx']] = -0.06 * (1 - p)
            if 'act_lz' in act_map: ctrl[i, act_map['act_lz']] = 0.03 * (1 - p)

            # Fingers: fully open
            for ai, _ in rh_acts + lh_acts:
                jid = model.actuator_trnid[ai, 0]
                if jid >= 0:
                    ctrl[i, ai] = model.jnt_range[jid][0]

        else:
            idx = i - REACH
            t = idx / max(GRASP_FRAMES - 1, 1)

            # Base: approach then hold then lift
            if t < 0.25:
                # APPROACH: move inward
                p = t / 0.25
                approach = -0.05 * p
                if 'act_rx' in act_map: ctrl[i, act_map['act_rx']] = approach
                if 'act_lx' in act_map: ctrl[i, act_map['act_lx']] = -approach
            elif t < 0.70:
                # CLOSE + GRASP: hold position
                if 'act_rx' in act_map: ctrl[i, act_map['act_rx']] = -0.05
                if 'act_lx' in act_map: ctrl[i, act_map['act_lx']] = 0.05
            else:
                # LIFT: move up
                lp = (t - 0.70) / 0.30
                if 'act_rx' in act_map: ctrl[i, act_map['act_rx']] = -0.05
                if 'act_lx' in act_map: ctrl[i, act_map['act_lx']] = 0.05
                if 'act_rz' in act_map: ctrl[i, act_map['act_rz']] = 0.08 * lp
                if 'act_lz' in act_map: ctrl[i, act_map['act_lz']] = 0.08 * lp

            # Fingers: staged curl
            if t < 0.20:
                curl = 0
            elif t < 0.45:
                curl = ((t - 0.20) / 0.25) * 0.7
            elif t < 0.65:
                curl = 0.7 + ((t - 0.45) / 0.20) * 0.25
            else:
                curl = 0.95  # maximum grip for lifting

            for ai, _ in rh_acts + lh_acts:
                jid = model.actuator_trnid[ai, 0]
                if jid >= 0:
                    jr = model.jnt_range[jid]
                    ctrl[i, ai] = jr[0] + curl * (jr[1] - jr[0])

    # Smooth
    ctrl = uniform_filter1d(ctrl, size=5, axis=0)

    # === Simulate with physics ===
    print("\nSimulating with contact physics...")
    renderer = mujoco.Renderer(model, height=480, width=640)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 0.28]
    cam.distance = 0.50
    cam.azimuth = 145
    cam.elevation = -25

    fps = 15
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(os.path.join(OUT_DIR, 'physics_grasp.mp4'), fourcc, fps, (640, 480))

    mujoco.mj_resetData(model, data)
    bottle_z_init = None
    phases = ['REACH', 'APPROACH', 'CLOSE', 'GRASP', 'LIFT']

    for i in range(TOTAL):
        data.ctrl[:] = ctrl[i]

        # Multiple physics steps per control step (more stable)
        for _ in range(15):
            mujoco.mj_step(model, data)

        bz = data.xpos[bottle_id][2] if bottle_id >= 0 else 0
        if bottle_z_init is None:
            bottle_z_init = bz

        # Determine phase
        if i < REACH:
            phase = "REACH"
        else:
            t = (i - REACH) / max(GRASP_FRAMES - 1, 1)
            if t < 0.25: phase = "APPROACH"
            elif t < 0.45: phase = "CLOSE"
            elif t < 0.70: phase = "GRASP"
            else: phase = "LIFT"

        # Render
        renderer.update_scene(data, cam)
        img = renderer.render()
        bgr = img[:, :, ::-1].copy()
        cv2.putText(bgr, f"[{phase}] {i} bottle_z={bz:.3f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)

        # Contact info
        n_contacts = data.ncon
        if n_contacts > 0:
            cv2.putText(bgr, f"contacts: {n_contacts}", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 255), 1)

        writer.write(bgr)

        if i % 30 == 0:
            print(f"  Frame {i:3d}/{TOTAL} [{phase:8s}] bottle_z={bz:.3f} contacts={n_contacts}")

        if i in [0, REACH//2, REACH-1, REACH+GRASP_FRAMES//5,
                 REACH+2*GRASP_FRAMES//5, REACH+3*GRASP_FRAMES//5,
                 REACH+4*GRASP_FRAMES//5, TOTAL-1]:
            cv2.imwrite(os.path.join(OUT_DIR, f'kf_{i:04d}_{phase.lower()}.jpg'), bgr)

    writer.release()

    # Results
    bz_final = data.xpos[bottle_id][2] if bottle_id >= 0 else 0
    lift = bz_final - bottle_z_init
    success = lift > 0.02

    print(f"\n=== Results ===")
    print(f"  Bottle z: {bottle_z_init:.3f} → {bz_final:.3f} ({lift*100:+.1f}cm)")
    print(f"  Grasp success: {'YES!' if success else 'No'}")

    # Comparison video with HaMeR overlay
    overlays_dir = os.path.join(BASE_DIR, 'output/pick_bottle_video')
    overlays = sorted([f for f in os.listdir(overlays_dir) if f.endswith('_overlay.jpg')])

    cap = cv2.VideoCapture(os.path.join(OUT_DIR, 'physics_grasp.mp4'))
    pw, ph = 640, 480
    w_cmp = cv2.VideoWriter(os.path.join(OUT_DIR, 'comparison.mp4'), fourcc, fps, (pw*2, ph+25))

    for i in range(TOTAL):
        ov_idx = max(0, min(int((i - REACH) / GRASP_FRAMES * len(overlays)), len(overlays) - 1))
        ov = cv2.resize(cv2.imread(os.path.join(overlays_dir, overlays[ov_idx])), (pw, ph))
        ret, sim = cap.read()
        if not ret: break
        panels = np.hstack([ov, sim])
        label = np.ones((25, pw*2, 3), dtype=np.uint8) * 35
        cv2.putText(label, "Video + HaMeR", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        cv2.putText(label, "Physics Simulation", (pw+5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        w_cmp.write(np.vstack([label, panels]))
    w_cmp.release(); cap.release()

    # GIF
    import subprocess
    subprocess.run(['ffmpeg', '-y', '-i', os.path.join(OUT_DIR, 'comparison.mp4'),
                    '-vf', 'fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
                    os.path.join(OUT_DIR, 'comparison.gif')], capture_output=True)

    print(f"\nDone! {OUT_DIR}/")
    for fn in sorted(os.listdir(OUT_DIR)):
        print(f"  {fn} ({os.path.getsize(os.path.join(OUT_DIR, fn))/1024:.0f} KB)")


if __name__ == '__main__':
    main()
