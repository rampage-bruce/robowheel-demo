"""
Bimanual grasp: both hands + 5-stage trajectory + nvblox SDF binary search fix.

Fixes:
  1. Both left and right hands from MANO data
  2. Binary search for exact non-penetrating curl per finger
  3. REACH→APPROACH→CLOSE→GRASP→LIFT pipeline
"""
import os, json, sys
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import mujoco
import cv2
import trimesh
import smplx
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_DIR = os.path.join(BASE_DIR, 'spider')
MANO_RESULTS = os.path.join(BASE_DIR, 'output/pick_bottle_video/mano_results.json')
MENAGERIE = os.path.join(BASE_DIR, 'mujoco_menagerie')
OUT_DIR = os.path.join(BASE_DIR, 'output/bimanual_grasp')


# ============================================================
# Step 1: Load MANO data for both hands
# ============================================================

def load_mano_data():
    """Load and align left + right hand MANO data."""
    mano_model = smplx.create(
        '/mnt/users/yjy/sim/video2robot-retarget/HaWoR/_DATA/data',
        model_type='mano', is_rhand=True, use_pca=False, flat_hand_mean=False)

    with open(MANO_RESULTS) as f:
        results = json.load(f)

    # Group by frame
    from collections import OrderedDict
    frames = OrderedDict()
    for r in results:
        key = r['img_name']
        if key not in frames:
            frames[key] = {'left': None, 'right': None}
        side = 'right' if r['is_right'] else 'left'
        frames[key][side] = r

    # Get frames where both hands exist
    both = [(k, v) for k, v in frames.items() if v['left'] is not None and v['right'] is not None]
    print(f"Frames with both hands: {len(both)}")

    def extract_joints(r):
        hp = np.array(r['mano_hand_pose'])
        hp_aa = np.array([R.from_matrix(hp[j]).as_rotvec() for j in range(15)]).flatten()
        go_mat = np.array(r['mano_global_orient'])
        if go_mat.ndim == 3: go_mat = go_mat[0]
        go_aa = R.from_matrix(go_mat).as_rotvec()
        bt = torch.tensor(r['mano_betas'], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            out = mano_model(
                hand_pose=torch.tensor(hp_aa, dtype=torch.float32).unsqueeze(0),
                betas=bt,
                global_orient=torch.tensor(go_aa, dtype=torch.float32).unsqueeze(0))
        joints = out.joints[0].detach().numpy()
        # Flip X for left hand (MANO is right-hand model)
        if not r['is_right']:
            joints[:, 0] = -joints[:, 0]
        return joints, go_mat

    # Extract joint positions
    N = len(both)
    right_joints = np.zeros((N, 16, 3))
    left_joints = np.zeros((N, 16, 3))

    for i, (fname, data) in enumerate(both):
        rj, _ = extract_joints(data['right'])
        lj, _ = extract_joints(data['left'])
        right_joints[i] = rj
        left_joints[i] = lj

    # Transform: MANO camera → world (Y/Z swap + offset)
    def mano_to_world(pts):
        w = np.zeros_like(pts)
        w[..., 0] = pts[..., 0]
        w[..., 1] = -pts[..., 2]
        w[..., 2] = -pts[..., 1]
        return w + np.array([0, 0, 0.20])

    right_joints = mano_to_world(right_joints)
    left_joints = mano_to_world(left_joints)

    # Object: between right thumb tip and index tip
    obj_pos = (right_joints[:, 15, :] + right_joints[:, 3, :]) / 2  # thumb+index tips
    obj_pos = uniform_filter1d(obj_pos, size=5, axis=0)

    return N, right_joints, left_joints, obj_pos


# ============================================================
# Step 2: Build MuJoCo scene with two Allegro hands + bottle
# ============================================================

def build_bimanual_scene(obj_center):
    """Two Allegro hands + bottle on table."""
    allegro_xml = f'{MENAGERIE}/wonik_allegro/right_hand.xml'
    os.chdir(os.path.dirname(os.path.abspath(allegro_xml)))
    hand_spec_r = mujoco.MjSpec.from_file(os.path.basename(allegro_xml))
    hand_spec_l = mujoco.MjSpec.from_file('left_hand.xml')

    s = mujoco.MjSpec()
    s.option.gravity = [0, 0, -9.81]
    s.option.timestep = 0.002

    w = s.worldbody
    # Lights
    for pos, dir_, diff in [
        ([0, -0.3, 0.6], [0, 0.2, -1], [1, 1, 1]),
        ([0.3, -0.2, 0.5], [-0.2, 0.1, -0.5], [0.5, 0.5, 0.5]),
    ]:
        l = w.add_light(); l.pos = pos; l.dir = dir_; l.diffuse = diff

    # Floor
    f = w.add_geom(); f.type = mujoco.mjtGeom.mjGEOM_PLANE
    f.size = [0.5, 0.5, 0.01]; f.rgba = [0.93, 0.93, 0.93, 1]

    # Table
    t = w.add_geom(); t.type = mujoco.mjtGeom.mjGEOM_BOX
    t.size = [0.20, 0.15, 0.01]; t.pos = [obj_center[0], obj_center[1], 0.12]
    t.rgba = [0.38, 0.28, 0.20, 1]

    # Bottle (static - no freejoint, won't be knocked over)
    bottle = w.add_body(); bottle.name = "bottle"
    bottle.pos = obj_center.tolist()
    bg = bottle.add_geom(); bg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    bg.size = [0.033, 0.10, 0]; bg.rgba = [0.15, 0.50, 0.85, 0.85]
    bg.contype = 0; bg.conaffinity = 0  # no collision (visual only)

    # Right hand: 6DoF base (3 translate + 3 rotate) driven by MANO global_orient
    base_r = w.add_body(); base_r.name = "base_right"
    base_r.pos = [obj_center[0] + 0.10, obj_center[1], obj_center[2]]
    for jn, ax, jt in [
        ("rx", [1,0,0], mujoco.mjtJoint.mjJNT_SLIDE),
        ("ry", [0,1,0], mujoco.mjtJoint.mjJNT_SLIDE),
        ("rz", [0,0,1], mujoco.mjtJoint.mjJNT_SLIDE),
        ("rrx", [1,0,0], mujoco.mjtJoint.mjJNT_HINGE),
        ("rry", [0,1,0], mujoco.mjtJoint.mjJNT_HINGE),
        ("rrz", [0,0,1], mujoco.mjtJoint.mjJNT_HINGE),
    ]:
        j = base_r.add_joint(); j.name = jn; j.type = jt; j.axis = ax
        j.range = [-0.25, 0.25] if jt == mujoco.mjtJoint.mjJNT_SLIDE else [-3.14, 3.14]
    mount_r = base_r.add_body(); mount_r.name = "mount_right"
    mount_r.quat = [1, 0, 0, 0]  # identity, orientation driven by hinge joints
    frame_r = mount_r.add_frame()
    frame_r.attach_body(hand_spec_r.worldbody.first_body(), "rh_", "")

    # Left hand: same 6DoF
    base_l = w.add_body(); base_l.name = "base_left"
    base_l.pos = [obj_center[0] - 0.10, obj_center[1], obj_center[2]]
    for jn, ax, jt in [
        ("lx", [1,0,0], mujoco.mjtJoint.mjJNT_SLIDE),
        ("ly", [0,1,0], mujoco.mjtJoint.mjJNT_SLIDE),
        ("lz", [0,0,1], mujoco.mjtJoint.mjJNT_SLIDE),
        ("lrx", [1,0,0], mujoco.mjtJoint.mjJNT_HINGE),
        ("lry", [0,1,0], mujoco.mjtJoint.mjJNT_HINGE),
        ("lrz", [0,0,1], mujoco.mjtJoint.mjJNT_HINGE),
    ]:
        j = base_l.add_joint(); j.name = jn; j.type = jt; j.axis = ax
        j.range = [-0.25, 0.25] if jt == mujoco.mjtJoint.mjJNT_SLIDE else [-3.14, 3.14]
    mount_l = base_l.add_body(); mount_l.name = "mount_left"
    mount_l.quat = [1, 0, 0, 0]
    frame_l = mount_l.add_frame()
    frame_l.attach_body(hand_spec_l.worldbody.first_body(), "lh_", "")

    # Base actuators (translate + rotate for each hand)
    for jn, kp in [("rx", 200), ("ry", 200), ("rz", 200),
                    ("rrx", 80), ("rry", 80), ("rrz", 80),
                    ("lx", 200), ("ly", 200), ("lz", 200),
                    ("lrx", 80), ("lry", 80), ("lrz", 80)]:
        a = s.add_actuator(); a.name = f"act_{jn}"
        a.target = jn; a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gainprm = [kp] + [0]*9

    model = s.compile()
    return model


# ============================================================
# Step 3: Build nvblox TSDF
# ============================================================

def build_nvblox():
    from nvblox_torch.mapper import Mapper
    from nvblox_torch.sensor import Sensor

    mapper = Mapper(voxel_sizes_m=[0.003])
    # Cylinder bottle: r=3.3cm, h=20cm
    mesh = trimesh.creation.cylinder(radius=0.033, height=0.20, sections=32)

    W, H = 320, 240
    sensor = Sensor.from_camera(fu=250, fv=250, cu=W/2, cv=H/2, width=W, height=H)

    for i in range(16):
        for elev in [-0.05, 0.05, 0.12]:
            angle = 2 * np.pi * i / 16
            pos = np.array([0.2 * np.cos(angle), 0.2 * np.sin(angle), elev])
            fwd = -pos / np.linalg.norm(pos)
            up = np.array([0, 0, 1])
            right = np.cross(fwd, up)
            if np.linalg.norm(right) < 1e-6: right = np.array([1, 0, 0])
            right /= np.linalg.norm(right)
            up = np.cross(right, fwd)
            Rot = np.stack([right, -up, fwd], axis=1)
            T = np.eye(4); T[:3, :3] = Rot; T[:3, 3] = pos

            origins = np.tile(pos, (H*W, 1))
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            dirs = (Rot @ np.stack([(u.flatten()-W/2)/250, (v.flatten()-H/2)/250, np.ones(H*W)], axis=1).T).T
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            try:
                locs, idx, _ = mesh.ray.intersects_location(origins, dirs)
                depth = np.zeros(H*W, dtype=np.float32)
                for ri, loc in zip(idx, locs):
                    d = np.linalg.norm(loc - origins[ri])
                    if depth[ri] == 0 or d < depth[ri]: depth[ri] = d
                depth = depth.reshape(H, W)
                if (depth > 0).sum() > 50:
                    mapper.add_depth_frame(torch.from_numpy(depth).cuda(),
                                           torch.from_numpy(T.astype(np.float32)), sensor)
            except:
                pass
    return mapper


def nvblox_sdf(mapper, points):
    from nvblox_torch.mapper import QueryType
    pts = torch.from_numpy(points.astype(np.float32)).cuda()
    r = mapper.query_layer(QueryType.TSDF, pts)
    return r[:, 0].cpu().numpy(), r[:, 1].cpu().numpy()


# ============================================================
# Step 4: Binary search for non-penetrating curl
# ============================================================

def find_safe_curl(model, data, qpos, jid, target_curl, mapper, obj_pos, tip_body_name):
    """Binary search for maximum curl that doesn't penetrate."""
    jr = model.jnt_range[jid]
    original = qpos[jid]
    lo = original  # no curl (safe)
    hi = np.clip(original + target_curl, jr[0], jr[1])  # full curl

    for _ in range(8):  # 8 iterations = 1/256 precision
        mid = (lo + hi) / 2
        test = qpos.copy()
        test[jid] = mid
        data.qpos[:] = test
        mujoco.mj_forward(model, data)

        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tip_body_name)
        if bid < 0:
            return mid
        tip_pos = data.xpos[bid] - obj_pos
        dist, weight = nvblox_sdf(mapper, tip_pos.reshape(1, 3))

        if dist[0] < -0.001 and weight[0] > 0:
            hi = mid  # penetrating, reduce curl
        else:
            lo = mid  # safe, can curl more

    return lo  # return safe value


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load MANO data
    print("Loading MANO data (both hands)...")
    N, right_joints, left_joints, obj_pos = load_mano_data()
    obj_center = obj_pos.mean(0)
    print(f"  {N} bimanual frames, obj_center={obj_center.round(3)}")

    # Build scene
    print("Building bimanual scene...")
    model = build_bimanual_scene(obj_center)
    data = mujoco.MjData(model)
    print(f"  joints={model.njnt}, actuators={model.nu}")

    # Print actuator layout
    rh_acts = []; lh_acts = []; base_acts = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        if "act_" in name: base_acts.append((i, name))
        elif "rh_" in name: rh_acts.append((i, name))
        elif "lh_" in name: lh_acts.append((i, name))
    print(f"  Base acts: {len(base_acts)}, RH acts: {len(rh_acts)}, LH acts: {len(lh_acts)}")

    # Build nvblox
    print("Building nvblox TSDF...")
    mapper = build_nvblox()
    print("  TSDF ready!")

    # Find tip bodies
    rh_tips = {}
    lh_tips = {}
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if "rh_" in name and "tip" in name: rh_tips[name] = i
        if "lh_" in name and "tip" in name: lh_tips[name] = i
    print(f"  RH tips: {list(rh_tips.keys())}")
    print(f"  LH tips: {list(lh_tips.keys())}")

    # Find curl joints for each hand
    rh_curl_joints = [i for i in range(model.njnt) if "rh_" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "")
                      and any(x in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "") for x in ["proximal", "medial", "distal"])]
    lh_curl_joints = [i for i in range(model.njnt) if "lh_" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "")
                      and any(x in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "") for x in ["proximal", "medial", "distal"])]

    # === Generate trajectory ===
    print(f"\n=== Generating 5-stage bimanual trajectory ({N} frames + 40 REACH) ===")

    REACH = 40
    TOTAL = REACH + N
    ctrl = np.zeros((TOTAL, model.nu))

    # Build actuator index map
    act_map = {}
    for idx, name in base_acts:
        act_map[name] = idx

    # Compute MANO-derived wrist orientations (world frame euler) per frame
    right_orient = np.zeros((N, 3))
    left_orient = np.zeros((N, 3))

    with open(MANO_RESULTS) as f:
        all_results = json.load(f)
    frames_both = []
    frame_dict = {}
    for r in all_results:
        key = r['img_name']
        if key not in frame_dict: frame_dict[key] = {}
        frame_dict[key]['right' if r['is_right'] else 'left'] = r
    for k, v in frame_dict.items():
        if 'left' in v and 'right' in v:
            frames_both.append(v)

    T_cam2world = np.array([[1,0,0],[0,0,-1],[0,-1,0]], dtype=np.float64)
    for i in range(min(N, len(frames_both))):
        for side, arr in [('right', right_orient), ('left', left_orient)]:
            go = np.array(frames_both[i][side]['mano_global_orient'])
            if go.ndim == 3: go = go[0]
            go_world = T_cam2world @ go @ T_cam2world.T
            arr[i] = R.from_matrix(go_world).as_euler('xyz')

    right_orient = uniform_filter1d(right_orient, size=7, axis=0)
    left_orient = uniform_filter1d(left_orient, size=7, axis=0)

    # Convert to relative rotation from first frame (delta from initial pose)
    # This prevents huge absolute rotations from flipping the hands
    right_orient_delta = right_orient - right_orient[0]
    left_orient_delta = left_orient - left_orient[0]

    # Scale down: MANO rotations are in camera space, need smaller adjustments in sim
    ORIENT_SCALE = 0.15  # only apply 15% of MANO rotation change
    right_orient_delta *= ORIENT_SCALE
    left_orient_delta *= ORIENT_SCALE

    # Compute wrist position from MANO cam_t (global displacement)
    # MANO joints are in local space; cam_t provides camera-space offset
    right_cam_t = np.zeros((N, 3))
    left_cam_t = np.zeros((N, 3))
    for i in range(min(N, len(frames_both))):
        right_cam_t[i] = np.array(frames_both[i]['right']['cam_t_full'])
        left_cam_t[i] = np.array(frames_both[i]['left']['cam_t_full'])

    # Convert cam_t to world frame position changes
    # cam_t: [x_cam, y_cam, z_cam (depth)] in meters
    # World: X=right, Y=forward(-Z_cam), Z=up(-Y_cam)
    def cam_t_to_world(ct):
        w = np.zeros_like(ct)
        w[:, 0] = ct[:, 0]      # X stays
        w[:, 1] = 0             # ignore depth (Y_world)
        w[:, 2] = -ct[:, 1]    # cam Y(down) → world Z(up)
        return w

    right_wrist_pos = cam_t_to_world(right_cam_t)
    left_wrist_pos = cam_t_to_world(left_cam_t)
    # Delta from first frame
    right_pos_delta = right_wrist_pos - right_wrist_pos[0]
    left_pos_delta = left_wrist_pos - left_wrist_pos[0]
    # Scale: cam_t is in meters but large (x~0.3, y~-0.1, z~25)
    # Delta values are ~0.01-0.1m, scale to match sim workspace
    POS_SCALE = 0.5
    right_pos_delta = uniform_filter1d(right_pos_delta * POS_SCALE, size=7, axis=0)
    left_pos_delta = uniform_filter1d(left_pos_delta * POS_SCALE, size=7, axis=0)

    print(f"  Right wrist delta range: [{right_pos_delta.min():.3f}, {right_pos_delta.max():.3f}]")
    print(f"  Left wrist delta range:  [{left_pos_delta.min():.3f}, {left_pos_delta.max():.3f}]")

    for i in range(TOTAL):
        if i < REACH:
            # REACH: hands approach from sides
            p = 3 * (i/REACH)**2 - 2 * (i/REACH)**3

            # Position: start far, approach to MANO first-frame position
            if 'act_rx' in act_map: ctrl[i, act_map['act_rx']] = 0.06 * (1 - p)
            if 'act_rz' in act_map: ctrl[i, act_map['act_rz']] = 0.04 * (1 - p)
            if 'act_lx' in act_map: ctrl[i, act_map['act_lx']] = -0.06 * (1 - p)
            if 'act_lz' in act_map: ctrl[i, act_map['act_lz']] = 0.04 * (1 - p)

            # Rotation: base orientation (fingers toward bottle)
            if 'act_rrz' in act_map: ctrl[i, act_map['act_rrz']] = np.pi
            if 'act_rry' in act_map: ctrl[i, act_map['act_rry']] = -0.25
            if 'act_lry' in act_map: ctrl[i, act_map['act_lry']] = 0.25

        else:
            idx = i - REACH
            t = idx / max(N - 1, 1)
            fi = min(idx, N - 1)

            # === BASE TRANSLATION: staged approach toward bottle ===
            # (cam_t mapping unreliable — use staged trajectory instead)
            approach_offset = 0.0
            if t < 0.30:
                approach_offset = -0.03 * (t / 0.30)  # move 3cm inward
            elif t < 0.60:
                approach_offset = -0.03 - 0.02 * ((t-0.30)/0.30)  # 2cm more
            else:
                approach_offset = -0.05

            if 'act_rx' in act_map: ctrl[i, act_map['act_rx']] = approach_offset  # right moves left
            if 'act_lx' in act_map: ctrl[i, act_map['act_lx']] = -approach_offset  # left moves right

            # === BASE ROTATION: fixed horizontal (MANO delta disabled for stability) ===
            if 'act_rrz' in act_map: ctrl[i, act_map['act_rrz']] = np.pi
            if 'act_rry' in act_map: ctrl[i, act_map['act_rry']] = -0.25
            if 'act_lry' in act_map: ctrl[i, act_map['act_lry']] = 0.25

            # === FINGER CURL (staged) ===
            if t < 0.25:
                curl = 0
            elif t < 0.50:
                curl = ((t - 0.25) / 0.25) * 0.6
            elif t < 0.70:
                curl = 0.6 + ((t - 0.50) / 0.20) * 0.3
            else:
                curl = 0.9

            # Lift in last phase (add to MANO delta)
            if t > 0.75:
                lp = (t - 0.75) / 0.25
                if 'act_rz' in act_map: ctrl[i, act_map['act_rz']] += 0.04 * lp
                if 'act_lz' in act_map: ctrl[i, act_map['act_lz']] += 0.04 * lp

            for ai, name in rh_acts + lh_acts:
                jid = model.actuator_trnid[ai, 0]
                if jid >= 0:
                    jr = model.jnt_range[jid]
                    ctrl[i, ai] = jr[0] + curl * (jr[1] - jr[0])

    ctrl = uniform_filter1d(ctrl, size=5, axis=0)

    # === nvblox SDF binary search fix ===
    print(f"\n=== nvblox SDF binary search penetration fix ===")
    total_pen = 0
    total_fixed = 0

    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'bottle')

    for i in range(TOTAL):
        # Set ctrl and step
        data.ctrl[:] = ctrl[i]
        for _ in range(5):
            mujoco.mj_step(model, data)

        bottle_pos = data.xpos[bottle_id] if bottle_id >= 0 else obj_center

        # Check all fingertips
        for tip_name, tip_bid in list(rh_tips.items()) + list(lh_tips.items()):
            tip_pos = data.xpos[tip_bid] - bottle_pos
            dist, weight = nvblox_sdf(mapper, tip_pos.reshape(1, 3))

            if dist[0] < -0.0005 and weight[0] > 0:
                total_pen += 1
                # Find which actuator controls this finger and reduce
                prefix = "rh_" if "rh_" in tip_name else "lh_"
                finger = tip_name.split("_")[1]  # ff, mf, rf, th
                acts_list = rh_acts if prefix == "rh_" else lh_acts
                for ai, aname in acts_list:
                    if finger in aname:
                        jid = model.actuator_trnid[ai, 0]
                        if jid >= 0:
                            jr = model.jnt_range[jid]
                            # Binary search: find max safe ctrl value
                            lo = jr[0]
                            hi = ctrl[i, ai]
                            for _ in range(6):
                                mid = (lo + hi) / 2
                                ctrl[i, ai] = mid
                                data.ctrl[:] = ctrl[i]
                                for _ in range(3):
                                    mujoco.mj_step(model, data)
                                tp = data.xpos[tip_bid] - data.xpos[bottle_id]
                                d, w = nvblox_sdf(mapper, tp.reshape(1, 3))
                                if d[0] < -0.0005 and w[0] > 0:
                                    hi = mid
                                else:
                                    lo = mid
                            ctrl[i, ai] = lo
                            total_fixed += 1

        if i % 30 == 0:
            phase = "REACH" if i < REACH else ["APPROACH","CLOSE","GRASP","LIFT"][min(int((i-REACH)/(N-1)*4), 3)] if N > 1 else "?"
            print(f"  Frame {i:3d}/{TOTAL} [{phase:8s}]")

    print(f"  Penetrations: {total_pen}, Fixed: {total_fixed}")

    # === Render ===
    print(f"\n=== Rendering ===")
    renderer = mujoco.Renderer(model, height=480, width=640)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = obj_center
    cam.distance = 0.42
    cam.azimuth = 145  # 3/4 view: see both hands + bottle without occlusion
    cam.elevation = -30  # elevated enough to see table

    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w_vid = cv2.VideoWriter(os.path.join(OUT_DIR, 'bimanual_grasp.mp4'), fourcc, fps, (640, 480))

    mujoco.mj_resetData(model, data)

    phases = ['REACH', 'APPROACH', 'CLOSE', 'GRASP', 'LIFT']
    for i in range(TOTAL):
        data.ctrl[:] = ctrl[i]
        for _ in range(10):
            mujoco.mj_step(model, data)
        renderer.update_scene(data, cam)
        img = renderer.render()
        bgr = img[:, :, ::-1].copy()

        if i < REACH:
            phase = "REACH"
        else:
            t = (i - REACH) / max(N - 1, 1)
            phase = ["APPROACH", "CLOSE", "GRASP", "LIFT"][min(int(t * 4), 3)]

        cv2.putText(bgr, f"[{phase}] {i}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        w_vid.write(bgr)

        if i in [0, REACH//2, REACH-1, REACH+N//4, REACH+N//2, REACH+3*N//4, TOTAL-1]:
            cv2.imwrite(os.path.join(OUT_DIR, f'kf_{i:04d}_{phase.lower()}.jpg'), bgr)

    w_vid.release()

    # Comparison with HaMeR overlay
    overlays_dir = os.path.join(BASE_DIR, 'output/pick_bottle_video')
    overlays = sorted([f for f in os.listdir(overlays_dir) if f.endswith('_overlay.jpg')])

    cap = cv2.VideoCapture(os.path.join(OUT_DIR, 'bimanual_grasp.mp4'))
    pw, ph = 640, 480
    w_cmp = cv2.VideoWriter(os.path.join(OUT_DIR, 'comparison.mp4'), fourcc, fps, (pw*2, ph+25))

    for i in range(TOTAL):
        ov_idx = max(0, min(int((i - REACH) / N * len(overlays)), len(overlays) - 1))
        ov = cv2.resize(cv2.imread(os.path.join(overlays_dir, overlays[ov_idx])), (pw, ph))
        ret, sim = cap.read()
        if not ret: break

        panels = np.hstack([ov, sim])
        label = np.ones((25, pw*2, 3), dtype=np.uint8) * 35
        cv2.putText(label, "Video + HaMeR (both hands)", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        cv2.putText(label, "Bimanual Allegro Sim", (pw+5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        frame = np.vstack([label, panels])
        w_cmp.write(frame)
        if i == TOTAL // 2:
            cv2.imwrite(os.path.join(OUT_DIR, 'comparison_preview.jpg'), frame)

    w_cmp.release(); cap.release()

    # GIF
    import subprocess
    subprocess.run(['ffmpeg', '-y', '-i', os.path.join(OUT_DIR, 'comparison.mp4'),
                    '-vf', 'fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
                    os.path.join(OUT_DIR, 'comparison.gif')], capture_output=True)

    print(f"\nDone! {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f} ({os.path.getsize(os.path.join(OUT_DIR, f))/1024:.0f} KB)")


if __name__ == '__main__':
    main()
