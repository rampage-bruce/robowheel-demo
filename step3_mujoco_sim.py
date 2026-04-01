"""
Step 3+4: MuJoCo physics validation + video rendering.
Replays Franka trajectory (from MANO-derived grasp) in MuJoCo with a bottle object.
Generates MP4 showing the robot grasping.
"""
import os, json, copy
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRANKA_XML = os.path.join(BASE_DIR, 'mujoco_menagerie/franka_emika_panda/panda.xml')
TRAJ_PATH = os.path.join(BASE_DIR, 'output/grasp_full_traj.json')
GRASP_PATH = os.path.join(BASE_DIR, 'output/grasp_poses.json')
OUT_DIR = os.path.join(BASE_DIR, 'output/franka_sim')

def create_scene_xml():
    """Create MuJoCo XML with Franka + table + bottle."""
    return f"""
    <mujoco model="franka_grasp">
      <include file="{FRANKA_XML}"/>

      <option gravity="0 0 -9.81" timestep="0.002"/>

      <worldbody>
        <!-- Table -->
        <body name="table" pos="0.4 0 0.25">
          <geom type="box" size="0.3 0.4 0.25" rgba="0.35 0.25 0.18 1"
                friction="1 0.005 0.0001"/>
        </body>

        <!-- Bottle (cylinder on table) -->
        <body name="bottle" pos="0.4 0 0.57">
          <joint name="bottle_free" type="free"/>
          <geom name="bottle_body" type="cylinder" size="0.025 0.06"
                rgba="0.3 0.6 0.9 0.8" mass="0.3" friction="1 0.005 0.0001"/>
          <geom name="bottle_cap" type="cylinder" size="0.012 0.015"
                pos="0 0 0.075" rgba="0.2 0.4 0.7 0.9" mass="0.05"/>
        </body>
      </worldbody>
    </mujoco>
    """


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load trajectory
    with open(TRAJ_PATH) as f:
        traj_data = json.load(f)

    with open(GRASP_PATH) as f:
        grasp_data = json.load(f)

    print(f"Trajectory: {traj_data['total_steps']} steps")
    print(f"Phases: {list(traj_data['phases'].keys())}")
    print(f"Grasp source: MANO frame {grasp_data['source_frame']}")
    print(f"Grasp width: {grasp_data['franka_grasp_width_m']*100:.1f} cm")

    # Build full joint trajectory with gripper states
    approach_traj = traj_data['phases']['approach']
    grasp_traj = traj_data['phases']['grasp']
    lift_traj = traj_data['phases']['lift']

    gripper_width = grasp_data['franka_grasp_width_m']
    gripper_open = 0.04   # fully open (each finger)
    gripper_close = max(gripper_width / 2, 0.001)  # close to grasp width

    # Combine: 7-DoF arm + 2-DoF gripper (left finger, right finger)
    full_traj = []  # (joint7 + gripper2)
    gripper_states = []

    # Approach: gripper open
    for js in approach_traj:
        full_traj.append(js)
        gripper_states.append(gripper_open)

    # Grasp: gripper closing gradually
    for i, js in enumerate(grasp_traj):
        full_traj.append(js)
        t = i / max(len(grasp_traj) - 1, 1)
        g = gripper_open * (1 - t) + gripper_close * t
        gripper_states.append(g)

    # Lift: gripper closed
    for js in lift_traj:
        full_traj.append(js)
        gripper_states.append(gripper_close)

    print(f"Full trajectory: {len(full_traj)} steps")

    # Load MuJoCo model - need to chdir for mesh file resolution
    xml_str = create_scene_xml()
    orig_dir = os.getcwd()
    os.chdir(os.path.dirname(FRANKA_XML))
    model = mujoco.MjModel.from_xml_string(xml_str)
    os.chdir(orig_dir)
    data = mujoco.MjData(model)

    # Find joint and actuator indices
    # Franka has joints: joint1-7 + finger_joint1 + finger_joint2
    arm_joint_ids = []
    for i in range(7):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{i+1}')
        if jid >= 0:
            arm_joint_ids.append(jid)
    print(f"Arm joint IDs: {arm_joint_ids}")

    finger_joint1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'finger_joint1')
    finger_joint2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'finger_joint2')
    print(f"Finger joints: {finger_joint1}, {finger_joint2}")

    # Find actuator indices
    n_actuators = model.nu
    print(f"Actuators: {n_actuators}")
    for i in range(n_actuators):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  actuator {i}: {name}")

    # Get bottle body ID for tracking
    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'bottle')

    # Setup renderer
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.4, 0.0, 0.55]
    cam.distance = 1.2
    cam.azimuth = 135
    cam.elevation = -25

    # Video writer
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        os.path.join(OUT_DIR, 'franka_grasp.mp4'), fourcc, fps, (640, 480))

    # Also a front view
    cam_front = mujoco.MjvCamera()
    cam_front.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_front.lookat[:] = [0.4, 0.0, 0.55]
    cam_front.distance = 1.0
    cam_front.azimuth = 180
    cam_front.elevation = -20

    writer_front = cv2.VideoWriter(
        os.path.join(OUT_DIR, 'franka_grasp_front.mp4'), fourcc, fps, (640, 480))

    # Reset
    mujoco.mj_resetData(model, data)

    # Initial bottle position
    bottle_z_init = None

    # Simulate trajectory
    steps_per_frame = 5  # slow down playback
    frame_idx = 0

    for t_idx, (js, g_state) in enumerate(zip(full_traj, gripper_states)):
        # Set arm joint positions via control
        for i in range(min(7, n_actuators)):
            data.ctrl[i] = js[i]

        # Set gripper
        if n_actuators > 7:
            data.ctrl[7] = g_state  # finger1
        if n_actuators > 8:
            data.ctrl[8] = g_state  # finger2

        # Step physics multiple times per trajectory point
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)

        # Record bottle height
        bottle_z = data.xpos[bottle_id][2]
        if bottle_z_init is None:
            bottle_z_init = bottle_z

        # Render
        renderer.update_scene(data, cam)
        img = renderer.render()
        writer.write(img[:, :, ::-1])

        renderer.update_scene(data, cam_front)
        img_front = renderer.render()
        writer_front.write(img_front[:, :, ::-1])

        if t_idx % 50 == 0:
            phase = "approach" if t_idx < len(approach_traj) else \
                    "grasp" if t_idx < len(approach_traj) + len(grasp_traj) else "lift"
            print(f"  Step {t_idx}/{len(full_traj)}: {phase}, "
                  f"bottle_z={bottle_z:.3f}, gripper={g_state:.3f}")

        # Save key frames
        if t_idx in [0, len(approach_traj)-1,
                     len(approach_traj)+len(grasp_traj)-1,
                     len(full_traj)-1]:
            cv2.imwrite(os.path.join(OUT_DIR, f'keyframe_{t_idx:04d}.jpg'),
                        img[:, :, ::-1])

    writer.release()
    writer_front.release()

    # Validation result
    bottle_z_final = data.xpos[bottle_id][2]
    lifted = bottle_z_final > bottle_z_init + 0.03  # lifted at least 3cm

    result = {
        "source": "MANO_HaMeR → CuRobo → MuJoCo",
        "source_frame": grasp_data["source_frame"],
        "grasp_width_cm": round(grasp_data["franka_grasp_width_m"] * 100, 2),
        "bottle_z_initial": round(float(bottle_z_init), 4),
        "bottle_z_final": round(float(bottle_z_final), 4),
        "lift_height_cm": round(float(bottle_z_final - bottle_z_init) * 100, 2),
        "success": bool(lifted),
        "total_steps": len(full_traj),
    }

    with open(os.path.join(OUT_DIR, 'validation_result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Bottle initial z: {bottle_z_init:.4f}")
    print(f"Bottle final z:   {bottle_z_final:.4f}")
    print(f"Lift height:      {(bottle_z_final - bottle_z_init)*100:.1f} cm")
    print(f"SUCCESS: {lifted}")
    print(f"\nOutput: {OUT_DIR}/")
    for f_name in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f_name))
        print(f"  {f_name} ({size/1024:.0f} KB)")


if __name__ == '__main__':
    main()
