"""Visualize one GMR robot-motion .pkl file.

Purpose:
    Replay a robot-motion file produced by smplx_to_robot.py,
    smplx_to_robot_dataset.py, or another compatible exporter. This is useful
    for inspecting individual retarget results and filter false positives.

Typical usage:
    conda run --no-capture-output -n gmr python scripts/vis_robot_motion.py \
        --robot unitree_g1_24dof \
        --robot_motion_path data/retarget_data/g1_24dof/example.pkl

Keys:
    =  speed up
    -  slow down
"""

import argparse
import os

# Select the desktop GLFW backend before importing MuJoCo.  This avoids an
# accidental EGL/GL backend selection on machines that have an X display.
if os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "glfw"

from general_motion_retargeting import RobotMotionViewer, load_robot_motion

env = None
speed_factor = 1.0

def keyboard_callback(keycode):
    global speed_factor
    if chr(keycode) == '=':
        speed_factor = min(4.0, speed_factor * 2)
        env.set_speed(speed_factor)
        print(f"Speed: {speed_factor:.2f}x")
    elif chr(keycode) == '-':
        speed_factor = max(0.25, speed_factor / 2)
        env.set_speed(speed_factor)
        print(f"Speed: {speed_factor:.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot_motion_path", type=str, required=True)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default="videos/example.mp4")
    parser.add_argument("--no-camera-follow", action="store_true",
                        help="Keep the initial camera fixed instead of following the robot")
    parser.add_argument("--camera-lookat", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        default=(0.0, 0.0, 0.7),
                        help="Initial camera target in world coordinates")
    parser.add_argument("--camera-distance", type=float, default=None)
    parser.add_argument("--show_body_frames", action="store_true",
                        help="Draw XYZ frames on every robot body/link")
    parser.add_argument("--show_joint_axes", action="store_true",
                        help="Show MuJoCo joint axes")
    parser.add_argument("--show_sites", action="store_true",
                        help="Draw XML sites, including heel/forefoot/knee guards")
    args = parser.parse_args()

    if not os.path.exists(args.robot_motion_path):
        raise FileNotFoundError(f"Motion file {args.robot_motion_path} not found")

    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(args.robot_motion_path)

    env = RobotMotionViewer(robot_type=args.robot,
                            motion_fps=motion_fps,
                            camera_follow=not args.no_camera_follow,
                            camera_lookat=args.camera_lookat,
                            camera_distance=args.camera_distance,
                            record_video=args.record_video, video_path=args.video_path,
                            show_body_frames=args.show_body_frames,
                            show_joint_axes=args.show_joint_axes,
                            show_sites=args.show_sites,
                            keyboard_callback=keyboard_callback)

    frame_idx = 0
    while True:
        env.step(motion_root_pos[frame_idx],
                motion_root_rot[frame_idx],
                motion_dof_pos[frame_idx],
                rate_limit=True)
        frame_idx += 1
        if frame_idx >= len(motion_root_pos):
            frame_idx = 0
    env.close()
