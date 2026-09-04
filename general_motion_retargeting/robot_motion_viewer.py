import os
import time
import mujoco as mj
import mujoco.viewer as mjv
import imageio
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from loop_rate_limiters import RateLimiter
import numpy as np
from rich import print


def draw_frame(
    pos,
    mat,
    v,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = v.user_scn.geoms[v.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + pos_offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        if joint_name is not None:
            geom.label = joint_name  # 这里赋名字
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + pos_offset,
            to=pos + pos_offset + size * (mat @ fix)[:, i],
        )
        v.user_scn.ngeom += 1

class RobotMotionViewer:
    def __init__(self,
                robot_type,
                camera_follow=True,
                motion_fps=30,
                transparent_robot=0,
                camera_azimuth=-140,
                camera_elevation=-20,
                camera_distance=None,
                camera_lookat=(0.0, 0.0, 0.8),
                # video recording
                record_video=False,
                video_path=None,
                video_width=640,
                video_height=480,
                keyboard_callback=None,
                show_body_frames=False,
                show_joint_axes=False,
                show_sites=False,
                ):
        
        self.robot_type = robot_type
        self.xml_path = ROBOT_XML_DICT[robot_type]
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        self.camera_azimuth = camera_azimuth
        self.camera_elevation = camera_elevation
        self.camera_distance = self.viewer_cam_distance if camera_distance is None else camera_distance
        self.camera_lookat = np.asarray(camera_lookat, dtype=float)
        mj.mj_step(self.model, self.data)
        
        self.motion_fps = motion_fps
        self.speed_factor = 1.0
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.record_video = record_video
        self.show_body_frames = bool(show_body_frames)
        self.show_joint_axes = bool(show_joint_axes)
        self.show_sites = bool(show_sites)


        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False, 
            key_callback=keyboard_callback
            )      

        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot
        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = self.show_joint_axes
        self._set_camera(self.camera_lookat)

    def _draw_debug_overlay(self):
        """Draw body coordinate frames and MuJoCo sites in the user scene."""
        if not (self.show_body_frames or self.show_sites):
            return
        self.viewer.user_scn.ngeom = 0
        if self.show_body_frames:
            for body_id in range(1, self.model.nbody):
                if self.viewer.user_scn.ngeom + 3 >= self.viewer.user_scn.maxgeom:
                    break
                name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
                draw_frame(
                    self.data.xpos[body_id].copy(),
                    self.data.xmat[body_id].reshape(3, 3).copy(),
                    self.viewer,
                    0.055,
                    joint_name=None,
                )
        if self.show_sites:
            for site_id in range(self.model.nsite):
                if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom:
                    break
                name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_SITE, site_id) or f"site_{site_id}"
                geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
                color = [1.0, 0.25, 0.05, 1.0] if ("guard" in name or "contact" in name) else [0.1, 0.85, 1.0, 1.0]
                mj.mjv_initGeom(
                    geom,
                    type=mj.mjtGeom.mjGEOM_SPHERE,
                    size=[0.012, 0.012, 0.012],
                    pos=self.data.site_xpos[site_id],
                    mat=np.eye(3).flatten(),
                    rgba=color,
                )
                geom.label = ""
                self.viewer.user_scn.ngeom += 1
        
        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            self.video_path = video_path
            video_dir = os.path.dirname(self.video_path)
            
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.mp4_writer = imageio.get_writer(self.video_path, fps=self.motion_fps)
            print(f"Recording video to {self.video_path}")
            
            # Initialize renderer for video recording
            self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)

    def _set_camera(self, lookat):
        self.viewer.cam.lookat[:] = lookat
        self.viewer.cam.distance = self.camera_distance
        self.viewer.cam.elevation = self.camera_elevation
        self.viewer.cam.azimuth = self.camera_azimuth

    def _follow_camera_position(self, lookat):
        self.viewer.cam.lookat[:] = lookat
        
    def step(self, 
            # robot data
            root_pos, root_rot, dof_pos, 
            # human data
            human_motion_data=None, 
            show_human_body_name=False,
            # scale for human point visualization
            human_point_scale=0.1,
            # human pos offset add for visualization    
            human_pos_offset=np.array([0.0, 0.0, 0]),
            # rate limit
            rate_limit=True, 
            follow_camera=None,
            ):
        """
        by default visualize robot motion.
        also support visualize human motion by providing human_motion_data, to compare with robot motion.
        
        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.
        """
        
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot # quat need to be scalar first! for mujoco
        self.data.qpos[7:] = dof_pos
        
        mj.mj_forward(self.model, self.data)

        self._draw_debug_overlay()
        
        if follow_camera is None:
            follow_camera = self.camera_follow

        if follow_camera:
            # Track the robot's torso area rather than the base origin.  This
            # keeps the full humanoid in frame for motions whose root is near
            # the ground (as in the NE01 retargeted dataset).
            base_pos = self.data.xpos[self.model.body(self.robot_base).id]
            self._follow_camera_position(base_pos + np.array([0.0, 0.0, 0.35]))
        
        if human_motion_data is not None:
            # Clean custom geometry
            if not (self.show_body_frames or self.show_sites):
                self.viewer.user_scn.ngeom = 0
            # Draw the task targets for reference
            for human_body_name, (pos, rot) in human_motion_data.items():
                draw_frame(
                    pos,
                    R.from_quat(rot, scalar_first=True).as_matrix(),
                    self.viewer,
                    human_point_scale,
                    pos_offset=human_pos_offset,
                    joint_name=human_body_name if show_human_body_name else None
                    )

        self.viewer.sync()
        if rate_limit is True:
            self.rate_limiter.sleep()

    def set_speed(self, factor):
        """Set playback speed. factor=1.0 is real-time, 2.0 is 2x, 0.5 is half speed."""
        self.speed_factor = max(0.25, min(4.0, factor))
        self.rate_limiter = RateLimiter(frequency=self.motion_fps * self.speed_factor, warn=False)

        if self.record_video:
            # Use renderer for proper offscreen rendering
            self.renderer.update_scene(self.data, camera=self.viewer.cam)
            img = self.renderer.render()
            self.mp4_writer.append_data(img)
    
    def close(self):
        self.viewer.close()
        time.sleep(0.5)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")
