# 导入机器人运动学求解库 mink，用于构建层级 IK 与任务求解
import mink
# 导入 MuJoCo 物理仿真库，用于加载机器人模型与正运动学计算
import mujoco as mj
# 导入 numpy 数值计算库
import numpy as np
# 导入 json 库，用于读取 IK 配置文件
import json
# 从 scipy 空间变换模块导入旋转类与球面插值类，用于姿态处理
from scipy.spatial.transform import Rotation as R, Slerp
# 从当前包参数模块导入机器人 XML 路径字典与 IK 配置字典
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
# 导入足部支撑任务与离地间隙任务类
from .foot_support_task import FootClearanceTask, FootSupportTask
# 导入 rich 打印库，用于格式化终端输出
from rich import print


class GeneralMotionRetargeting:
    """
    通用运动重定向 (General Motion Retargeting, GMR) 主类
    功能：将人体动捕数据映射为人形机器人关节轨迹，包含两级 IK、足部接触约束、
         骨盆稳定、时序平滑、穿模修正等完整重定向管线
    """

    def __init__(
        self,
        src_human,                # 源人体数据标识（对应配置文件中的人体类型）
        tgt_robot,                # 目标机器人型号（如 ne01）
        actual_human_height=None, # 实际动捕演员身高，用于缩放骨骼
        solver="proxqp",         # QP 求解器，默认 proxqp
        damping=5e-1,            # 阻尼系数，防止 IK 奇异
        verbose=True,            # 是否打印详细初始化信息
        use_velocity_limit=False,# 是否启用关节速度限制
        velocity_limit=3*np.pi,  # 关节速度上限（弧度/秒）
        motion_fps=50.0,         # 输出动作帧率
        legacy_mode=False,       # 兼容旧模式开关（非层级 IK 模式）
    ):
        # ========== 1. 加载机器人 MuJoCo 模型 ==========
        # 根据机器人型号从字典中获取对应的 XML 模型文件路径
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        # 保存目标机器人型号
        self.tgt_robot = tgt_robot
        if verbose:
            print("Use robot model: ", self.xml_file)
        # 从 XML 文件加载 MuJoCo 模型对象
        self.model = mj.MjModel.from_xml_path(self.xml_file)

        # ========== 2. 打印并保存机器人自由度（DoF）信息 ==========
        if verbose:
            print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        # 字典：关节名称 -> 自由度索引
        self.robot_dof_names = {}
        # 遍历所有自由度
        for i in range(self.model.nv):
            # 通过自由度 ID 反查所属关节名称
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")

        # ========== 3. 打印并保存机器人刚体（body）信息 ==========
        if verbose:
            print("[GMR] Robot body names and their IDs:")
        # 字典：刚体名称 -> 刚体 ID
        self.robot_body_names = {}
        for i in range(self.model.nbody):
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")

        # ========== 4. 打印并保存机器人电机（执行器）信息 ==========
        if verbose:
            print("[GMR] Robot Motor (Actuator) names and their IDs:")
        # 字典：电机名称 -> 电机 ID
        self.robot_motor_names = {}
        for i in range(self.model.nu):
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # ========== 5. 加载 IK 重定向配置文件 ==========
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])

        # ========== 6. 根据实际人体身高计算整体缩放比例 ==========
        if actual_human_height is not None:
            # 缩放比 = 实际身高 / 配置假定身高
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0

        # 对分段骨骼缩放表整体乘以身高比例
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] *= ratio

        # ========== 7. 保存核心配置参数 ==========
        # 两级 IK 匹配表（table1 粗对齐，table2 精细对齐）
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        # 人体根骨骼名称（通常为 pelvis）
        self.human_root_name = ik_config["human_root_name"]
        # 机器人根连杆名称（通常为 base_link）
        self.robot_root_name = ik_config["robot_root_name"]
        # 是否启用两级 IK
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        # 人体分段骨骼缩放系数表
        self.human_scale_table = ik_config["human_scale_table"]
        # 地平面高度向量（Z 方向）
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])
        # 机器人根节点与人体根节点的位置偏移（坐标系对齐用）
        self.robot_root_to_human_root_offset = np.asarray(
            ik_config.get("robot_root_to_human_root_offset", [0.0, 0.0, 0.0]),
            dtype=float,
        )
        # 校验偏移向量维度
        if self.robot_root_to_human_root_offset.shape != (3,):
            raise ValueError("robot_root_to_human_root_offset must be a 3D vector")

        # 初始帧重定向迭代次数（第一帧需要更多迭代收敛）
        self.initial_frame_retarget_passes = max(
            1, int(ik_config.get("initial_frame_retarget_passes", 1))
        )
        # 重定向调用计数
        self.retarget_call_count = 0
        # 单帧 IK 最大迭代次数
        self.max_iter = 10
        # 动作输出帧率
        self.motion_fps = float(motion_fps)
        if self.motion_fps <= 0:
            raise ValueError("motion_fps must be positive")
        # 单帧时间步长 dt = 1/fps
        self.motion_dt = 1.0 / self.motion_fps
        # 旧模式兼容标志
        self.legacy_mode = bool(legacy_mode)

        # 上一帧、上两帧关节位置，用于时序平滑
        self._q_prev = None
        self._q_prev2 = None
        # 层级 IK 求解失败计数
        self.hierarchy_failures = 0

        # ========== 8. 足部接触检测相关变量 ==========
        # 左右脚历史位置列表，用于计算速度
        self._foot_history = {"left_foot": [], "right_foot": []}
        # 接触高度阈值：脚低于该高度视为可能接触
        self.foot_contact_height_threshold = float(ik_config.get("foot_contact_height_threshold", 0.035))
        # 接触速度阈值：脚水平速度低于该值视为稳定接触
        self.foot_contact_speed_threshold = float(ik_config.get("foot_contact_speed_threshold", 0.35))
        # 垂直接触速度阈值：排除跳跃下降/快速穿越地面的瞬时近地事件
        self.foot_contact_vertical_speed_threshold = float(
            ik_config.get("foot_contact_vertical_speed_threshold", 0.18)
        )
        # 进入接触所需连续帧数（防抖）
        self.foot_contact_enter_frames = int(ik_config.get("foot_contact_enter_frames", 2))
        # 退出接触所需连续帧数（防抖）
        self.foot_contact_exit_frames = int(ik_config.get("foot_contact_exit_frames", 3))
        # 左右脚接触状态布尔值
        self.foot_contact_state = {"left_foot": False, "right_foot": False}
        # 支撑脚锁定的 XY 平面坐标
        self._foot_lock_xy = {"left_foot": None, "right_foot": None}
        # 足部模式：AIR 腾空 / HEEL_CONTACT 脚跟接触 / TOE_CONTACT 脚尖接触 / FLAT_CONTACT 全掌接触
        self._foot_mode = {"left_foot": "AIR", "right_foot": "AIR"}
        # 接触过渡过程总帧数
        self.foot_contact_transition_frames = int(ik_config.get("foot_contact_transition_frames", 5))
        # 退出支撑时单独使用更长的释放窗口；只平滑脚目标，不继续锁定 root XY。
        self.foot_contact_release_frames = max(
            1, int(ik_config.get("foot_contact_release_frames", 8))
        )
        # 接触过渡混合系数 0~1，平滑切换支撑约束
        self._foot_contact_blend = {"left_foot": 0.0, "right_foot": 0.0}
        # 接触锚点世界坐标（踩地点，支撑全程固定）
        self._foot_contact_anchor = {"left_foot": None, "right_foot": None}
        # 当前接触枢轴模式（脚跟/脚尖/全掌）
        self._foot_contact_pivot_mode = {"left_foot": None, "right_foot": None}
        # 进入接触计数器
        self._foot_enter_count = {"left_foot": 0, "right_foot": 0}
        # 退出接触计数器
        self._foot_exit_count = {"left_foot": 0, "right_foot": 0}
        # 接触候选期的多帧锚点样本；达到该帧数后才正式锁定接触点
        self.foot_contact_anchor_frames = max(
            1, int(ik_config.get("foot_contact_anchor_frames", 3))
        )
        self._foot_pending_anchor_samples = {"left_foot": [], "right_foot": []}
        self._foot_pending_mode = {"left_foot": None, "right_foot": None}
        # 人体脚相对当前接触 episode 起点的累积 XY 位移参考
        self.foot_contact_anchor_drift_threshold = float(
            ik_config.get("foot_contact_anchor_drift_threshold", 0.05)
        )
        self._foot_source_anchor_xy = {"left_foot": None, "right_foot": None}
        self._foot_impact_contact = {"left_foot": False, "right_foot": False}
        # 新脚落地且另一只脚仍支撑时，短暂冻结根部 XY，避免双支撑切换拉动整机。
        self._root_xy_hold_frames = 0

        # 动态估计的人体地面高度 Z 值（运行中持续更新最小值）
        self._human_floor_z = np.inf
        # 足部翻滚角度限制（弧度）
        self.foot_rocking_limit = float(ik_config.get("foot_rocking_limit_rad", np.deg2rad(12.0)))

        # ========== 9. 躯干与骨盆稳定相关参数 ==========
        # 是否强制躯干保持直立
        self.upright_torso_orientation = bool(ik_config.get("upright_torso_orientation", False))
        # 是否启用骨盆姿态稳定
        self.stabilize_pelvis_orientation = bool(ik_config.get("stabilize_pelvis_orientation", False))
        # 骨盆横滚（roll）缩放系数
        self.pelvis_roll_scale = float(ik_config.get("pelvis_roll_scale", 0.0))
        # 骨盆俯仰（pitch）缩放系数
        self.pelvis_pitch_scale = float(ik_config.get("pelvis_pitch_scale", 0.35))
        # 骨盆 roll 低通滤波截止频率
        self.pelvis_roll_cutoff_hz = float(ik_config.get("pelvis_roll_cutoff_hz", 3.0))
        # 骨盆 roll 角度上限
        self.pelvis_roll_limit = float(ik_config.get("pelvis_roll_limit_rad", np.deg2rad(3.0)))
        # 躯干侧倾补偿混合系数
        self.torso_compensation_blend = float(ik_config.get("torso_compensation_blend", 0.8))
        # 躯干世界坐标系 roll 缩放系数
        self.torso_world_roll_scale = float(ik_config.get("torso_world_roll_scale", 0.15))

        # 单支撑骨盆侧移限制
        self.pelvis_lateral_single_limit = float(ik_config.get("pelvis_lateral_single_limit", 0.11))
        # 双支撑骨盆侧移限制
        self.pelvis_lateral_double_limit = float(ik_config.get("pelvis_lateral_double_limit", 0.06))
        # 骨盆侧移修正混合系数
        self.pelvis_lateral_blend = float(ik_config.get("pelvis_lateral_blend", 0.45))

        # 摆动脚最小离地间隙（防止穿模）
        self.swing_foot_clearance = float(ik_config.get("swing_foot_clearance", 0.002))
        # 滤波后的骨盆 roll 角度
        self._pelvis_roll_filtered = None

        # ========== 10. 髋部稳定相关参数 ==========
        # 是否启用髋部稳定
        self.hip_stabilization_enabled = bool(ik_config.get("hip_stabilization_enabled", False))
        # 髋部低通滤波截止频率
        self.hip_filter_cutoff_hz = float(ik_config.get("hip_filter_cutoff_hz", 6.0))
        # 支撑相髋关节 yaw 角度限制
        self.stance_hip_yaw_limit = float(ik_config.get("stance_hip_yaw_limit_rad", np.deg2rad(10.0)))
        # 支撑髋 yaw 限制混合系数
        self.stance_hip_yaw_blend = float(ik_config.get("stance_hip_yaw_blend", 0.65))
        # 双支撑 yaw 衰减系数
        self.double_support_yaw_blend = float(ik_config.get("double_support_yaw_blend", 0.15))
        # 髋部滤波后角度缓存
        self._hip_filtered = {}
        # 支撑相髋部 yaw 锚点
        self._hip_contact_anchor = {"left_foot": None, "right_foot": None}

        # ========== 11. 骨骼缩放与 IK 层级配置 ==========
        # 骨骼父子关系映射，用于链式缩放
        self.human_scale_parents = ik_config.get("human_scale_parents", {})
        # IK 任务优先级层级列表
        self.ik_priority_levels = ik_config.get("ik_priority_levels", [])
        # QP 求解器名称
        self.solver = solver
        # IK 阻尼系数
        self.damping = damping

        # 人体关节 -> table1 任务对象映射
        self.human_body_to_task1 = {}
        # 人体关节 -> table2 任务对象映射
        self.human_body_to_task2 = {}
        # table1 位置偏移字典
        self.pos_offsets1 = {}
        # table1 旋转偏移字典
        self.rot_offsets1 = {}
        # table2 位置偏移字典
        self.pos_offsets2 = {}
        # table2 旋转偏移字典
        self.rot_offsets2 = {}
        # table1 任务误差记录
        self.task_errors1 = {}
        # table2 任务误差记录
        self.task_errors2 = {}
        # 任务对象 -> 对应机器人连杆名称
        self.task_frame_names = {}
        # 任务对象 -> 权重元组 (位置权重, 旋转权重)
        self.task_costs = {}

        # ========== 12. IK 约束限制 ==========
        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        # 如果启用速度限制，构建速度约束并加入限制列表
        if use_velocity_limit:
            velocity_limits = self.build_actuated_joint_velocity_limits(velocity_limit)
            self.ik_limits.append(mink.VelocityLimit(self.model, velocity_limits))

        # ========== 13. 初始化各子模块 ==========
        self.setup_retarget_configuration()   # 初始化重定向配置与两级任务
        self.setup_foot_support()             # 初始化足部支撑任务
        self.setup_hip_stabilization()        # 初始化髋部稳定模块

        # 全局地面偏移量
        self.ground_offset = 0.0

    def build_actuated_joint_velocity_limits(self, max_velocity):
        """
        为所有驱动关节构建统一的速度上限字典
        :param max_velocity: 最大角速度（弧度/秒）
        :return: 关节名称 -> 速度上限 的字典
        """
        velocity_limits = {}
        # 遍历所有电机
        for actuator_id in range(self.model.nu):
            # 获取电机驱动的关节 ID
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue
            # 跳过浮动基（自由关节），不限制整体速度
            joint_type = self.model.jnt_type[joint_id]
            if joint_type == mj.mjtJoint.mjJNT_FREE:
                continue
            # 获取关节名称
            joint_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, joint_id)
            velocity_limits[joint_name] = max_velocity
        return velocity_limits

    def setup_retarget_configuration(self):
        """
        初始化重定向运动学配置，构建 table1 / table2 两级 FrameTask 任务
        每个任务对应一个机器人连杆，跟踪人体对应关节的位姿
        """
        # 创建 mink 运动学配置对象，绑定机器人模型
        self.configuration = mink.Configuration(self.model)

        self.tasks1 = []  # table1 任务列表（粗对齐）
        self.tasks2 = []  # table2 任务列表（精细对齐）

        # ========== 构建 table1 任务 ==========
        for frame_name, entry in self.ik_match_table1.items():
            # entry 格式：[人体关节名, 位置权重, 旋转权重, 位置偏移, 旋转偏移]
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            # 保存位置偏移（减去地面高度偏移）
            self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
            # 保存旋转偏移（四元数转 Rotation 对象）
            self.rot_offsets1[body_name] = R.from_quat(rot_offset, scalar_first=True)

            # 权重不全为 0 则创建 FrameTask
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,      # 机器人连杆名称
                    frame_type="body",          # 帧类型为 body
                    position_cost=pos_weight,   # 位置跟踪权重
                    orientation_cost=rot_weight,# 姿态跟踪权重
                    lm_damping=1,               # Levenberg-Marquardt 阻尼
                )
                self.tasks1.append(task)
                # 记录任务 -> 连杆名映射
                self.task_frame_names[task] = frame_name
                # 记录任务权重
                self.task_costs[task] = (pos_weight, rot_weight)
                # 记录人体关节 -> 任务映射
                self.human_body_to_task1[body_name] = task
                # 初始化误差列表
                self.task_errors1[task] = []

        # ========== 构建 table2 任务（逻辑与 table1 完全一致，独立偏移） ==========
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
            self.rot_offsets2[body_name] = R.from_quat(rot_offset, scalar_first=True)

            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.tasks2.append(task)
                self.task_frame_names[task] = frame_name
                self.task_costs[task] = (pos_weight, rot_weight)
                self.human_body_to_task2[body_name] = task
                self.task_errors2[task] = []

    def setup_foot_support(self):
        """
        初始化足部支撑相关任务：
        - FootSupportTask：支撑相贴地约束
        - FootClearanceTask：摆动相离地间隙约束
        - 姿态保持任务：摆动脚姿态锁定
        - 足部碰撞几何组、局部枢轴向量
        """
        self.foot_support_tasks = {}          # 支撑任务字典
        self.foot_clearance_tasks = {}        # 离地间隙任务字典
        self.foot_orientation_hold_tasks = {} # 姿态保持任务字典
        self.foot_support_frame_heights = {}  # 支撑帧高度
        self.foot_pivot_local = {}            # 局部枢轴向量（相对于 toe_link）
        self.foot_geom_groups = {}            # 足部碰撞几何组

        # 仅 NE01 机器人启用足部专用逻辑
        if self.tgt_robot != "ne01":
            return

        # 执行一次正运动学，获取初始几何位置
        mj.mj_forward(self.model, self.configuration.data)

        for side in ("left", "right"):
            # 足底四个碰撞几何体名称（后左、后右、前左、前右）
            geom_names = [
                f"{side}_foot_rear_left_collision",
                f"{side}_foot_rear_right_collision",
                f"{side}_foot_front_left_collision",
                f"{side}_foot_front_right_collision",
            ]
            foot_name = f"{side}_foot"

            # 创建支撑任务：保证足底贴合地面
            task = FootSupportTask(
                self.model,
                geom_names,
                ground_height=float(self.ground[2]),
                cost=200.0,
            )
            self.foot_support_tasks[foot_name] = task

            # 创建离地间隙任务：保证摆动脚高于地面+安全间隙
            self.foot_clearance_tasks[foot_name] = FootClearanceTask(
                self.model,
                geom_names,
                clearance_height=float(self.ground[2]) + self.swing_foot_clearance,
                cost=300.0,
            )

            # 创建足部姿态保持任务：只约束旋转，不约束位置
            self.foot_orientation_hold_tasks[foot_name] = mink.FrameTask(
                frame_name=f"{side}_toe_link",
                frame_type="body",
                position_cost=0.0,
                orientation_cost=300.0,
                lm_damping=1.0,
            )

            # 获取四个几何体的 ID
            geom_ids = np.asarray([self.model.geom(name).id for name in geom_names])
            # 分组：前两个后跟，后两个前掌，四个全掌
            self.foot_geom_groups[foot_name] = {
                "HEEL_CONTACT": geom_ids[:2],
                "TOE_CONTACT": geom_ids[2:],
                "FLAT_CONTACT": geom_ids,
            }

            # 获取脚趾连杆 ID 及其在父节点下的位置
            toe_body_id = self.model.body(f"{side}_toe_link").id
            toe_in_parent = self.model.body_pos[toe_body_id]

            # 计算每个碰撞几何的足底最低点位置（几何中心 - 半径）
            sole_points = self.model.geom_pos[geom_ids].copy()
            sole_points[:, 2] -= self.model.geom_size[geom_ids, 0]

            # 计算三种接触模式下的局部枢轴点（相对于 toe_link 坐标系）
            self.foot_pivot_local[foot_name] = {
                "HEEL_CONTACT": np.mean(sole_points[:2], axis=0) - toe_in_parent,
                "TOE_CONTACT": np.mean(sole_points[2:], axis=0) - toe_in_parent,
                "FLAT_CONTACT": np.mean(sole_points, axis=0) - toe_in_parent,
            }

            # 计算脚趾连杆到足底的高度差
            toe_z = self.configuration.data.xpos[toe_body_id, 2]
            sole_z = np.min(task.compute_error(self.configuration) + float(self.ground[2]))
            self.foot_support_frame_heights[foot_name] = float(toe_z - sole_z)

    def setup_hip_stabilization(self):
        """
        初始化髋部稳定模块，记录髋部与腰部关节在 qpos 中的索引
        """
        self.hip_qpos_indices = {}
        # 腰部偏航关节索引
        self.waist_yaw_qpos_index = int(
            self.model.jnt_qposadr[self.model.joint("WAIST_YAW_JOINT").id]
        )
        # 躯干横滚关节索引
        self.torso_roll_qpos_index = int(
            self.model.jnt_qposadr[self.model.joint("TORSO_ROLL_JOINT").id]
        )

        if not self.hip_stabilization_enabled:
            return

        # 记录左右髋 roll、yaw 关节的 qpos 索引
        for side in ("L", "R"):
            for axis in ("ROLL", "YAW"):
                name = f"HIP_{axis}_{side}_JOINT"
                joint_id = self.model.joint(name).id
                self.hip_qpos_indices[(side.lower(), axis.lower())] = int(
                    self.model.jnt_qposadr[joint_id]
                )

    def update_targets(self, human_data, offset_to_ground=False):
        """
        【核心入口】根据单帧人体数据，更新所有 IK 任务的目标位姿
        流程：骨骼缩放 → 两级偏移 → 地面对齐 → 接触检测 → 骨盆修正 → 设置任务目标
        :param human_data: 人体动捕数据，键为关节名，值为 [位置, 四元数]
        :param offset_to_ground: 是否强制对齐地面
        """
        # 转换为 numpy 数组格式
        human_data = self.to_numpy(human_data)
        # 对人体骨骼进行分段链式缩放，适配机器人尺寸
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)

        # 保存未加偏移的基准人体数据（用于 table2 独立计算，避免偏移叠加）
        base_human_data = {k: [v[0].copy(), v[1].copy()] for k, v in human_data.items()}

        # ========== 处理 table1 目标 ==========
        # 叠加 table1 专属局部偏移
        human_data = self.offset_human_data(base_human_data, self.pos_offsets1, self.rot_offsets1)
        # 应用机器人根节点与人体根节点的偏移
        human_data = self.apply_robot_root_to_human_root_offset(human_data)
        # 应用全局地面偏移
        human_data = self.apply_ground_offset(human_data)

        # 可选：强制整体对齐地面
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)

        # 动态更新人体地面估计值（取所有足部关键点最低高度）
        self._update_human_floor(human_data)
        # 计算人体地面与仿真地面的高度差
        ground_shift = self._human_floor_z - float(self.ground[2])
        # 整体下移人体，使脚底贴合仿真地面
        human_data = self._shift_human_height(human_data, ground_shift)

        # 更新足部接触状态机
        self._update_foot_contact(human_data)
        # 计算骨盆侧移修正量，限制骨盆横向摆动
        lateral_shift = self._compute_pelvis_lateral_shift(human_data)
        # 水平面平移人体数据
        human_data = self._shift_human_xy(human_data, lateral_shift)

        # 保存缩放后的人体数据
        self.scaled_human_data = human_data

        # 设置 table1 所有任务的目标位姿
        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                # 对骨盆姿态进行缩放稳定处理
                rot = self._project_ne01_pelvis_orientation(task, body_name, rot)
                # 对躯干姿态进行直立约束
                rot = self._project_ne01_torso_orientation(task, body_name, rot)
                # 对支撑脚目标进行接触锚点投影
                pos, rot = self._project_contact_foot_target(body_name, pos, rot)
                # 设置任务目标 SE3 位姿
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

        # ========== 处理 table2 目标（独立偏移，互不污染） ==========
        if self.use_ik_match_table2:
            # 从基准数据重新叠加 table2 专属偏移，不复用 table1 结果
            table2_data = self.offset_human_data(base_human_data, self.pos_offsets2, self.rot_offsets2)
            table2_data = self.apply_robot_root_to_human_root_offset(table2_data)
            table2_data = self.apply_ground_offset(table2_data)
            table2_data = self._shift_human_height(table2_data, ground_shift)
            table2_data = self._shift_human_xy(table2_data, lateral_shift)

            # 设置 table2 所有任务目标
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = table2_data[body_name]
                rot = self._project_ne01_pelvis_orientation(task, body_name, rot)
                rot = self._project_ne01_torso_orientation(task, body_name, rot)
                pos, rot = self._project_contact_foot_target(body_name, pos, rot)
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

    def retarget(self, human_data, offset_to_ground=False):
        """
        【主函数】单帧运动重定向入口
        :return: 求解后的机器人关节位置 qpos
        """
        # 第一步：更新所有 IK 目标
        self.update_targets(human_data, offset_to_ground)

        # 确定本帧迭代次数：第一帧多轮迭代收敛，后续每帧 1 次
        retarget_passes = (
            self.initial_frame_retarget_passes
            if self.retarget_call_count == 0
            else 1
        )

        for _ in range(retarget_passes):
            if self.legacy_mode:
                # 旧模式：分别迭代求解 table1、table2
                if self.use_ik_match_table1:
                    self._solve_task_group(self.tasks1, self.error1)
                if self.use_ik_match_table2:
                    self._solve_task_group(self.tasks2, self.error2)
            else:
                # 新模式：字典序层级 IK 求解（优先级严格分层）
                self.solve_lexicographic_ik()

        # 调用计数 +1
        self.retarget_call_count += 1
        if self._root_xy_hold_frames > 0:
            self._root_xy_hold_frames -= 1

        # ========== 后处理修正模块 ==========
        self._stabilize_pelvis_qpos()      # 骨盆姿态稳定滤波
        self._stabilize_hips()             # 髋部角度平滑与限制
        self._correct_swing_foot_clearance()# 摆动/落地过渡脚离地间隙修正（防穿模）
        self._correct_support_height()     # 支撑相高度闭式修正（整体升降）
        self._correct_contact_xy()         # 支撑相 XY 闭式修正（整体平移）

        # 保存历史关节位置，用于下一帧时序正则
        self._q_prev2 = self._q_prev
        self._q_prev = self.configuration.data.qpos.copy()

        # 返回当前关节位置
        return self.configuration.data.qpos.copy()

    def solve_lexicographic_ik(self):
        """
        字典序（层级优先级）IK 求解
        核心思想：高优先级任务先求解，求解完成后冻结为等式约束，
                 低优先级任务只能在高优先级的零空间内优化，
                 保证高优先级任务不会被低优先级破坏
        """
        # 构建按优先级排序的任务层级列表
        levels = self._build_priority_levels()
        # 约束列表，每完成一层就将该层任务作为约束加入
        constraints = []
        if self._root_xy_hold_frames > 0:
            # 双支撑切换的短窗口内锁住浮动基 XY，避免新脚目标把整机横向/纵向
            # 拉开；腿部关节仍可在零空间内完成收脚和落地。
            constraints.append(mink.DofFreezingTask(self.model, [0, 1]))
        solved_any = False

        for tasks in levels:
            if not tasks:
                continue
            try:
                # 求解当前层级 IK，得到关节速度
                velocity = mink.solve_ik(
                    self.configuration,
                    tasks,
                    self.motion_dt,
                    self.solver,
                    self.damping,
                    limits=self.ik_limits,
                    constraints=constraints,
                )
                # 将速度积分到位姿上，更新关节角度
                self.configuration.integrate_inplace(velocity, self.motion_dt)
                # 将本层任务冻结为约束，加入约束列表供下层使用
                constraints.extend(
                    self._freeze_task(task)
                    for task in tasks
                    if not isinstance(task, FootSupportTask)
                )
                solved_any = True
            except Exception:
                # 求解失败则跳过本层，保留上层有效解，不崩溃
                self.hierarchy_failures += 1
                continue

        # ========== 最低优先级：时序平滑正则项 ==========
        if self._q_prev is not None:
            try:
                # 创建姿态任务，目标为上一帧关节角度
                posture = mink.PostureTask(self.model, cost=1e-3, lm_damping=1.0)
                posture.set_target(self._q_prev)
                # 在已有约束下求解，仅在冗余自由度内平滑
                velocity = mink.solve_ik(
                    self.configuration,
                    [posture],
                    self.motion_dt,
                    self.solver,
                    self.damping,
                    limits=self.ik_limits,
                    constraints=constraints,
                )
                self.configuration.integrate_inplace(velocity, self.motion_dt)
            except Exception:
                # 平滑失败静默跳过，不影响主解
                pass

        # 如果所有层级都失败，回退到 table1 普通求解保底
        if not solved_any and self.use_ik_match_table1:
            self._solve_task_group(self.tasks1, self.error1)

    def _freeze_task(self, task):
        """
        将已求解的任务冻结为零空间约束（增益置 0，仅保留等式约束）
        作用：下层任务不能改变上层任务的求解结果
        """
        # 支撑任务有专门的冻结方法
        if isinstance(task, FootSupportTask):
            return task.frozen_copy()

        pos_cost, rot_cost = self.task_costs[task]
        # 创建同构型任务，增益设为 0，仅维持当前残差不变
        frozen = mink.FrameTask(
            frame_name=self.task_frame_names[task],
            frame_type="body",
            position_cost=task.cost[:3],
            orientation_cost=task.cost[3:],
            gain=0.0,       # 零增益 = 不主动减小误差，仅保持
            lm_damping=0.0,
        )
        # 目标位姿与原任务一致
        frozen.set_target(task.transform_target_to_world)
        return frozen

    def _build_priority_levels(self):
        """
        根据配置的优先级层级列表，将任务分组到不同优先级
        未在配置中的任务统一放到最后一层
        """
        # 按连杆去重，同一连杆只保留权重最高的任务
        primary_by_frame = {}
        for task in self.tasks1 + self.tasks2:
            frame_name = self.task_frame_names[task]
            current = primary_by_frame.get(frame_name)
            # 位置权重大的保留
            if current is None or self.task_costs[task][0] > self.task_costs[current][0]:
                primary_by_frame[frame_name] = task

        all_tasks = list(primary_by_frame.values())

        # 足底正式支撑约束必须高于躯干、髋部和手臂任务。
        # 过渡期由脚目标 blend 处理，只有锚点已建立且 blend 完成的脚才进入该层。
        support_level = [
            task
            for name, task in self.foot_support_tasks.items()
            if self.foot_contact_state.get(name, False)
            and self._foot_contact_blend.get(name, 0.0) >= 1.0
            and self._foot_contact_anchor.get(name) is not None
        ]

        # 如果没有配置优先级，直接分为 table1、table2 两层
        if not self.ik_priority_levels:
            return ([support_level] if support_level else []) + [self.tasks1, self.tasks2]

        remaining = set(all_tasks)
        levels = []

        # 按配置顺序构建层级
        for frame_names in self.ik_priority_levels:
            names = set(frame_names)
            level = [
                task for task in all_tasks
                if task in remaining and self.task_frame_names[task] in names
            ]
            if level:
                levels.append(level)
                remaining.difference_update(level)

        # 剩余未分配任务放到最后一层
        if remaining:
            levels.append([task for task in all_tasks if task in remaining])

        return ([support_level] if support_level else []) + levels

    def _update_foot_contact(self, human_data):
        """
        足部接触状态机更新
        判据：高度低于阈值 + 水平速度/垂直速度均低于阈值 → 接触候选
        连续满足进入帧数后进入候选期，再用多帧中位数确定正式锚点。
        未形成锚点就退出的短片段会被丢弃，避免跳跃或噪声触发支撑修正。
        """
        for name in self._foot_history:
            if name not in human_data:
                continue
            pos = np.asarray(human_data[name][0], dtype=float)
            hist = self._foot_history[name]
            hist.append(pos.copy())
            # 只保留最近 4 帧
            if len(hist) > 4:
                del hist[:-4]

            # 计算水平速度与有符号垂直速度（+Z 为向上）
            speed = 0.0 if len(hist) < 2 else np.linalg.norm((hist[-1] - hist[-2])[:2]) / self.motion_dt
            vertical_speed = 0.0 if len(hist) < 2 else abs(float(hist[-1][2] - hist[-2][2])) / self.motion_dt

            side = name.split("_", 1)[0]
            # 收集足跟、足尖等所有足部关键点，取最低高度
            foot_points = [
                np.asarray(human_data[key][0], dtype=float)
                for key in (f"{side}_heel", f"{side}_big_toe", f"{side}_small_toe")
                if key in human_data
            ]
            support_z = min([p[2] for p in foot_points], default=pos[2])
            source_foot_xy = np.mean([p[:2] for p in foot_points], axis=0) if foot_points else pos[:2].copy()
            source_drift = 0.0
            if self.foot_contact_state[name] and self._foot_source_anchor_xy[name] is not None:
                drift_mode = self._foot_pending_mode[name] or self._foot_mode[name]
                drift_point = self._select_source_contact_point(name, drift_mode, human_data)
                source_drift = float(
                    np.linalg.norm(drift_point[:2] - self._foot_source_anchor_xy[name])
                )

            # 是否满足接触候选条件
            near_ground = support_z <= float(self.ground[2]) + self.foot_contact_height_threshold
            candidate = (
                near_ground
                and speed <= self.foot_contact_speed_threshold
                and vertical_speed <= self.foot_contact_vertical_speed_threshold
                and not (
                    self.foot_contact_state[name]
                    and source_drift > self.foot_contact_anchor_drift_threshold
                )
            )
            if candidate:
                self._foot_enter_count[name] += 1
                self._foot_exit_count[name] = 0
                # 连续满足帧数达标 → 进入接触状态
                if self._foot_enter_count[name] >= self.foot_contact_enter_frames:
                    entering_contact = not self.foot_contact_state[name]
                    self.foot_contact_state[name] = True
                    if entering_contact:
                        # 只有双脚接近地面/已有支撑时才冻结根部 XY。普通跑步的
                        # 单脚落地仍允许根部推进，跳跃落地则避免另一条腿被连带拉动。
                        other_name = "right_foot" if name == "left_foot" else "left_foot"
                        other_near_ground = (
                            other_name in human_data
                            and float(human_data[other_name][0][2])
                            <= float(self.ground[2]) + self.foot_contact_height_threshold
                        )
                        other_support = any(
                            other != name
                            and self.foot_contact_state.get(other, False)
                            for other in self.foot_contact_state
                        )
                        if other_near_ground or other_support:
                            self._root_xy_hold_frames = max(
                                self._root_xy_hold_frames,
                                self.foot_contact_transition_frames + 2,
                            )
                        self._foot_impact_contact[name] = bool(
                            other_near_ground or other_support
                        )
                        # 进入接触候选期：先分类，但暂不使用单帧位置建立锚点
                        mode = self._classify_foot_contact(name, human_data)
                        # 如果人体没有脚跟脚趾关键点，用机器人几何判断
                        if mode == "FLAT_CONTACT" and not self._has_human_heel_toe(name, human_data):
                            mode = self._classify_robot_foot_contact(name)
                        # 释放期内重新接触必须开启新 episode。若保留旧 anchor，
                        # 新落脚会被拉回上一次世界接触位置。
                        self._foot_contact_anchor[name] = None
                        self._foot_contact_pivot_mode[name] = None
                        self._foot_lock_xy[name] = None
                        self._foot_pending_anchor_samples[name] = []
                        self._foot_pending_mode[name] = mode
                        self._foot_mode[name] = mode
                        self._foot_source_anchor_xy[name] = self._select_source_contact_point(
                            name, mode, human_data
                        )[:2].copy()
                        # 立即以当前帧的机器人足底几何建立临时锚点。
                        # 若等待多帧后才生成 anchor，接触过渡期间脚会先跟随人体
                        # 目标移动，随后突然切换到锚点，造成落地跳变和两腿前后分开。
                        geom_ids = self.foot_geom_groups[name][mode]
                        if other_near_ground or other_support:
                            # 双脚落地/跳跃冲击：以机器人当前足底为锚点，避免
                            # 人体脚踝关键点与 NE01 足底几何差异造成腿部拉开。
                            provisional_anchor = np.mean(
                                self.configuration.data.geom_xpos[geom_ids], axis=0
                            )
                        else:
                            # 普通单脚步态保留源运动的接触位置，避免改变正常
                            # 快走/快跑的根部推进轨迹。
                            provisional_anchor = self._select_source_contact_point(
                                name, mode, human_data
                            )
                        self._start_foot_contact(
                            name, mode, anchor=provisional_anchor
                        )
                    if self._foot_contact_anchor[name] is None:
                        # 使用当前无旧锚点约束的机器人足底几何建立新 episode。
                        # 直接使用人体 heel/toe 的绝对坐标会把人体关键点与
                        # NE01 足底长度差异错误地固化成世界锚点。
                        mode = self._foot_pending_mode[name] or self._foot_mode[name]
                        geom_ids = self.foot_geom_groups[name][mode]
                        sample = np.mean(
                            self.configuration.data.geom_xpos[geom_ids], axis=0
                        )
                        self._foot_pending_anchor_samples[name].append(sample.copy())
                        if len(self._foot_pending_anchor_samples[name]) >= self.foot_contact_anchor_frames:
                            anchor = np.median(
                                np.asarray(self._foot_pending_anchor_samples[name]), axis=0
                            )
                            self._start_foot_contact(name, mode, anchor=anchor)
            else:
                self._foot_enter_count[name] = 0
                self._foot_exit_count[name] += 1
                # 水平或垂直速度超阈值持续数帧后退出支撑，
                # 但保留 anchor/mode 供后续 blend 平滑释放。
                if self._foot_exit_count[name] >= self.foot_contact_exit_frames:
                    self.foot_contact_state[name] = False
                    self._foot_pending_anchor_samples[name] = []
                    self._foot_pending_mode[name] = None

            # 更新接触过渡混合系数（平滑切入/切出）
            if self.foot_contact_state[name]:
                enter_step = 1.0 / max(self.foot_contact_transition_frames, 1)
                self._foot_contact_blend[name] = min(1.0, self._foot_contact_blend[name] + enter_step)
            else:
                release_step = 1.0 / self.foot_contact_release_frames
                self._foot_contact_blend[name] = max(0.0, self._foot_contact_blend[name] - release_step)
            # 只在退出渐变完成后清除锚点，避免接触姿态单帧跳回人体原始姿态。
            if not self.foot_contact_state[name] and self._foot_contact_blend[name] <= 1e-9:
                self._foot_lock_xy[name] = None
                self._foot_mode[name] = "AIR"
                self._foot_contact_anchor[name] = None
                self._foot_contact_pivot_mode[name] = None
                self._foot_pending_anchor_samples[name] = []
                self._foot_pending_mode[name] = None
                self._foot_source_anchor_xy[name] = None
                self._foot_impact_contact[name] = False

    def _update_human_floor(self, human_data):
        """
        动态估计人体地面高度：取所有足部关键点的最低 Z 值
        全程只降不升，适应不同动捕数据的地面基准
        """
        support_heights = []
        for side in ("left", "right"):
            for point in ("heel", "big_toe", "small_toe"):
                key = f"{side}_{point}"
                if key in human_data:
                    support_heights.append(float(human_data[key][0][2]))

        # 如果没有细分关键点，回退到整脚位置
        if not support_heights:
            support_heights = [
                float(human_data[name][0][2])
                for name in ("left_foot", "right_foot")
                if name in human_data
            ]

        if support_heights:
            self._human_floor_z = min(self._human_floor_z, min(support_heights))

    @staticmethod
    def _shift_human_height(human_data, shift):
        """整体上下平移人体所有关节的 Z 坐标"""
        for pos, _ in human_data.values():
            pos[2] -= shift
        return human_data

    @staticmethod
    def _shift_human_xy(human_data, shift):
        """整体水平平移人体所有关节的 XY 坐标"""
        if np.allclose(shift, 0.0):
            return human_data
        for pos, _ in human_data.values():
            pos[:2] += shift
        return human_data

    def _compute_pelvis_lateral_shift(self, human_data):
        """
        计算骨盆侧向位移修正量，限制骨盆横向摆动幅度
        避免动捕中骨盆过度侧移导致机器人重心超出支撑多边形
        """
        if not self.stabilize_pelvis_orientation:
            return np.zeros(2)

        # 收集当前所有支撑脚的锁定 XY 坐标
        active_locks = [
            self._foot_lock_xy[name]
            for name, active in self.foot_contact_state.items()
            if active and self._foot_lock_xy[name] is not None
        ]
        if not active_locks:
            return np.zeros(2)

        # 获取骨盆位置与朝向
        pelvis_pos, pelvis_quat = human_data[self.human_root_name]
        yaw = R.from_quat(pelvis_quat, scalar_first=True).as_euler("zyx")[0]
        # 骨盆侧向单位向量
        lateral_axis = np.array([-np.sin(yaw), np.cos(yaw)])

        # 支撑中心
        support_center = np.mean(active_locks, axis=0)
        # 骨盆在侧向方向上偏离支撑中心的距离
        lateral_offset = float(np.dot(pelvis_pos[:2] - support_center, lateral_axis))

        # 根据单/双支撑选择限制阈值
        limit = (
            self.pelvis_lateral_double_limit
            if len(active_locks) > 1
            else self.pelvis_lateral_single_limit
        )
        # 裁剪到限制范围内
        bounded = float(np.clip(lateral_offset, -limit, limit))
        # 返回需要修正的位移向量
        return self.pelvis_lateral_blend * (bounded - lateral_offset) * lateral_axis

    @staticmethod
    def _select_source_contact_point(name, mode, human_data):
        """Return the source-side contact point used to seed a new episode."""
        side = name.split("_", 1)[0]
        if mode == "HEEL_CONTACT" and f"{side}_heel" in human_data:
            return np.asarray(human_data[f"{side}_heel"][0], dtype=float).copy()
        if mode == "TOE_CONTACT":
            toes = [
                np.asarray(human_data[key][0], dtype=float)
                for key in (f"{side}_big_toe", f"{side}_small_toe")
                if key in human_data
            ]
            if toes:
                return np.mean(toes, axis=0)
        flat_points = [
            np.asarray(human_data[key][0], dtype=float)
            for key in (
                f"{side}_heel",
                f"{side}_big_toe",
                f"{side}_small_toe",
            )
            if key in human_data
        ]
        if flat_points:
            return np.mean(flat_points, axis=0)
        return np.asarray(human_data[name][0], dtype=float).copy()

    def _classify_foot_contact(self, name, human_data):
        """
        根据人体脚跟、脚尖高度差，判断接触模式
        - 脚跟低 → HEEL_CONTACT
        - 脚尖低 → TOE_CONTACT
        - 接近 → FLAT_CONTACT
        """
        side = name.split("_", 1)[0]
        toe_names = (f"{side}_big_toe", f"{side}_small_toe")
        heel_name = f"{side}_heel"

        toes = [human_data[n][0][2] for n in toe_names if n in human_data]
        heel = human_data[heel_name][0][2] if heel_name in human_data else None

        if heel is None or not toes:
            return "FLAT_CONTACT"

        toe_height = float(np.mean(toes))
        if heel + 0.025 < toe_height:
            return "HEEL_CONTACT"
        if toe_height + 0.025 < heel:
            return "TOE_CONTACT"
        return "FLAT_CONTACT"

    @staticmethod
    def _has_human_heel_toe(name, human_data):
        """检查人体数据中是否包含脚跟、脚趾关键点"""
        side = name.split("_", 1)[0]
        return (
            f"{side}_heel" in human_data
            and (f"{side}_big_toe" in human_data or f"{side}_small_toe" in human_data)
        )

    def _classify_robot_foot_contact(self, name):
        """
        根据机器人足底几何高度判断接触模式（人体数据不足时的回退）
        """
        heights = self.foot_clearance_tasks[name].sole_heights(self.configuration)
        rear_height = float(np.mean(heights[:2]))
        front_height = float(np.mean(heights[2:]))

        if rear_height + 0.008 < front_height:
            return "HEEL_CONTACT"
        if front_height + 0.008 < rear_height:
            return "TOE_CONTACT"
        return "FLAT_CONTACT"

    def _start_foot_contact(self, name, mode, anchor=None):
        """
        接触建立瞬间：记录世界坐标系下的接触锚点与枢轴模式
        支撑全程锚点固定，保证脚底不滑移
        """
        geom_ids = self.foot_geom_groups[name][mode]
        # 取候选期多帧几何中心的中位数作为锚点；无候选样本时回退到当前帧
        if anchor is None:
            anchor = np.mean(self.configuration.data.geom_xpos[geom_ids], axis=0)
        else:
            anchor = np.asarray(anchor, dtype=float).copy()
        # Z 轴强制对齐地面
        anchor[2] = float(self.ground[2])
        self._foot_contact_anchor[name] = anchor
        self._foot_contact_pivot_mode[name] = mode
        # 锁定 XY 坐标
        self._foot_lock_xy[name] = anchor[:2].copy()
        self._foot_mode[name] = mode
        self._foot_pending_anchor_samples[name] = []
        self._foot_pending_mode[name] = None

    def _project_contact_foot_target(self, body_name, pos, quat):
        """
        支撑脚目标投影：
        1. 姿态逐步投影到水平（roll/pitch 归零，保留 yaw）
        2. 位置根据锚点 + 局部枢轴反推，保证接触点不滑动
        """
        if body_name not in self.foot_contact_state:
            return pos, quat

        # state=False 后仍在 blend 释放期内保留接触目标，
        # 直到 blend=0 才恢复完整人体脚姿态。
        blend = self._foot_contact_blend[body_name]
        if not self.foot_contact_state[body_name] and blend <= 1e-9:
            return pos, quat

        euler = R.from_quat(quat, scalar_first=True).as_euler("zyx")
        mode = self._foot_mode[body_name]

        # 目标全掌放平姿态：roll=0, pitch=0, 保留原 yaw
        flat_quat = R.from_euler("zyx", [euler[0], 0.0, 0.0]).as_quat(scalar_first=True)
        # 球面插值，从原姿态平滑过渡到水平姿态
        blended_quat = Slerp(
            [0.0, 1.0],
            R.from_quat(np.asarray([quat, flat_quat]), scalar_first=True),
        )([blend])[0].as_quat(scalar_first=True)

        projected = np.asarray(pos, dtype=float).copy()
        anchor = self._foot_contact_anchor[body_name]
        pivot_mode = self._foot_contact_pivot_mode[body_name]

        # 冲击落地退出后不再把旧世界锚点混入位置目标；普通步态仍使用
        # 原有释放 blend，兼顾脚滑抑制和连续性。
        if (
            anchor is not None
            and pivot_mode is not None
            and (
                self.foot_contact_state[body_name]
                or not self._foot_impact_contact[body_name]
            )
        ):
            # 获取局部枢轴向量
            pivot_local = self.foot_pivot_local[body_name][pivot_mode]
            # 根据锚点反算脚趾连杆位置：p_toe = p_anchor - R_foot * r_pivot
            anchored_position = anchor - R.from_quat(
                blended_quat, scalar_first=True
            ).apply(pivot_local)

            projected[:2] += blend * (anchored_position[:2] - projected[:2])
            projected[2] += blend * (anchored_position[2] - projected[2])

        return projected, blended_quat

    def _project_ne01_torso_orientation(self, task, body_name, quat):
        """
        躯干姿态约束：强制保持直立，roll/pitch 归零，只保留 yaw
        避免人体弯腰驼背导致机器人躯干前倾失衡
        """
        if not self.upright_torso_orientation or self.tgt_robot != "ne01":
            return quat
        if self.task_frame_names.get(task) != "TORSO_LINK" and body_name != "spine3":
            return quat

        yaw = R.from_quat(quat, scalar_first=True).as_euler("zyx")[0]
        return R.from_euler("z", yaw).as_quat(scalar_first=True)

    def _project_ne01_pelvis_orientation(self, task, body_name, quat):
        """
        骨盆姿态缩放：按系数缩小 roll 和 pitch，保留 yaw
        抑制人体骨盆过度倾斜，增强行走稳定性
        """
        if not self.stabilize_pelvis_orientation or self.tgt_robot != "ne01":
            return quat
        if self.task_frame_names.get(task) != "base_link" and body_name != self.human_root_name:
            return quat

        yaw, pitch, roll = R.from_quat(quat, scalar_first=True).as_euler("zyx")
        return R.from_euler(
            "zyx",
            [yaw, pitch * self.pelvis_pitch_scale, roll * self.pelvis_roll_scale],
        ).as_quat(scalar_first=True)

    def _correct_support_height(self):
        """
        支撑相高度闭式修正（Root-Z 投影）
        不改变腿部关节角度，整体上下平移浮动基座，
        使支撑脚最低点刚好接触地面，避免脚踝旋转导致滑移
        """
        # 找出完全进入支撑的脚
        active_names = [
            name for name in self.foot_support_tasks
            if self.foot_contact_state.get(name, False)
            and self._foot_contact_blend[name] >= 1.0
            and self._foot_contact_anchor.get(name) is not None
        ]
        if not active_names:
            return

        # 取所有支撑脚中最低的足底高度
        lowest = min(
            float(np.min(self.foot_clearance_tasks[name].sole_heights(self.configuration)))
            for name in active_names
        )
        # 计算需要修正的高度差
        correction = float(self.ground[2]) - lowest
        if abs(correction) <= 1e-8:
            return

        # 直接修改 qpos 中浮动基的 Z 坐标（第 3 个分量）
        qpos = self.configuration.data.qpos.copy()
        qpos[2] += correction
        self.configuration.update(qpos)

    def _correct_contact_xy(self):
        """
        支撑相 XY 闭式修正（Root-XY 投影）
        整体水平平移浮动基座，消除足部接触点与锚点的水平残差
        单支撑精确消除，双支撑最小二乘平均
        """
        if self._root_xy_hold_frames > 0:
            return

        residuals = []
        for name, active in self.foot_contact_state.items():
            anchor = self._foot_contact_anchor[name]
            mode = self._foot_contact_pivot_mode[name]
            if (
                not active
                or self._foot_contact_blend[name] < 1.0
                or anchor is None
                or mode is None
            ):
                continue

            geom_ids = self.foot_geom_groups[name][mode]
            # 当前实际接触点位置
            point_xy = np.mean(self.configuration.data.geom_xpos[geom_ids, :2], axis=0)
            # 与锚点的残差
            residuals.append(point_xy - anchor[:2])

        if not residuals:
            return

        # 只有仍处于支撑状态的脚参与 root XY 锁定。
        correction = np.mean(residuals, axis=0)
        qpos = self.configuration.data.qpos.copy()
        # 浮动基前两维是 X、Y
        qpos[:2] -= correction
        self.configuration.update(qpos)

    def _correct_swing_foot_clearance(self):
        """
        摆动脚离地间隙修正
        冻结骨盆（root 6 自由度全锁），仅通过腿部关节抬起脚，
        保证足底最低点高于地面+安全间隙，防止穿模
        """
        # 筛选出穿透地面的摆动脚
        tasks = [
            task
            for name, task in self.foot_clearance_tasks.items()
            if self._foot_contact_blend[name] < 1.0
            and not (
                self.foot_contact_state[name] and self._foot_impact_contact[name]
            )
            and task.compute_error(self.configuration)[0] < 0.0
        ]
        if not tasks:
            return

        # 约束：冻结浮动基座全部 6 个自由度
        constraints = [mink.DofFreezingTask(self.model, [0, 1, 2, 3, 4, 5])]

        # 更新姿态保持任务的目标为当前姿态
        for name, task in self.foot_clearance_tasks.items():
            if task in tasks:
                self.foot_orientation_hold_tasks[name].set_target(
                    self.configuration.get_transform_frame_to_world(
                        f"{name.split('_', 1)[0]}_toe_link", "body"
                    )
                )

        # 最多迭代 6 次逐步抬高
        for _ in range(6):
            active_clearance = [task for task in tasks if task.compute_error(self.configuration)[0] < -1e-5]
            if not active_clearance:
                break

            orientation_holds = [
                self.foot_orientation_hold_tasks[name]
                for name, task in self.foot_clearance_tasks.items()
                if task in active_clearance
            ]

            try:
                velocity = mink.solve_ik(
                    self.configuration,
                    active_clearance + orientation_holds,
                    self.motion_dt,
                    self.solver,
                    self.damping,
                    limits=self.ik_limits,
                    constraints=constraints,
                )
                self.configuration.integrate_inplace(velocity, self.motion_dt)
            except Exception:
                break

    def _stabilize_hips(self):
        """
        髋部稳定：
        1. 低通滤波平滑髋部角度
        2. 支撑相限制髋部 yaw 幅度，避免腿部过度外旋
        """
        if not self.hip_stabilization_enabled:
            return

        qpos = self.configuration.data.qpos.copy()
        # 一阶低通滤波系数
        alpha = 1.0 - np.exp(-2.0 * np.pi * self.hip_filter_cutoff_hz * self.motion_dt)
        double_support = all(self.foot_contact_state.values())

        for side in ("left", "right"):
            key = f"{side}_foot"
            short_side = side[0]
            yaw_index = self.hip_qpos_indices[(short_side, "yaw")]
            roll_index = self.hip_qpos_indices[(short_side, "roll")]

            for axis, index in (("yaw", yaw_index), ("roll", roll_index)):
                filter_key = (side, axis)
                previous = self._hip_filtered.get(filter_key, qpos[index])
                # 指数平滑
                qpos[index] = previous + alpha * (qpos[index] - previous)
                self._hip_filtered[filter_key] = qpos[index]

            # 支撑相限制髋 yaw
            if self.foot_contact_state[key]:
                if self._hip_contact_anchor[key] is None:
                    self._hip_contact_anchor[key] = qpos[yaw_index]

                anchor = self._hip_contact_anchor[key]
                # 裁剪到限制范围内
                bounded = np.clip(
                    qpos[yaw_index],
                    anchor - self.stance_hip_yaw_limit,
                    anchor + self.stance_hip_yaw_limit,
                )
                # 混合平滑限制
                qpos[yaw_index] = (
                    (1.0 - self.stance_hip_yaw_blend) * qpos[yaw_index]
                    + self.stance_hip_yaw_blend * bounded
                )

                # 双支撑进一步衰减 yaw
                if double_support:
                    qpos[yaw_index] *= 1.0 - self.double_support_yaw_blend
            else:
                # 摆动相清空锚点
                self._hip_contact_anchor[key] = None

        self.configuration.update(qpos)

    def _stabilize_pelvis_qpos(self):
        """
        骨盆姿态稳定：
        1. 低通滤波 roll 角度，抑制高频抖动
        2. 限制 roll 幅度
        3. 躯干关节补偿侧倾，保持上半身直立
        """
        if not self.stabilize_pelvis_orientation:
            return

        qpos = self.configuration.data.qpos.copy()
        yaw, pitch, roll = R.from_quat(qpos[3:7], scalar_first=True).as_euler("zyx")

        # 一阶低通滤波
        alpha = 1.0 - np.exp(-2.0 * np.pi * self.pelvis_roll_cutoff_hz * self.motion_dt)
        if self._pelvis_roll_filtered is None:
            self._pelvis_roll_filtered = roll
        self._pelvis_roll_filtered += alpha * (roll - self._pelvis_roll_filtered)

        # 限制 roll 幅度
        controlled_roll = float(np.clip(
            self._pelvis_roll_filtered,
            -self.pelvis_roll_limit,
            self.pelvis_roll_limit,
        ))

        # 写回骨盆四元数
        qpos[3:7] = R.from_euler(
            "zyx", [yaw, pitch, controlled_roll]
        ).as_quat(scalar_first=True)

        # ========== 躯干侧倾补偿 ==========
        # 当前骨盆+腰部的总朝向
        base_rotation = R.from_quat(qpos[3:7], scalar_first=True)
        pre_torso = base_rotation * R.from_euler(
            "z", qpos[self.waist_yaw_qpos_index]
        )
        torso_yaw = pre_torso.as_euler("zyx")[0]

        # 目标躯干姿态：抵消部分骨盆侧倾
        desired_torso = R.from_euler(
            "zyx",
            [torso_yaw, 0.0, controlled_roll * self.torso_world_roll_scale],
        )
        # 计算相对旋转角度
        relative = pre_torso.inv() * desired_torso
        compensation = float(relative.as_euler("xyz")[0])

        # 限制在关节范围内
        torso_joint_id = self.model.joint("TORSO_ROLL_JOINT").id
        target_torso = np.clip(
            compensation,
            self.model.jnt_range[torso_joint_id, 0],
            self.model.jnt_range[torso_joint_id, 1],
        )
        # 混合平滑
        current_torso = qpos[self.torso_roll_qpos_index]
        qpos[self.torso_roll_qpos_index] = (
            (1.0 - self.torso_compensation_blend) * current_torso
            + self.torso_compensation_blend * target_torso
        )

        self.configuration.update(qpos)

    def _solve_task_group(self, tasks, error_fn):
        """
        旧模式：迭代求解一组任务，直到误差收敛或达到最大迭代次数
        """
        curr_error = error_fn()
        self._solve_ik_once(tasks)
        next_error = error_fn()

        num_iter = 0
        while curr_error - next_error > 0.001 and num_iter < self.max_iter:
            curr_error = next_error
            self._solve_ik_once(tasks)
            next_error = error_fn()
            num_iter += 1

    def _solve_ik_once(self, tasks):
        """单次 IK 求解并积分"""
        dt = self.motion_dt
        velocity = mink.solve_ik(
            self.configuration,
            tasks,
            dt,
            self.solver,
            self.damping,
            limits=self.ik_limits,
        )
        self.configuration.integrate_inplace(velocity, dt)

    def error1(self):
        """计算 table1 所有任务的总误差范数"""
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )

    def error2(self):
        """计算 table2 所有任务的总误差范数"""
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )

    @staticmethod
    def to_numpy(human_data):
        """将人体数据转换为 numpy 数组"""
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data

    def scale_human_data(self, human_data, human_root_name, scale_table):
        """
        链式骨骼缩放：从根节点开始，沿父子关系逐段缩放骨骼长度
        只缩放位移，不改变旋转，保证关节角度不变
        """
        root_pos, root_quat = human_data[human_root_name]
        root_scale = scale_table.get(human_root_name, 1.0)
        scaled = {human_root_name: [root_scale * root_pos, root_quat]}

        pending = set(human_data)
        pending.remove(human_root_name)

        while pending:
            progressed = False
            for name in list(pending):
                # 获取父节点名称，默认根节点
                parent = self.human_scale_parents.get(name, human_root_name)
                if parent not in scaled:
                    continue

                # 父节点在原始数据中的位置
                parent_source = human_data.get(parent, human_data[human_root_name])[0]
                scale = scale_table.get(name, scale_table.get(parent, 1.0))

                # 子节点位置 = 父节点缩放后位置 + 缩放后的骨骼向量
                pos = scaled[parent][0] + scale * (human_data[name][0] - parent_source)
                scaled[name] = [pos, human_data[name][1]]

                pending.remove(name)
                progressed = True

            # 如果有节点找不到父节点，直接按根节点比例缩放兜底
            if not progressed:
                for name in pending:
                    scale = scale_table.get(name, root_scale)
                    scaled[name] = [scaled[human_root_name][0] + scale * (human_data[name][0] - root_pos), human_data[name][1]]
                break

        return scaled

    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """
        对人体数据施加局部坐标系偏移
        旋转偏移先乘，位置偏移在局部坐标系下施加后转到世界系
        """
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]

            rotation_offset = rot_offsets.get(body_name, R.identity())
            position_offset = pos_offsets.get(body_name, np.zeros(3))

            # 更新姿态：原旋转 * 局部偏移旋转
            updated_quat = (R.from_quat(quat, scalar_first=True) * rotation_offset).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat

            # 位置偏移从局部转到世界系叠加
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(position_offset)
            offset_human_data[body_name][0] = pos + global_pos_offset

        return offset_human_data

    def apply_robot_root_to_human_root_offset(self, human_data):
        """
        应用根节点偏移：对齐人体骨盆与机器人基座的几何原点差异
        """
        if np.allclose(self.robot_root_to_human_root_offset, 0.0):
            return human_data

        pos, quat = human_data[self.human_root_name]
        root_rot = R.from_quat(quat, scalar_first=True)
        # 偏移在局部坐标系下施加
        human_data[self.human_root_name][0] = pos - root_rot.apply(
            self.robot_root_to_human_root_offset
        )
        return human_data

    def offset_human_data_to_ground(self, human_data):
        """
        强制对齐地面：找到最低的脚，整体下移使脚底刚好接触地面
        """
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf
        lowest_body_name = None

        for body_name in human_data.keys():
            if "foot" not in body_name.lower():
                continue
            pos, _ = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name

        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # 整体下移 + 预留 0.1m 余量
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])

        return offset_human_data

    def set_ground_offset(self, ground_offset):
        """设置全局地面偏移量"""
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        """应用全局地面偏移"""
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data
