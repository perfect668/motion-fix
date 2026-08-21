"""
将单段 SMPL-X 人体动作重定向到单台人形机器人，用于调试验证与快速导出。

功能定位：
    本脚本是 smplx_to_robot_dataset.py 的单文件配套工具。
    可用于查看单个 SMPL-X 动作文件、调试坐标系/IK 问题、
    可选打开机器人动作可视化查看器，也可选保存单段机器人动作 .pkl 文件。

典型使用命令：
    conda run --no-capture-output -n gmr python scripts/smplx_to_robot.py \
        --smplx_file data/smplx_data/selected_10_12h/example.npz \
        --robot unitree_g1_24dof \
        --save_path /tmp/example_g1_24dof.pkl \
        --max_frames 300

无界面快速导出命令：
    conda run --no-capture-output -n gmr python scripts/smplx_to_robot.py \
        --smplx_file data/smplx_data/selected_10_12h/example.npz \
        --robot ne01 \
        --save_path /tmp/example_ne01.pkl \
        --headless

注意事项：
    批量生成数据集请使用 smplx_to_robot_dataset.py，
    因为它会保存 local_body_pos 和 link_body_list 供下游筛选使用。
    本脚本主要用于可视化检查和针对性调试。
"""

# ========== 基础库导入 ==========
import argparse          # 命令行参数解析
import pathlib           # 路径对象处理，方便跨平台拼接路径
import os                # 操作系统接口，用于创建目录、路径处理
import time              # 时间相关工具，用于计算实际渲染帧率

import numpy as np       # 数值计算库，处理数组、矩阵运算
import torch             # PyTorch 深度学习框架，用于正运动学批量计算

# ========== 项目核心模块导入 ==========
# 导入通用运动重定向主类 GMR
from general_motion_retargeting import GeneralMotionRetargeting as GMR
# 导入机器人动作可视化查看器（基于 MuJoCo）
from general_motion_retargeting import RobotMotionViewer
# 导入正运动学模型类，用于批量计算连杆空间位置
from general_motion_retargeting.kinematics_model import KinematicsModel
# 导入 SMPL-X 相关工具：加载文件、离线快速预处理
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

# rich 库：终端彩色美化打印
from rich import print


def adjust_qpos_to_ground(qpos_list, xml_file, height_adjust=True, root_origin_offset=True):
    """
    调整机器人整段动作的根节点位置，使脚底最低点贴合地面，并可选将初始帧根节点XY对齐到原点。
    
    参数:
        qpos_list: np.array, 形状 [帧数, 7+关节数]，每帧前3位是根位置，3-7位是根四元数(wxyz)，后面是关节角度
        xml_file: str, 机器人 MuJoCo XML 模型文件路径，用于构建正运动学模型
        height_adjust: bool, 是否调整高度使最低点贴地
        root_origin_offset: bool, 是否将首帧根节点XY平移到世界原点
    
    返回:
        qpos_list: 调整后的全帧关节位姿
        root_deltas: 每帧根节点的平移偏移量，用于同步平移人体参考数据
    """
    # 复制输入数据，避免修改原数组
    qpos_list = qpos_list.copy()
    
    # 提取根节点位置（前3维：x, y, z）
    root_pos = qpos_list[:, :3].copy()
    # 保存原始根位置，用于计算偏移量
    original_root_pos = root_pos.copy()
    
    # 提取根节点旋转四元数，并将 wxyz 顺序转换为 xyzw 顺序（适配运动学模型输入格式）
    root_rot_xyzw = qpos_list[:, 3:7][:, [1, 2, 3, 0]].copy()
    
    # 提取所有驱动关节的角度
    dof_pos = qpos_list[:, 7:].copy()

    if height_adjust:
        # 构建机器人正运动学模型，用于计算所有连杆的空间位置
        kinematics_model = KinematicsModel(xml_file, device="cpu")
        # 关闭梯度计算，仅做前向推理
        with torch.no_grad():
            # 批量执行正运动学，得到所有身体连杆的位置和姿态
            body_pos, _ = kinematics_model.forward_kinematics(
                torch.as_tensor(root_pos, dtype=torch.float32),
                torch.as_tensor(root_rot_xyzw, dtype=torch.float32),
                torch.as_tensor(dof_pos, dtype=torch.float32),
            )
            # 找出整段动作中所有连杆的最低高度值（Z轴最小值）
            lowest_height = torch.min(body_pos[..., 2]).item()
        
        # 整体向下平移根节点，使动作最低点刚好落在地面 Z=0 处
        root_pos[:, 2] -= lowest_height

    if root_origin_offset:
        # 将首帧根节点的 XY 坐标平移到原点 (0,0)，整段动作同步平移
        root_pos[:, :2] -= root_pos[0, :2]

    # 将调整后的根位置写回 qpos 数组
    qpos_list[:, :3] = root_pos
    
    # 返回调整后的 qpos 和根节点偏移量（偏移量用于同步移动人体参考骨架）
    return qpos_list, root_pos - original_root_pos


def offset_human_motion(human_motion_list, root_deltas):
    """
    根据机器人根节点的平移偏移量，同步平移每帧人体动捕数据，
    保证可视化时人体和机器人的相对位置一致。
    
    参数:
        human_motion_list: list，每帧是一个字典，键为关节名，值为 [位置, 四元数]
        root_deltas: np.array，每帧根节点的平移偏移向量
    
    返回:
        adjusted_motion: 平移后的人体动作列表
    """
    adjusted_motion = []
    # 逐帧遍历人体数据与对应偏移量
    for human_data, delta in zip(human_motion_list, root_deltas):
        frame_data = {}
        for body_name, (pos, quat) in human_data.items():
            # 位置加上偏移量，旋转保持不变
            frame_data[body_name] = [np.asarray(pos).copy() + delta, np.asarray(quat).copy()]
        adjusted_motion.append(frame_data)
    return adjusted_motion


if __name__ == "__main__":
    # 获取当前脚本所在目录的绝对路径
    HERE = pathlib.Path(__file__).parent

    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser()
    
    # SMPL-X 动捕文件路径（必填）
    parser.add_argument(
        "--smplx_file",
        help="待加载的 SMPL-X 动作文件路径。",
        type=str,
        required=True,
    )
    
    # 目标机器人型号（从支持的列表中选择）
    parser.add_argument(
        "--robot",
        choices=["ne01", "unitree_g1", "unitree_g1_with_hands", "unitree_g1_24dof", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1",
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung", "fourier_gr3"],
        default="unitree_g1",
    )
    
    # 机器人动作保存路径（可选）
    parser.add_argument(
        "--save_path",
        default=None,
        help="机器人动作文件的保存路径。",
    )
    
    # 是否循环播放动作
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="循环播放动作。",
    )

    # 是否录制视频
    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="录制播放过程为视频。",
    )

    # 是否限制播放帧率，与原始人体动作帧率保持一致
    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="限制重定向后动作的播放帧率，与原始人体动作帧率保持一致。",
    )
    
    # 无界面模式（不打开可视化窗口，仅后台计算导出）
    parser.add_argument(
        "--headless",
        default=False,
        action="store_true",
        help="不打开 MuJoCo 可视化查看器。",
    )
    
    # 仅处理前 N 帧，用于快速验证
    parser.add_argument(
        "--max_frames",
        default=None,
        type=int,
        help="仅重定向前 N 帧，用于快速验证效果。",
    )

    # 目标输出帧率
    parser.add_argument(
        "--tgt_fps",
        default=50,
        type=float,
        help="对齐源动作时使用的目标帧率。",
    )

    # 关闭高度自动对齐（不调整根节点使最低点贴地）
    parser.add_argument(
        "--no_height_adjust",
        default=False,
        action="store_true",
        help="不自动升降机器人动作，使身体最低点刚好落在地面上。",
    )

    # 关闭根节点原点对齐
    parser.add_argument(
        "--no_root_origin_offset",
        default=False,
        action="store_true",
        help"不将初始帧根节点的 XY 位置对齐到世界坐标原点。",
    )

    # 解析所有命令行参数
    args = parser.parse_args()


    # SMPL-X 人体模型文件夹路径（项目 assets 目录下）
    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    
    # ========== 加载 SMPL-X 动作轨迹 ==========
    # 加载 npz 动捕文件，返回：逐帧数据、人体模型对象、SMPL 输出、实际演员身高
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    
    # 帧率对齐：将原始动捕数据插值到目标帧率，得到对齐后的逐帧人体关节数据
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.tgt_fps
    )
    
   
    # ========== 初始化运动重定向系统 ==========
    retarget = GMR(
        actual_human_height=actual_human_height,  # 传入演员实际身高，用于骨骼缩放
        src_human="smplx",                        # 源人体数据类型为 SMPL-X
        tgt_robot=args.robot,                     # 目标机器人型号
        use_velocity_limit=args.robot == "ne01",  # 仅 ne01 机器人启用关节速度限制
        motion_fps=args.tgt_fps,                  # 动作输出帧率
    )
    
    # 确定要处理的帧索引范围（跳过第0帧通常是T-pose初始帧）
    frame_indices = range(1, len(smplx_data_frames)) if len(smplx_data_frames) > 1 else range(len(smplx_data_frames))
    # 如果指定了最大帧数，截取前 N 帧
    if args.max_frames is not None:
        frame_indices = list(frame_indices)[: args.max_frames]
    
    # 用于保存每帧重定向后的机器人关节位姿
    qpos_list = []
    # 用于保存每帧缩放对齐后的人体数据（可视化时做对比参考）
    human_motion_list = []
    
    # ========== 逐帧执行运动重定向 ==========
    for frame_idx in frame_indices:
        # 调用 GMR 主函数，输入单帧人体数据，输出机器人 qpos
        qpos = retarget.retarget(smplx_data_frames[frame_idx])
        qpos_list.append(qpos.copy())
        
        # 保存当前帧缩放后的人体数据，用于后续可视化对比
        human_motion_list.append(
            {body_name: [data[0].copy(), data[1].copy()] for body_name, data in retarget.scaled_human_data.items()}
        )

    # 转换为 numpy 数组，方便批量处理
    qpos_list = np.asarray(qpos_list)
    
    # ========== 后处理：地面高度对齐 + 原点对齐 ==========
    qpos_list, root_deltas = adjust_qpos_to_ground(
        qpos_list,
        retarget.xml_file,
        # ne01 机器人内部已有地面接触逻辑，默认不做额外高度对齐
        height_adjust=not args.no_height_adjust and args.robot != "ne01",
        root_origin_offset=not args.no_root_origin_offset,
    )
    # 同步平移人体参考数据，保证人和机器人位置对齐
    human_motion_list = offset_human_motion(human_motion_list, root_deltas)

    # ========== 初始化可视化查看器 ==========
    robot_motion_viewer = None
    if not args.headless:
        try:
            # 创建机器人动作查看器
            robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                                    motion_fps=aligned_fps,
                                                    transparent_robot=0,        # 机器人不透明
                                                    record_video=args.record_video,
                                                    video_path=f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4",)
        except Exception as e:
            # 查看器创建失败时打印警告，自动切换为无界面模式
            print(f"警告：无法创建可视化查看器 ({e})，将以无界面模式运行。")

    # ========== 可视化播放循环 ==========
    if robot_motion_viewer is not None:
        frame_idx = 0          # 当前播放帧索引
        fps_counter = 0        # 帧率统计计数器
        fps_start_time = time.time()  # 帧率统计起始时间
        fps_display_interval = 2.0    # 每隔2秒打印一次实际帧率

        while True:
            # 取出当前帧的机器人位姿
            qpos = qpos_list[frame_idx]
            
            # 步进一帧可视化，同时渲染人体参考骨架
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=human_motion_list[frame_idx],
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,  # 不显示人体关节名称标签
                rate_limit=args.rate_limit,
                follow_camera=False,        # 相机不跟随机器人移动
            )

            # 帧索引自增
            frame_idx += 1
            fps_counter += 1
            
            # 每隔固定时间打印一次实际渲染帧率
            if time.time() - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (time.time() - fps_start_time)
                print(f"实际渲染帧率: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = time.time()
            
            # 循环播放模式：帧索引取模循环
            if args.loop:
                frame_idx %= len(qpos_list)
            # 非循环模式：播放完所有帧后退出
            elif frame_idx >= len(qpos_list):
                break
            
    # ========== 保存机器人动作到文件 ==========
    if args.save_path is not None:
        import pickle
        # 获取保存目录
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            # 目录不存在则自动创建
            os.makedirs(save_dir, exist_ok=True)
        
        # 提取根节点位置数组
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # 提取根节点旋转，并将 wxyz 顺序转换为 xyzw 顺序（下游常用格式）
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        # 提取所有驱动关节角度
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        
        # 单文件调试脚本不计算局部连杆位置，数据集脚本才会生成
        local_body_pos = None
        body_names = None
        
        # 组装保存数据字典
        motion_data = {
            "fps": aligned_fps,       # 动作帧率
            "root_pos": root_pos,     # 根节点位置 [N, 3]
            "root_rot": root_rot,     # 根节点旋转四元数 [N, 4] (xyzw)
            "dof_pos": dof_pos,       # 关节角度 [N, 关节数]
            "local_body_pos": local_body_pos,  # 局部连杆位置（本脚本为空）
            "link_body_list": body_names,      # 连杆名称列表（本脚本为空）
        }
        
        # 以二进制写入 pickle 文件
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"已保存到 {args.save_path}")
            
      
    
    # 关闭可视化查看器，释放资源
    if robot_motion_viewer is not None:
        robot_motion_viewer.close()
