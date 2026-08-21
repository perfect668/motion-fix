### 1.GMR项目的主入口为：xsens_bvh_to_robot.py

1. 
| 参数名 | 作用 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--bvh_file` | 输入的 BVH 动捕文件路径 | **无（必填）** | 你的人体动作源文件 |
| `--robot` | 目标机器人型号 | `unitree_h1_2` | 可选：`unitree_g1` / `unitree_h1_2` / `Q1` / `X1` |
| `--record_video` | 是否录制可视化视频 | `False` | 加此参数则开启录屏 |
| `--video_path` | 视频输出路径 | `videos/example.mp4` | 录制视频的保存位置 |
| `--rate_limit` | 是否按原始帧率限速播放 | `True` | 开启后可视化会按动捕原始帧率播放，不会快进 |
| `--save_path` | 重定向结果保存路径 | `None`（不保存） | 指定后会将机器人动作存为 pkl 文件 |
| `--scale` | 长度单位缩放系数 | `0.01` | 用于将 BVH 单位转为米：若 BVH 是厘米单位，`0.01` 正确；若 BVH 是米单位需改为 `1.0` |
| `--reset_to_zero` | 是否将首帧位置 / Z 轴旋转归零 | `False` | 开启后动作从原点出发，初始朝向对齐 |
| `--start` / `--end` | 处理的起止帧号 | `None`（全量处理） | 只处理 BVH 中的一段时使用 |
| `--bvh_format` | BVH 文件格式 | `3DSM` | 目前仅支持 3ds Max 导出的 BVH 格式 |
2. 需要改脚本的场景
只有特殊需求才需要修改脚本本身：
想调整视频分辨率：解开 video_width / video_height 的注释并修改数值
想给人体模型加位置偏移：解开 human_pos_offset 并设置偏移量
想在保存的 pkl 里加入更多数据（比如关节位置、力矩）：在 motion_data 字典里补充字段
支持其他 BVH 格式：需要去 load_xsens_file 工具函数里修改解析逻辑
想看实时运行 FPS：解开 print(f"Actual rendering FPS...") 的注释

### 2.GMR项目的核心算法类：motion_retarget.py

1. 基于MuJoCo 机器人运动学 + mink 逆运动学 (IK) 求解器，实现人体动捕数据到人形机器人关节运动的映射。
    核心作用为：
        1.加载目标机器人的 MuJoCo 模型，读取关节、连杆、驱动器信息
        2.加载「人体-机器人」骨骼匹配配置，构建分层IK任务
        3.对输入的人体动捕数据做尺寸缩放、位姿偏移、地面对齐等预处理
        4.通过二次规划求解逆运动学，输出机器人每帧的广义位姿 qpos(根位置+根旋转+所有关节角度)
        5.支持关节限位、速度限位、迭代收敛控制，保证重定向结果物理可行

2. if verbose:	如果 verbose 为 True，则执行下面缩进的代码块（通常是打印信息）；如果为 False，则跳过。

3. 核心方法详解
    3.1 build_actuated_joint_velocity_limits
    构建所有驱动关节的速度限制字典，自动排除根自由关节。
    3.2 setup_retarget_configuration
    根据 IK 匹配表，创建两组 mink 的 FrameTask（位置 + 姿态跟踪任务），每个任务带独立权重。
    3.3 update_targets
    对输入的人体动捕数据做完整预处理，并更新 IK 任务的目标位姿。
    3.4 retarget（对外主接口）
    每帧调用一次，输入人体数据，输出机器人广义位姿 qpos。
    3.5 _solve_task_group
    迭代求解一组 IK 任务，直到误差收敛或达到最大迭代次数。
    3.6 _solve_ik_once
    单次 IK 求解：调用 mink 求解器得到关节速度，积分到位姿更新。
    3.7 误差计算方法
    计算一组任务的总误差（所有任务误差向量拼接后的 L2 范数），用于判断收敛。

4. 数据预处理辅助方法
    4.1 to_numpy
    把输入人体数据中的列表转为 numpy 数组。
    4.2 scale_human_data
    局部坐标系下的人体尺寸缩放：只缩放位置，不改变旋转，保证姿态不变、整体尺寸按比例缩放。
    4.3 offset_human_data
    给每个骨骼施加局部坐标系下的位置和旋转偏移，用于修正人体与机器人的关节零位差异。
    4.4 apply_robot_root_to_human_root_offset
    修正机器人根节点与人体根节点的语义位置差异（比如机器人根在骨盆中心，人体根在骨盆上沿）。
    4.5 offset_human_data_to_ground
    自动对齐地面：找到人体最低点（脚），整体下移让脚刚好离地一点，防止浮空。
    4.6 地面偏移设置与应用

5. 绝大多数调参都不需要动这个脚本，改配置文件或构造函数参数即可：

| 参数类别 | 调整位置 | 作用 | 调参建议 |
| :--- | :--- | :--- | :--- |
| IK 求解器类型 | `__init__` 的 `solver` | 二次规划求解后端 | 默认 `daqp` 即可，速度快、稳定性好 |
| IK 阻尼系数 | `__init__` 的 `damping` | 控制 IK 稳定性与精度 | 动作抖、求解报错 → 调大；跟踪精度差 → 调小 |
| 关节速度限制 | `__init__` 的 `use_velocity_limit` + `velocity_limit` | 限制单帧关节角速度 | 动作太突兀、关节超速 → 开启并调小 |
| 人体身高缩放 | 构造函数传入 `actual_human_height` | 适配不同身高的动捕数据 | 填动捕演员真实身高（米），尺寸不对优先调这个 |
| 骨骼匹配关系、权重、偏移 | IK 配置 JSON 文件 | 定义人体骨骼与机器人的对应关系、跟踪权重 | 重定向姿态不对优先调这里 |
| 首帧迭代次数 | IK 配置 JSON 文件 | 首帧收敛次数 | 首帧姿态差 → 调大 |
| 地面高度、根偏移 | IK 配置 JSON 文件 | 坐标系对齐 | 浮空 / 穿地 → 调地面相关参数 |


6. 只有深度定制算法逻辑时才需要改：
    想增加第三组 IK 任务、增加新的任务类型（比如质心任务、关节平滑任务）
    想修改 IK 迭代收敛条件（比如误差阈值、最大迭代次数）
    想增加新的人体数据预处理步骤（比如平滑滤波、镜像翻转）
    想替换 IK 求解逻辑、不用 mink 改用别的库
    想输出更多信息（比如每帧误差、关节力矩等）
    想支持新的约束类型（比如碰撞避免、支撑多边形约束）


### 3.GMR项目的可视化回放脚本：vis_robot_motion.py
    这是 GMR 项目的离线运动回放可视化脚本，核心用途是加载并预览已经完成重定向、保存为 .pkl 格式的机器人运动数据，无需实时运行重定向算法。
    快速检查重定向结果的质量（姿态是否自然、有无穿模浮空、动作是否流畅）
    排查重定向异常，定位问题帧
    录制演示视频
    支持倍速调节、循环播放，方便反复观察细节


1. 日常使用：不需要修改脚本，全部通过命令行控制
    常规场景直接传参即可，无需改动代码：
    切换机器人：修改 --robot
    换回放文件：修改 --robot_motion_path
    录制视频：添加 --record_video 并指定 --video_path

2. 需要修改脚本的场景
    只有定制播放行为时才需要改脚本：
    开启相机跟随：机器人移动范围大、会走出视野 → 把 camera_follow=False 改成 True
    修改倍速范围：想支持更快 / 更慢的播放速度 → 修改 min(4.0, ...) 和 max(0.25, ...) 的上下限
    只播放一遍就退出：不想循环播放 → 把 while True 改成 while frame_idx < len(motion_root_pos):，去掉帧索引归零逻辑
    增加快捷键功能：比如加空格暂停、左右键单帧步进 → 在 keyboard_callback 里补充按键判断
    关闭帧率限速：想最快速度跑完 → 把 rate_limit=True 改成 False
    同步显示人体参考模型：需要额外传入人体数据，在 env.step() 里补充 human_motion_data 参数
   
### 4.GMR 项目的IK 匹配核心配置文件：bvh_xsens_to_g1.json
    （JSON 格式，不是 Python 脚本），是整个动作重定向效果的「调参中枢」。它完整定义了人体骨骼与机器人连杆的对应关系、跟踪权重、零位偏移、尺寸缩放比例，直接决定重定向后的姿态是否自然、位置是否准确。

### 5.GMR 项目的重定向后统一质量过滤脚本：filter_robot_motion.py
    它输入批量生成的机器人运动 .pkl 数据集，从运动学合理性、物理可行性、重定向质量三个维度自动打分，将所有动作划分为 pass（合格）、tag（轻微瑕疵，可保留）、reject（不合格，丢弃） 三档，并输出完整的质检报告和分类软链接目录。

1. 判定规则：

| 类别 | 代表参数 | 含义 |
| :--- | :--- | :--- |
| 基础校验 | `min_frames=30` | 最少帧数，低于 30 帧的过短动作直接拒绝 |
| 脚部接触与穿地 | `foot_contact_height=0.05` | 低于 0.05 米判定为脚接触地面 |
| 脚部接触与穿地 | `foot_penetration_reject=-0.06` | 脚穿地超过 6cm 直接判定不合格 |
| 脚部接触与穿地 | `foot_slide_speed_reject=0.15` | 支撑脚滑动速度超过 0.15m/s 判定不合格 |
| 腾空检测 | `double_air_height=0.08` | 双脚都高于 8cm 判定为腾空 |
| 腾空检测 | `double_air_reject_frames=30` | 连续腾空 30 帧以上判定不合格 |
| 根节点姿态 | `root_tilt_reject_deg=65` | 躯干倾斜超过 65° 判定为摔倒 / 躺卧 |
| 根节点姿态 | `root_height_span_reject=0.4` | 根高度变化超过 0.4 米判定为攀爬类高难度动作 |
| 语义动作检测 | `knee_near_ground_height=0.08` | 膝盖低于 8cm 判定为接近跪地 |
| 语义动作检测 | `sitting_root_knee_gap=0.15` | 骨盆与膝盖高度差小于 15cm 判定为坐姿 |
| 语义动作检测 | `hand_near_ground_height=0.10` | 手低于 10cm 判定为手撑地爬行 |
| 速度与平滑度 | `root_speed_reject=4.0` | 根速度超过 4m/s 直接拒绝 |
| 速度与平滑度 | `joint_speed_reject=30.0` | 关节速度超过 30rad/s 直接拒绝 |
| 速度与平滑度 | `joint_delta_reject=0.75` | 单帧关节跳变超过 0.75rad 直接拒绝 |
| 关节限位 | `joint_limit_margin_deg=5` | 距离关节限位 5° 以内判定为接近极限 |
| 手臂质量 | `arm_side_correct_ratio=0.95` | 手臂左右侧正确率低于 95% 判定为异常 |
| 手臂质量 | `arm_limit_ratio_reject=0.50` | 手臂关节超过 50% 时间顶限位判定为异常 |

2. 严格版报告（标准输出）

| 文件 / 目录 | 内容 |
| :--- | :--- |
| `summary.json` | 整体统计：总数、各档位数量、各拒绝原因计数、所有阈值 |
| `details.json` | 每个文件的完整质检详情：状态、原因、所有量化指标 |
| `report.csv` | 表格版报告，可直接用 Excel 打开筛选 |
| `pass.txt` / `tag.txt` / `reject.txt` | 对应档位的文件路径列表 |
| `pass_motions/` / `tag_motions/` / `reject_motions/` | 对应档位的软链接目录，直接存放指向原始 pkl 的符号链接 |

3. 日常使用：无需改脚本，全部通过命令行控制
    常用参数：
    --robot：必须和生成 pkl 时的机器人型号完全一致，否则映射全错
    --input：输入数据集路径，支持单文件或文件夹
    --output_dir：报告输出目录，默认自动生成在 filter_reports/ 下
    --disable_arm_quality：关闭手臂专项检测，加快速度
    --no_relaxed_report：只生成严格报告，不生成宽松训练版
    --no_symlink_dirs：不生成软链接目录，只输出报告文件
    --relaxed_dynamic_actions：开启动态动作宽松，适合高动态训练集
    --max_files N：只处理前 N 个文件，快速测试阈值是否合理



### 6.自操作流程：使用自有smpl文件转化为适用机器人的pkl


1. scripts/convert_smpl_to_ne01.sh为自动运行脚本
    # 命令如下：
    cd ~/桌面/GMR

    bash scripts/convert_smpl_to_ne01.sh \
    ~/桌面/gmr_ne01/smpl/diverse_actions_8h_v3 \
    ~/桌面/GMR/data/retarget_data/ne01/diverse_actions_8h_v3 \
    2 cuda:0

    # 参数含义：
    第1个参数：SMPL数据目录
    第2个参数：输出PKL目录
    第3个参数：CPU进程数，默认2
    第4个参数：设备，默认cuda:0
    
    # 例如以后换了一套SMPL数据：
    cd ~/桌面/GMR

    bash scripts/convert_smpl_to_ne01.sh \
    ~/桌面/new_dataset/smpl \
    ~/桌面/GMR/data/retarget_data/ne01/new_dataset \
    2 cuda:0

    # 脚本会自动完成：
    SMPL .npz 转换为GMR需要的SMPL-X字段。
    使用GMR重定向到NE01。
    固定输出为 50 FPS。
    生成包含 local_body_pos 和 link_body_list 的训练PKL。
    关闭GMR内置动作过滤，确保输入全部处理。
    检查输入和输出文件数量。
    自动跳过已经存在的PKL，支持中断后重复执行。
    中间SMPL-X文件保存在：
    ~/桌面/GMR/work/smplx_to_ne01/<输出目录名>