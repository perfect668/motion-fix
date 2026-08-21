# GMR—NE01 人形运动重定向代码改进归档说明

## 项目概述

本项目将 SMPL/SMPL-X 人体动作转换为 NE01 人形机器人的 MuJoCo 关节轨迹，供后续强化学习跟踪控制器训练和评估。完整链路为：人体数据加载与重采样、人体骨架尺度归一化、Table1/Table2 目标生成、层级 QP IK、NE01 足部接触与地面修正、50 Hz PKL 输出。

对人体关键刚体 `i`，机器人 FrameTask 的微分 IK 目标为：

```text
J_i(q) Δq ≈ -α_i e_i(q)
```

其中 `e_i` 是位置/姿态误差，`J_i` 是机器人雅可比，`Δq` 是一次广义坐标增量。基线 GMR 虽然区分 `ik_match_table1` 和 `ik_match_table2`，但本质上是两次普通加权 IK 顺序调用，并不是真正的词典序优化。

基线存在的系统性问题包括：IK 时间步与输出帧率不一致；Table2 offset 串接 Table1 结果；人体缩放以 pelvis 为中心而不是沿骨链递归；NE01 的 root 高度与人体目标地面不一致；足部只用单一最低点和 `toe_link` 处理，不能表达 heel-strike、toe-off 和刚性脚掌；支撑脚和摆动脚混用同一地面修正；低优先级任务可破坏高优先级脚/base 任务；最终表现为抬髋、骨盆扭动、躯干后仰、脚部穿模、落地跳变、后跟打滑和动作目标畸变。

## 变更清单

### 1. 统一 50 Hz 时间基准

① **文件与代码区段**：`general_motion_retargeting/motion_retarget.py`，`__init__`、`_solve_ik_once`、`solve_lexicographic_ik`；`scripts/smplx_to_robot.py`；`scripts/smplx_to_robot_dataset.py`。

② **修改前**：IK 使用 MuJoCo XML 内部 timestep，通常为 0.002 s，而动作输出按 50 Hz 生成，即帧间隔 0.02 s。速度限制实际按另一条时间尺度计算，出现动作过快、关节速度约束失真和帧间跳变。

③ **修改方案**：新增 `motion_fps` 和 `motion_dt=1/motion_fps`，所有 IK 积分和速度限制统一使用 `motion_dt`；入口默认目标频率为 50 Hz，PKL 的 `fps` 同步保存为 50；NE01 自动启用速度限制。

④ **修改动机**：离散速度约束是 `Δq/Δt`。如果求解器使用的 `Δt` 与轨迹真实采样周期不一致，求解器优化的不是实际输出动作。

⑤ **改进效果**：输出轨迹和 MuJoCo RL 控制器时间基准一致，关节速度更可控，减少高频抖动和单帧姿态突变。

### 2. 实现真正的 lexicographic/hierarchical QP

① **文件与代码区段**：`motion_retarget.py:solve_lexicographic_ik`、`_freeze_task`、`_build_priority_levels`。

② **修改前**：先求 Table1，再求 Table2；第二阶段仍可改变第一阶段的 base、脚部和关键姿态结果，导致优先级反转、IK 分支跳变和脚部目标被破坏。

③ **修改方案**：按 `ik_priority_levels` 分层求解。第 `k` 层求解时约束所有高层任务的零空间：`J_high Δq=0`。高层任务求解完成后创建零增益冻结任务，低层只能在高层已达到的结果附近优化。

④ **修改动机**：增大任务权重只是加权和，不等于词典序；真正的层级 QP 必须保证低层不能牺牲高层目标。

⑤ **改进效果**：base/脚部优先于躯干、下肢细节和上肢，减少髋膝跳解、足部漂移和躯干目标被低层破坏的问题。

### 3. 更换求解器并增加时序正则

① **文件与代码区段**：`motion_retarget.py` 默认 solver、`solve_lexicographic_ik` 中的上一帧 `PostureTask`。

② **修改前**：基线默认 DAQP，冗余自由度没有上一帧连续性约束，同一人体姿态附近可能在多个等价 IK 解间跳转，表现为骨盆扭动和关节尖峰。

③ **修改方案**：NE01 默认使用 ProxQP；每帧把上一帧 qpos 作为低权重 posture target，仅在高优先级任务零空间内进行时序平滑；增加 `hierarchy_failures` 统计。

④ **修改动机**：冗余 IK 的解通常不唯一。加入小的时间连续性代价可以稳定解分支，但不应覆盖高优先级几何约束。

⑤ **改进效果**：动作轨迹更连续、可重复，RL 训练数据中的非物理尖峰减少，求解失败不再完全静默。

### 4. 修复 Table1/Table2 offset 串行叠加

① **文件与代码区段**：`motion_retarget.py:update_targets`。

② **修改前**：Table2 复用已经经过 Table1 offset 的人体数据，位置、旋转和地面偏移被重复施加，造成脚、pelvis、base 目标整体偏移。

③ **修改方案**：保存未加表偏移的 `base_human_data`；Table1 和 Table2 分别从该副本独立应用自身的 position offset、rotation offset、root offset 和 ground shift。

④ **修改动机**：两张表是并行任务观测，不是连续坐标变换。串行使用会把两个独立补偿错误相加。

⑤ **改进效果**：Table2 精细任务不再重复改变 Table1 坐标系，减少整体抬髋、脚目标漂移和姿态畸变。

### 5. 将人体缩放改为骨链递归缩放

① **文件与代码区段**：`motion_retarget.py:scale_human_data`；`smplx_to_ne01.json:human_scale_parents`。

② **修改前**：所有人体关节相对 pelvis 直接缩放。大腿、小腿、躯干比例误差被混合到 root，脚和膝位置逐级偏离，IK 为满足脚目标会抬高或扭曲髋部。

③ **修改方案**：root 单独缩放；子节点按父子链递归计算：

```text
child_global' = parent_global' + scale_child * (child_global - parent_global)
```

配置明确 pelvis、spine3、左右髋、膝、脚、肩、肘、腕的父节点。

④ **修改动机**：人体形态缩放应保持局部骨段语义，而不是把所有点投射到 root 中心。NE01 腿部末端误差会沿髋—膝—踝链放大。

⑤ **改进效果**：膝、脚、肩与人体目标的几何关系更稳定，降低髋部异常抬升和腿部折叠。

### 6. 修复人体与机器人地面基准

① **文件与代码区段**：`motion_retarget.py:_update_human_floor`、`_shift_human_height`、`_correct_support_height`；`scripts/smplx_to_robot.py` 高度后处理。

② **修改前**：机器人按整段动作最低点统一平移，但人体目标没有同步减去同一地面基准。支撑脚投影后 root 下移而 pelvis/hip/knee 目标仍在旧高度，产生“强制抬髋”和浮空。

③ **修改方案**：从 heel、big toe、small toe（缺失时退回 foot）估计人体地面，所有人体目标同步减去 ground shift；NE01 不再执行其他机器人使用的整段最低点统一平移，支撑阶段仅按当前活动足底点修正 root-z。

④ **修改动机**：人体目标和机器人输出必须处于同一世界地面坐标系；整段最低点无法区分跳跃、摆动和支撑相。

⑤ **改进效果**：root-z 与人体髋膝高度一致，抬髋、浮空和支撑阶段高度跳变明显减少。

### 7. 新增 NE01 四点脚底几何任务

① **文件与代码区段**：新增 `general_motion_retargeting/foot_support_task.py`；`motion_retarget.py:setup_foot_support`。

② **修改前**：足部主要以 `toe_link` frame 或单一最低点表达接触，无法判断后跟、前掌和全脚是否同时合理，容易出现点接触、脚掌翘起和穿模。

③ **修改方案**：从 `assets/ne01/ne01.xml` 读取每只脚四个 collision geom；计算每个球形 geom 的最低点高度及 z 向雅可比；建立 `FootSupportTask` 和 `FootClearanceTask`，分别服务于支撑高度检查和摆动脚防穿透。

④ **修改动机**：NE01 足底是刚体，接触面由多个点共同定义；单点最低点不能表达脚掌是否平整。

⑤ **改进效果**：足部接触拥有明确的 rear/front/flat 几何语义，可直接和 MuJoCo 地面碰撞以及 RL 接触奖励对应。

### 8. 新增足部接触状态机与滞回

① **文件与代码区段**：`motion_retarget.py:_update_foot_contact`、`_classify_foot_contact`、`_classify_robot_foot_contact`。

② **修改前**：单帧最低点判断接触，临界帧容易 AIR/CONTACT 来回切换；摆动脚可能被提前吸到地面，支撑脚可能被误判为 AIR。

③ **修改方案**：高度阈值 3.5 cm、水平速度阈值 0.35 m/s；连续 2 帧进入、连续 3 帧退出；接触建立后 5 帧 blend 过渡。优先使用人体 heel/toe 高度差分类，缺失时使用机器人四点高度分类。

④ **修改动机**：接触是有持续时间的状态而不是单帧事件。滞回消除测量噪声，blend 将状态切换连续化。

⑤ **改进效果**：减少足部 AIR 误判、接触瞬间吸地和 root 突跳，落地过程更连续。

### 9. 使用真实后跟/前掌接触枢轴生成脚目标

① **文件与代码区段**：`motion_retarget.py:_start_foot_contact`、`_project_contact_foot_target`、`foot_pivot_local` 计算。

② **修改前**：锁定 `toe_link` XY 或直接复制人体脚姿态。脚从 heel-strike 放平时绕错误中心旋转，后跟向后滑；人体可弯曲足部姿态被硬套到 NE01 刚体脚上。

③ **修改方案**：接触开始时记录 rear/front collision 点世界坐标；利用 MuJoCo 局部 geom 坐标得到相对 `toe_link` 的枢轴向量；按

```text
p_toe = p_anchor - R_foot r_pivot
```

生成目标位置，保留 yaw，并把 roll/pitch 在接触 transition 内逐步投影到水平。

④ **修改动机**：刚体绕错误 frame 旋转必然产生切向位移；支撑脚应优先满足机器人实际接触点，而不是复制人体足部变形。

⑤ **改进效果**：单支撑后跟锚点误差达到数值零级，相比原先百毫米级滑移和一次 517 mm 异常滑移明显改善。

### 10. 用 root-XY 闭式投影锁定接触点

① **文件与代码区段**：`motion_retarget.py:_correct_contact_xy`。

② **修改前**：同时使用独立 FootAnchorTask、完整 foot FrameTask 和四点贴地 QP，约束重复且浮动基座自由度无充分速度界，出现米级 root 跳变和不可行层级。

③ **修改方案**：主 IK 保持完整脚目标；末端只根据活动接触点水平残差平移 root。单支撑精确消除残差，双支撑取两脚残差的最小二乘平均，不再把独立锚点 QP 放进主层级。

④ **修改动机**：root XY 平移能在不改变腿部关节相对构型的情况下移动所有接触点，是消除世界接触残差最直接的自由度；重复完整任务会导致不可行。

⑤ **改进效果**：单支撑脚不再随脚掌姿态后移，避免通过踝关节和髋膝链补偿导致的姿态扭曲。

### 11. 支撑相 root-z 闭式贴地

① **文件与代码区段**：`motion_retarget.py:_correct_support_height`。

② **修改前**：最终贴地阶段再次调用四点 IK，踝关节变化会破坏之前已经固定的后跟 XY，导致脚掌放平时二次滑动。

③ **修改方案**：主 IK 完成脚掌姿态后，读取活动支撑脚的最低 sole height，只对 root-z 加 `ground-lowest`，不再改变髋、膝、踝关节。

④ **修改动机**：残余高度误差属于整体平移问题；使用踝关节修正会改变接触切向位置，违反支撑点固定的物理约束。

⑤ **改进效果**：支撑脚贴地不会再次扭动踝关节，降低落地吸附和后跟滑移。

### 12. 摆动脚防穿透并保持姿态

① **文件与代码区段**：`motion_retarget.py:_correct_swing_foot_clearance`、`foot_orientation_hold_tasks`；`foot_support_task.py:FootClearanceTask`。

② **修改前**：只在支撑相修正穿模；摆动脚落地前可能已经陷入地面，随后支撑修正一次性托起，形成跳变和脚掌扭曲。

③ **修改方案**：AIR/过渡相持续检查最低点，目标为 ground+2 mm；冻结 root 6 DoF；建立当前 `toe_link` 姿态保持任务，只允许腿链抬脚；每次重新选择最低点并最多迭代 6 次。

④ **修改动机**：刚性脚的四角不是同一个点，抬起一个角会暴露另一个最低角；若不冻结姿态，roll/pitch 与高度会振荡。

⑤ **改进效果**：摆动阶段穿模明显减少，落地阶段不再通过突然整体抬高解决长期穿透。

### 13. 骨盆、躯干和支撑髋稳定

① **文件与代码区段**：`motion_retarget.py:_project_ne01_pelvis_orientation`、`_project_ne01_torso_orientation`、`_stabilize_pelvis_qpos`、`_stabilize_hips`。

② **修改前**：人体骨盆 roll、躯干后仰和髋 yaw 被直接复制，NE01 上表现为 T 型骨架左右晃动、身体后仰和髋部扭转。

③ **修改方案**：启用竖直躯干和骨盆稳定；骨盆 roll/pitch 分别缩放为 0.3/0.5，roll 低通并限制在约 ±3°；支撑期对 hip yaw/roll 进行 6 Hz 平滑，yaw 约束 ±10°；双支撑进一步减小 yaw 摆动。

④ **修改动机**：人体骨盆运动与 NE01 足底支撑和关节自由度不等价，直接复制会产生机器人不可稳定实现的目标；应保留必要 yaw，同时压制会破坏支撑的侧向分量。

⑤ **改进效果**：骨盆左右摆动、躯干后仰和髋部扭动减轻，为 RL 控制器提供更可跟踪的躯干/骨盆轨迹。

### 14. 修复 SMPL/SMPL-X 兼容与递归转换

① **文件与代码区段**：`general_motion_retargeting/utils/smpl.py`、`scripts/smpl_to_smplx.py`。

② **修改前**：beta 维度只有固定假设；递归目录遍历不完整；单帧数组可能被 `squeeze()` 成标量；长动作一次性推理造成内存压力；异常 `(300,)` beta 直接崩溃。

③ **修改方案**：兼容 `(10,)`、`(16,)`、逐帧 beta 和可识别展平 beta；修复单帧姿态形状；长动作分块；递归扫描目录；输出保留 AMASS 相对路径；保留 heel/toe 辅助点。

④ **修改动机**：输入层形状错误会在 IK 之前中断，或者直接丢失接触检测所需关节，无法把数据问题与算法问题分开。

⑤ **改进效果**：批处理可覆盖多层数据目录，转换失败更容易定位，足部接触辅助信息得到保留。

### 15. 新增非平地/异常动作预筛

① **文件与代码区段**：`scripts/filter_nonflat_motions.py`。

② **修改前**：楼梯、地形交互、趴下、倒立动作直接按平地 NE01 模型重定向，产生不可收敛、整机抬升、脚强行托地和目标畸变。

③ **修改方案**：递归扫描 NPZ；依据脚底高度跨度识别楼梯/非平地；依据 pelvis→spine3 方向、pelvis 高度和躯干姿态识别趴下/倒立/地面交互；输出 CSV 并可复制通过样本，保留原相对路径。

④ **修改动机**：无地形模型的平地 IK 无法表达多高度支撑和环境碰撞；筛选不满足建模假设的数据比让 QP 强行补偿更可靠。

⑤ **改进效果**：批量失败率下降，异常动作不会混入平地 RL 训练集，筛选过程可审计和人工复核。

## 配置文件变更汇总

静态参数位于 `general_motion_retargeting/ik_configs/smplx_to_ne01.json`；算法行为位于 Python 代码，两者需要区分。

### Table1/Table2 与任务权重

Table1 负责 base、左右脚和关键刚体的粗对齐；Table2 负责补充位置/姿态细化。二者不串行叠加 offset。NE01 层级配置为：

```text
Level 1: base_link, left_toe_link, right_toe_link
Level 2: TORSO_LINK
Level 3: HIP_ROLL_L/R_LINK, KNEE_PITCH_L/R_LINK
Level 4: shoulder/yaw, elbow, wrist
```

脚和 base 的位置任务权重最高；躯干优先于下肢细节；上肢位于低层。Table2 的 toe 位置/姿态权重用于精细脚目标，但不会重复施加 Table1 offset。

### 足部参数

```json
{
  "foot_contact_height_threshold": 0.035,
  "foot_contact_speed_threshold": 0.35,
  "foot_contact_enter_frames": 2,
  "foot_contact_exit_frames": 3,
  "foot_contact_transition_frames": 5,
  "swing_foot_clearance": 0.002
}
```

这些参数只定义接触状态机和摆动安全间隙；四点 geom、pivot 和雅可比由 NE01 MJCF 自动读取。

### 躯干、骨盆与髋参数

```json
{
  "upright_torso_orientation": true,
  "stabilize_pelvis_orientation": true,
  "pelvis_roll_scale": 0.3,
  "pelvis_pitch_scale": 0.5,
  "pelvis_roll_limit_rad": 0.05235988,
  "hip_stabilization_enabled": true,
  "hip_filter_cutoff_hz": 6.0,
  "stance_hip_yaw_limit_rad": 0.17453293,
  "double_support_yaw_blend": 0.15
}
```

### 缩放、根偏移和输出参数

`human_scale_parents` 明确骨链父子关系；`human_height_assumption=1.8` 用于人体尺度归一化；`robot_root_to_human_root_offset` 定义 NE01 `base_link` 与人体 `pelvis` 的固定几何偏移；`output_fps=50` 规定输出轨迹采样率。

## 核心知识点附录

### 1. Table1 粗对齐与 Table2 精细 IK

两张表是并行目标集合。正确目标生成是：

```text
T1_target = F(human_raw, offset_table1)
T2_target = F(human_raw, offset_table2)
```

而不是：

```text
T2_target = F(T1_target, offset_table2)
```

后者会把补偿重复施加到同一个坐标链。

### 2. 骨盆缩放与骨段缩放

骨盆/root 缩放改变整个人体的根位置，骨段缩放改变父子局部向量。递归缩放保持局部骨长语义：

```text
p_child' = p_parent' + s_child (p_child - p_parent)
```

这是 NE01 腿链稳定的基础。

### 3. 动态相对地面接触

先从人体 heel/toe/foot 估计动作地面，再用脚底相对高度和水平速度做状态判断。进入/退出滞回避免噪声，blend 避免离散吸附。绝对 z 单独判断会把人体整体移动误认为脚离地。

### 4. Hierarchical QP 规则

第 `k` 层优化：

```text
min ||W_k(J_k Δq + α_k e_k)||²
subject to J_j Δq = 0, j < k
           configuration limits
           velocity limits
```

它与单纯提高高层 cost 不同：低层不能牺牲高层已经达到的结果。

### 5. 刚性脚接触

人体 heel、跖趾和足弓可以变形，NE01 脚掌不能。heel-strike 应固定后跟并允许脚掌围绕后跟转动；flat support 应使四点接近地面；toe-off 只能在脱离支撑后释放接触约束。用 `toe_link` 原点代替接触点会把姿态修正直接变成滑移。

## 当前遗留风险 & 后续可优化方向

1. 双支撑的两只脚间距、人体 pelvis 轨迹和 NE01 腿长可能不完全兼容，当前采用两个接触残差平均；代表性走路测试中双支撑最大约 18.8 mm、95 分位约 12.8 mm。
2. 极端跳跃/快跑中，在冻结 root、保持脚姿态和关节限位同时存在时，摆动脚个别帧仍可能残留约 14 mm 穿透。可继续研究全脚 clearance QP、局部踝限位自适应和相位预处理。
3. 固定高度/速度阈值对高速跑、慢动作和跳跃并非最优，可增加垂直速度、足底面积置信度和自适应阈值。
4. 双支撑可进一步使用二维刚体 Procrustes（root XY+水平 yaw）配准两脚锚点，并在脚间距不可实现时输出明确警告。
5. `foot_rocking_limit_rad` 与当前姿态 Slerp 投影仍有参数语义重叠，未来可拆成 heel-phase pitch limit、toe-phase pitch limit 和 flat-phase roll/pitch hard bound。
6. 非平地筛选目前是启发式规则，若加入地形高度图，应将 ground plane 扩展为时空地形接触约束。
7. QP 失败目前计数并继续处理，未来应保存失败前 qpos、回滚失败层并输出逐帧任务残差。

## 全局改造总结

本次改造修复的不是单一权重，而是一条从输入到控制器的连锁管线：

```text
SMPL/SMPL-X 形状与目录兼容
        ↓
50 Hz 时间一致性
        ↓
骨链递归尺度 + 独立 Table1/Table2 offset
        ↓
人体/机器人地面基准统一
        ↓
AIR/HEEL/TOE/FLAT 接触状态机
        ↓
真实接触枢轴与刚性脚目标
        ↓
base/脚 → 躯干 → 下肢 → 上肢的层级 QP
        ↓
摆动脚姿态保持和最低点防穿透
        ↓
支撑 root-z 与接触 root-XY 投影
        ↓
50 Hz NE01 PKL → MuJoCo RL 跟踪控制器
```

几何层解决了 root 缩放、offset 串接和地面不一致；目标生成层把人体可变形脚转换成 NE01 刚性脚的接触枢轴；状态层解决 AIR 误判和落地跳变；层级 IK 保护 base、脚和躯干优先级；末端投影避免贴地修正再次破坏后跟 XY。最终输出不只是“能求出 qpos”，而是具有明确接触语义、统一时间尺度和较好 RL 可跟踪性的 NE01 运动轨迹。

归档时应将本文件与以下内容一并保存：`motion_retarget.py`、`foot_support_task.py`、`smplx_to_ne01.json`、SMPL/SMPL-X 转换脚本、筛选 CSV、代表性 PKL 以及可视化结果。

## 代码级前后对照（补充归档证据）

上一版说明缺少代码证据。本节逐项贴出基线代码和当前代码，路径、函数和代码区段均对应当前工作区。

### 1. 主类初始化：solver、动作周期和运行状态

文件：/home/user/桌面/GMR/general_motion_retargeting/motion_retarget.py，GeneralMotionRetargeting.__init__，原始约第 10—30 行，当前约第 10—40、100—150 行。

修改前：

~~~python
def __init__(self, src_human, tgt_robot,
             actual_human_height=None,
             solver="daqp",
             damping=5e-1,
             verbose=True,
             use_velocity_limit=False,
             velocity_limit=3*np.pi):
    ...
    self.max_iter = 10
~~~

修改后：

~~~python
def __init__(self, src_human, tgt_robot,
             actual_human_height=None,
             solver="proxqp",
             damping=5e-1,
             verbose=True,
             use_velocity_limit=False,
             velocity_limit=3*np.pi,
             motion_fps=50.0,
             legacy_mode=False):
    ...
    self.tgt_robot = tgt_robot
    self.motion_fps = float(motion_fps)
    if self.motion_fps <= 0:
        raise ValueError("motion_fps must be positive")
    self.motion_dt = 1.0 / self.motion_fps
    self.legacy_mode = bool(legacy_mode)
    self._q_prev = None
    self._q_prev2 = None
    self.hierarchy_failures = 0
    self._foot_history = {"left_foot": [], "right_foot": []}
    self.foot_contact_state = {"left_foot": False, "right_foot": False}
~~~

修改原因：基线没有真实动作周期、上一帧状态和接触状态；用 MuJoCo 0.002 s 作为 50 Hz 动作步长会错误计算速度。修改后 ProxQP、50 Hz 时间步、失败统计和时序状态成为后续算法的基础。效果是速度限制与输出轨迹一致，减少帧间跳变。

### 2. Table1/Table2 顺序 IK 改为层级 QP

文件：motion_retarget.py 的 retarget，新增 solve_lexicographic_ik、_freeze_task、_build_priority_levels，原始约第 300 行、当前约第 360—470 行。

修改前：

~~~python
for _ in range(retarget_passes):
    if self.use_ik_match_table1:
        self._solve_task_group(self.tasks1, self.error1)
    if self.use_ik_match_table2:
        self._solve_task_group(self.tasks2, self.error2)
~~~

修改后：

~~~python
for _ in range(retarget_passes):
    if self.legacy_mode:
        if self.use_ik_match_table1:
            self._solve_task_group(self.tasks1, self.error1)
        if self.use_ik_match_table2:
            self._solve_task_group(self.tasks2, self.error2)
    else:
        self.solve_lexicographic_ik()
~~~

~~~python
def solve_lexicographic_ik(self):
    levels = self._build_priority_levels()
    constraints = []
    for tasks in levels:
        try:
            velocity = mink.solve_ik(
                self.configuration, tasks, self.motion_dt,
                self.solver, self.damping,
                limits=self.ik_limits, constraints=constraints)
            self.configuration.integrate_inplace(velocity, self.motion_dt)
            constraints.extend(
                self._freeze_task(task) for task in tasks
                if not isinstance(task, FootSupportTask))
        except Exception:
            self.hierarchy_failures += 1
~~~

~~~python
def _freeze_task(self, task):
    frozen = mink.FrameTask(
        frame_name=self.task_frame_names[task],
        frame_type="body",
        position_cost=task.cost[:3],
        orientation_cost=task.cost[3:],
        gain=0.0,
        lm_damping=0.0)
    frozen.set_target(task.transform_target_to_world)
    return frozen
~~~

修改原因：原实现第二次 Table2 求解可以破坏第一阶段的脚和 base。当前通过高层任务零增益冻结实现 J_high Δq=0，低层只能在高层零空间内优化。效果是 base/脚优先，减少 IK 跳变、脚漂移和髋部错误分支。

### 3. Table2 offset 独立生成

文件：motion_retarget.py:update_targets，原始约第 230—260 行，当前约第 314—350 行。

修改前：

~~~python
human_data = self.scale_human_data(...)
human_data = self.offset_human_data(
    human_data, self.pos_offsets1, self.rot_offsets1)
...
for body_name in self.human_body_to_task2.keys():
    pos, rot = human_data[body_name]
    task.set_target(...)
~~~

修改后：

~~~python
base_human_data = {
    k: [v[0].copy(), v[1].copy()]
    for k, v in human_data.items()}
human_data = self.offset_human_data(
    base_human_data, self.pos_offsets1, self.rot_offsets1)
...
table2_data = self.offset_human_data(
    base_human_data, self.pos_offsets2, self.rot_offsets2)
table2_data = self.apply_robot_root_to_human_root_offset(table2_data)
table2_data = self.apply_ground_offset(table2_data)
table2_data = self._shift_human_height(table2_data, ground_shift)
table2_data = self._shift_human_xy(table2_data, lateral_shift)
~~~

修改原因：两张表是并行目标，旧代码把 Table1 offset 再传给 Table2，造成位置/姿态重复补偿。当前两表都从未偏移副本生成。效果是目标坐标一致，减少抬髋和脚/base 漂移。

### 4. 人体缩放和动态地面

文件：motion_retarget.py 的 scale_human_data、_update_human_floor、_shift_human_height，当前约第 320—538、866—910 行。

修改前：

~~~python
scaled_root_pos = human_scale_table[human_root_name] * root_pos
for body_name in human_data.keys():
    human_data_local[body_name] = (
        human_data[body_name][0] - root_pos
    ) * human_scale_table[body_name]
    human_data_global[body_name] = (
        human_data_local[body_name] + scaled_root_pos,
        human_data[body_name][1])
~~~

~~~python
human_data = self.apply_ground_offset(human_data)
if offset_to_ground:
    human_data = self.offset_human_data_to_ground(human_data)
~~~

修改后：

~~~python
self._update_human_floor(human_data)
ground_shift = self._human_floor_z - float(self.ground[2])
human_data = self._shift_human_height(human_data, ground_shift)
~~~

~~~python
root_scale = human_scale_table.get(human_root_name, 1.0)
scaled = {human_root_name: [root_scale * root_pos, root_quat]}
pending = set(human_data) - {human_root_name}
while pending:
    progressed = False
    for name in list(pending):
        parent = self.human_scale_parents.get(name, human_root_name)
        if parent not in scaled:
            continue
        parent_source = human_data.get(
            parent, human_data[human_root_name])[0]
        scale = human_scale_table.get(
            name, human_scale_table.get(parent, 1.0))
        pos = scaled[parent][0] + scale * (
            human_data[name][0] - parent_source)
        scaled[name] = [pos, human_data[name][1]]
        pending.remove(name)
        progressed = True
~~~

修改原因：旧实现所有关节以 pelvis 为中心缩放，骨段误差累积到脚；地面只修机器人，人体目标未同步，导致支撑时抬髋。当前沿父子骨链递归缩放，并将同一 ground shift 施加到人体目标。效果是髋膝脚高度关系稳定，减少抬髋和浮空。

### 5. 四点足底任务

文件：新增 /home/user/桌面/GMR/general_motion_retargeting/foot_support_task.py；接入 motion_retarget.py:setup_foot_support，当前约第 234—289 行。

修改前：该文件不存在，基线只用完整 foot FrameTask，无法取得四个 NE01 collision 点的真实最低高度或雅可比。

修改后：

~~~python
geom_names = [
    f"{side}_foot_rear_left_collision",
    f"{side}_foot_rear_right_collision",
    f"{side}_foot_front_left_collision",
    f"{side}_foot_front_right_collision"]
task = FootSupportTask(
    self.model, geom_names,
    ground_height=float(self.ground[2]), cost=200.0)
self.foot_clearance_tasks[f"{side}_foot"] = FootClearanceTask(
    self.model, geom_names,
    clearance_height=float(self.ground[2])
        + self.swing_foot_clearance,
    cost=300.0)
~~~

~~~python
class FootSupportTask(Task):
    def compute_error(self, configuration):
        centers_z = configuration.data.geom_xpos[self.geom_ids, 2]
        radii = self.model.geom_size[self.geom_ids, 0]
        return centers_z - radii - self.ground_height

class FootClearanceTask(Task):
    def sole_heights(self, configuration):
        return (configuration.data.geom_xpos[self.geom_ids, 2]
                - self.model.geom_size[self.geom_ids, 0])
    def compute_error(self, configuration):
        heights = self.sole_heights(configuration)
        self._active_index = int(np.argmin(heights))
        return np.array([
            heights[self._active_index] - self.clearance_height])
~~~

修改原因：NE01 脚是刚体，四个支撑点共同定义接触。旧单点逻辑会产生点接触、翘起和另一角穿模。当前任务提供 rear/front/flat 几何和最低点检查。效果是足底约束与 MuJoCo 碰撞几何一致。

### 6. 接触状态机与 heel/toe 枢轴

文件：motion_retarget.py 的 _update_foot_contact、_classify_foot_contact、_start_foot_contact、_project_contact_foot_target，当前约第 473—636 行。

修改前：没有 enter/exit 滞回和世界接触锚点，toe_link 直接追踪人体脚目标。heel-strike 后放平脚掌时，后跟绕错误中心后移。

修改后：

~~~python
near_ground = support_z <= float(self.ground[2])               + self.foot_contact_height_threshold
candidate = near_ground and speed <= self.foot_contact_speed_threshold
if candidate:
    self._foot_enter_count[name] += 1
    if self._foot_enter_count[name] >= self.foot_contact_enter_frames:
        entering_contact = not self.foot_contact_state[name]
        self.foot_contact_state[name] = True
        if entering_contact:
            mode = self._classify_foot_contact(name, human_data)
            self._start_foot_contact(name, mode)
else:
    self._foot_exit_count[name] += 1
    if self._foot_exit_count[name] >= self.foot_contact_exit_frames:
        self.foot_contact_state[name] = False
        self._foot_mode[name] = "AIR"
~~~

~~~python
def _start_foot_contact(self, name, mode):
    geom_ids = self.foot_geom_groups[name][mode]
    anchor = np.mean(
        self.configuration.data.geom_xpos[geom_ids], axis=0)
    anchor[2] = float(self.ground[2])
    self._foot_contact_anchor[name] = anchor
    self._foot_contact_pivot_mode[name] = mode
    self._foot_lock_xy[name] = anchor[:2].copy()
    self._foot_mode[name] = mode
~~~

~~~python
anchored_position = anchor - R.from_quat(
    blended_quat, scalar_first=True).apply(pivot_local)
projected[:2] = anchored_position[:2]
projected[2] += blend * (
    anchored_position[2] - projected[2])
~~~

修改原因：接触是持续状态，必须使用滞回；刚体姿态变化应绕实际 rear/front collision 点，而不是 toe frame 原点。效果是 AIR 误判减少，单支撑后跟锚点达到数值零级，旧版百毫米级滑移和一次 517 mm 异常滑移被消除。

### 7. 支撑 root-z、root-XY 与摆动脚修正

文件：motion_retarget.py 的 _correct_support_height、_correct_contact_xy、_correct_swing_foot_clearance，当前约第 657—740 行。

修改前：支撑相再次用四点 QP 改踝关节；摆动脚只在支撑相修正，导致摆动阶段穿模、落地瞬间托起和后跟二次滑移。

修改后：

~~~python
lowest = min(float(np.min(
    self.foot_clearance_tasks[name].sole_heights(
        self.configuration))) for name in active_names)
correction = float(self.ground[2]) - lowest
qpos = self.configuration.data.qpos.copy()
qpos[2] += correction
self.configuration.update(qpos)
~~~

~~~python
residuals = []
for name, active in self.foot_contact_state.items():
    if not active:
        continue
    geom_ids = self.foot_geom_groups[name][
        self._foot_contact_pivot_mode[name]]
    point_xy = np.mean(
        self.configuration.data.geom_xpos[geom_ids, :2], axis=0)
    residuals.append(point_xy - self._foot_contact_anchor[name][:2])
if residuals:
    qpos = self.configuration.data.qpos.copy()
    qpos[:2] -= np.mean(residuals, axis=0)
    self.configuration.update(qpos)
~~~

~~~python
constraints = [mink.DofFreezingTask(
    self.model, [0, 1, 2, 3, 4, 5])]
for _ in range(6):
    active_clearance = [
        task for task in tasks
        if task.compute_error(self.configuration)[0] < -1e-5]
    if not active_clearance:
        break
    velocity = mink.solve_ik(
        self.configuration,
        active_clearance + orientation_holds,
        self.motion_dt, self.solver, self.damping,
        limits=[mink.ConfigurationLimit(self.model)],
        constraints=constraints)
~~~

修改原因：高度误差属于 root-z 平移，水平接触误差属于 root-XY 平移，不能用踝关节同时解决；摆动脚需要冻结 root 并保持脚姿态，只允许腿链抬脚。效果是支撑贴地不再重新旋转脚，单支撑后跟固定，摆动穿模和落地跳变减少。

### 8. 骨盆、躯干和支撑髋稳定

文件：motion_retarget.py 的 _project_ne01_pelvis_orientation、_project_ne01_torso_orientation、_stabilize_pelvis_qpos、_stabilize_hips，当前约第 638—830 行。

修改前：

~~~python
pos, rot = human_data[body_name]
task.set_target(mink.SE3.from_rotation_and_translation(
    mink.SO3(rot), pos))
~~~

修改后：

~~~python
rot = self._project_ne01_pelvis_orientation(task, body_name, rot)
rot = self._project_ne01_torso_orientation(task, body_name, rot)
pos, rot = self._project_contact_foot_target(body_name, pos, rot)
task.set_target(mink.SE3.from_rotation_and_translation(
    mink.SO3(rot), pos))
~~~

~~~python
yaw, pitch, roll = R.from_quat(
    quat, scalar_first=True).as_euler("zyx")
return R.from_euler(
    "zyx", [yaw,
            pitch * self.pelvis_pitch_scale,
            roll * self.pelvis_roll_scale]
).as_quat(scalar_first=True)
~~~

修改原因：人体骨盆侧摆和躯干后仰不等价于 NE01 的稳定目标。当前保留 yaw、缩放 pitch、限制/低通 roll，并在支撑期平滑 hip yaw/roll。效果是 T 型骨盆左右晃动、身体后仰和髋部扭动减轻。

### 9. 单文件与批处理入口

文件：/home/user/桌面/GMR/scripts/smplx_to_robot.py 第 130—215 行；scripts/smplx_to_robot_dataset.py 的 process_file、main。

修改前：

~~~python
tgt_fps = 30
smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
    smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
retarget = GMR(...)
...
height_adjust=not args.no_height_adjust
~~~

批处理基线固定 30 Hz、固定开启整段 HEIGHT_ADJUST。

修改后：

~~~python
parser.add_argument("--tgt_fps", default=50, type=float)
smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
    smplx_data, body_model, smplx_output, tgt_fps=args.tgt_fps)
retarget = GMR(
    actual_human_height=actual_human_height,
    src_human="smplx", tgt_robot=args.robot,
    use_velocity_limit=args.robot == "ne01",
    motion_fps=args.tgt_fps)
qpos_list, root_deltas = adjust_qpos_to_ground(
    qpos_list, retarget.xml_file,
    height_adjust=not args.no_height_adjust
        and args.robot != "ne01")
~~~

~~~python
retargeter = GMR(
    ..., use_velocity_limit=use_velocity_limit
        or tgt_robot == "ne01",
    motion_fps=tgt_fps)
HEIGHT_ADJUST = tgt_robot != "ne01"
~~~

修改原因：NE01 地面已在重定向内部处理，再按整段最低点平移会重复修改；其他机器人继续保留原行为。效果是单文件/批处理一致，NE01 不再被入口后处理重新抬升或压低。

### 10. utils/smpl.py：分块 FK 和精确重采样

文件：/home/user/桌面/GMR/general_motion_retargeting/utils/smpl.py，load_smplx_file、get_smplx_data_offline_fast。

修改前：

~~~python
smplx_output = body_model(
    betas=torch.tensor(betas).float().view(1, -1),
    global_orient=torch.tensor(
        smplx_data["root_orient"]).float(),
    body_pose=torch.tensor(
        smplx_data["pose_body"]).float(),
    transl=torch.tensor(smplx_data["trans"]).float(),
    return_full_pose=True)
~~~

~~~python
src_fps = smplx_data["mocap_frame_rate"].item()
frame_skip = int(src_fps / tgt_fps)
global_orient = smplx_output.global_orient.squeeze()
~~~

修改后：

~~~python
output_parts = {"global_orient": [],
                "full_pose": [], "joints": []}
body_model.eval()
with torch.no_grad():
    for start in range(0, num_frames, 512):
        end = min(start + 512, num_frames)
        output = body_model(
            betas=betas_tensor,
            global_orient=root_orient[start:end],
            body_pose=pose_body[start:end],
            transl=trans[start:end],
            return_full_pose=True,
            return_verts=False)
        for name in output_parts:
            output_parts[name].append(
                getattr(output, name).detach().cpu())
~~~

~~~python
global_orient = smplx_output.global_orient.detach().cpu().numpy()     .reshape(num_frames, 3)
new_num_frames = max(1, int(np.floor(
    (num_frames - 1) * tgt_fps / src_fps)) + 1)
target_time = np.arange(new_num_frames) * (src_fps / tgt_fps)
~~~

修改原因：长动作一次性 FK 会耗尽内存；squeeze 导致单帧旋转掉维；整数 frame skip 不能保证 50 Hz。效果是长动作可转换、单帧不再出现 rot_vec shape ()，重采样频率严格一致。

### 11. smpl_to_smplx.py：beta 和递归路径

文件：/home/user/桌面/GMR/scripts/smpl_to_smplx.py，convert_smpl_to_smplx、process_directory。

修改前：

~~~python
betas = data_dict["betas"]
if betas.shape == (10,):
    data_dict["betas"] = np.concatenate(
        [betas, np.zeros(6, dtype=betas.dtype)])
elif betas.shape not in [(16,), (1, 16)]:
    raise ValueError("Unexpected betas shape...")
~~~

~~~python
for filename in tqdm(os.listdir(src_folder)):
    if filename.endswith(".npz"):
        input_path = os.path.join(src_folder, filename)
        output_path = os.path.join(tgt_folder, filename)
~~~

修改后：

~~~python
betas = np.asarray(data_dict["betas"])
if betas.ndim == 2 and betas.shape[0] == 1:
    betas = betas.reshape(-1)
if betas.shape == (10,):
    betas = np.concatenate(
        [betas, np.zeros(6, dtype=betas.dtype)])
elif betas.ndim != 1 or betas.size < 16:
    raise ValueError(
        "Expected a 10-element legacy SMPL vector or "
        "an SMPL-X vector with at least 16 elements.")
data_dict["betas"] = betas
~~~

~~~python
for dirpath, dirnames, filenames in os.walk(src_folder):
    dirnames.sort()
    for filename in sorted(filenames):
        if filename.endswith(".npz"):
            input_paths.append(os.path.join(dirpath, filename))
for input_path in tqdm(input_paths):
    relative_path = os.path.relpath(input_path, src_folder)
    output_path = os.path.join(tgt_folder, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
~~~

修改原因：基线拒绝 (300,) beta 且只扫描一层目录，造成转换中断和文件漏转。当前允许至少 16 个 shape 系数并保留 AMASS 相对路径。

### 12. JSON 配置具体前后变化

文件：/home/user/桌面/GMR/general_motion_retargeting/ik_configs/smplx_to_ne01.json。

新增键：

~~~diff
+ "output_fps": 50,
+ "foot_contact_height_threshold": 0.035,
+ "foot_contact_speed_threshold": 0.35,
+ "foot_contact_enter_frames": 2,
+ "foot_contact_exit_frames": 3,
+ "foot_contact_transition_frames": 5,
+ "swing_foot_clearance": 0.002,
+ "upright_torso_orientation": true,
+ "stabilize_pelvis_orientation": true,
+ "pelvis_roll_scale": 0.3,
+ "pelvis_pitch_scale": 0.5,
+ "pelvis_roll_limit_rad": 0.05235988,
+ "hip_stabilization_enabled": true,
+ "hip_filter_cutoff_hz": 6.0,
+ "stance_hip_yaw_limit_rad": 0.17453293,
+ "double_support_yaw_blend": 0.15
~~~

新增层级和父子骨链：

~~~diff
+ "human_scale_parents": {
+   "spine3": "pelvis",
+   "left_hip": "pelvis", "left_knee": "left_hip",
+   "left_foot": "left_knee",
+   "right_hip": "pelvis", "right_knee": "right_hip",
+   "right_foot": "right_knee"
+ },
+ "ik_priority_levels": [
+   ["base_link", "left_toe_link", "right_toe_link"],
+   ["TORSO_LINK"],
+   ["HIP_ROLL_L_LINK", "KNEE_PITCH_L_LINK",
+    "HIP_ROLL_R_LINK", "KNEE_PITCH_R_LINK"],
+   ["SHOULDER_YAW_L_LINK", "ELBOW_PITCH_L_LINK",
+    "HAND_YAW_L_LINK", "SHOULDER_YAW_R_LINK",
+    "ELBOW_PITCH_R_LINK", "HAND_YAW_R_LINK"]
+ ]
~~~

已有参数调整：

~~~diff
- "initial_frame_retarget_passes": 5,
+ "initial_frame_retarget_passes": 15,
- "spine3": 0.9,
+ "spine3": 0.94,
- "left_hip": 0.86,
+ "left_hip": 0.93,
- "left_knee": 0.86,
+ "left_knee": 0.87,
- "left_shoulder": 0.8,
+ "left_shoulder": 0.88,
- Table2 knee position cost: 10,
+ Table2 knee position cost: 20,
- Table2 toe orientation cost: 50,
+ Table2 toe orientation cost: 25,
- left_toe_link offset: [0.0, 0.02, 0.0],
+ left_toe_link offset: [0.02575903, 0.02972709, 0.03050858],
- right_toe_link offset: [0.0, -0.02, 0.0],
+ right_toe_link offset: [0.02022062, -0.02592166, 0.03537731]
~~~

修改原因：这些是静态 scale、weight、offset；offset 的独立应用、层级冻结和接触状态机属于 Python 算法代码，不能混写成“只调权重”。

## 代码验收记录与限制

代表性 Loop_Forward_Walk_001__A018 当前验证：

~~~text
输出频率：50 Hz
hierarchy_failures：0
单支撑后跟锚点误差：接近数值零
双支撑最大接触误差：约 18.8 mm
双支撑 95 分位：约 12.8 mm
极端摆动脚残余穿透：约 14 mm
~~~

曾验证但未保留的失败实现是：把独立 FootAnchorTask、四点支撑 QP 和完整 foot FrameTask 同时放入主 QP。该实现出现约 517 mm 后跟滑移、层级失败和米级 root 跳变，当前代码已删除该重复约束。

## 全局改造总结

~~~text
输入 beta/目录/单帧形状兼容
        ↓
SMPL-X 分块 FK + 50 Hz 时间采样
        ↓
骨链递归缩放 + Table1/Table2 独立 offset
        ↓
人体与 NE01 地面基准统一
        ↓
AIR/HEEL/TOE/FLAT 接触状态机
        ↓
真实 rear/front 枢轴的刚性脚目标
        ↓
base/脚 → 躯干 → 下肢 → 上肢层级 QP
        ↓
摆动脚姿态保持和最低点防穿透
        ↓
支撑 root-z、接触 root-XY 投影
        ↓
50 Hz NE01 PKL → MuJoCo RL tracking controller
~~~

本文件现在同时记录：每个改动的绝对位置、基线代码片段、当前代码片段、原理根因、修改动机、仿真现象、改进效果、JSON 前后值和验收限制。

### 13. XSens/BVH 输出增加 FK 身体轨迹

文件：/home/user/桌面/GMR/scripts/xsens_bvh_to_robot.py，原始约第 194—215 行；新增 /home/user/桌面/GMR/general_motion_retargeting/motion_data.py。

修改前：

~~~python
local_body_pos = None
body_names = None
motion_data = {
    "fps": motion_fps,
    "root_pos": root_pos,
    "root_rot": root_rot,
    "dof_pos": dof_pos,
    "local_body_pos": local_body_pos,
    "link_body_list": body_names,
}
~~~

修改后：

~~~python
parser.add_argument(
    "--fk_device", default="auto",
    help="Device for local-body FK: auto, cpu, cuda, or cuda:0.")
...
motion_data = {
    "fps": motion_fps,
    "root_pos": root_pos,
    "root_rot": root_rot,
    "dof_pos": dof_pos,
}
motion_data = enrich_robot_motion_with_fk(
    motion_data, robot=args.robot, device=args.fk_device)
~~~

新增 FK 函数：

~~~python
def enrich_robot_motion_with_fk(motion_data, robot, device="auto"):
    resolved_device = resolve_torch_device(device)
    kinematics_model = KinematicsModel(
        str(ROBOT_XML_DICT[robot]), device=resolved_device)
    dof_pos = np.asarray(
        motion_data["dof_pos"], dtype=np.float32)
    frame_count = dof_pos.shape[0]
    root_pos = torch.zeros(
        (frame_count, 3), device=resolved_device)
    root_rot = torch.zeros(
        (frame_count, 4), device=resolved_device)
    root_rot[:, -1] = 1.0
    with torch.no_grad():
        local_body_pos, _ = kinematics_model.forward_kinematics(
            root_pos, root_rot,
            torch.from_numpy(dof_pos).to(device=resolved_device))
    enriched_motion = dict(motion_data)
    enriched_motion["local_body_pos"] = (
        local_body_pos.detach().cpu().numpy())
    enriched_motion["link_body_list"] = list(
        kinematics_model.body_names)
    return enriched_motion
~~~

修改原因：基线 XSens/BVH PKL 的 local_body_pos 和 link_body_list 是 None，下游 body-space 跟踪和动作质量分析无法使用。当前在导出时按机器人 FK 自动补齐，支持 CPU/CUDA 选择。

### 14. 预筛脚本的实际判别代码

文件：/home/user/桌面/GMR/scripts/filter_nonflat_motions.py，新增。

修改前：没有任何代码在重定向前检查楼梯、非平地、倒立或趴下，异常动作会直接进入平地 NE01 IK。

修改后：

~~~python
foot_z = np.stack(foot_z, axis=1)
floor = float(np.percentile(foot_z, 10))
low_envelope = np.min(foot_z, axis=1)
contact = low_envelope <= floor + 0.045
...
terrain_span = float(
    np.percentile(contact_heights, 95)
    - np.percentile(contact_heights, 5))
~~~

~~~python
torso_axis = spine - pelvis
torso_norm = np.linalg.norm(
    torso_axis, axis=1, keepdims=True).clip(min=1e-6)
torso_up_z = (torso_axis / torso_norm)[:, 2]
pelvis_gap = pelvis[:, 2] - foot_floor
inverted_fraction = float(
    np.mean(torso_up_z < -0.2))
prone_fraction = float(
    np.mean((torso_up_z < 0.45)
            & (pelvis_gap < 0.45))
)
flags = []
if terrain_span > 0.10:
    flags.append("nonflat_or_stairs")
if inverted_fraction > 0.10:
    flags.append("inverted_or_handstand")
if prone_fraction > 0.20:
    flags.append("prone_or_floor_interaction")
~~~

修改原因：平地模型不能解释多高度地形或人体与物体接触。该脚本输出 CSV，并可保留相对路径复制 accepted 数据，避免把不适合的动作混入 RL 数据集。

### 15. 批处理内存保护参数

文件：/home/user/桌面/GMR/scripts/smplx_to_robot_dataset.py，check_memory、process_file、main。

修改前：

~~~python
def check_memory(threshold_gb=30):
    ...
while check_memory():
    time.sleep(60*2)
~~~

修改后：

~~~python
def check_memory(threshold_gb=4):
    ...
while check_memory(min_available_memory_gb):
    print("[PAUSE] Paused processing ...")
    time.sleep(60*2)
...
parser.add_argument(
    "--min_available_memory_gb",
    default=4, type=float)
parser.add_argument(
    "--tgt_fps", default=50, type=float)
~~~

修改原因：SMPL-X 长序列和多 worker 会造成内存峰值，原固定阈值无法适配当前数据批处理。当前阈值可配置，低内存时暂停而不是继续分配导致整个转换中断。

## 代码改动索引

| 编号 | 文件 | 函数/区段 | 前后代码位置 |
|---|---|---|---|
| 1 | motion_retarget.py | __init__ | 原约 10—30；现约 10—150 |
| 2 | motion_retarget.py | retarget / solve_lexicographic_ik | 原约 300；现约 353—470 |
| 3 | motion_retarget.py | update_targets | 原约 230—260；现约 310—350 |
| 4 | motion_retarget.py | scale_human_data / floor | 原约 866；现约 518—538、866—910 |
| 5 | foot_support_task.py | FootSupportTask / FootClearanceTask | 新增全文件 |
| 6 | motion_retarget.py | contact / pivot | 现约 473—636 |
| 7 | motion_retarget.py | support / swing correction | 现约 657—740 |
| 8 | motion_retarget.py | pelvis/torso/hip stabilization | 现约 638—830 |
| 9 | smplx_to_robot.py / dataset.py | FPS/height/FK | 现约 130—230 |
| 10 | utils/smpl.py | load/FK/resample | 原约 63、213；现约 64、233 |
| 11 | smpl_to_smplx.py | beta/walk | 原约 21、80；现约 21、65 |
| 12 | smplx_to_ne01.json | weights/offset/parameters | 全文件 1—263 |
| 13 | xsens_bvh_to_robot.py / motion_data.py | output FK enrichment | 原约 194；新增 motion_data.py |
| 14 | filter_nonflat_motions.py | inspect_motion/main | 新增全文件 |
| 15 | smplx_to_robot_dataset.py | check_memory/main | 原约 45；现约 45、276 |

以上每个索引项均已在本文件前文提供修改前代码、修改后代码、缺陷根因、修改动机和效果。
