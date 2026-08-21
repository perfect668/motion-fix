# NE01 GMR 算法改进说明

## 1. 文档目的

本文档说明本地 GMR 项目针对 NE01 机器人所做的运动重定向改进，包括：

- 原始 GMR 的求解方式及其问题；
- 真正的 lexicographic/hierarchical QP 实现；
- 50 Hz 时间基准和速度限制修正；
- 人体骨链缩放方式修正；
- table2 独立 offset 修复；
- 足部接触检测、刚性脚姿态、四点支撑和贴地修正；
- NE01 躯干后仰修正；
- 非平地、趴下和倒立动作的预筛；
- 修改文件、配置参数、运行命令和当前局限。

本文档描述的是当前工作区中的实际实现，而不是仅停留在设计层面的方案。

## 修改文件清单（先看这里）

本次 NE01 算法改进实际新增或修改了以下文件。

### A. 核心算法文件

#### 1. 修改：`/home/user/桌面/GMR/general_motion_retargeting/motion_retarget.py`

这是本次改进的主文件，修改内容包括：

- 将默认 QP solver 从 DAQP 改为 ProxQP；
- 新增 `motion_fps` 和 `motion_dt`，使用真实动作帧间隔求解；
- 新增 `legacy_mode`，保留旧式 table1/table2 顺序求解入口；
- 实现 `solve_lexicographic_ik()` 层级 IK；
- 实现 `_freeze_task()`，用 `J Delta_q = 0` 保持高优先级任务；
- 实现 `_build_priority_levels()`，从 JSON 配置构建任务层级；
- 修复 table2 使用错误 offset 的问题；
- 将人体缩放改为父子骨链递归缩放；
- 增加人体 heel、big toe、small toe 辅助点保留；
- 实现足部接触检测和连续帧迟滞；
- 实现支撑脚 XY 锁定和 yaw-only 平脚姿态；
- 接入 NE01 四点脚底支撑任务；
- 实现 `_correct_support_height()` 最终接触投影；
- 实现 `_project_ne01_torso_orientation()` 躯干竖直投影；
- 增加上一帧姿态正则、层级失败计数和关节速度限制时间修正。

#### 2. 新增：`/home/user/桌面/GMR/general_motion_retargeting/foot_support_task.py`

这是新增加的四点足底任务文件，主要内容为：

- 定义 `FootSupportTask`；
- 从 MuJoCo model 中查找 NE01 四个脚底 collision geom；
- 计算每个球形 collision geom 最低点到地面的高度误差；
- 调用 `mj.mj_jac()` 计算四个支撑点的 z 向雅可比；
- 将四点误差组织为可直接交给 Mink/ProxQP 的 Task。

#### 3. 修改：`/home/user/桌面/GMR/general_motion_retargeting/ik_configs/smplx_to_ne01.json`

这是 NE01 的 IK 参数配置文件，修改内容包括：

- 保留并更新 NE01 的人工/AutoIK scale、位置 offset、姿态 offset 和任务权重；
- 增加 `output_fps=50`；
- 增加足部接触高度、速度和迟滞参数；
- 增加 `upright_torso_orientation=true`；
- 增加 `human_scale_parents` 骨链父子关系；
- 增加 `ik_priority_levels` 层级顺序；
- 将 `TORSO_LINK` 提升到下肢和上肢任务之前；
- 配置 base、足部、下肢、上肢的位置和姿态 cost。

### B. SMPL-X 转换入口文件

#### 4. 修改：`/home/user/桌面/GMR/scripts/smplx_to_robot.py`

单文件重定向入口，修改内容包括：

- 默认目标频率改为 50 Hz；
- 创建 GMR 时传入 `motion_fps`；
- NE01 自动启用关节速度限制；
- NE01 不再执行“整段动作按全局最低点统一平移”的旧式高度修正；
- 保留其他机器人的原高度修正行为。

#### 5. 修改：`/home/user/桌面/GMR/scripts/smplx_to_robot_dataset.py`

批量数据集重定向入口，修改内容包括：

- 默认目标频率改为 50 Hz；
- 每个 worker 创建 GMR 时传入 `motion_fps`；
- NE01 自动启用速度限制；
- NE01 跳过整段最低点高度修正；
- PKL 中保存实际的 50 Hz FPS；
- 保持相对目录结构输出。

### C. 数据质量预筛文件

#### 6. 新增：`/home/user/桌面/GMR/scripts/filter_nonflat_motions.py`

这是新增的重定向前预筛脚本，主要内容为：

- 递归扫描 SMPL-X NPZ；
- 根据脚底接触高度跨度判断楼梯或非平地动作；
- 根据 `pelvis -> spine3` 位置向量判断倒立；
- 根据 pelvis 高度和躯干方向判断趴下或地面交互；
- 输出 CSV 报告；
- 可将通过筛选的数据复制到新目录，并保留原相对路径。

### D. 输入数据兼容文件

以下文件主要用于此前的 SMPL/SMPL-X 输入转换和数据兼容，不属于层级 QP 本体，但会影响完整数据流水线。

#### 7. 修改：`/home/user/桌面/GMR/general_motion_retargeting/utils/smpl.py`

- 修复不同 beta 维数的 SMPL-X model 创建；
- 修复单帧姿态数组被 `squeeze()` 成标量的问题；
- 对长动作分块调用 SMPL-X，降低主机内存占用；
- 修正/增强目标 FPS 重采样；
- 保留 toe、heel 等接触检测所需关节。

#### 8. 修改：`/home/user/桌面/GMR/scripts/smpl_to_smplx.py`

- 将目录遍历改为递归遍历；
- 输出目录保留输入 AMASS 相对路径；
- 兼容 `(10,)`、`(16,)`、逐帧 beta 和展平 beta；
- 改进异常 beta 形状的错误提示和转换逻辑。

### E. 本次说明文档

#### 9. 新增：`/home/user/桌面/GMR/NE01_GMR算法改进说明.md`

即当前文档，用于记录上述代码、配置、数学原理、运行命令、验证结果和已知限制。

### 文件关系

```text
smplx_to_robot.py / smplx_to_robot_dataset.py
                |
                v
        motion_retarget.py
          |             |
          v             v
smplx_to_ne01.json  foot_support_task.py

filter_nonflat_motions.py
        |
        v
重定向前筛除非平地/趴下/倒立动作
```

如果只关注本次 NE01 算法本身，最关键的是以下三个文件：

```text
/home/user/桌面/GMR/general_motion_retargeting/motion_retarget.py
/home/user/桌面/GMR/general_motion_retargeting/foot_support_task.py
/home/user/桌面/GMR/general_motion_retargeting/ik_configs/smplx_to_ne01.json
```

## 2. 原始 GMR 的基本原理

GMR 将每一帧人体动作转成若干人体关键刚体的世界位姿：

```text
human body name -> (position, orientation)
```

然后通过 IK 配置表，将人体刚体与机器人刚体对应起来，例如：

```text
pelvis    -> base_link
spine3    -> TORSO_LINK
left_foot -> left_toe_link
left_knee -> KNEE_PITCH_L_LINK
```

对于机器人配置 `q` 和机器人 frame 位姿函数 `T_i(q)`，每个 FrameTask 的目标是使机器人位姿接近人体目标位姿 `T_i*`。在微分 IK 中，一阶近似为：

```text
J_i(q) Delta_q = -alpha_i e_i(q)
```

其中：

- `e_i(q)` 是位置和姿态误差；
- `J_i(q)` 是任务雅可比；
- `Delta_q` 是当前 IK 步的广义坐标增量；
- `alpha_i` 是任务增益。

加权最小二乘形式为：

```text
min ||W_i (J_i Delta_q + alpha_i e_i)||^2
```

原始实现先求 `ik_match_table1`，再求 `ik_match_table2`。这种顺序调用并不是真正的层级 IK，因为第二次求解仍然可以破坏第一次已经达到的结果。

## 3. 总体改进结构

当前 NE01 重定向流程为：

```text
SMPL-X 输入
  -> 50 Hz 重采样
  -> 骨链递归缩放
  -> table1/table2 独立 offset
  -> 足部接触状态估计
  -> NE01 躯干和足部目标投影
  -> lexicographic QP
  -> 支撑脚四点最终接触投影
  -> 关节限位和速度限制后的 qpos
  -> PKL 输出
```

## 4. 修改文件总览

| 文件 | 修改内容 |
|---|---|
| `general_motion_retargeting/motion_retarget.py` | 层级 QP、50 Hz 时间步长、骨链缩放、table2 offset、接触状态机、足部目标、躯干投影、最终支撑修正 |
| `general_motion_retargeting/foot_support_task.py` | 新增 NE01 四点脚底支撑任务 |
| `general_motion_retargeting/ik_configs/smplx_to_ne01.json` | NE01 层级、接触参数、骨链父子关系、躯干竖直配置和 IK 权重/offset |
| `scripts/smplx_to_robot.py` | 单文件默认 50 Hz、传递 `motion_fps`、NE01 不再执行整段最低点平移 |
| `scripts/smplx_to_robot_dataset.py` | 批处理默认 50 Hz、传递 `motion_fps`、NE01 不再执行整段最低点平移 |
| `scripts/filter_nonflat_motions.py` | 新增非平地、楼梯、趴下、倒立动作预筛 |
| `general_motion_retargeting/utils/smpl.py` | SMPL-X 加载、长动作分块和重采样相关兼容修正 |
| `scripts/smpl_to_smplx.py` | SMPL 到 SMPL-X 转换的递归遍历和 beta 形状兼容修正 |

核心算法修改集中在前三个文件。

## 5. 50 Hz 时间基准

### 5.1 原问题

原代码在 IK 中使用 MuJoCo XML 的内部 timestep：

```text
dt = model.opt.timestep = 0.002 s
```

但输出动作是一帧一帧生成的。如果输出为 50 Hz，真实帧间隔应为：

```text
motion_dt = 1 / 50 = 0.02 s
```

使用 `0.002 s` 计算速度限制，会把每帧动作错误地解释成 500 Hz 动作，导致速度约束和输出数据时间尺度不一致。

### 5.2 当前实现

`GeneralMotionRetargeting` 新增：

```python
motion_fps: float = 50.0
self.motion_dt = 1.0 / self.motion_fps
```

所有帧间速度限制和 IK 积分都使用 `motion_dt`。MuJoCo 内部仿真 timestep 与动作输出 timestep 不再混用。

NE01 单文件和批处理入口默认 `--tgt_fps 50`，PKL 中的 `fps` 同样保存为 50。

## 6. Lexicographic/Hierarchical QP

### 6.1 优先级配置

NE01 当前优先级为：

```text
动态 Level 0：当前支撑脚的四点地面任务
Level 1：base_link、left_toe_link、right_toe_link
Level 2：TORSO_LINK
Level 3：左右髋、膝
Level 4：左右肩、肘、手腕
最后：上一帧姿态正则
```

`TORSO_LINK` 必须在上肢之前求解。原先 torso 放在肩肘之后时，上肢任务已经冻结了躯干相关自由度，torso 任务无法再纠正后仰。

### 6.2 每层 QP

每层使用 Mink 和 ProxQP 求解：

```text
min Delta_q  sum ||W_i (J_i Delta_q + alpha_i e_i)||^2

subject to:
    joint lower <= q + Delta_q <= joint upper
    velocity lower <= Delta_q / dt <= velocity upper
    A_prev Delta_q = 0
```

其中 `A_prev Delta_q = 0` 来自已经求解的高优先级任务。

### 6.3 为什么冻结任务使用 gain=0

如果直接把之前的 FrameTask 当成等式约束，Mink 会要求：

```text
J Delta_q = -alpha e
```

这表示低优先级求解时仍必须继续消除高优先级误差。在任务冗余或机器人自由度不足时，这很容易不可行。

当前实现会复制已经求解的 FrameTask，并设置：

```python
gain = 0.0
```

等式变成：

```text
J Delta_q = 0
```

其意义是低优先级只能在高优先级任务的一阶零空间中运动，不能恶化已经取得的高优先级结果。这是当前实现的 lexicographic/null-space 约束。

### 6.4 为什么使用 ProxQP

层级任务会产生冗余等式约束。DAQP 在 NE01 任务栈上出现过不可行或数值失败，ProxQP 对冗余约束更稳定，因此当前默认 solver 改为：

```python
solver="proxqp"
```

`hierarchy_failures` 用于记录层级求解失败次数，便于批处理质量检查。

### 6.5 table1/table2 去重

同一个机器人 frame 可能同时出现在 table1 和 table2。如果把两份 FrameTask 都作为硬约束，会造成重复约束。

当前实现每个机器人 frame 只选择一个主任务：

- 优先选择位置 cost 更完整的任务；
- table2 用来补充 table1 中缺失的位置约束；
- 避免同一 frame 被两份任务重复冻结。

## 7. table2 独立 offset 修复

原实现虽然读取了：

```text
pos_offsets2
rot_offsets2
```

但更新 table2 target 时仍然使用已经套用 table1 offset 的人体数据，因此 table2 自己的 offset 实际不生效。

当前处理方式是保留缩放后的基础人体数据：

```python
base_human_data
```

然后分别计算：

```text
table1_data = offset(base_human_data, offsets1)
table2_data = offset(base_human_data, offsets2)
```

这样两张表的位置和姿态 offset 完全独立。

## 8. 骨链递归缩放

### 8.1 原问题

原方法把所有人体关键点都相对 pelvis 单独缩放：

```text
p_j' = p_root' + s_j (p_j - p_root)
```

这种做法会让膝、脚、肘、腕的缩放互相独立，破坏相邻骨段的几何连续性。

### 8.2 当前方法

配置中新增 `human_scale_parents`，例如：

```text
left_foot <- left_knee <- left_hip <- pelvis
left_wrist <- left_elbow <- left_shoulder <- spine3 <- pelvis
```

缩放按骨链递归计算：

```text
p_j' = p_parent(j)' + s_j (p_j - p_parent(j))
```

这样大腿、小腿、上臂和前臂的缩放都以真实父节点为起点，不会因所有点都直接连接 pelvis 而扭曲骨架。

SMPL-X 的 heel、big toe、small toe 等辅助关节也会被保留，用于接触判断。

## 9. 足部接触状态估计

### 9.1 为什么不能直接复制人体脚姿态

人体脚包含前掌、足弓和脚趾的柔性变化；NE01 脚是刚体。支撑时直接复制人体 ankle/foot 姿态会造成：

- 前掌点接触；
- 后跟翘起；
- 四点高度不一致；
- 脚底穿模或浮空。

因此人体脚姿态只用于估计接触状态和 yaw，不能在支撑时完整复制 roll/pitch。

### 9.2 相对地面检测

原先失败的原因是直接使用经过缩放和 offset 后的绝对脚高，并设置固定 `0.055 m` 阈值。测试动作中左右脚最低高度分别约为 `0.0603 m` 和 `0.0551 m`，导致 240 帧全部被识别为 AIR。

当前使用 heel、big toe、small toe 的运行最低值估计人体地面：

```text
z_floor = min(z_floor, heel_z, big_toe_z, small_toe_z)
```

接触候选条件为：

```text
support_z <= z_floor + 0.035 m
and
foot_xy_speed <= 0.35 m/s
```

### 9.3 真正的连续帧迟滞

当前状态机使用：

```text
连续 2 帧满足条件 -> 进入 FLAT_CONTACT
连续 3 帧不满足条件 -> 退出到 AIR
```

这样能避免阈值附近的一帧抖动让接触状态频繁切换。

### 9.4 支撑时的目标姿态

支撑脚处理为：

```text
保留人体脚 yaw
roll = 0
pitch = 0
XY 固定为接触开始时的位置
Z 使用机器人真实脚底几何决定
```

当前运行阶段统一使用 FLAT_CONTACT，避免将人体脚趾滚动直接复制给刚性机器人脚。

## 10. NE01 四点脚底支撑任务

### 10.1 使用的碰撞点

NE01 MJCF 每只脚有四个球形碰撞几何：

```text
rear-left
rear-right
front-left
front-right
```

`FootSupportTask` 直接读取这些 geom，而不是只观察某个 body frame。

### 10.2 四点误差

对于第 `i` 个球形支撑 geom：

```text
e_i(q) = geom_center_z(q) - sphere_radius_i - ground_z
```

当 `e_i = 0` 时，该球形碰撞点的最低点正好位于地面。

四点误差向量为：

```text
e_foot = [e_rear_left, e_rear_right, e_front_left, e_front_right]
```

如果四个误差同时为零，则脚底不穿地、不浮空，并且刚性脚自然与地面平行。

### 10.3 雅可比

对每个 geom 中心调用 MuJoCo：

```python
mj.mj_jac(...)
```

取线速度雅可比的 z 行：

```text
J_i = J_position_i[z, :]
```

因此四点支撑任务可直接进入 Mink QP。

### 10.4 自动计算 toe frame 支撑高度

旧实现手填 `foot_support_frame_height=0.055`，但 IK 控制的是 `toe_link`，不是 ankle frame。

当前从 MJCF 自动计算：

```text
toe_frame_height = toe_link_z - sole_lowest_z
```

NE01 左右脚计算结果均约为：

```text
0.02001 m
```

这避免了 frame 选错导致的人为浮空。

### 10.5 最终接触投影

四点支撑任务先作为最高优先级 QP 任务求解。它没有冻结为后续层级的硬等式，因为四点高度和 toe frame 完整位姿之间存在强冗余，直接同时冻结会造成数值不可行。

所有跟踪层完成后，`_correct_support_height()` 再执行一次：

1. 只求解当前支撑脚的四点任务；
2. 重新计算四点实际高度；
3. 用四点误差中位数做最大 `50 mm` 的 root-z 残差校正。

这里的 root-z 修正不是原来的“整段动作按全局最低点统一抬升”，而是：

- 仅在检测到支撑脚的帧执行；
- 足底姿态和四点任务已经先求解；
- root-z 只处理最终的毫米级公共残差。

测试跑步动作中，支撑点高度约为：

```text
-0.50 mm 到 +0.54 mm
```

支撑脚四点最大高度差约为：

```text
1.04 mm
```

## 11. NE01 躯干后仰修复

### 11.1 原因

人体 `spine3` 是三轴姿态，但 NE01 躯干只有：

```text
WAIST_YAW_JOINT
TORSO_ROLL_JOINT
```

没有 torso pitch。直接把人体 spine3 三轴姿态压到 NE01 可用轴上，会把部分人体姿态映射成视觉上的持续后仰。

此外，旧层级顺序把 `TORSO_LINK` 放在肩、肘任务之后。上肢任务先被冻结后，torso 已经没有足够自由度进行修正。

### 11.2 当前处理

当 `upright_torso_orientation=true` 时：

```text
TORSO_LINK target yaw = human spine3 yaw
TORSO_LINK target pitch = 0
TORSO_LINK target roll = 0
```

也就是保留人体转身方向，但不强行复制 NE01 无法合理表达的 torso pitch/roll。

同时将 `TORSO_LINK` 提升到下肢和上肢任务之前。

测试中：

```text
TORSO_ROLL_JOINT：约 0.112 rad -> 约 0.009 rad
hierarchy_failures：0
```

## 12. 时序正则与速度限制

### 12.1 速度限制

NE01 自动启用 actuated joint velocity limits。速度限制只作用于受执行器控制的关节，不作用于 floating base。

速度使用真实帧间隔：

```text
v = Delta_q / 0.02
```

### 12.2 上一帧姿态正则

每帧保存：

```text
q_prev
q_prev2
```

当前实际使用 `q_prev` 创建低权重 PostureTask：

```text
cost = 1e-3
```

它位于所有关键点任务之后，只在剩余冗余自由度中抑制关节跳变，不覆盖高优先级跟踪结果。

`q_prev2` 已保留供后续显式加速度正则使用，当前版本尚未将二阶项直接写入 QP。

## 13. NE01 后处理变化

原入口脚本会计算整段动作所有机器人 body 的最低高度，然后给所有帧统一增加 root-z offset。

这种做法只能消除最深穿模，不能解决：

- 某些帧浮空；
- 点接触；
- 后跟翘起；
- 左右脚不同接触状态。

当前 NE01 跳过该整段最低点后处理：

```python
height_adjust = robot != "ne01"
```

NE01 的地面关系改由逐帧接触状态、四点任务和最终接触投影处理。其他机器人仍保留原行为。

## 14. 非平地和异常动作预筛

新增脚本：

```text
scripts/filter_nonflat_motions.py
```

### 14.1 非平地/楼梯

脚本从 heel、big toe、small toe、foot 计算低点，并在低速接触帧统计支撑高度分布：

```text
terrain_span = percentile95(contact_height) - percentile5(contact_height)
```

当：

```text
terrain_span > 0.10 m
```

标记为：

```text
nonflat_or_stairs
```

### 14.2 倒立

不使用 SMPL-X 关节局部旋转轴判断人体 up，因为该局部轴不一定对应人体语义上的竖直方向。

脚本使用位置向量：

```text
torso_axis = normalize(spine3_position - pelvis_position)
```

统计 `torso_axis_z < -0.2` 的帧比例。比例超过 10% 时标记：

```text
inverted_or_handstand
```

### 14.3 趴下/地面交互

当躯干接近水平且 pelvis 与脚底高度差较小时，认为可能存在趴下或地面交互：

```text
torso_axis_z < 0.45
and
pelvis_z - foot_floor_z < 0.45 m
```

满足帧比例超过 20% 时标记：

```text
prone_or_floor_interaction
```

### 14.4 预筛局限

该脚本是轻量运动学启发式，不读取场景、物体 mesh 或接触力，因此不能可靠识别所有物体交互，例如：

- 坐椅子；
- 扶墙；
- 搬箱子；
- 使用工具；
- 与另一个人交互。

这类语义仍需要动作名称过滤、场景元数据或人工复核。

## 15. 配置参数说明

当前 NE01 新增的主要参数为：

```json
{
  "output_fps": 50,
  "foot_contact_height_threshold": 0.035,
  "foot_contact_speed_threshold": 0.35,
  "foot_contact_enter_frames": 2,
  "foot_contact_exit_frames": 3,
  "foot_rocking_limit_rad": 0.20943951,
  "upright_torso_orientation": true
}
```

含义如下：

| 参数 | 含义 |
|---|---|
| `output_fps` | NE01 输出频率 |
| `foot_contact_height_threshold` | toe/heel 相对估计地面的接触高度阈值 |
| `foot_contact_speed_threshold` | 脚 XY 速度接触阈值 |
| `foot_contact_enter_frames` | 连续多少帧后确认接触 |
| `foot_contact_exit_frames` | 连续多少帧后释放接触 |
| `foot_rocking_limit_rad` | 保留的 heel/toe rocking 上限；当前平地支撑统一投影为 flat |
| `upright_torso_orientation` | 是否将 NE01 torso 目标投影为仅保留 yaw |

## 16. 运行命令

### 16.1 单文件转换

```bash
cd /home/user/桌面/GMR

conda run --no-capture-output -n gmr python \
  scripts/smplx_to_robot.py \
  --smplx_file <输入.npz> \
  --robot ne01 \
  --save_path <输出.pkl> \
  --headless \
  --tgt_fps 50
```

### 16.2 批处理

```bash
cd /home/user/桌面/GMR

conda run --no-capture-output -n gmr python \
  scripts/smplx_to_robot_dataset.py \
  --robot ne01 \
  --src_folder <SMPL-X 输入目录> \
  --tgt_folder <PKL 输出目录> \
  --tgt_fps 50 \
  --num_cpus 2 \
  --device cuda:0
```

### 16.3 可视化

```bash
cd /home/user/桌面/GMR

conda run --no-capture-output -n gmr python \
  scripts/vis_robot_motion.py \
  --robot ne01 \
  --robot_motion_path <动作.pkl>
```

### 16.4 数据预筛

```bash
cd /home/user/桌面/GMR

conda run --no-capture-output -n gmr python \
  scripts/filter_nonflat_motions.py \
  --src_folder <原始 SMPL-X 目录> \
  --report <筛选报告.csv> \
  --accepted_folder <通过筛选的数据目录> \
  --body_model_folder /home/user/桌面/GMR/assets/body_models
```

`accepted_folder` 可省略。省略时脚本只生成 CSV 报告，不复制文件。

## 17. Legacy 回退

`GeneralMotionRetargeting` 保留：

```python
legacy_mode=True
```

启用后恢复 table1 后 table2 的旧式顺序求解，便于 A/B 对比。

注意：`legacy_mode` 只切换 IK 求解流程，不会自动恢复旧版 NE01 JSON、旧骨链缩放或旧入口后处理。若要做严格源码基线，必须使用独立的原始 Git worktree 和原始配置。

## 18. 已验证结果

跑步转站立样例：

```text
输出频率：50 Hz
帧数：240
层级 QP 失败：0
NaN/Inf：无
左脚支撑帧：176
右脚支撑帧：170
支撑点高度范围：约 -0.50 mm 到 +0.54 mm
支撑脚四点最大高度差：约 1.04 mm
TORSO_ROLL_JOINT：约 0.112 rad 降到约 0.009 rad
```

## 19. 当前已知限制

1. 当前层级 IK 是逐层一阶 null-space 形式，不是一次性求完整轨迹的全局时域优化。
2. 接触地面假设为水平平面，不支持显式楼梯或斜坡地形。
3. 没有引入动力学、质心稳定裕度、接触力或 GRF。
4. `q_prev2` 尚未形成显式加速度二阶代价。
5. 自碰撞、机器人与环境碰撞尚未作为 QP 约束加入。
6. 非平地/物体交互筛选是启发式，不能代替场景标注。
7. 四点任务当前依赖 NE01 MJCF 中固定命名的球形脚底 collision geom。

这些限制不影响当前平地 SMPL-X 到 NE01 PKL 的基本运行，但在楼梯、斜坡、趴地、倒立和物体交互动作中不应直接使用输出结果。
