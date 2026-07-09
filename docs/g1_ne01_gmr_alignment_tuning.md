# G1 / NE01 GMR Alignment And Tuning Notes

本文档记录 `smplx_to_robot_dataset.py` 通过 GMR 将 SMPL-X 重定向到 `unitree_g1` 和 `ne01` 时，当前需要关注的机器人运动学几何尺寸、NE01 参数调试结论，以及后续风险点。

当前优化边界：

- 不在重定向后的 robot-motion `.pkl` 上做数据后处理。
- 通过 GMR 算法内部语义对齐、IK task 权重、IK solver 约束来改善 NE01。
- `human_scale_table` 保持每个 body 一个 scalar，不引入 per-axis scale。
- 以 G1 已验证的重定向效果为参照，让 NE01 在运动学语义和任务权重上尽量靠近 G1，同时保留 NE01 自身 DoF/限位差异。

## 计算口径

- 单位均为 `m`。
- 数据来自当前 MuJoCo XML 的 `qpos0` 构型：
  - G1: `assets/unitree_g1/g1_mocap_29dof.xml`
  - NE01: `assets/ne01/ne01.xml`
- GMR task frame 来自当前 IK config：
  - G1: `general_motion_retargeting/ik_configs/smplx_to_g1.json`
  - NE01: `general_motion_retargeting/ik_configs/smplx_to_ne01.json`
- 本文只讨论 GMR 算法/IK task 所需的几何量，不使用 mesh 外形尺寸、惯量中心或 XML body tree 的折线链长作为主尺度。
- `human_scale_table` 保持当前各项 scalar，不引入 per-axis scale。

GMR 当前核心目标是让 robot task frame 跟踪缩放后的 SMPL-X joint：

```text
human_target_i = pelvis_pos + scale_i * (human_joint_i - pelvis_pos) + pos_offset_i
robot_task_i   tracks human_target_i
```

因此，机器人尺寸需要与 SMPL-X 的 `pelvis -> joint` 或 `joint -> joint` 语义对齐。

## Root 语义

G1 和 NE01 的 MuJoCo freejoint root 不是同一语义：

| robot | MuJoCo/GMR root | 语义 |
|---|---|---|
| G1 | `pelvis` | 接近人体/SMPL-X pelvis 语义 |
| NE01 | `base_link` | 更接近髋俯仰安装平面，不等同于人体 pelvis |

NE01 已增加 GMR 内部 root 语义偏移：

```json
"robot_root_to_human_root_offset": [0.0235979, 0.0, 0.1071685]
```

定义：

```text
virtual_pelvis_ne01 = base_link + [0.0235979, 0, 0.1071685]
base_link_target    = human_pelvis_target - R_root * offset
```

这个 offset 只移动 NE01 root task 目标，不整体平移 hip/knee/foot 等人体目标。这样 IK 看到的是：`base_link` 位于语义 pelvis 的下方，而不是直接把 `base_link` 当 SMPL-X pelvis。

offset 来源于 hip task 对齐：

```text
G1   pelvis -> hip_task = [0.000000, +/-0.116452, -0.133165]
NE01 base   -> hip_task = [0.023598, +/-0.126556, -0.025996]
offset = [0.023598, 0.0, 0.107169]
```

## 当前 GMR Task Frame

| SMPL-X task | G1 robot frame | NE01 robot frame |
|---|---|---|
| `pelvis` | `pelvis` | `base_link` |
| `left_hip` / `right_hip` | `left_hip_roll_link` / `right_hip_roll_link` | `HIP_ROLL_L_LINK` / `HIP_ROLL_R_LINK` |
| `left_knee` / `right_knee` | `left_knee_link` / `right_knee_link` | `KNEE_PITCH_L_LINK` / `KNEE_PITCH_R_LINK` |
| `left_foot` / `right_foot` | `left_toe_link` / `right_toe_link` | `left_toe_link` / `right_toe_link` |
| `spine3` | `torso_link` | `TORSO_LINK` |
| `left_shoulder` / `right_shoulder` | `left_shoulder_yaw_link` / `right_shoulder_yaw_link` | `SHOULDER_YAW_L_LINK` / `SHOULDER_YAW_R_LINK` |
| `left_elbow` / `right_elbow` | `left_elbow_link` / `right_elbow_link` | `ELBOW_PITCH_L_LINK` / `ELBOW_PITCH_R_LINK` |
| `left_wrist` / `right_wrist` | `left_wrist_yaw_link` / `right_wrist_yaw_link` | `HAND_YAW_L_LINK` / `HAND_YAW_R_LINK` |

当前 NE01 scalar 对齐参数：

| body group | scale |
|---|---:|
| `pelvis`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_foot`, `right_foot` | 0.86 |
| `spine3` | 0.90 |
| `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist` | 0.80 |

当前 NE01 下肢 foot position weight 已与 G1 对齐为 `100`；foot offset 暂未额外调整。

## 原始 Root Frame 下的 Task 位置

这些数值是 robot task frame 相对各自 MuJoCo root 的位置。注意 NE01 这里仍是 `base_link` 口径，不能直接当人体 pelvis 口径。

| task | G1 xyz | NE01 xyz |
|---|---:|---:|
| `left_hip` | `[0.000000, 0.116452, -0.133165]` | `[0.023598, 0.126556, -0.025996]` |
| `right_hip` | `[0.000000, -0.116452, -0.133165]` | `[0.023598, -0.126556, -0.025996]` |
| `left_knee` | `[-0.000002, 0.118601, -0.439296]` | `[0.066113, 0.126556, -0.314765]` |
| `right_knee` | `[-0.000002, -0.118601, -0.439296]` | `[0.066113, -0.126556, -0.314765]` |
| `left_foot` | `[0.099998, 0.118506, -0.776864]` | `[0.100020, 0.126556, -0.634362]` |
| `right_foot` | `[0.099998, -0.118506, -0.776864]` | `[0.100020, -0.126556, -0.634362]` |

`ne01.xml` 更新后，NE01 左右 toe task 已对称。

## Virtual Pelvis 对齐后的腿部尺寸

NE01 下肢应使用 `virtual_pelvis_ne01` 口径与 G1/SMPL-X pelvis 语义比较：

```text
NE01 virtual_pelvis -> task = NE01 base_link -> task - robot_root_to_human_root_offset
```

| quantity | G1 | NE01 virtual pelvis | NE01/G1 |
|---|---:|---:|---:|
| `pelvis -> left_hip` dist | 0.176901 | 0.183709 | 1.038 |
| `pelvis -> left_knee` dist | 0.455024 | 0.442552 | 0.973 |
| `pelvis -> left_foot` dist | 0.792187 | 0.756124 | 0.954 |
| `left_hip -> left_knee` dist | 0.306138 | 0.291883 | 0.953 |
| `left_knee -> left_foot` dist | 0.352068 | 0.321390 | 0.913 |
| `left_hip -> left_foot` dist | 0.651423 | 0.613147 | 0.941 |

按 Z 向高度看：

| quantity | G1 z | NE01 virtual pelvis z | NE01/G1 abs(z) |
|---|---:|---:|---:|
| `pelvis -> left_hip` | -0.133165 | -0.133164 | 1.000 |
| `pelvis -> left_knee` | -0.439296 | -0.421934 | 0.960 |
| `pelvis -> left_foot` | -0.776864 | -0.741530 | 0.955 |

结论：修正 root 语义后，NE01 不是原始 `base_link -> foot` 口径下看起来那样“腿短很多”。对齐到 virtual pelvis 后，NE01 toe task 的 Z 向高度约为 G1 的 `95.5%`，差约 `3.5 cm`。

## 左右宽度

| quantity | G1 | NE01 | NE01/G1 |
|---|---:|---:|---:|
| hip width | 0.232904 | 0.253112 | 1.087 |
| knee width | 0.237202 | 0.253112 | 1.067 |
| foot width | 0.237013 | 0.253112 | 1.068 |

结论：NE01 下肢 Y 向宽度比 G1 大约 `6.7% ~ 8.7%`。由于当前 `human_scale_table` 仍是每个 body 一个 scalar，不能同时独立匹配 Y 向宽度和 Z 向高度；当前优化选择先用 root 语义 offset 修正最大误差来源。

## 足底接触几何

foot task 是 toe body，不等于实际足底接触面。脚底悬空判断需要额外看 toe 到 contact surface 的偏移。

| robot | side | contact bbox center rel toe | bbox size x/y | lowest surface z rel toe |
|---|---|---:|---:|---:|
| G1 | left/right | `[-0.065, 0.000, -0.010]` | `0.170 / 0.060` | -0.015 |
| NE01 | left/right | `[-0.065, 0.000, -0.015]` | `0.170 / 0.060` | -0.020 |

结论：NE01 的 foot contact surface 比 toe task 低约 `2.0 cm`，G1 低约 `1.5 cm`。因此同样的 toe 高度下，NE01 实际足底比 G1 更低约 `0.5 cm`。脚端悬空判断不能只比较 toe task，需要考虑这个 toe-to-sole 偏移。

## 上肢和躯干注意事项

当前配置中：

```text
G1   spine3 -> torso_link
NE01 spine3 -> TORSO_LINK
```

在 NE01 virtual pelvis 口径下，`TORSO_LINK` 的 Z 位置约为 `0.044331`，与 G1 `pelvis -> torso_link` 的 `0.044000` 基本对齐。`WAIST_YAW` 会落在 virtual pelvis 下方，不适合直接作为“人体 spine3 高度”的几何参照。

NE01 手臂自由度少于 G1，因此上肢 IK 权重只做有限幅度向 G1 靠拢。当前上肢姿态权重不是统一值，而是按 DoF 可表达性分层：

| task group | G1 table1 rot | NE01 table1 rot | G1 table2 rot | NE01 table2 rot | NE01 table2 pos |
|---|---:|---:|---:|---:|---:|
| shoulder | 10 | 7 | 5 | 4 | 10 |
| elbow | 10 | 2 | 5 | 1 | 10 |
| wrist/hand | 10 | 0 | 5 | 0 | 10 |

上肢 task 段当前尺寸：

| quantity | G1 | NE01 | NE01/G1 |
|---|---:|---:|---:|
| shoulder width | 0.293603 | 0.309853 | 1.055 |
| wrist width | 0.297313 | 0.309853 | 1.042 |
| shoulder -> elbow | 0.082050 | 0.109171 | 1.331 |
| elbow -> wrist | 0.184281 | 0.178006 | 0.966 |

## GMR 调参含义

1. NE01 与人体/G1 root 对齐时，应使用 `robot_root_to_human_root_offset` 修正 `base_link` 和语义 pelvis 的高度/前后差。
2. 下肢高度比较应使用 NE01 virtual pelvis 口径，不应直接用 `base_link -> foot` 与 G1 `pelvis -> foot` 比。
3. 当前保持 `human_scale_table` scalar，不做 per-axis scale；因此 Y 向宽度和 Z 向高度的残余差异只能通过 task 权重、offset 和 root 语义对齐来折中。
4. `spine3` 应映射到 `TORSO_LINK`，不要映射到 `WAIST_YAW`。
5. foot/toe task 与实际足底接触面存在固定偏移；脚底悬空评估需要使用 toe-to-sole 偏移。
6. 这些修正属于 GMR/IK 配置与算法内部语义对齐，不是在重定向后对导出的轨迹做后处理。

## 当前 NE01 参数调试结论

### 几何对齐

NE01 已按 G1 语义完成以下对齐：

- root 语义：`base_link` 不直接等同于人体 pelvis，通过 `robot_root_to_human_root_offset = [0.0235979, 0.0, 0.1071685]` 建立 virtual pelvis。
- 下肢尺度：当前 `pelvis/hip/knee/foot = 0.86`，结合 virtual pelvis 后，NE01 `pelvis -> toe task` 的 Z 向高度约为 G1 的 `95.5%`。
- 躯干尺度：`spine3 = 0.90`，并使用 `TORSO_LINK` 作为 spine3 task frame。
- 上肢尺度：`shoulder/elbow/wrist = 0.80`，在保持手部位置语义的同时避免过度放大 NE01 上肢目标。
- 足端位置权重：NE01 foot position weight 已从更强的早期设置回到 G1 风格的 `100`。
- foot offset 暂未继续调整，避免把 toe task 与足底接触几何混在一起。

当前效果：NE01 重定向动作整体已经与 G1 比较相似，动作连续性和“丝滑程度”较早期参数明显改善。

### 上肢 IK

NE01 每臂只有：

```text
shoulder pitch / roll / yaw + elbow pitch + hand yaw
```

G1 每臂额外有 wrist roll / pitch / yaw。NE01 的 `HAND_YAW_*` 更接近 G1 wrist roll，不能表达完整 wrist pitch/yaw。因此不能简单把 G1 的 wrist 姿态跟踪权重搬到 NE01。

当前策略：

- 保留 hand/wrist position tracking，table2 位置权重为 `10`。
- 关闭 `HAND_YAW_L_LINK` / `HAND_YAW_R_LINK` 对 `left_wrist/right_wrist` 的姿态跟踪，orientation weight 为 `0`。
- 将 elbow orientation 作为弱约束：table1/table2 为 `2/1`。
- shoulder orientation 保持中等偏强约束：table1/table2 为 `7/4`。

调试结论：

- 对手臂动作较多的动作，G1 会大量使用 wrist pitch/yaw；NE01 缺失这些 DoF 时，如果强追 wrist 姿态，会把不可达误差转嫁给 shoulder/elbow。
- 降低/关闭 hand orientation 后，代表动作中 shoulder 限位邻域占比显著下降，hand yaw 不再长期贴限位，同时 hand position error 没有恶化。
- 在 hand orientation 保持 `0/0`、elbow orientation 保持 `2/1` 的前提下，对 shoulder orientation 做 sweep，`7/4` 是当前 10 个上肢代表动作上的最佳点。相对 `5/3`，shoulder/elbow 平均姿态误差下降约 `12.8%`，joint limit occupancy 基本不变，arm `p95 |dq|` 仅增加约 `0.8%`，hand position p95 error 下降约 `10.3%`。
- 这属于 IK task 语义调整，不是导出数据后的平滑或裁剪。

### 腿部与足端 IK

当前参数下，NE01 重定向结果的整体观感已经接近 G1：动作比较丝滑，足端接触也基本可接受。这里的“足端接触可接受”包括两层含义：

- foot/toe 到地面的高度距离没有明显系统性悬空。
- 脚底板姿态整体能跟随动作语义，没有明显长期翻脚或脚掌姿态错误。

因此当前不建议继续优先调整 foot offset 或大幅降低 foot position/orientation 权重。足端 task 仍保持 G1 风格：

```text
table1 foot: position/orientation = 100/10
table2 foot: position/orientation = 100/50
```

仍需关注的问题是：个别动作中仍会出现关节或 root 突变。当前判断这不是单纯的足端接触几何问题，也不是后处理可以正确解决的问题，而是 GMR IK 多解/跨分支导致的。

## 当前残余问题

当前残余跳变主要来自 IK 多解/跨分支：

- **腿部多解**：足端 position/orientation、hip/knee task、root task 同时作用时，IK 可能在相邻帧选择不同的髋 roll/yaw、膝、踝组合。
- **手臂多解**：NE01 缺少 G1 的 wrist pitch/yaw；即使关闭 hand orientation，shoulder/elbow/hand position 仍可能在大幅摆臂、交叉手臂、抬手绕身等动作中形成多组可行解。

这类问题的关键不是“某个单独权重不对”，而是缺少显式的跨帧解连续性偏好。当前 GMR 每帧以上一帧 configuration 为初值，但目标冲突或多解区域足够强时，仅靠初值仍不能保证选择同一 IK 分支。

## 当前建议配置策略

当前参数总体可作为下一版 NE01 重定向的基线。建议暂时保持：

- 下肢/root/torso 几何对齐和 scale 不再大幅调整。
- foot offset 暂不调整。
- foot position/orientation 权重保持当前 G1 风格设置。
- hand orientation 保持 `0/0`。
- elbow orientation 保持 `2/1`。
- shoulder orientation 保持当前 sweep 后的 `7/4`。

对于残余关节/root 突变，不建议把低速度限制作为主要方案。原因：

- 速度限制过低会削弱跳跃、快速摆腿、挥手、舞蹈等动态动作，导致动作发钝、跟踪滞后。
- 速度限制过高又很难真正限制多解切换时的瞬时跳变。
- 当前 `VelocityLimit` 主要约束 actuated joints，不直接解决 freejoint root 的高频抖动或突变。

更合理的后续方向仍应放在 GMR/IK 求解层：

1. **IK 分支连续性代价**
   在 IK QP 中加入相对上一帧 joint displacement 的软惩罚，优先选择与上一帧同一分支的解，而不是只依赖上一帧初值。

2. **多解区域的 task 自适应权重**
   当某些 task 误差、关节限位距离或目标速度表明进入多解/冲突区域时，临时降低容易诱发跨解的姿态项，例如 foot orientation 或上肢局部 orientation，同时保留关键 position tracking。

3. **关节限位邻域的稳定化**
   在接近限位时提高限位回避或连续性权重，避免 IK 为了追 task 突然切换到另一组髋/膝/踝或肩/肘解。

4. **root 连续性约束**
   root/freejoint 需要单独的速度、加速度或 tracking smoothness 约束。仅限制 actuated joints 不能完全解决 root 高频突变。

## 风险点

1. **IK 多解/跨分支**
   当前主要残余问题。腿部和手臂都存在多解，明显跳变多来自相邻帧切换 IK 分支。

2. **速度限制不是充分解**
   velocity limit 可以压制部分数值尖峰，但阈值难选。低阈值损伤动态动作，高阈值无法稳定限制跨解跳变。

3. **foot orientation 与足端接触的平衡**
   当前足端接触可接受，不宜为了个别跳变过度降低 foot orientation 或改 foot offset。否则可能引入脚掌姿态错误或落脚漂移。

4. **root 突变仍需单独关注**
   root task 由 pelvis target、足端约束和全身姿态共同影响。即使 joint motion 变平滑，root 仍可能因为目标冲突出现高频抖动。

5. **human_scale_table scalar 的表达限制**
   当前每个 body 只有一个 scalar，无法同时完美匹配 NE01/G1 的 Y 向宽度、Z 向高度和前后 offset。后续若需要更细对齐，需要引入更复杂的 GMR 几何语义，而不是简单 per-axis 后处理。

6. **NE01 上肢 DoF 缺失**
   hand/wrist 姿态不能强行追 G1/SMPL-X 完整腕部姿态。否则会重新引入 shoulder/elbow 限位和僵硬问题。

7. **评估不能只看可视化主观感受**
   建议每轮至少统计 max/p95 `|dq|`、root 线/角速度与加速度、关节限位邻域占比、foot task error、toe-to-sole 高度偏移。
