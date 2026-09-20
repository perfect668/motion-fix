# stairs_0037：有限表面接触修复

基于 `wholebody-v4-cleanup` 的 `ef5233ada2266ed1d3d68a90868ba0b430d69655`。
本次保留 V4/V3 求解核心、目标权重、关节限制、碰撞几何和场景尺寸。

## 交付包证据

原输出共 401 帧、50 Hz，记录的 QP 失败数为 0。根部最大单帧平移为
0.1984865411 m（零基帧号 180）。这不是修复后的运动指标。

对每个网格接触，独立计算人体点与 `surface_point_solver` 的欧氏距离：

| 通道 | 原网格接触记录数 | 超过 0.05 m 的记录数 | 最大距离 / m |
| --- | ---: | ---: | ---: |
| left_heel | 9 | 0 | 0.047049 |
| right_heel | 28 | 11 | 0.469882 |
| left_toe | 29 | 0 | 0.048116 |
| right_toe | 31 | 0 | 0.047036 |
| left_knee | 98 | 98 | 1.213660 |
| right_knee | 58 | 58 | 0.613281 |
| left_shin | 9 | 6 | 0.092508 |
| right_shin | 42 | 13 | 0.279738 |

旧实现只检查到三角形所在平面的法向距离，因而会把远离三角形边界的点
当成接触；滞回也使用同一个错误距离。另有零基帧 100–105 的右脚，网格
覆盖后的脚跟/脚尖满足旧分数下的同面条件，但 `flat_foot` 仍保留地板阶段
的 0。输出的 `terrain_primitives` 只有地板，楼梯来自网格补充逻辑；因此
本补丁优先处理实际执行的网格路径，没有扩展修改盒体接触调度。

## 修改

1. 网格接触进入条件、置信度和滞回均使用到有限三角形的完整距离。
   旧三角形超过接触距离时不能继续锁定。
2. 新网格表面不再继承不相关地板表面的置信度。
3. 网格覆盖后重新计算平脚条件。用物体身份、法向与平面间距判断共面，
   允许同一踏面上不同三角形组成的支撑，拒绝不同高度台阶。
4. 接触锚点在换到不同物体/平面时重新建立；同一平面的三角形接缝保留锚点。
5. 只有脚跟、脚尖均无接触时才启用腾空朝向；脚尖支撑不等同于腾空。
6. 增加 7 个回归测试和 `scripts/audit_contact_schedule.py`，审计不依赖
   原输出中可能具有误导性的 signed_distance。

## 实际完成的验证及边界

第一轮仓库测试结果：60 passed（包含新增 7 项）；另有一条 PyTorch JIT 弃用提示。
验证环境为 Python 3.12、MuJoCo 3.3.7、Mink 0.0.13。本补丁未修改依赖版本。

使用交付包保存的人体接触代理点、USD 与物体变换重放接触阶段，不进行
SMPL-X 重建，也不重新求解 qpos。重建网格接触点与原记录最大差异约
1.23e-14 m。旧代码重放得到的所有通道状态及表面编号与交付包一致，
不一致数为 0。

修复后：原来 186 个超距网格接触全部消失；原来距离合格的 118 个
帧—通道接触全部保留，没有新增或丢失这些接触。重新计算置信度后，
右脚有 4 帧满足平脚启用条件，不沿用旧分数强行启用全部 6 帧。

这是接触阶段的验证，不代表已经证明根部跳变、机器人支撑间隙或整段
动作质量改善。交付包缺少 SMPL-X 模型和原碰撞缓存，而且 README 使用
仓库中不存在的 `scripts/grail_retarget_scene.py`，不能把当前仓库的完整
重跑宣称为原执行环境的精确复现。下一步应在原环境用相同输入对照重跑。
脚步支撑关系修正后仍可能存在骨架尺寸、源动作接触缺失、根部连续性等问题。

## 本地复跑

在本仓库根目录、原有 gmr 环境中，设置交付包解压后的目录。SMPL-X 模型
仍按仓库约定位于 `assets/body_models/`。显式指定 USD，避免原机器绝对路径。

```bash
BUNDLE='/你的路径/交付包_grail_stairs_0037_v4'
conda run --no-capture-output -n gmr python scripts/grail_to_robot_wholebody_v4.py \
  --motion "$BUNDLE/input/grail_stairs_0037_input.pkl" \
  --object_asset "$BUNDLE/input/stairs_0037.usd" \
  --save_path outputs/grail_stairs_0037_contact_fixed.pkl \
  --tgt_fps 50

conda run --no-capture-output -n gmr python scripts/audit_contact_schedule.py \
  --motion outputs/grail_stairs_0037_contact_fixed.pkl

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

完整动作验收应比较同配置的修复前后输出：接触时脚底间隙、切向滑动、
碰撞穿透、根部及关节增量、QP 失败与上楼完成情况；同时复跑平地和快速
动作。不要仅凭“接触检测无超距”就判定整段动作合格。

## 第二轮：竖脚及脚背朝向的反馈（2026-09-20）

用户新截图显示前脚近乎竖直靠在台阶边缘。截图不能确定接触相位、帧号，
也不能证实脚背与碰撞表面的确切距离；因此尚未将某个故障帧精确复现。
但 V4 的 `FootFrameTask` 有两个可独立验证的问题：

- 旧法向误差为 `T.T @ normal`，脚底法向朝上、朝下都可得到零误差。
  绕脚前向轴翻转 180 度时，旧的前向误差也为零。侧翻 90 度时，
  法向项的纠偏梯度退化。只提高这个目标的权重无法消除歧义。
- `flat_foot=0` 且至少一个脚部通道接触时，朝向任务的激活值为零。
  在脚尖、脚跟单独支撑阶段，位置目标可能存在而朝向没有目标。

修复将 V4 足部朝向改为 SO(3) 旋转对数误差，使用相匹配且小角度稳定的
解析雅可比。整脚支撑使用支撑面法向和人体脚前向构成目标旋转；非整脚
支撑使用人体脚姿态，以原腾空阶段同等强度的弱目标保持朝向。单脚尖
接触允许蹬离倾斜，腾空允许翻转，不全程压平脚底。V3 的旧法向任务、
求解器权重、碰撞尺寸和位置接触任务均未修改。

新增 9 个用例，包括实际调用 Mink/DAQP 将 90/180 度脚姿态纠正至目标、
解析雅可比有限差分验证、斜坡与部分支撑姿态保持，以及腾空倒置动作保持。
这些是受控模型验证，不是该楼梯完整动作的效果验收。
第二轮完整仓库测试结果为 69 passed，仍只有一条 PyTorch JIT 弃用提示。

新输出每帧增加 `terrain_diagnostics[i]["foot_orientation"]`，按左右脚记录：

- `mode`：flat、partial、airborne；
- `activation`：朝向任务的激活值；
- `sole_normal_world`、`target_normal_world`：实际与目标足底法向；
- `sole_target_angle_deg`、`frame_error_deg`：法向夹角和完整姿态误差。

后续应提供本轮运行生成的诊断 PKL 与截图帧号；旧交付包不是新截图对应
的输出。仍需用完整人体模型重跑，检查位置接触是否成立以及姿态误差是否
下降，不能仅凭上述单脚测试宣称整段动作已修复。


## 第三轮：actual 输出确认的接触相位错位（2026-09-20）

对用户提供的 `grail_stairs_0037_ac07b5c_actual.pkl` 逐帧检查后，问题可以
进一步定位：脚部网格接触一旦成立时，记录的楼梯表面法向均为向上的踏面
法向；截图中的“脚侧/脚背顶住立面”并不是 heel/toe 被错误绑定到竖直
riser，而是 **碰撞约束已经接触楼梯、脚底支撑任务却尚未激活**。

典型的零基帧 272：右脚 heel/toe 仍是 floor/NONE，足部朝向模式为
airborne、激活仅 0.15；实际足底法向和目标法向相差 84.25 度。同时右脚
踝碰撞体到楼梯碰撞壳的最小距离约 0.004 m，前脚掌碰撞体到同一楼梯约
0.014 m。下一帧 273 右脚尖才进入楼梯 mesh 接触。也就是说，MuJoCo
碰撞先把脚挡在台阶前，而弱朝向目标不足以阻止求解器把脚旋到近乎竖直。

本轮修正接触/碰撞的相位关系：

1. heel/toe 支撑改成显式踏面查询：忽略三角形绕序，竖直投影覆盖时选择
   脚下最高且可达的向上表面，竖直 riser 不能成为足底支撑面。
2. 足部硬接触捕获距离设为 0.06 m，与场景碰撞开始活跃的距离同量级；
   仍然使用有限三角形距离，不能把无限平面当接触。
3. 在 0.12 m 内增加 support-approach 元数据。这个阶段不创建位置锚点，
   因而不会把摆动脚提前粘在台阶上，只逐渐加强有向 SO(3) 的脚姿态目标，
   最高激活 0.65。
4. 增加反向绕序踏面、重叠台阶最高可达面、预接触窗口、捕获距离以及
   预接触朝向增益的回归测试。

本轮修改仍需在原 GRAIL/SMPL-X 运行环境完整重跑。actual PKL 足够定位
故障相位，但不包含完整源资产和人体模型依赖，不能在当前工具环境重算
401 帧并据此宣称视觉效果已经通过。


## 第四轮：terrain-native 重构（2026-09-20）

第三轮继续证明：只在 V4 的 heel/toe 最近表面查询、FootFrameTask 权重和
碰撞激活阈值上做局部修复，不能从结构上排除“脚侧顶住台阶立面”的局部解。
本轮不再让机器人当前姿态决定“它正在踩哪个面”，而是把 HoloSoMo /
OmniRetarget 的 climbing 思路放到主干：

1. 从最终对齐后的真实楼梯 mesh 提取连通、近似共面的向上 support patch，
   并把解析 floor 放进同一个 patch map。三角面绕序不影响支撑面语义。
2. 在机器人优化开始前，用整段 source human + source terrain 推断左右脚的
   stance / swing / free 序列。每个 stance episode 固定一个 support patch；
   swing episode 提前知道下一次 landing patch，并生成脚底 clearance corridor。
3. 场景 interaction pool 不再只是全局最近点。约 75% 的楼梯场景采样预算
   优先给可支撑的上表面，同时保留一部分普通 mesh 点描述 riser / edge。
   Interaction Laplacian 使用按边长衰减的稀疏权重，减少远处环境边稀释。
4. NE01 每只脚使用 4 个现有 sole guard site 作为一个刚性脚底 patch。
   stance 时四点共同受同一支撑平面约束，并在 episode 开始后做 robot-relative
   tangential sticking；不再依赖单独的 FootFrameTask 去猜脚底方向。
5. swing 时四个 sole 点都受到未来 landing patch 的高度走廊约束。MuJoCo
   scene collision 仍保留，但只作为不可穿透的可行性边界，不再承担接触规划。
6. 如果整段 source/scene 分析完全没有得到非 floor 的 stance episode，
   入口直接报错，不允许静默退回旧的“碰撞先顶住脚、随后再补接触”路径。

新增核心文件：

- `general_motion_retargeting/terrain_native_geometry.py`
- `general_motion_retargeting/terrain_native_planner.py`
- `general_motion_retargeting/terrain_native_tasks.py`
- `general_motion_retargeting/wholebody_terrain_native.py`

GRAIL V4 入口 `scripts/grail_to_robot_wholebody_v4.py` 已直接切换到
`TerrainNativeRetargeter`。原 V4 的多格式适配、场景资产、MuJoCo 组合模型、
CoACD、输出格式与诊断外壳继续复用；旧 reactive foot contact 不再控制
GRAIL 楼梯脚部。

新增 `tests/test_terrain_native_planner.py` 覆盖：反向 triangle winding、
floor -> swing -> stair 的整段支撑规划，以及 contact schedule 必须来自预先
规划的 support patch 而不是机器人侧最近表面。

当前可在无 MuJoCo/Mink 完整环境下验证的纯几何/规划逻辑已用合成一级台阶
检查通过。完整仓库测试和 `grail_stairs_0037` 的 401 帧重新求解仍必须在
原 gmr + SMPL-X + 场景资产环境执行；在得到新的 PKL 前，不把视觉效果宣称
为已通过。

### 本轮验收重点

新输出应重点检查：

- `terrain_diagnostics[i]["terrain_native"][side]["mode"]`
- stance: `patch_id`, `normal_spread`, `anchor_error`
- swing: `landing_patch_id`, `minimum_clearance`
- `terrain_native_hard_rows`
- `minimum_scene_distance` / `maximum_penetration`

楼梯验收不再以“某个 heel/toe 最近三角形是否正确”为主，而以三条结构性
条件为准：stance 四个 sole 点必须属于同一 support patch；swing sole 不得
被 riser 截获；support transition 前后 q/root 不得发生异常跳变。
