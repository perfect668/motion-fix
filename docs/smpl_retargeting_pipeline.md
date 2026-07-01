# SMPL/SMPL-X 到机器人动作与筛选流程

本文梳理从 Sonic SMPL 或 AMASS 等 SMPL/SMPL-X 数据源，到 GMR 机器人动作 `pkl`，再到训练数据筛选的完整链路。目标是让数据生成过程可复现、可追溯、可验证。

## 总览

主链路按数据格式分为五段：

```text
源数据
  Sonic SMPL .pkl/.joblib/.npz
  或 AMASS/GMR-compatible SMPL .npz
        |
        v
GMR SMPL .npz
  poses, trans, betas, mocap_framerate, gender
        |
        v
SMPL-X .npz
  root_orient, pose_body, trans, betas, mocap_frame_rate, gender
        |
        v
Robot motion .pkl
  fps, root_pos, root_rot(xyzw), dof_pos, local_body_pos, link_body_list
        |
        v
Filter report / selected robot-motion folders
  strict pass/tag/reject
  relaxed keep/pass/tag/reject
```

对应脚本：

| 阶段 | 脚本 | 作用 |
| --- | --- | --- |
| Sonic 源数据筛选 | `scripts/select_sonic_smpl_subset.py` | 从巨大 Sonic SMPL 目录筛选 10-12h 源数据子集，输出软链接和 manifest。 |
| Sonic -> GMR SMPL | `scripts/gear_sonic_smpl_to_gmr_smpl.py` | 读取 Sonic SMPL，统一为 `smpl_to_smplx.py` 可处理的 GMR SMPL `.npz`。 |
| SMPL -> SMPL-X | `scripts/smpl_to_smplx.py` | 将 SMPL `poses` 拆成 SMPL-X 风格的 `root_orient` 和 `pose_body`。 |
| SMPL-X -> robot | `scripts/smplx_to_robot_dataset.py` | 批量重定向到指定机器人，生成 robot-motion `.pkl`。 |
| 单文件调试 | `scripts/smplx_to_robot.py` | 单条 SMPL-X 可视化、快速保存，用于排查 IK 和姿态问题。 |
| 重定向结果筛选 | `scripts/filter_robot_motion.py` | 对 robot-motion 做质量筛选，输出报告和软链接目录。 |
| 回放检查 | `scripts/vis_robot_motion.py`, `scripts/vis_robot_motion_dataset.py` | 回放单条或文件夹 robot-motion。 |

机器人 XML 和 SMPL-X IK config 的映射在 `general_motion_retargeting/params.py` 中维护：

- `ROBOT_XML_DICT[robot]`
- `IK_CONFIG_DICT["smplx"][robot]`

实际 IK 由 `general_motion_retargeting/motion_retarget.py` 中的 `GeneralMotionRetargeting` 执行。

## 1. Sonic SMPL 源数据筛选

Sonic 原始数据通常位于：

```text
~/robot_software/drl/wbc/GR00T-WholeBodyControl/data/smpl_filtered
```

这些文件很大，筛选脚本默认只创建软链接，不复制 `.pkl`。

推荐命令：

```bash
conda run -n gear_sonic_train python scripts/select_sonic_smpl_subset.py \
  --src_folder ~/robot_software/drl/wbc/GR00T-WholeBodyControl/data/smpl_filtered \
  --out_folder data/sonic_smpl_data/selected_10_12h \
  --target_hours 11 \
  --min_hours 10 \
  --max_hours 12 \
  --overwrite
```

输出：

```text
data/sonic_smpl_data/selected_10_12h/
  motions/                    # 指向 Sonic 原始 .pkl 的软链接
  manifest.csv                # 每条数据的来源、类别、时长、质量指标
  selected_sonic_smpl.txt     # 原始文件路径列表
  summary.json                # 总时长、各类别时长、拒绝原因统计
  selection_principles.md     # 筛选原则和本次结果
```

主要筛选原则：

- 类别比例：locomotion 50%、idle 15%、upper_body 20%、transition_mild 10%、dynamic 5%。
- 排除 jump、dance、kneel、sit、crawl、lie、on_ground 等不适合平地双足控制的动作。
- 检查 `pose_aa`、`transl`、`fps`、NaN/Inf、时长、root speed、vertical span。
- 控制单个 motion family 和 actor 的时长占比，避免数据过度集中。

## 2. Sonic SMPL 转 GMR SMPL

Sonic SMPL 常见字段：

```text
pose_aa    (T, 72) 或等价 pose 字段
transl     (T, 3)
fps
betas      可选
```

`scripts/gear_sonic_smpl_to_gmr_smpl.py` 会统一输出：

```text
poses              (T, 72)
trans              (T, 3)
betas              (10,)
mocap_framerate    scalar
gender             scalar string
source_file
source_pose_key
source_trans_key
coord_transform
```

Sonic 数据使用 Y-up。默认坐标转换为 GMR Z-up：

```text
[x, y, z] -> [x, -z, y]
```

同时会对 root orientation 应用同样的坐标系旋转。

推荐命令：

```bash
conda run -n gear_sonic_train python scripts/gear_sonic_smpl_to_gmr_smpl.py \
  --src_folder data/sonic_smpl_data/selected_10_12h/motions \
  --tgt_folder data/smpl_data/selected_10_12h \
  --coord_transform sonic_yup_to_gmr_zup \
  --overwrite
```

如果输入已经是 GMR 坐标系，可使用：

```bash
--coord_transform none
```

## 3. AMASS 或其它 SMPL 数据入口

如果数据已经是 GMR 兼容的 SMPL `.npz`，可以直接进入下一步 `smpl_to_smplx.py`。

最低要求字段：

```text
poses              (T, >=66), 推荐 (T, 72)
trans              (T, 3)
mocap_framerate    或后续能改名为 mocap_frame_rate
betas              可选，推荐 (10,)
gender             可选
```

如果是 AMASS 风格 SMPL-X 数据，且已经包含：

```text
root_orient
pose_body
trans
betas
mocap_frame_rate
gender
```

则可以跳过 `smpl_to_smplx.py`，直接运行 `smplx_to_robot_dataset.py`。

## 4. GMR SMPL 转 SMPL-X

`scripts/smpl_to_smplx.py` 是格式桥接脚本，不做真实 SMPL 到 SMPL-X 模型拟合。它主要执行：

- `poses[:, :3] -> root_orient`
- `poses[:, 3:66] -> pose_body`
- `mocap_framerate -> mocap_frame_rate`
- `betas` 从 `(10,)` padding 到 `(16,)`
- 删除原始 `poses`

推荐命令：

```bash
conda run -n gmr python scripts/smpl_to_smplx.py \
  --src_folder data/smpl_data/selected_10_12h \
  --tgt_folder data/smplx_data/selected_10_12h \
  --gender neutral
```

输出 SMPL-X `.npz` 字段：

```text
root_orient        (T, 3)
pose_body          (T, 63)
trans              (T, 3)
betas              (16,)
mocap_frame_rate   scalar
gender             scalar string
```

## 5. SMPL-X 批量重定向到机器人

核心脚本：

```text
scripts/smplx_to_robot_dataset.py
```

推荐命令，以 `unitree_g1_24dof` 为例：

```bash
conda run --no-capture-output -n gmr python -u scripts/smplx_to_robot_dataset.py \
  --src_folder data/smplx_data/selected_10_12h \
  --tgt_folder data/retarget_data/g1_24dof/sonic_smpl_selected_10_12h \
  --robot unitree_g1_24dof \
  --num_cpus 8 \
  --device cpu
```

常用机器人 key：

```text
unitree_g1
unitree_g1_24dof
unitree_g1_with_hands
ne01
fourier_gr3
unitree_h1
unitree_h1_2
```

脚本行为：

- 加载每个 SMPL-X `.npz`。
- 使用 `get_smplx_data_offline_fast(..., tgt_fps=30)` 对齐到约 30 FPS。
- 构造 `GeneralMotionRetargeting(src_human="smplx", tgt_robot=...)`。
- 按帧求 IK，得到 MuJoCo `qpos = [root_pos(3), root_rot_wxyz(4), dof_pos(D)]`。
- 保存时将 root quaternion 从 `wxyz` 转为 `xyzw`。
- 通过 FK 计算 `local_body_pos` 和 `link_body_list`，供后续过滤复用。
- 默认做贴地高度修正和初始 root XY 归零。

输出 robot-motion `.pkl` 字段详见 `docs/smplx_to_robot_motion_format.md`：

```python
{
    "fps": aligned_fps,
    "root_pos": root_pos,          # (T, 3)
    "root_rot": root_rot_xyzw,     # (T, 4), 文件中为 xyzw
    "dof_pos": dof_pos,            # (T, D)
    "local_body_pos": local_body_pos,
    "link_body_list": body_names,
}
```

注意：`smplx_to_robot_dataset.py` 会读取 `assets/hard_motions/0.txt` 和 `1.txt` 中列出的 hard motions，默认跳过这些动作。可用 `--disable_hard_motion_filter` 关闭。

## 6. 单文件重定向调试

当怀疑坐标系、姿态、IK config 或机器人模型有问题时，先用单文件脚本验证：

```bash
conda run --no-capture-output -n gmr python scripts/smplx_to_robot.py \
  --smplx_file data/smplx_data/selected_10_12h/welcoming_003__A098_M.npz \
  --robot ne01 \
  --save_path /tmp/welcoming_ne01.pkl \
  --max_frames 300
```

如只想跑转换并保存，不打开 viewer：

```bash
--headless
```

单文件脚本适合快速定位：

- 源数据姿态是否正常。
- 坐标系转换是否正确。
- 某个 robot 的 IK config 是否导致手臂/腿部反关节。
- root 高度或朝向是否异常。

批量生产数据仍建议使用 `smplx_to_robot_dataset.py`，因为它会保存完整 `local_body_pos` 和 `link_body_list`。

## 7. 重定向后数据筛选

核心脚本：

```text
scripts/filter_robot_motion.py
```

推荐命令：

```bash
conda run --no-capture-output -n gmr python -u scripts/filter_robot_motion.py \
  --robot unitree_g1_24dof \
  --input data/retarget_data/g1_24dof/sonic_smpl_selected_10_12h \
  --output_dir data/retarget_data/g1_24dof/sonic_smpl_selected_10_12h/filter_report
```

该脚本输出两套结果：

### Strict 报告

```text
summary.json
details.json
report.csv
pass.txt
tag.txt
reject.txt
pass_motions/
tag_motions/
reject_motions/
```

Strict 适合做质量审计，`reject` 会比较严格。

### Relaxed 训练报告

```text
relaxed_summary.json
relaxed_report.csv
relaxed_keep.txt
relaxed_pass.txt
relaxed_tag.txt
relaxed_reject.txt
relaxed_keep_motions/
relaxed_pass_motions/
relaxed_tag_motions/
relaxed_reject_motions/
```

Relaxed 用于训练集构建。它只把严重问题保留为 reject，部分 strict reject 会降为 tag，从而提高可用数据量。

默认严重原因包括：

```text
arm_ik_limit_residual
arm_side_anomaly
crawling_or_hand_support_like
fall_or_lie_like
foot_penetration_persistent
foot_penetration_severe
high_elevation_or_climb_like
joint_speed_spike
kneeling_like
long_airborne
nan_or_inf
root_speed_spike
sitting_like
```

实际训练通常优先使用：

```text
filter_report/relaxed_keep_motions
```

该目录包含 relaxed pass + relaxed tag 的软链接。

## 8. 筛选规则摘要

筛选脚本检查：

- 帧数过短、NaN/Inf。
- 足端穿透、足端打滑。
- 长时间双脚离地。
- root 大倾角、倒地、趴地。
- 跪地、坐姿、手撑地/爬行。
- root speed spike。
- joint speed / acceleration spike。
- 关节接近极限比例。
- 左右手侧向异常和手臂 IK 极限残留。

详细阈值和解释见：

```text
docs/robot_motion_filtering.md
```

## 9. 可视化与抽查

回放单条 robot-motion：

```bash
conda run --no-capture-output -n gmr python scripts/vis_robot_motion.py \
  --robot unitree_g1_24dof \
  --robot_motion_path data/retarget_data/g1_24dof/sonic_smpl_selected_10_12h/filter_report/relaxed_keep_motions/example.pkl
```

回放文件夹：

```bash
conda run --no-capture-output -n gmr python scripts/vis_robot_motion_dataset.py \
  --robot unitree_g1_24dof \
  --robot_motion_folder data/retarget_data/g1_24dof/sonic_smpl_selected_10_12h/filter_report/relaxed_keep_motions
```

快捷键：

```text
space  暂停/继续
[      上一个动作
]      下一个动作
=      加速
-      减速
```

## 10. 推荐目录组织

建议保持以下结构：

```text
data/
  sonic_smpl_data/
    selected_10_12h/
      motions/
      manifest.csv
      summary.json
      selection_principles.md
  smpl_data/
    selected_10_12h/
  smplx_data/
    selected_10_12h/
  retarget_data/
    g1_24dof/
      sonic_smpl_selected_10_12h/
        *.pkl
        filter_report/
    g1/
      sonic_smpl_selected_10_12h/
    ne01/
      sonic_smpl_selected_10_12h/
```

命名建议：

- 源数据选择结果：`selected_10_12h`
- robot retarget 输出：`sonic_smpl_selected_10_12h`
- 筛选输出：放在 robot-motion 数据集目录下的 `filter_report/`

## 11. 机器人迁移注意事项

切换机器人时需要同步修改：

```text
--robot
--tgt_folder
--output_dir
```

并确认：

- `general_motion_retargeting/params.py` 中有对应 `ROBOT_XML_DICT`。
- `IK_CONFIG_DICT["smplx"]` 中有对应 IK config。
- `scripts/filter_robot_motion.py` 的 `BODY_ROLE_HINTS` 能正确定位左右脚、膝盖、手。
- 如果 `BODY_ROLE_HINTS` 没覆盖，脚本会尝试按 body 名称推断，但建议对训练机器人显式配置。

## 12. 常见问题

### 机器人朝向或根姿态异常

优先检查源数据坐标系转换。Sonic 源默认需要：

```bash
--coord_transform sonic_yup_to_gmr_zup
```

如果输入已经是 GMR/AMASS 常规 Z-up 数据，错误地再次转换会导致朝向异常。

### 左右手或腿映射异常

先用单文件 `smplx_to_robot.py` + viewer 重现，再检查：

- `general_motion_retargeting/ik_configs/smplx_to_<robot>.json`
- `ik_match_table1/2`
- robot XML 的 body 名称和左右侧定义
- `filter_robot_motion.py` 中 arm side quality 的报告指标

### 过滤通过率过低

先看：

```text
filter_report/summary.json
filter_report/relaxed_summary.json
filter_report/report.csv
```

如果 strict pass 很少，不代表数据完全不可用。训练集通常看 relaxed keep 的时长。

### 想复现某次筛选

保留以下文件：

```text
selection_principles.md
manifest.csv
summary.json
filter_report/summary.json
filter_report/relaxed_summary.json
```

这些文件记录了源路径、筛选参数、类别、时长和过滤原因。

## 13. 推荐端到端命令模板

以 Sonic -> `unitree_g1_24dof` 为例：

```bash
conda run -n gear_sonic_train python scripts/select_sonic_smpl_subset.py \
  --src_folder ~/robot_software/drl/wbc/GR00T-WholeBodyControl/data/smpl_filtered \
  --out_folder data/sonic_smpl_data/selected_10_12h \
  --target_hours 11 --min_hours 10 --max_hours 12 --overwrite

conda run -n gear_sonic_train python scripts/gear_sonic_smpl_to_gmr_smpl.py \
  --src_folder data/sonic_smpl_data/selected_10_12h/motions \
  --tgt_folder data/smpl_data/selected_10_12h \
  --coord_transform sonic_yup_to_gmr_zup --overwrite

conda run -n gmr python scripts/smpl_to_smplx.py \
  --src_folder data/smpl_data/selected_10_12h \
  --tgt_folder data/smplx_data/selected_10_12h \
  --gender neutral

conda run --no-capture-output -n gmr python -u scripts/smplx_to_robot_dataset.py \
  --src_folder data/smplx_data/selected_10_12h \
  --tgt_folder data/retarget_data/g1_24dof/sonic_smpl_selected_10_12h \
  --robot unitree_g1_24dof \
  --num_cpus 8 \
  --device cpu

conda run --no-capture-output -n gmr python -u scripts/filter_robot_motion.py \
  --robot unitree_g1_24dof \
  --input data/retarget_data/g1_24dof/sonic_smpl_selected_10_12h \
  --output_dir data/retarget_data/g1_24dof/sonic_smpl_selected_10_12h/filter_report
```

最终训练可用目录通常是：

```text
data/retarget_data/g1_24dof/sonic_smpl_selected_10_12h/filter_report/relaxed_keep_motions
```
