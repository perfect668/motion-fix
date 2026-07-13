# Robot Motion Filtering

## 目标

`robot-motion` 数据在完成重定向后，通常还需要再做一轮质量筛选，去掉不适合平地双足控制的数据。这里的目标不是做动作识别，而是尽量稳定地过滤掉以下几类样本：

- 足端打滑明显，接触约束不可靠。
- 长时间双脚离地，接近跳跃、攀爬、悬挂类动作。
- 机器人根部姿态异常，例如趴地、摔倒、长期大角度倾斜。
- 膝盖、手臂经常贴近地面，接近跪地、爬行、手撑地动作。
- 非足端 body 依赖外部物体支撑，离开墙、桌、椅、扶手等环境后无法自平衡的动作。
- 速度、加速度或关节极限占比异常，通常意味着重定向质量较差。

当前仓库里的统一入口是 [scripts/filter_robot_motion.py](/home/user/robot_software/drl/retarget/GMR/scripts/filter_robot_motion.py)。它会先生成严格的 `pass / tag / reject` 三类结果，然后再生成面向训练集使用的 relaxed 结果，把部分非严重 reject 降级为 tag，从而得到 `relaxed_keep` 数据。

## 输入数据

脚本面向 GMR 生成的 `robot-motion pkl` 文件，默认依赖以下字段：

- `fps`
- `root_pos`
- `root_rot`
- `dof_pos`

如果 `local_body_pos` 和 `link_body_list` 已经存在，脚本会优先复用它们；否则会根据对应机器人 XML 做一次前向运动学来恢复全身 link 轨迹。

脚本会根据 `--robot` 从 `general_motion_retargeting.params.ROBOT_XML_DICT` 读取机器人 XML。左右脚、膝盖、手部 body 名称优先使用脚本里的 `BODY_ROLE_HINTS`；若没有显式配置，会基于 body 名称中的 left/right、foot/knee/wrist 等关键词做推断。

当输入是目录时，脚本会递归扫描 `.pkl`，但会跳过已有报告和筛选软链接目录，避免把历史报告再次纳入输入。跳过目录包括 `filter_report`、`quality_report`、`pass_motions`、`tag_motions`、`reject_motions`、`relaxed_keep_motions` 等。

## 判定规则

### 1. 无效数据

- 帧数少于 `30`，直接 `reject`
- 任意字段存在 `NaN` 或 `Inf`，直接 `reject`
- 无法定位左右足端 body，直接 `reject`

### 2. 外部支撑依赖

如果动作语义表明除足端外还有 body 与外部环境发生支撑性力交互，直接 `reject` 为 `external_support_dependency`。典型例子：

- `wall_leaning_idle_270_R_001__A285`
- `idle_to_wall_leaning_*`
- `drinking_wall_leaning_mug_*`
- `lean_on`、`lean_against`、`resting_on`、`supported_by` 等依赖墙、桌、椅、门、扶手、杆、梯子的动作

该规则优先使用 motion family / filename 语义。原因是当前 `robot-motion pkl` 不包含外部物体几何，单靠机器人轨迹很难可靠判断“背后是否有墙”。这类动作和 jump/run/dance 不同：高动态动作可以是自包含的，而外部支撑动作在无对应环境时静态平衡条件本身不成立。

### 3. 足端接触与打滑

接触帧定义：

- 足端高度 `< 0.05 m`
- 且足端竖直速度 `|vz| < 0.15 m/s`

打滑判定：

- 接触状态下，足端水平速度 `> 0.15 m/s` 连续 `10` 帧及以上，`reject`
- 单次接触段内，足端水平累计漂移 `> 0.10 m`，`reject`
- 接触状态下，足端水平速度 `> 0.10 m/s` 连续 `6` 帧及以上，或漂移 `> 0.06 m`，`tag`

地面穿透判定：

- 任意足端高度 `< -0.06 m`，`reject`
- 任意足端高度 `< -0.03 m` 持续超过 `10` 帧，`reject`

### 4. 悬空、跳跃、攀爬类

双脚同时离地定义：

- 左右足端高度都 `> 0.08 m`

判定方式：

- 双脚同时离地连续超过 `30` 帧，`reject`
- 双脚同时离地连续 `20` 到 `30` 帧，`tag` 为 `jump_like`

高度变化判定：

- 根部高度 `95% 分位 - 5% 分位 > 0.40 m`
- 且根部高于本段中位高度 `0.18 m` 以上的帧占比 `> 30%`
- 满足以上条件时，判为 `high_elevation_or_climb_like`，直接 `reject`

如果根部高度变化超过 `0.25 m`，但没有达到拒绝阈值，则 `tag`

### 5. 倒地、趴地、跪地、爬行

根部倾角判定：

- 用根部四元数的 `z` 轴与世界竖直方向夹角衡量
- 倾角 `> 65 deg` 且持续 `20` 帧以上，`reject`
- 倾角 `> 50 deg` 且持续 `12` 帧以上，`tag`

跪地判定：

- 左右膝任一高度 `< 0.08 m` 的帧占比 `> 30%`，`reject`

手撑地 / 爬行判定：

- 左右手任一高度 `< 0.10 m` 的帧占比 `> 30%`，`reject`

坐姿判定：

- 计算 `root.z - avg(left_knee.z, right_knee.z)`。
- 当该值 `< 0.15 m` 且持续不少于 `15` 帧时，判为 `sitting_like`，直接 `reject`。

这些阈值已适度放宽以提升通过率。如果目标数据集包含跪地、翻滚、撑地站起等动作，可进一步调整。

### 6. 手臂侧向与 IK 质量

默认启用 `arm quality` 检查，可通过 `--disable_arm_quality` 关闭。

检查方式：

- 使用局部 FK 或 `local_body_pos`，计算左右手相对 pelvis/root 的横向位置。
- 默认期望左手在正侧、右手在负侧。
- 如果左右手侧向正确比例 `< 0.95`，认为存在左右臂侧向异常。

同时会统计部分手臂关节贴近 IK 极限的比例：

- `left_shoulder_yaw_joint` 接近上限
- `left_elbow_joint` 接近上限
- `right_shoulder_yaw_joint` 接近下限
- `right_elbow_joint` 接近上限

如果手臂侧向异常，同时最大手臂极限占比 `>= 0.50`，判为 `arm_ik_limit_residual`；否则判为 `arm_side_anomaly`。两者在 strict 和 relaxed 结果中默认都属于严重 reject 原因。

### 7. 速度、加速度、关节极限

根部速度：

- 最大值 `> 4.0 m/s`，`reject`
- 最大值 `> 3.2 m/s`，`tag`

关节速度：

- 单帧最大关节速度 `> 37 rad/s`，且超过阈值的帧数不少于 `10`，`reject`
- 单帧最大关节速度 `> 30 rad/s`，且超过阈值的帧数不少于 `10`，`tag`

关节加速度：

- 单帧最大关节加速度 `> 800 rad/s^2`，且超过阈值的帧数不少于 `10`，`reject`
- 单帧最大关节加速度 `> 640 rad/s^2`，且超过阈值的帧数不少于 `10`，`tag`

关节极限占比：

- 如果某个关节在超过 `10%` 的帧里，离上下限不足 `5 deg`
- 且这类关节数量不少于 `3`
- 则给出 `joint_near_limits` 的 `tag`

这个规则更偏向“质量提醒”，默认不直接丢弃。

## 输出结果

脚本会在输出目录生成 strict 报告和 relaxed 报告。

### Strict 报告

- `summary.json`：总体统计和阈值，包含以下字段：
  - `status_counts`：各类别文件数量
  - `frame_counts`：各类别累计帧数
  - `duration_sec`：各类别累计时长（秒），可直接用于计算训练数据量
  - `reject_reason_counts` / `tag_reason_counts`：各拒绝 / 标记原因的出现次数
- `details.json`：每条动作的详细判定结果
- `report.csv`：便于人工筛查的表格（含 `num_frames`、`fps` 等 metrics 列）
- `pass.txt`：通过动作列表
- `tag.txt`：需要人工复核或单独分桶的动作
- `reject.txt`：建议直接排除的动作
- `pass_motions/`、`tag_motions/`、`reject_motions/`：默认生成的软链接目录，方便后续拷贝、读取或可视化抽查。

### Relaxed 训练报告

默认还会生成 relaxed 报告：

- `relaxed_summary.json`
- `relaxed_report.csv`
- `relaxed_keep.txt`
- `relaxed_pass.txt`
- `relaxed_tag.txt`
- `relaxed_reject.txt`
- `relaxed_keep_motions/`
- `relaxed_pass_motions/`
- `relaxed_tag_motions/`
- `relaxed_reject_motions/`

Relaxed 逻辑：

- 如果 strict reject 原因属于严重原因，仍保留为 relaxed reject。
- 如果 strict reject 原因不属于严重原因，则降为 relaxed tag。
- relaxed keep = relaxed pass + relaxed tag。

默认严重原因来自脚本中的 `DEFAULT_RELAXED_SEVERE_REASONS`：

```text
arm_ik_limit_residual
arm_side_anomaly
crawling_or_hand_support_like
external_support_dependency
fall_or_lie_like
foot_penetration_persistent
foot_penetration_severe
high_elevation_or_climb_like
joint_position_jump
joint_second_diff_spike
joint_speed_spike
kneeling_like
long_airborne
nan_or_inf
root_speed_spike
sitting_like
```

以下前缀也始终视为严重原因：

```text
exception:
missing_required_bodies:
too_short:
```

可用 `--relaxed_severe_reasons` 覆盖严重原因集合；可用 `--no_relaxed_report` 只生成 strict 报告。

`relaxed_report.csv` 包含 `filter_schema_version` 和 `filter_policy`。对应的
`relaxed_summary.json` 还保存实际 thresholds 和 severe reasons，使下游数据选择脚本
可以验证报告契约，而不需要重新实现质量规则。

使用 `--relaxed_dynamic_actions` 时，`long_airborne`、`high_elevation_or_climb_like`、root/joint speed spike 等动态相关 reject 会在 relaxed 报告中降为 tag，但 `external_support_dependency`、`arm_side_anomaly` 和关节跳变仍然保持 relaxed reject。

## 使用方式

批量扫描文件夹：

```bash
conda run --no-capture-output -n gmr python -u scripts/filter_robot_motion.py \
  --robot unitree_g1 \
  --input retargeting_data/amass_g1 \
  --output_dir retargeting_data/amass_g1/filter_report
```

单文件：

```bash
conda run --no-capture-output -n gmr python -u scripts/filter_robot_motion.py \
  --robot ne01 \
  --input /tmp/ne01_walk_test.pkl \
  --output_dir /tmp/ne01_filter_report
```

快速抽样验证：

```bash
conda run --no-capture-output -n gmr python -u scripts/filter_robot_motion.py \
  --robot ne01 \
  --input retargeting_data/lafan1_ne01 \
  --max_files 20
```

只生成 csv/json/txt，不生成软链接目录：

```bash
conda run --no-capture-output -n gmr python -u scripts/filter_robot_motion.py \
  --robot unitree_g1 \
  --input retargeting_data/amass_g1 \
  --output_dir retargeting_data/amass_g1/filter_report \
  --no_symlink_dirs
```

关闭手臂质量检查：

```bash
conda run --no-capture-output -n gmr python -u scripts/filter_robot_motion.py \
  --robot unitree_g1 \
  --input retargeting_data/amass_g1 \
  --disable_arm_quality
```

随后可把软链接子目录传给 `vis_robot_motion_dataset.py`：

```bash
conda run --no-capture-output -n gmr python scripts/vis_robot_motion_dataset.py \
  --robot unitree_g1 \
  --robot_motion_folder retargeting_data/amass_g1/filter_report/relaxed_keep_motions
```

命令行运行结束后，终端会直接打印 strict 各类别的文件数、帧数和时长，例如：

```text
  pass  :   312 files     93600 frames    3120.0 s  (52.0 min)
  tag   :    48 files     11520 frames     384.0 s  (6.4 min)
  reject:    90 files     18000 frames     600.0 s  (10.0 min)
```

relaxed 时长请查看：

```text
filter_report/relaxed_summary.json
```

## 建议的使用策略

- strict `pass`：质量最干净，适合做高置信度动作库。
- strict `tag`：建议人工抽查或单独建桶，例如 `jump_like`、`large_root_height_change`。
- strict `reject`：用于质量审计，不建议直接训练使用。
- relaxed `relaxed_keep_motions`：默认推荐训练入口，包含 relaxed pass + relaxed tag。
- relaxed `relaxed_reject_motions`：默认从平地双足训练数据中排除。

第一版清洗建议先保守一些，宁可多放进 `tag`，不要过早把边界动作都删掉。等你对目标机器人控制器的稳定区间更清楚之后，再收紧阈值会更稳。
