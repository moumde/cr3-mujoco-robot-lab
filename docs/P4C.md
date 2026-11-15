# P4c LeRobot 动作片段训练与闭环验证

## Git 与资产

从 `feat/gripper@b962a1dd1dc2e201916ab4e7bc4d91a66afca0c0` 开始。核对已验收 P4a/P4b 历史、交接报告、双工作树状态和 CR3 资产归档，P4b 原始/导出数据和归档 SHA-256 通过。85 项回归通过后，无冲突合并已验收的两项到本机 `main@75106355459af2f283e21593734ca172655162a4`。检查时 `docs/开发共识.md` 实际位于 gripper 工作树且未追踪，保留原文件。main 缺少独立运行所需忽略资产，合并只带来追踪文件。CR3 工位源和 checkpoint 仍可从 `outputs/p4b/p4a-p3c-local-backup.tar` 恢复，备份本次重新核验。P4c 实验留在功能分支，O6 保持原样。

## 数据训练语义

P4b 每行视觉帧的动作来自同一时刻起的首个 20 ms 控制区间。单行 7 维只覆盖五分之一的 100 ms 图像周期，10 Hz 单动作日志不足以复原原控制。P4c 以图像对应的 50 Hz 控制索引 k 为锚，逐区间核查 `observed_time_s/time_s/action_end_s`，使用同一回合 `[k,k+5)` 内连续 5 条真实下发动作，按行展开为 35 维。末尾不足 5 条的帧丢弃，无填充、跨回合或合成未来指令。每帧状态仍为 20 维。LeRobot 编码 600×400 双 RGB / 10 Hz，训练和推理均缩放为 48×32。阶段 ID 用于误差分组，模型只接收双图和本体状态。

旧 P4b 的 1,474 帧可构造 1,469 个完整片段，其中 6 个来自提前终止的降光回合（均属实际等待命令），旧失败与其他旧轨迹保留原状，本轮训练不使用它们。扩展面板使用 5501–5512 Train、5513–5515 Validation、5516–5518 Test，另有固定位置双朝向 Test seed 5500。每个随机位置由原 LARGE 每轴 ±25 mm 工位采样。同 seed 的两朝向保持同划分。采集使用同一 RGB Expert；仅成功回合进入导出。P3c 17700–17739 继续封存。

## 扩展采集与训练配置

RGB Expert 在 38/38 回合完整成功，包含 24 Train、6 Validation、8 Test，控制样本 41,013 条，仿真累计 820.26 s，单回合 21.10–22.04 s。采集进程意外退出后按原清单接续最后六回合，详见[调试记录](P4C_DEBUG.md)。完整原始数据约 12 GiB，位于忽略目录。固定专家演示由 50 Hz 控制日志物理回放生成，逐步积分状态核对到 1e−12。

LeRobot 0.7.0 v3 导出使用 10 FPS 的 H.264 双路视频。训练缓存由实际 Loader 解码，按原布局保留 35 维闭环目标；Test 解码用于格式检查，其缓存不写入训练入口。小型模型以六通道输入、三层浅卷积、两层 128 维 MLP 和 20 维本体状态预测动作片段。Train 统计做本体及动作标准化。固定 seed 11、AdamW .001、batch 128、30 epoch、按 Validation 归一化 MSE 选择快照。GPU 驱动不可用，本轮使用 CPU，显存峰值无实测。

三分区分别为 5,164 / 1,284 / 1,739 个完整片段，共 8,187 个，省略 32 个短尾帧。Parquet 的每个片段、前置状态、阶段标签和时间戳均已与原控制日志逐项核对。

最佳 checkpoint 在第 30 epoch，Train/Validation MSE 0.001558/0.001862，134,859 参数，CPU 耗时 16.46 s。验证关节 MAE J1–J6 为 .001431/.000512/.002161/.002267/.000065/.009268 rad，夹爪 .01707。Grasp 阶段夹爪误差 .09358，Release .04760。全阶段和 Train MAE 见 `train/training.json`。Loader 实际读取 Train/Validation/Test 所有双 RGB 及帧索引，通过 10 Hz 检查。

## 闭环

控制器从相同种子和 grasp_ready 起步，每 100 ms 以当前双 RGB 和本体状态推理，五条目标逐个执行，未调用 IK 或仿真状态替换。阶段窗口来自 Train 的阶段起点中位数，仅用于评分。测试相同固定双向和三个留出随机 XY 双向。

| 方法 | 固定 | 随机 | 凸起 / 平面 | 双侧 / 确认 / 100 mm / 稳定落台 |
| --- | ---: | ---: | --- | --- |
| RGB Expert | 2/2 | 6/6 | 4/4；4/4 | 8/8 各项 |
| RGB + 本体 BC | 0/2 | 0/6 | 0/4；0/4 | 0/8 各项 |

BC 的 Initialize、Approach、Pre-Grasp、Grasp、Lift、Reorient 时间窗口分别到达 8/8、8/8、7/8、7/8、7/8、5/8；Transfer、Place、Release、Retreat 均 0/8。没有建立双侧夹持或完成抬升的回合。五次在进入 Reorient 时零件仍在桌面，被原评分标为搬运中桌面支撑；另三次是左指垫/物体提前接触、右指垫/桌面接触、物体/地面接触。BC 直接预测执行器目标，失败分类中无 IK 调用或 RGB 定位器输出。不能将失败直接归为定位错误。

回合 000 在开头 30 个控制步内已出现 J6 最大 .188 rad 和夹爪 .596 的请求差；回合 001 为 .174 rad 和 .726。与同回合专家按时间对照，这些差异先于接触。回合 000 在 Grasp 时间窗口开头请求闭合 1.551（限幅后 1），专家为 .150。所有 BC 回合都出现边界限幅。离线模仿误差低并未转化为反馈恢复能力；当前证据支持早期动作偏差和未能抓取，尚无法分离图像编码损失、闭环分布偏移及网络结构各自贡献。保留失败轨迹，未追加训练或进入 P5a。

## 运行与产物

```bash
export PYTHONPATH=src MUJOCO_GL=egl MESA_SHADER_CACHE_DISABLE=true MPLCONFIGDIR=/tmp/cr3-mpl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=outputs/p4c/reproduction
.venv/bin/python outputs/p4c/collect_panel.py "$out/shard-a" --through-seed 5510
.venv/bin/python outputs/p4c/collect_panel.py "$out/shard-b" --from-seed 5511
.venv/bin/python outputs/p4c/assemble_raw.py "$out/raw" "$out/shard-a" "$out/shard-b"
bash outputs/p4c/run_offline.sh "$out"
.venv/bin/python outputs/p4c/visualize_run.py "$out" --replay
.venv/bin/python outputs/p4c/visualize_run.py "$out"
.venv/bin/python -m pytest -q tests
(cd outputs/p4c && sha256sum -c --quiet SHA256SUMS)
```

85 项项目与历史回归测试通过，包含动作片段边界和时间错误检测。数据、checkpoint、原始控制、失败序列、训练 JSON 和运行日志位于 `outputs/p4c/`。专家物理重放 GIF 为 `expert-replay.gif`，闭环演示为 `eval/episode-000.gif` 与 `eval/episode-001.gif`，图表为 `loss.png`、`action-error.png`、`stages.png`、`scene-distribution.png`。等待验收。
