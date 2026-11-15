# P4b 仿真示范采集与 LeRobot v3 导出

## 基线与资产

从 `feat/gripper@d1c5476604e5cae2bbd8674b64b1f02760cfd7e1` 起步。该提交已归档 P4a 代码、P4A.md、P4A_DEBUG.md 及 PROJECT_HANDOFF.md。检查时 `main@e63488b` 已集成到 P3c-10.7-R1，P4a 提交尚未进入 main。主工作树另有用户未追踪文件 `docs/开发共识.md`，保留原样。本轮无分支合并。

main 工作树缺少忽略目录中的 CR3 源文件、冻结 BC 与 Uniform checkpoint。已在本机保存 `outputs/p4b/p4a-p3c-local-backup.tar` 和校验和，含 CR3 源、P3c-6 至 R1 所需实验产物及 P4a 演示。使用独立临时源码副本，从该归档恢复 CR3 源和两个 checkpoint 后，成功加载 15 qpos 的工位及 γ=.25 策略。该备份是本机副本，尚未验证机外持久备份。CR3 资产继续遵循 P0 的来源与再分发规则，只在本机使用。

## 数据与时间约定

| 字段 | 顺序与语义 |
| --- | --- |
| 图像 | Inspection、Localization 两路 RGB，原始 600×400×3 uint8，来自完全相同的 MuJoCo 状态。Overview 不进入数据集。 |
| observation.state（20） | joint1–6 角度 rad、同序角速度 rad/s、夹爪测得开口 m、TCP 世界位置 XYZ m、世界姿态四元数 wxyz。 |
| action（7） | joint1–6 经偏置补偿与执行器限幅后实际保持的目标 rad，随后为渐进限幅后的实际闭合命令 [0,1]。原始控制日志另存含重力偏置、限幅前的目标，未把 RL TCP 请求或计划路点记成执行动作。 |
| Episode metadata | 场景种子、模式、初始朝向、任务文本、采样率、时长、实际感知时刻、估计和无效原因、成功及终止。LeRobot 通过 episode index 对应外部 manifest。 |
| Oracle Debug / Replay | 真值评分在独立 sidecar。原始回放另有完整初始积分状态和每步后状态，供仿真复现。上述字段不导出为 LeRobot observation。 |

MuJoCo 2 ms；控制保持 20 ms。记录器在每条命令前采机器人状态，同一时刻的 action 是接下来这一控制区间实际使用的指令。根据 MuJoCo 整数物理 tick，每 50 个物理步采集双 RGB，对应 10 FPS；50 Hz/20 FPS 的采样时刻有一半落在控制区间内部，本轮采用整除频率以保留原有批量积分和严格动作观测对齐。LeRobot 时间戳从零开始，Loader 逐帧与真实采样时刻核对。感知决策原图另存为 PNG，包括在 0.60 s 提前终止的回合，该瞬间超出其最后一个周期训练帧（0.50 s）。编码在采集完成后离线执行。

## 实测结果

| 模式/工位 | 完整成功 | 帧数 |
| --- | ---: | ---: |
| RGB Expert 固定双向 | 2/2 | 428 |
| RGB Expert 随机双向 | 2/2 | 427 |
| RGB + Learned Grasp 随机双向 | 2/2 | 398 |
| RGB Expert 固定降光 20% 双向 | 1/2 | 221 |

Train 含固定几何 seed 5400 的正常和降光回合，以及随机 seed 5401，共 6 回合、1,076 帧。Validation 保留不同随机 seed 5402，双向 2 回合、398 帧。该小样本划分的控制模式与场景同时变化，仅用于接口校验。降光平面回合在 Initialize 终止，`localization_geometry_failed`，保存两幅实际决策原图、估计、30 条控制和 6 个双 RGB 帧。RGB + Learned Grasp 仍只在执行策略内部使用特权状态。

8/8 回合逐控制比较开启与关闭周期拍摄的控制、物理状态和最终评分，一致；另用原始动作回放全部回合，与原积分状态按 1e−12 绝对容差逐步核对。10 FPS 数据用来播放、学习对齐，不足以重建 50 Hz 全部控制。LeRobot 0.7.0 实际 API 生成 v3.0 Parquet+AV1 MP4，`create → add_frame → save_episode → finalize`，两套 local root 按原划分读取。Loader 使用 PyAV 逐帧验证两路 1,474 对图像、时间、帧索引、任务、状态、动作。视频验证为 600×400、10 FPS。保留完整失败及终止信息于 manifest。官方格式说明见 [LeRobot Dataset v3](https://huggingface.co/docs/lerobot/v0.5.0/lerobot-dataset-v3)。

## 命令与产物

```bash
export PYTHONPATH=src MUJOCO_GL=egl MESA_SHADER_CACHE_DISABLE=true MPLCONFIGDIR=/tmp/cr3-mpl
export HF_HUB_OFFLINE=1 HF_HOME=/tmp/cr3-p4b-hf OPENBLAS_NUM_THREADS=1
out=outputs/p4b
lerobot_py="$HOME/miniconda3/envs/lerobot_sim/bin/python"
.venv/bin/python -m cr3_lab.demonstrations.collect "$out/repro-raw"
.venv/bin/python -m cr3_lab.demonstrations.check "$out/repro-raw" --episode episode-000
"$lerobot_py" -m cr3_lab.demonstrations.export export "$out/repro-raw" "$out/repro-lerobot"
"$lerobot_py" -m cr3_lab.demonstrations.export verify "$out/repro-raw" "$out/repro-lerobot"
sha256sum -c "$out/backup.sha256"
(cd "$out" && sha256sum -c --quiet SHA256SUMS)
```

Loader 示例（同一 LeRobot 环境）：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/cr3-motor-validation", root="outputs/p4b/lerobot/validation",
                     episodes=[0], video_backend="pyav")
for frame in ds:
    state = frame["observation.state"]
    image = frame["observation.images.inspection_rgb"]
```

本机数据目录为 `outputs/p4b/raw/`、`outputs/p4b/lerobot/`。83 项自动化测试通过，包括 P4a、P2b 回归和新增采样/命令语义测试。详见[技术调试记录](P4B_DEBUG.md)。未训练模型，独立 Test 封存，等待验收。
