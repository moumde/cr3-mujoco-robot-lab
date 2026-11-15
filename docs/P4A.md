# P4a RGB Vision + Learned Grasp & Lift Integration

基于 `858c61732b28739e9c904e3f277f49709810fa1b`，在 `feat/gripper` 集成 **RGB Perception + Privileged-State Learned Grasp + Motion Planning**。冻结 P2b RGB 模块与原相机、P3c-6 BC、P3c-10.6 Uniform Residual BC（Gain=.25）、执行器、物理和任务判据。SAC 750 步权重继续归档。未训练新模型，独立 Test 17700–17739 继续封存。

## 连续仿真与信息边界

- 两套控制复用同一个 `load_scene`、`MotorExpert`、CR3 六个关节和命名 TCP site、桌面/电机/夹爪几何，以及 50 Hz 控制与每周期 10 个 2 ms 积分步。工具坐标来自同一个 TCP，姿态规划采用向下工具轴。Gradual 夹爪、IK 和重力偏置沿用原实现。
- `p4a.py` 扩展 `VisionEpisode`，与基线共用 reset、视觉取帧、标定和估计。场景在开头 reset 一次，稳定到 grasp_ready，取得两幅 RGB 后在同一模型和同一 `MjData` 附接 RL 逻辑，初始化观测、控制目标和账本。附接过程无物理步、无 `qpos/qvel/ctrl` 写入。训练分支使用 40/43 维特权状态执行原控制器。RGB XY 提供名义 housing 与抓取几何，抓取动作由特权观测驱动。无效 RGB 在抓取前退出。
- 双侧接触、确认、100 mm 抬升和连续 20 步保持全部成立才进入接管。用当前实际关节位置作为 IK 种子，从当前 TCP 和所测工具偏航开始。后续复用 `vision_motion.execute` 的旋转、搬运、放置、释放、撤离；视觉轴 +Y/−Y 对应 +90°/−90°。规划只读 RGB、机器人状态和名义夹持变换。真实 housing 偏差只在评分侧记录。失败停止，无 Oracle 回退。放置评分沿用 ≤15 mm、≤10°、稳定落台及无碰撞和失握。

## 实测

先完成固定种子 19000 双朝向，再以冻结的种子 19001–19010 在 LARGE 每轴 ±25 mm 区域抽取十个 XY，双朝向各一回合。配对共享相机和同 seed/朝向复位，初始 XY 逐项一致。结果共 44 回合。

| 方法 | 固定完整成功 | 随机完整成功 | 固定 / 随机平均误差（mm） | 固定 / 随机平均角误差（°） | 固定 / 随机平均仿真时间（s） |
| --- | ---: | ---: | --- | --- | --- |
| RGB Expert | 2/2 | 20/20 | 6.580 / 6.576 | .0322 / .0322 | 21.38 / 21.46 |
| RGB + Learned Grasp | 2/2 | 20/20 | 6.561 / 6.613 | .0325 / .0341 | 20.22 / 20.34 |

两方法每组朝向均全正确，固定/随机 XY 平均误差 .255/.214 mm，最大 .323/.350 mm。两方法各 22/22 达成双侧、抓取确认、100 mm 抬升、Reorient、Transfer、Place、稳定落台和完成撤退，无异常接触、失握或运行失败。学习型全部在接管前连续保持 20 步，专家沿用原有时序，其 Lift 阶段没有达到额外的 P3c 20 步静持门槛。学习型随机面板最大终点误差 6.839 mm/.0414°。平均本机耗时固定约 7.81 s（录 GIF）和随机 .588 s，专家 .685/.424 s。

首次接管周期记录最大指令差固定/随机 .00696/.00846 rad，TCP 有限差分速度最大 .00907/.00939 m/s。夹爪从约 .9999 到 1.0，无打开。固定两方向接管前最高抬升 120.4/118.6 mm。被动 housing-in-tool 对名义差固定最大 6.66 mm、随机最大 8.25 mm，未反馈给计划器。物体最终成功落入容差；尚未验证更广的夹持偏差和视觉条件。固定 GIF 分别在 `outputs/p4a/final/protrusion_facing_camera/motion.gif` 和 `outputs/p4a/final/flat_facing_camera/motion.gif`，同目录附两相机图。配对、相位迁移、实际关节、TCP、闭合值和物体被动偏差均在 `outputs/p4a/final/results.json`。

## 复现

```bash
export MUJOCO_GL=egl MESA_SHADER_CACHE_DISABLE=true MPLCONFIGDIR=/tmp/cr3-mpl PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m cr3_lab.tasks.p4a --output outputs/p4a/reproduction
.venv/bin/python outputs/p4a/summarize.py outputs/p4a/reproduction
.venv/bin/python -m pytest -q tests outputs/p3c-7/test_benchmark.py outputs/p3c-8/test_diagnose.py \
  outputs/p3c-9/test_residual.py outputs/p3c-10/test_smoke.py outputs/p3c-10.1/test_audit.py \
  outputs/p3c-10.4/test_dataset.py outputs/p3c-10.5/test_distill.py outputs/p3c-10.6/test_experiment.py \
  outputs/p3c-10.7/test_conservative_sac.py outputs/p3c-10.7-r1/test_r1.py
```

82 项自动测试通过，含 P2b-2 视觉与评分隔离回归，以及共享状态、±90° 接管规划测试。详见[调试记录](P4A_DEBUG.md)。等待验收。
