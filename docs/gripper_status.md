# 主线 P0 验证报告

## 模型与控制

- `src/cr3_lab/core/model.py` 将官方 CR3 Xacro 中的静态 URDF 链、关节原点、惯性张量和 STL 转为 MJCF。仅处理锁定版本的模型，不需要 ROS2。运行时可用 `build_mjcf()` 查看 XML，`load_model()` 加载纯机械臂，`load_model(gripper=True)` 加装两指夹爪。
- `JointPositionController.step(data, target_rad)` 将六维位置目标裁剪至 URDF 范围，写入对应的 `ctrl`，再积分 10 步。仿真步长 0.002 s，控制频率 50 Hz，重力沿世界 −Z，大小 9.81 m/s²。`state(data)` 从 `qpos` 和 `qvel` 读回关节角和角速度。六个关节依次为 `joint1` 至 `joint6`，正方向均绕各自局部 +Z 按右手定则。J3 范围 ±2.7039 rad，其余 ±6.28 rad。仿真零位为六关节全零，对应 MoveIt fake system 初值。SRDF 的命名姿态 `home` 为另一组角度。
- `assets/robots/gripper/parallel.xml` 为项目自建几何。两个手指沿局部 ±X 向内滑动，使用等位移约束及一个位置执行器。`ParallelGripper.command(data, closure)` 接收 [0, 1]，0 为打开，1 为闭合。也可调用 `open(data)` 和 `close(data)`。单指行程 0.037 m，名义开口 0.074 m。夹爪命令在下次积分时生效。

## 公共挂载契约

- `world` 与 `base_link` 轴向一致，沿用官方 `world → dummy_link → base_link` 固定变换，基座在 world 的 z=0.03 m。
- `flange` 同时为固定 body 与同名 site，两者位姿重合。该 frame 与官方链终点 `Link6` 重合，平移为零、旋转为单位阵。官方 J6 STL 主要位于局部 −Z，项目约定工具沿法兰 +Z 向外伸出。O6 挂载时可在 `flange` 下添加工具 body，在工具内定义其专用 `tcp`，并记录法兰到工具的刚体变换。
- 纯臂 `tcp` 与法兰重合。两指版 `tcp` 在法兰坐标 [0, 0, 0.10] m，轴向一致。`site_pose(data, name)` 返回世界坐标下的具名位姿，位置单位 m，四元数顺序 `wxyz`。

## 来源与再分发

[Dobot 官方固定版本](https://github.com/Dobot-Arm/DOBOT_6Axis_ROS2_V4/tree/7807ef8fb31e8f86a76cd474a4a9cddffef6eca9) 的 `cra_description/urdf/cr3_robot.xacro` 和 `cra_description/meshes/cr3/*.STL` 是模型来源。URDF 定义关节角限位，`cr3_moveit/config/` 提供 fake system 初值、命名姿态和 MoveIt 速度/加速度配置。

该提交的[根 LICENSE](https://github.com/Dobot-Arm/DOBOT_6Axis_ROS2_V4/blob/7807ef8fb31e8f86a76cd474a4a9cddffef6eca9/LICENSE) 为 MIT，MIT 文本要求随副本保留版权和许可声明。[模型包元数据](https://github.com/Dobot-Arm/DOBOT_6Axis_ROS2_V4/blob/7807ef8fb31e8f86a76cd474a4a9cddffef6eca9/cra_description/package.xml) 标注 BSD，未指定变体，仓库树中也未见模型目录的独立许可文本。在厂商明确模型文件许可前，官方 Xacro、STL 及模型衍生文件只在本机使用。`scripts/fetch_cr3.py` 从锁定版本取回文件与声明，保存在 Git 忽略的 `outputs/dobot_cr3/`。两指夹爪为项目自建，无第三方资产。

验证环境为 Python 3.12。直接依赖为 MuJoCo 4.0.1（Apache-2.0）、NumPy 2.5.4（BSD-3-Clause）、pytest 9.2.0（MIT，测试）和 Pillow 13.0.0（MIT-CMU，GIF）。

## 复现与实测

在仓库根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-core.txt
.venv/bin/python scripts/fetch_cr3.py
PYTHONPATH=src .venv/bin/python -m pytest -q -s
PYTHONPATH=src .venv/bin/python -m cr3_lab.gripper.demo
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python -m cr3_lab.gripper.demo --gif outputs/p0-motion.gif
```

有桌面的环境可以给演示入口加 `--viewer`。本次实测覆盖 headless 和 EGL GIF。

| 验证 | 实测 |
| --- | --- |
| 自动测试 | 5 passed，覆盖两种模型加载、惯性张量重建、重力、六关节索引与限位、位置执行器、裁剪、控制周期及夹爪开合 |
| 无夹爪 FK | 47 组，法兰及 TCP 最大误差 3.249e-16 m、9.786e-16 rad |
| 有夹爪 FK | 47 组，法兰及 TCP 最大误差 3.522e-16 m、9.786e-16 rad |
| 动态演示 | 4 s 后最大关节跟踪误差 0.01396 rad。3 s 闭合时两指位移 0.03700、0.03691 m，4 s 重新打开时为 0.000015、−0.000025 m |
| 渲染 | `outputs/p0-motion.gif`，600 × 400，另存闭合帧 `outputs/p0-closed.png` |

FK 基准独立读取官方 Xacro，以齐次矩阵沿 URDF 链累乘，用 `Rz(yaw) Ry(pitch) Rx(roll)` 表示固定原点旋转。测试覆盖零位、逐关节正负扰动、上下限角向量，以及种子 17 生成的 32 组合法角度。计算 TCP 参考时额外应用项目定义的工具偏移。

## 未解决事项

- 法兰方向已通过 URDF 和 J6 网格核对，真实法兰孔位、标定和动力学尚待确认。URDF effort/velocity 均为 0，当前执行器增益、阻尼、转子惯量与 ±60 N·m 力矩限制为仿真设定。控制器没有执行 MoveIt 速度和加速度限制，重力下存在静差。
- 当前碰撞掩码只启用地面与机械臂/夹爪接触。臂间自碰撞、夹爪与机械臂碰撞、指尖接触负载及稳定抓取均未验证。MuJoCo 网格碰撞采用凸包，指关节软约束在端点允许微小偏移。
- 交互 Viewer 入口尚未实测。进入 P1a 时需验证零件接触、抬升与翻转可行性，O6 使用上述法兰契约即可开始独立挂载。

# 主线 P1a 验证报告

## 实现与场景

- `src/cr3_lab/tasks/motor_scene.py` 复用 P0 MJCF，加入桌面、目标区域、相机坐标和自由刚体电机。桌面和零件尺寸、质量及任务接触摩擦集中在 `SceneConfig`，夹爪底层接口沿用 P0。
- `src/cr3_lab/tasks/ik.py` 在独立的 MuJoCo 数据中求解有界阻尼最小二乘 TCP IK。每个笛卡尔采样点以前一组关节角为初值，检查位置、旋转残差和关节范围。进场使用固定 IK 分支初值和三次平滑关节插值，后续使用平滑笛卡尔位置和 yaw 轨迹。标称关节速度上限 1.5 rad/s，笛卡尔速度上限 0.08 m/s，yaw 速度上限 0.7 rad/s。
- `src/cr3_lab/tasks/motor_expert.py` 负责状态机、接触检查和成功判定。执行仍用 P0 位置控制器，控制周期 20 ms，积分步长 2 ms。用关节偏置力除以位置增益得到前馈设定值偏移。
- `src/cr3_lab/tasks/p1a.py` 提供批量评估、逐回合 JSON、实时 Viewer 和带阶段字幕的 GIF。`tests/test_p1a.py` 覆盖复位坐标和相机标签、可达/不可达 IK、双朝向完整回合、实际旋转方向和失败报告。

World 采用右手系，+Z 向上。CR3 Base 与 World 同轴，原点为 World `(0, 0, 0.03) m`。桌面中心 XY 为 `(0.47, 0) m`，台面高 0.12 m，台面宽深各 0.40 m。物体局部原点为圆柱主体中心，局部 +Z 沿轴指向凸起端，另一端为平面。主体直径 52 mm、长度 76 mm、质量 180 g，凸起直径 24 mm、长度 15 mm、质量 18 g。主体直径小于夹爪 74 mm 开口。

固定初始中心为 World `(0.39, 0, 0.146) m`，目标中心为 `(0.50, 0, 0.146) m`。目标标记为 124 × 86 mm。相机位置 `(1.12, 0, 0.45) m`，朝向 `(0.44, 0, 0.24) m`，视线水平投影为 World −X。凸起端朝相机时，凸起法向为 World +X。平面端朝相机时，凸起法向为 World −X。目标凸起法向为从目标指向基座的水平单位向量，即 World −X。录像使用便于同时观察桌面和机械臂的斜视角。任务从 MuJoCo 读物体位姿。

P1a 仅在本场景将指垫沿圆柱轴线的宽度由 32 mm 增至 62 mm，减少偏载翻转，夹爪行程及力限幅沿用 P0。滑动摩擦系数为台面和零件 0.45、指垫 0.8。场景使用椭圆摩擦锥，`impratio=10`，求解容差 `1e-10`。桌面和零件为自建几何，沿用 P0 的官方模型下载及许可证处理方式，未新增依赖。

## 执行流程与判定

| 阶段 | 动作与检查 |
| --- | --- |
| Initialize | 复位场景并沉降，Oracle 读取电机轴向，按刚体质心确定抓取中心 |
| Approach | 用 IK 选择进场目标并平滑移动，检查关节范围和意外接触 |
| Pre-Grasp | 张开夹爪，沿世界 −Z 下探，使指垫覆盖圆柱中部 |
| Grasp | 闭合并确认两片指垫均与电机接触 |
| Lift | 抬升约 140 mm，确认双指接触且零件脱离桌面 |
| Reorient | 凸起端朝相机时绕世界 +Z 转约 180°。另一朝向保持朝向，仅按 Oracle 轴向修正微小偏差 |
| Transfer | 在抬升高度移至目标上方，持续检查接触和滑脱 |
| Place | 用 Oracle 测得的夹持偏移修正落点，将零件降至名义支承高度上方 2 mm |
| Release | 张开夹爪，零件靠重力落台 |
| Retreat | 抬起空夹爪并等待零件稳定，评估最终位置与朝向 |

抓取成立需要双侧指垫接触、脱离桌面且抬升超过 20 mm。搬运中双侧接触丢失持续 100 ms，或零件主体中心距 TCP 超过 35 mm，判为失握并中止。零件位姿只在复位时写入，夹取、旋转和放置都通过接触和重力完成。唯一的等式约束是 P0 的两指联动。

回合成功要求走完十阶段、抓住并抬升至少 100 mm、未掉落、无意外碰撞，且最终三维位置误差 ≤15 mm、凸起法向误差 ≤10°、稳定落台。100 mm 抬升阈值提供远大于半径 26 mm 的翻转净空。15 mm 接近目标区域相对零件的最小轴向余量，10° 用于限制端面偏转。稳定要求桌面支承且无指垫接触，连续至少 0.40 s 内线速度 ≤5 mm/s、角速度 ≤0.10 rad/s。阈值定义在 `ExpertConfig`。

## 实测

复用 P0 安装和模型下载命令。批量评估用种子 17–26，每种朝向 10 次。种子仅改变轴对称零件的初始局部滚转，位置和端面方向固定。各回合近似重复同一物理条件。

| 初始朝向 | 成功 | 掉落 / 意外碰撞 | 最大抬升高度 | 最终位置误差 | 法向误差 | 每回合仿真时间 |
| --- | --- | --- | --- | --- | --- | --- |
| 凸起端朝相机 | 10/10 | 0 / 0 | 138.155 mm | 0.311 mm | 0.0311° | 25.38 s |
| 平面端朝相机 | 10/10 | 0 / 0 | 138.211 mm | 0.253 mm | 0.0309° | 18.86 s |

表中连续指标按显示精度舍入，同一组内一致。20 次均完整结束并满足稳定条件，失败原因统计为空。自动化测试共 **11 passed**，含 5 项 P0 回归。Viewer 在本机 X11 实测平面端朝相机 1/1 成功。两个 GIF 均完成录制和解码检查。

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q -s
PYTHONPATH=src .venv/bin/python -m cr3_lab.tasks.p1a --episodes 10 --seed 17 --json outputs/p1a/evaluation.json
PYTHONPATH=src .venv/bin/python -m cr3_lab.tasks.p1a --orientation flat_facing_camera --episodes 1 --viewer
for orientation in protrusion_facing_camera flat_facing_camera; do
  MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python -m cr3_lab.tasks.p1a \
    --orientation "$orientation" --episodes 1 --gif "outputs/p1a/$orientation.gif"
done
```

产物在 `outputs/p1a/`。逐回合数据为 `evaluation.json`，两段演示为 `protrusion_facing_camera.gif` 和 `flat_facing_camera.gif`，同名 PNG 为撤离结束帧。

## 局限和后续验收

- 开发试跑曾在原指垫和默认摩擦锥配置下出现抬升后俯仰滑移和掉落。当前结果对应上述 P1a 指垫、质心对准和接触求解设置。对摩擦、指垫宽度和质量变化的敏感性尚未单独量化。
- 意外接触按 50 Hz 控制周期检查活动接触，短于一个周期的瞬态接触可能漏检。CR3 自碰撞沿用 P0 设置。进场使用固定场景下验证过的 IK 分支，未覆盖其他场景。
- P1a 等待正式验收。后续 P1b 建议先定义独立的固定种子 XY 测试集，并分别统计不可达、碰撞和失握，避免将本阶段两种固定位置结果外推到随机位置。

P1a 实现及实测报告提交为 `5b014167738b6054c5b3f0a1354589e07e46e8c5`。

# P1a-R1 工位布局修订

项目负责人在 P1a 可视化验收中发现，全零待命姿态和沿世界 X 轴摆放的工件与实际工位预期不一致。R1 将零件从机器人左侧搬到右侧，保留 P1a 十阶段状态机和 P0 公共运动学及法兰契约。

| 项目 | 原 P1a | P1a-R1 |
| --- | --- | --- |
| 零件初始中心，World m | (0.39, 0, 0.146) | (0.45, +0.09, 0.146) |
| 目标中心，World m | (0.50, 0, 0.146) | (0.45, −0.09, 0.146) |
| 初始凸起端法向 | ±X | ±Y，平行近侧桌边 |
| 目标凸起端法向 | −X | −X，独立于目标区域的 Y 坐标 |
| 待命姿态 | 六关节全零 | `grasp_ready`，张开指垫位于零件上方 |

## 待命、碰撞和重定向

`MotorExpert.grasp_ready` 在复位期间从 TCP 目标 `(0.45, 0.09, 0.29) m` 和工具 +Z 向下的旋转解 IK。关节按 J1–J6 排列，实测 `qpos = [0.480746, -0.406890, -0.815250, -0.348645, 1.570807, 2.051538] rad`。表中角度是求解结果，配置中的近似角仅用于选取 IK 分支。FK 得到 TCP `(0.449989, 0.089995, 0.290005) m`，比初始主体中心高约 144 mm。夹爪指垫的闭合方向平行 World X，初始行程和目标均为零，即完全张开。

重置后的关节速度为零。臂控制目标由关节角加上偏置力除以位置增益得到，六关节初始加速度绝对值最大约 `1.8e-6 rad/s²`。初始化仅检测到零件主体与桌面的预期接触。任务场景显式启用非相邻臂段网格和夹爪对上游臂段的碰撞检查，跳过设计上有搭接的相邻链节及法兰与工具的固定接口。公共模型的碰撞配置保持原样。

Lift 结束后读取 Oracle 凸起端法向，并在世界 XY 平面上用叉积与点积计算到 −X 的有符号夹角。Reorient 用该夹角规划连续 TCP yaw，零件靠夹持接触随动。右侧绿色目标区域上的箭头指向 World −X。

## 相机

- Overview 位于 World `(0.83, −0.94, 0.72) m`，看向 `(0.31, 0, 0.25) m`，垂直视场 39°。GIF 和 Viewer 共用此相机。检查了待命和结束帧，画面同时包含机械臂、桌面、零件和方向标记。
- Inspection 位于 World `(0.45, 0.39, 0.235) m`，看向 `(0.45, 0.09, 0.146) m`，垂直视场 30°。从 +Y 略向下观察，凸起朝 +Y 时可见凸起圆面及侧壁，朝 −Y 时可见完整平面。本阶段只生成图像用于视角验证。

## 重复评估

种子 17–26，每组 10 次。保持同一工件中心，种子仅改变轴对称工件的初始滚转。固定位置重复测试不代表位置泛化能力。

| 初始凸起端 | 成功 | 计划 / 实际 yaw | 最大抬升 | 最终位置误差 | 最终法向误差 |
| --- | --- | --- | --- | --- | --- |
| +Y，凸起端朝 Inspection | 10/10 | +89.998° / +89.457° | 138.377 mm | 0.235 mm | 0.0316° |
| −Y，平面端朝 Inspection | 10/10 | −90.002° / −89.466° | 138.378 mm | 0.252 mm | 0.0308° |

组内连续指标在表格精度下一致。20 回合均完成十阶段并稳定落台，掉落、意外接触和失败原因计数均为零。成功阈值沿用 P1a。自动化测试 **13 passed**，覆盖 P0 回归、初始轴线锁定为 ±Y、待命 FK 和动力学一致性、初始碰撞、双向实际旋转和右侧放置。

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q -s
PYTHONPATH=src .venv/bin/python -m cr3_lab.tasks.p1a --episodes 10 --seed 17 --json outputs/p1a-r1/evaluation.json
PYTHONPATH=src .venv/bin/python -m cr3_lab.tasks.p1a --orientation flat_facing_camera --episodes 1 --viewer
for orientation in protrusion_facing_camera flat_facing_camera; do
  MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python -m cr3_lab.tasks.p1a \
    --orientation "$orientation" --episodes 1 --gif "outputs/p1a-r1/$orientation.gif"
  MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python -m cr3_lab.tasks.p1a \
    --orientation "$orientation" --episodes 1 --inspection-png "outputs/p1a-r1/inspection-$orientation.png"
done
```

输出在 `outputs/p1a-r1/`，包括 `evaluation.json`、两种标签各自的 GIF、Inspection 端面截图及 Overview 首尾帧。自动测试和渲染沿用 P0 依赖。Viewer 实测平面端朝相机 1/1 成功，使用独立显示状态后正常退出。

已知局限包括 50 Hz 接触采样可能漏检瞬态，碰撞网格为凸包，待命和轨迹仅在当前工位验证。等待 P1a-R1 正式验收。

P1a-R1 实现及实测报告提交为 `516a5d921f5593ddc03501033b8b90c30e827cbc`。

# P1b 随机位置 Oracle 定向抓放

## 场景与实现

沿用 P1a-R1 工位。World +X 指向台内、+Y 为机器人左侧、+Z 向上。零件初始轴线平行 Y，凸起法向为 ±Y，放置目标中心为 `(0.45, −0.09, 0.146) m`，最终凸起端朝 −X。主体直径 52 mm、长 76 mm。主体与凸起的几何尺寸和质量统一定义在 `motor_scene.MotorDimensions`，可直接由其他场景引用。本阶段复用 P0 CR3、两指夹爪、已验收的接触参数和已有依赖。

`motor_scene.py` 新增以 `(0.45, 0.09) m` 为中心的复位模式。两轴独立均匀采样，轴向滚转使用独立随机流。两种朝向使用同一组 XY，便于配对比较。每回合采样一次，无重抽。

| 模式 | 相对中心的每轴偏移 | X 范围，m | Y 范围，m |
| --- | --- | --- | --- |
| fixed | 0 mm | 0.450 | 0.090 |
| small | ±10 mm | 0.440–0.460 | 0.080–0.100 |
| large | ±25 mm | 0.425–0.475 | 0.065–0.115 |

按外廓沿 X ±26 mm、沿 Y 最大 ±53 mm 保守计算，大范围内零件始终位于左半桌面，最靠近中线的外廓仍有 12 mm 余量。采样前验证整个范围。零件位姿只在复位时写入。

`motor_expert.py` 保留固定的、经碰撞检查的 `grasp_ready` 待命点。从沉降后的 Oracle 位姿和质心重算接近、预抓取、抓取和抬升路点。抬升后用实际端面法向计算旋转，在目标上方用实测的夹持偏移修正下放位置。每段笛卡尔轨迹按控制周期独立执行有界 IK。逐回合记录采样 XY、计划抓取 TCP、阶段、成功、放置误差、仿真时间及实耗时间。

`p1a.py` 的 `--spawn-range` 同时适用于批量评估、Viewer 和 GIF，默认 `fixed`。失败以阶段和具体原因记录，分类包括 `unreachable_ik`、`collision`、`grasp_failure`、`lost_grip`、关节限位/速度和放置验收。回合成功沿用 P1a-R1 判据，包括抬升至少 100 mm、最终位置误差 ≤15 mm、端面角度误差 ≤10°、稳定落台以及无失握或记录到的异常接触。

## 复现

开发检查用种子 17–26，每模式每朝向 10 回合。独立测试用 1000–1019，每模式每朝向 20 回合。

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
for profile in fixed small large; do
  PYTHONPATH=src .venv/bin/python -m cr3_lab.tasks.p1a \
    --spawn-range "$profile" --episodes 20 --seed 1000 \
    --json "outputs/p1b/test-$profile.json"
done
for entry in 'protrusion_facing_camera 1000' 'flat_facing_camera 1001'; do
  read -r orientation seed <<< "$entry"
  MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python -m cr3_lab.tasks.p1a \
    --spawn-range large --orientation "$orientation" --episodes 1 --seed "$seed" \
    --gif "outputs/p1b/large-$orientation.gif"
done
```

自动化测试 **17 passed**，包含 P0/P1a-R1 回归、采样可复现性及边界、±Y 轴向、不同初始 XY 下 Oracle 抓取目标变化，以及不可达错误分类。

## 实测结果

开发检查 60/60 成功。下表为独立测试，时间为各组回合实耗之和，包含规划和仿真，不含模型加载或渲染。误差区间按显示精度舍入。

| 模式 | 初始凸起端 | 不同 XY | 成功 | 位置误差，mm | 法向误差，° | 实耗，s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fixed | +Y | 1 | 20/20 | 0.2350 | 0.0316 | 7.90 |
| fixed | −Y | 1 | 20/20 | 0.2516 | 0.0308 | 10.98 |
| small | +Y | 20 | 20/20 | 0.2335–0.2369 | 0.0316 | 10.22 |
| small | −Y | 20 | 20/20 | 0.2495–0.2550 | 0.0308 | 8.03 |
| large | +Y | 20 | 20/20 | 0.2322–0.2737 | 0.0316–0.0317 | 7.94 |
| large | −Y | 20 | 20/20 | 0.2301–0.2614 | 0.0308 | 7.80 |

独立测试共 120/120 成功，无不可达、异常接触、抓取失败或失握。随机组共 80 回合、40 个不同 XY，按朝向配对。大范围实测采样覆盖 X 0.4261–0.4701 m、Y 0.0686–0.1138 m。逐回合结果为 `outputs/p1b/test-{fixed,small,large}.json`。大范围双朝向 Overview GIF 位于 `outputs/p1b/large-protrusion_facing_camera.gif` 和 `outputs/p1b/large-flat_facing_camera.gif`，均已检查首尾帧解码。

结果适用于当前工位、采样范围与物理配置。均匀抽样未覆盖连续区域的所有边界点，未量化摩擦和工件质量变化的影响。碰撞检查仍按 50 Hz 采样。P1b 等待正式验收。

P1b 实现及实测报告提交为 `f1be13f9d40adb7022691a944d60e4b623c03273`。

# P2a RGB 相机验证与视觉数据生成

## 相机与标定

保留 P1b 的任务几何、物理参数和专家。Overview 和 Inspection 沿用原参数，新增 Localization。采集在 Episode 复位、机械臂进入张开的 `grasp_ready` 后进行。两张 RGB 来自同一个仿真状态。

| 相机 | 用途 | World 位置，m | World 注视点，m | 垂直 FOV | RGB |
| --- | --- | --- | --- | --- | --- |
| Overview | 展示 | (0.83, −0.94, 0.72) | (0.31, 0, 0.25) | 39° | 演示使用 600 × 400 |
| Inspection | 端面 | (0.45, 0.39, 0.235) | (0.45, 0.09, 0.146) | 30° | 600 × 400 |
| Localization | XY 定位 | (0.72, 0.37, 0.49) | (0.45, 0.09, 0.146) | 31° | 600 × 400 |

World +Z 向上，零件局部 +Z 指向凸起。两台感知相机的内参为针孔矩阵 `K`，无畸变。主点为 (299.5, 199.5) px，Inspection 与 Localization 的 `fx=fy` 分别为 746.410 和 721.177 px。像素坐标原点为左上像素中心。外参定义 `p_optical = R_world_to_optical @ p_world + t_world_to_optical_m`，光学轴为右、下、前，由 MuJoCo 的右、上、后转换。完整矩阵保存在 `cameras.json`。

## 覆盖实测

固定点及每轴 ±10 mm、±25 mm 两个采样框均检查了双朝向。每个采样框另外覆盖四角和四条边的中点，共 16 个边界 XY、32 个边界场景。使用完整场景与单独渲染零件的几何 ID 像素比较可见轮廓。端面可见性用近端圆面或凸起周围环面上的 96 个点做射线检查。这些诊断只用于离线验证。

| 位置配置 | Inspection 最小边距 | Localization 最小边距 | 两视角最小轮廓可见比例 |
| --- | ---: | ---: | ---: |
| fixed | 106 px | 142 px | 100% |
| small | 95 px | 127 px | 100% |
| large | 77 px | 103 px | 100% |

样本中无出框、无环境或机械臂遮挡零件轮廓。Inspection 近端采样点最小可见比例为 90.6%，Localization 为 77.1%，斜视中较多端面采样点被零件自身遮住。Inspection 正面至少可见 3483 个凸起像素，平面朝相机时凸起可见像素为 0。RGB 拼图可直接看到凸起的圆面、侧壁以及另一朝向的平面。边界有可见透视变化，未妨碍本次样本中的目视区分。几何 ID 统计不衡量 RGB 分类准确率。

## 数据格式与划分

`outputs/p2a-v1/` 包含 `train/`、`validation/`、`test/` 下的无标注 PNG，以及 `manifest.jsonl`、`cameras.json`、`samples.png` 和 `coverage.json`。每行清单对应一对 RGB，包含 `split/profile/seed/case`、两台相机对应的相对路径、覆盖统计，以及 `oracle` 下的相机相对朝向标签、World XY、初始 XYZ 和 MuJoCo `wxyz` 四元数。标签由凸起法向与指向 Inspection 的向量点积获得。图像与标签分别保存。

| 划分 | 种子或配置 | 不同 XY | RGB 对 |
| --- | --- | ---: | ---: |
| Train | 2000–2011，small 和 large | 24 | 48 |
| Validation | 3000–3003，small 和 large | 8 | 16 |
| Test | 4000–4005，另含固定中心 4100 及边界种子 4200–4208 | 29 | 58 |

同一 XY 的双朝向和双相机图像属于同一划分，复用的场景种子也只出现在同一划分。共 61 个不同 XY、122 对、244 张 RGB。检查命令核验了文件可解码、分辨率与通道、XY 分组、四元数和相机相对标签，并生成中心和两组对角点的标注拼图。

```bash
MESA_SHADER_CACHE_DISABLE=true MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python -m cr3_lab.perception.dataset generate outputs/p2a-v1
PYTHONPATH=src .venv/bin/python -m cr3_lab.perception.dataset inspect outputs/p2a-v1
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m cr3_lab.tasks.p1a --spawn-range large --episodes 2 --seed 1000 --json outputs/p2a-oracle-regression.json
```

自动测试 **19 passed**，包含 P0、P1a-R1、P1b 回归和相机标定、采样边界、划分检查。独立运行 Oracle 大范围入口双朝向各 2/2 成功。沿用 P0 依赖和资产许可处理，生成数据保存在 Git 忽略目录。

## P2b 接口与限制

`perception.camera.VisualEstimate` 定义 World 米制 `position_xy`、Inspection 端面 `orientation` 和 [0,1] 置信度，供后续纯 RGB 模块交付。P2a 只验证相机和标签，没有实测 RGB 分类或 XY 预测误差。

Inspection 的视线接近圆柱轴，沿 Y 的位移主要改变表观尺度。端面深度还随凸起朝向变化，直接把图像质心投到桌面会产生系统偏差。P2b 可先用 Inspection RGB 的端面轮廓做分类，再用 Localization RGB 和固定标定，将主体特征与已知轴向、尺寸及中心高度联合拟合。用独立 Test 中的 Oracle 标签计算定位误差和分类指标。当前数据只有固定光照、材质、相机及待命状态，尚未验证对真实材质、遮挡、照明或姿态变化的泛化。等待 P2a 正式验收。

P2a 实现及实测报告提交为 `e4f9abda18edc2ae5f2eb0cd0e51f195f7ffb14b`。

# P2b-1 RGB 端面识别与 XY 定位

## 实现

`perception/rgb_estimator.py` 提供 `RgbMotorEstimator.predict(inspection_rgb, localization_rgb)`。两张输入为 uint8 RGB，标定由 `camera.load_calibration` 读取。返回 `VisualEstimate`，包含 World XY（m）、Inspection 相对端面标签、启发式置信度和失败原因。无效预测的 XY 为空，置信度为零。置信度反映颜色判据和几何残差的余量。

- Inspection 在标定投影得到的固定工位区域内提取暖色零件像素，再用偏红凸起像素占比分类。占比低于 2% 判平面，高于 10% 判凸起，中间区间拒判。
- Localization 提取暖色轮廓包围框，用已知轴向、圆柱及凸起尺寸、台面高度和 Inspection 判定的朝向构造三维轮廓。以包围框中心射线和主体中心高度平面交点为初值，最小二乘拟合投影框的四条边，只优化 World XY。几何残差超过 6 px 或超出工位允许范围时拒绝输出。
- `camera.py` 提供 World 到像素投影以及像素射线与水平面求交。推理只使用 RGB、标定和固定场景参数，评估模块在推理返回后读取 Oracle 标签。沿用 P2a 原始划分和依赖，未改动专家、模型或物理参数。

## 独立 Test 实测

阈值和几何拟合在 Train 与 Validation 阶段确定，Test 用于最终评估。Train 48/48 分类正确，定位中位误差 0.267 mm。Validation 16/16 分类正确，定位中位误差 0.344 mm。

| Test 分组 | 正确 / 总数 | XY 均值，mm | 中位，mm | P95，mm | 最大，mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全部 | 58/58 | 0.343 | 0.319 | 0.586 | 0.687 |
| 固定 | 2/2 | 0.259 | 0.259 | 0.321 | 0.328 |
| 随机范围（含边界） | 56/56 | 0.346 | 0.336 | 0.586 | 0.687 |
| small | 28/28 | 0.287 | 0.271 | 0.544 | 0.549 |
| large | 28/28 | 0.404 | 0.435 | 0.638 | 0.687 |
| 凸起端朝 Inspection | 29/29 | 0.357 | 0.361 | 0.574 | 0.687 |
| 平面端朝 Inspection | 29/29 | 0.328 | 0.284 | 0.586 | 0.666 |

混淆矩阵以真实凸起、平面为行，以预测凸起、平面、无效为列，结果为 `[[29,0,0],[0,29,0]]`。无效预测计入分类总分母。XY 误差为有效预测与 Oracle XY 的欧氏距离。Test 中没有误分类、无效预测或超过 5 mm 的样本，案例拼图展示误差最大的六例。最大误差来自 large 的 +X 边中点，凸起朝相机，真实 XY 为 `(0.475, 0.090) m`，预测为 `(0.474362, 0.089746) m`。同位置反向也出现略偏小的 X 估计。可能误差源包括像素量化、颜色阈值对暗边缘的裁剪和离散轮廓近似，尚未分离各项贡献。

## 复现与输出

```bash
PYTHONPATH=src .venv/bin/python -m cr3_lab.perception.evaluate_rgb outputs/p2a-v1 outputs/p2b-1 --split test
PYTHONPATH=src .venv/bin/python -m cr3_lab.perception.rgb_estimator \
  outputs/p2a-v1/cameras.json \
  outputs/p2a-v1/test/fixed-4100-center-protrusion_facing_camera-inspection.png \
  outputs/p2a-v1/test/fixed-4100-center-protrusion_facing_camera-localization.png
PYTHONPATH=src .venv/bin/python -m pytest -q
```

`outputs/p2b-1/results.json` 含逐样本预测与分组指标，`metrics.png` 展示分类混淆矩阵和定位误差，`examples.png` 为双相机案例。自动化测试 **23 passed**，覆盖 P0–P2a 回归及标定往返、双标签几何定位、无效预测统计。单对 RGB CLI 已实跑。

结果仅适用于这批固定材质、照明、相机和轴向的仿真数据。颜色阈值依赖渲染材质，遮挡、相似背景、视角或光照变化可能导致拒判或错误预测。置信度未经概率校准。本阶段等待正式验收。

P2b-1 实现及实测报告提交为 `a227b4ff1b089b048f04e60bd44c25bdb687d4f4`。

# P2b-2 RGB 驱动定向抓放

## 执行与隔离

`tasks.p2b2` 使用 P1b 场景复位并进入 `grasp_ready`，沉降 0.60 s 后采集同步 Inspection 和 Localization RGB。原 P2b-1 估计器输出 `VisualEstimate`，包含 World XY（m）、端面标签、置信度和无效原因。视觉运动规划器 `vision_motion.execute` 接收该估计、固定工位与零件几何、IK、待命 TCP 和关节角。它通过单向回调发送阶段、六关节位置目标和夹爪闭合度。无效估计在 Initialize 停止。

视觉路径走完 Initialize → Approach → Pre-Grasp → Grasp → Lift → Reorient → Transfer → Place → Release → Retreat。初始质心偏移由主体和凸起的名义质量与几何推导，旋转方向由 Inspection 标签决定。释放目标由抓取时的名义工具—主体变换推算。机器人状态适配器复用 P1b 的 50 Hz 位置伺服、关节偏置力前馈、两指执行器及 2 ms 积分。原 Oracle 专家的 Cartesian IK 插值提取为共享函数，Oracle 阶段和观察行为保持原样。

| 审查点 | RGB 执行的数据来源 |
| --- | --- |
| 初始位置、凸起法向 | 双 RGB、固定标定、已知水平 ±Y 轴向 |
| 抬升后的转向 | 初始视觉法向和计划腕部旋转 |
| 放置前夹持偏移 | 名义工具—主体变换 |
| 物体真值、free joint、物体接触 ID | 复位或独立评分，未送入规划器 |

`vision_score.py` 被动记录接触、抬升、失握、稳定及最终误差。评分结果不改变轨迹，也不中断命令。复位中的待命安全检查沿用 P1b。视觉执行可因 IK、速度约束或 MuJoCo 仿真告警停止。隔离测试禁用原 Oracle 执行入口，并篡改评估端的真实位置和姿态。固定和大范围回合的评分改变，全部关节和夹爪命令逐项一致。另有空白 RGB 测试验证在 Initialize 停止。

## 配对实测

固定、small（每轴 ±10 mm）、large（每轴 ±25 mm），凸起 ±Y，每组合 20 个配对回合，种子 5000–5019。检查了配对初始 XY 一致。物理参数、控制频率和成功阈值相同。RGB 使用 ≥100 mm 抬升、≤15 mm 最终位置误差、≤10° 法向误差、稳定及无失握和记录到的异常接触的原判据。

| 位置 / 初始凸起 | Oracle 成功 | RGB 成功 | Oracle / RGB 最终平均位置误差 |
| --- | ---: | ---: | ---: |
| fixed / +Y | 20/20 | 20/20 | 0.235 / 6.549 mm |
| fixed / −Y | 20/20 | 20/20 | 0.252 / 6.557 mm |
| small / +Y | 20/20 | 20/20 | 0.235 / 6.539 mm |
| small / −Y | 20/20 | 20/20 | 0.252 / 6.553 mm |
| large / +Y | 20/20 | 20/20 | 0.242 / 6.554 mm |
| large / −Y | 20/20 | 20/20 | 0.252 / 6.556 mm |

两种方法各 120/120 抓取并完整成功，无失握、异常碰撞或 IK 失败。Oracle / RGB 的平均最大抬升为 138.382 / 138.381 mm，平均最终法向误差 0.0312° / 0.0318°，累计回合实耗 54.31 / 56.37 s。RGB 耗时含渲染和推理。RGB 端面正确 120/120，初始 XY 平均、中位、最大误差为 0.248、0.231、0.529 mm。P2b-1 的离线 Test 结果仍为 58/58、XY 中位 0.319 mm。RGB 约 6.55 mm 的落点偏差远大于初始定位误差，与名义夹持变换无法修正闭合后的偏移相符。

## 独立降光测试与失败案例

large，种子 6000–6003，漫反射与环境光强度降低 20%，模型和估计器参数沿用标称设置。8/8 端面标签正确，4/8 返回有效位置且完成抓放。凸起端朝相机 4/4 成功，定位中位误差 3.455 mm。平面端朝相机 0/4，全部因 `localization_geometry_failed` 停在 Initialize。该拒判可能由暗边缘受颜色阈值裁剪导致定位框几何不一致。标称配对组未观察到操作失败。RGB 执行没有在线物体滑移、掉落或碰撞反馈，实际部署需要独立的夹爪和安全传感通道。接触按 50 Hz 检查，结果仅覆盖当前仿真工位。

## 复现与产物

```bash
export MUJOCO_GL=egl MESA_SHADER_CACHE_DISABLE=true PYTHONPATH=src
.venv/bin/python -m cr3_lab.tasks.p2b2 --episodes 20 --seed 5000 --output outputs/p2b2/paired
.venv/bin/python -m cr3_lab.tasks.p2b2 --mode rgb --profile large --episodes 4 --seed 6000 --dim --output outputs/p2b2/dim
.venv/bin/python -m cr3_lab.tasks.p2b2 --mode rgb --profile large --orientation protrusion_facing_camera --episodes 1 --seed 5000 --media --output outputs/p2b2/plus
.venv/bin/python -m cr3_lab.tasks.p2b2 --mode rgb --profile large --orientation flat_facing_camera --episodes 1 --seed 5001 --media --output outputs/p2b2/minus
.venv/bin/python -m cr3_lab.tasks.p2b2 --mode rgb --profile large --orientation flat_facing_camera --episodes 1 --seed 6000 --dim --media --output outputs/p2b2/dim-failure
.venv/bin/python -m pytest -q
```

包含失败回合的命令返回 1，并保存结果。各输出目录内 `results.json` 含逐回合结果。plus、minus 和 dim-failure 含 `motion.gif` 及双相机识别图 `rgb.png`，GIF 已验证解码。自动化测试 **26 passed**，含 P0–P2b-1 历史回归。沿用已有依赖与资产许可。等待 P2b-2 正式验收。

P2b-2 实现及实测报告提交为 `2bedf3c7abb359a5ed14413777368df50de6fae8`。

# P3a 夹爪闭合检查与 Gymnasium Reach

## 闭合实验

P0 执行器为单滑轨位置伺服，双指通过等位移约束联动。滑轨目标 0–37 mm，`kp=1500 N/m`、`kv=10 N·s/m`、执行器力限幅 ±35 N。Baseline 在 Grasp 首个 50 Hz 周期将归一化闭合目标从 0 跳到 1。渐进方案每周期增加 0.05（滑轨目标每周期增加 1.85 mm），两组均沿用 0.65 s Grasp 驻留。渐进指令从首个周期开始，在 0.38 s 发出第 20 个、即满闭合目标。该时间为指令到达时间，夹持工件时实际手指不会达到空载闭合终点。

`gripper.closure_probe` 复用 P2b-2 的轨迹执行路径。两组均用 Oracle 生成相同初始目标，隔离视觉误差对接触实验的影响。大范围种子 7000–7004、双朝向各五对。场景、零件质量、摩擦和求解器配置相同。Grasp 期间每 2 ms 记录滑轨速度、执行器力、工件速度和相对闭合开始的三维旋转角。指垫与工件接触力用 `mj_contactForce` 读取并按接触点求和。下表峰值先按回合计算，再取组均值。

| 实测指标 | Baseline | 渐进 |
| --- | ---: | ---: |
| 首次 / 双侧接触 | 8 / 10 ms | 122 / 128 ms |
| 峰值手指速度 | 1.997 m/s | 0.121 m/s |
| 峰值执行器力绝对值 | 35.00 N | 28.47 N |
| 接触后首 20 ms 法向力和峰值 | 17.41 N | 4.13 N |
| 整个闭合阶段法向力和峰值 | 27.76 N | 28.41 N |
| 整个闭合阶段接触力模长和峰值 | 27.83 N | 28.51 N |
| 峰值工件线速度 | 0.0966 m/s | 0.0259 m/s |
| 峰值工件角速度 | 1.953 rad/s | 0.258 rad/s |
| 峰值工件姿态变化 | 1.226° | 0.424° |
| 实际抓取 / 完整抓放 | 10/10 / 10/10 | 10/10 / 10/10 |
| 平均仿真回合时间 | 21.326 s | 21.326 s |
| 平均实耗 | 0.362 s | 0.355 s |

保留渐进闭合作为可复现实验方案，Reach 的夹爪指令也按每步最多 0.05 递增或递减。已验收 Oracle/RGB 入口继续使用 Baseline。渐进方案减轻初次接触冲击和工件晃动，整个闭合阶段峰值略有上升。本实验未验证其他夹持工况的效果。

## Reach 接口

`rl.reach.MotorReachEnv` 实现 Gymnasium 的 reset、step、Box spaces 和 info。复用 CR3、夹爪、MotorExpert 待命初始化、已有 TCP IK 与伺服控制器。默认任务为 `reach`，`task` 参数为后续 Grasp & Lift 预留入口，当前其他任务抛出 `NotImplementedError`。

| Observation 索引 | 量 | 单位与范围 |
| --- | --- | --- |
| 0–5 | CR3 关节角 | rad，URDF 限位各放宽 0.1 以容纳软限位偏差 |
| 6–11 | CR3 关节速度 | rad/s，Box 无有限上下限 |
| 12–14 | TCP World XYZ | m，无有限上下限 |
| 15–17 | 零件中心 World XYZ | m，无有限上下限，Oracle 特权 |
| 18–20 | 凸起端 World 单位法向 | [−1,1]，Oracle 特权 |
| 21–23 | Reach 目标 World XYZ | m，无有限上下限，由复位后 Oracle 中心加 65 mm 得到 |
| 24–26 | 放置目标 World XYZ | m，无有限上下限 |
| 27–28 | 两指滑轨位置 | m，Box [−0.005, 0.042]，含软限位余量 |
| 29 | 当前归一化闭合指令 | [0,1] |
| 30–32 | 当前命令 TCP World XYZ | m，无有限上下限 |

观测共 33 维 float32。动作 4 维 float32，范围 [−1,1]。前三维每轴最多移动目标 TCP 1.5 mm，工具姿态固定向下，第四维每步最多改变闭合指令 0.05。目标 TCP 限制在以默认零件中心 XY 为基准各 ±60 mm 的工作区，Z 在 `rest_z + 0.04` 到 `ready_z + 0.06` m。IK 不可达或单周期关节目标变化超过 0.03 rad 时保持上个 TCP 和关节目标。每次 step 积分 10 个 2 ms 物理步，即 20 ms。零件只在 reset 时初始化。

奖励为 `-min(距离_m, 1) - 0.01 × IK拒绝 + 成功 - 异常终止`。实际 TCP 距 Reach 目标 ≤8 mm 且无异常时成功终止。意外接触或仿真异常也终止。默认 150 步未终止则超时截断。info 返回原因、距离、步数、场景种子、朝向、成功和 IK 拒绝标记。接触检测按 50 Hz，Reach 只允许工件主体与桌面的支承接触。

Reset 沿用 fixed、small、large 和 ±Y 双朝向。Gymnasium RNG 决定场景种子和朝向，options 可显式传入 `scene_seed` 或 `orientation`。后续训练建议显式使用场景种子 8000–8099，独立测试 9000–9019。本轮训练种子只定义了范围，脚本检查使用测试种子。固定模式的位置始终相同。

## 实测与复现

- Gymnasium 1.2.3 和 Stable-Baselines3 2.7.1 的环境检查在三个位置模式均通过。Gymnasium 对无界 Box 给出提示。SB3 验证使用 PyTorch 2.7.1 CPU。
- 独立种子脚本 Reach 三组各 20/20。自动化测试 **30 passed**，包括 reset 一致性、动作与观测范围、160 个随机动作的有限性、20 ms 步长、成功终止、超时截断和采样钩子对状态的影响检查。
- 双朝向、三位置模式的抓放回归各两组种子，Oracle 和 RGB 各 **12/12**。本轮未运行 PPO 或 SAC。

```bash
.venv/bin/pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements-p3a.txt
# 此处为 P3a 历史命令，精确复现时在对应旧提交中执行。当前候选安装见 README。
export PYTHONPATH=src
.venv/bin/python -m cr3_lab.gripper.closure_probe --output outputs/p3a/closure.json
MPLCONFIGDIR=/tmp/cr3-p3a-mpl .venv/bin/python -m cr3_lab.rl.check_reach --output outputs/p3a/reach-check.json
MUJOCO_GL=egl MESA_SHADER_CACHE_DISABLE=true .venv/bin/python -m cr3_lab.tasks.p2b2 --episodes 2 --seed 5000 --output outputs/p3a/regression
.venv/bin/python -m pytest -q
```

已知局限包括 50 Hz 接触检查可能漏检瞬态、Reach 使用特权状态与固定工具姿态、工件接触仅在标称仿真参数下测试。后续建议先在 CPU 用低维 MLP 验证 Reach 的 PPO/SAC 收敛，按独立种子报告成功率、距离和样本效率，再实现 Grasp & Lift 的夹持和抬升判据。等待正式验收。

P3a 实现及实测报告提交为 `a2e0e37fe0caf7e1f7d1802df52f4f707b65fcfa`。

# P3b 渐进闭合与 PPO/SAC Reach 对比

## 抓放与环境回归

Oracle `tasks.p1a` 与 Oracle/RGB 配对入口 `tasks.p2b2` 默认使用渐进闭合，每 20 ms 增量最多 0.05。用 `--closure fast` 复现原始阶跃。打开仍立即发出开口目标。固定、±10 mm、±25 mm，双朝向各三回合，种子 5000–5002。两种闭合模式的 Oracle 和 RGB 分别各 18/18，未记录失握、意外接触或 IK 失败。渐进模式最大最终位置误差为 Oracle 0.249 mm、RGB 6.650 mm。保留原物理、视觉及成功条件。

Reach 动作为三维 `[dx, dy, dz]`，每维 [−1,1]、每周期最多 1.5 mm，夹爪保持张开。每步仍为 10 个 2 ms 物理步。原始低维观测减为 32 维，删除原索引 29 的闭合指令，命令 TCP 移至 29–31，其余索引与 P3a 一致。PPO、SAC 共享六维策略输入，即 `10 × (Reach目标 − 命令TCP)` 和 `10 × (命令TCP − 实际TCP)`，单位为 0.1 m。使用固定缩放，无训练统计量。Reach 目标由复位时 Oracle 位置生成，是明示的特权输入。

奖励改为 `5 × clip(d_prev − d_now, −0.01, 0.01) − 0.005 − 0.01×IK拒绝 + 成功 − 3×异常终止`，距离以 m 计并在计算前封顶 1 m。进度项强化接近目标，时间成本使静止和无进展动作产生负回报，异常终止惩罚高于等待到超时的时间成本。重复 IK 拒绝测试约 −0.015/步，目标保持。阈值仍为实际 TCP 距目标 ≤8 mm，异常接触和仿真故障终止，150 步超时。奖励与成功标记未作为策略输入。独立测试 IK 启发式逐步按当前目标差值规划，120/120 成功，平均 50 步。Gymnasium 和 SB3 检查三个模式均通过。自动化测试覆盖接近奖励、拒绝、保持张开、渐进闭合及历史回归。

## 训练与划分

`rl.train_reach` 在 CPU 使用 SB3 的 MlpPolicy，两层各 64 单元，gamma 0.99，学习率 0.0003。PPO 每次收集 500 步、batch 100、每批 rollout 更新 5 个 epoch。SAC replay buffer 100k、预热 1000 步、batch 128、每 4 步更新一次。其余参数沿用 SB3 2.7.1 默认。训练 reset 均匀选 fixed/small/large、随机选择端面，场景种子从 8000–8099 抽样。学习曲线用验证种子 8500–8503，覆盖三模式、双朝向，共 24 回合。正式 Test 用 9000–9019，共 120 回合，每种组合 20 次。fixed 在不同划分中的 XY 相同，随机模式按独立场景种子采样。

10k 步 Smoke 用种子 11。PPO 验证成功率 0% → 8.3%，最终距离 80.41 → 22.81 mm、奖励 −0.750 → −0.339。SAC 成功率 0% → 0%，距离 98.65 → 87.28 mm、奖励 −0.841 → −0.785，改善有限。两者环境检查、非有限值检查和 checkpoint 加载均通过，重载后确定性动作一致。训练实耗（含周期验证）为 19.04 / 26.60 s。正式实验冻结上述配置，采用训练随机种子 11、22、33，各 50k 环境步。

## 复现

```bash
export PYTHONPATH=src MPLCONFIGDIR=/tmp/cr3-p3b-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p outputs/p3b
.venv/bin/python -m pytest -q
.venv/bin/python -m cr3_lab.rl.check_reach --output outputs/p3b/reach-check.json
for mode in gradual fast; do
  MUJOCO_GL=egl MESA_SHADER_CACHE_DISABLE=true .venv/bin/python -m cr3_lab.tasks.p2b2 \
    --closure "$mode" --episodes 3 --seed 5000 --output "outputs/p3b/grasp-$mode"
done
for method in random ik; do
  .venv/bin/python -m cr3_lab.rl.evaluate_reach "$method" "outputs/p3b/$method.json"
done
for method in ppo sac; do
  .venv/bin/python -m cr3_lab.rl.train_reach "$method" "outputs/p3b/smoke-$method" \
    --steps 10000 --seed 11 >"outputs/p3b/smoke-$method.log" 2>&1
  for seed in 11 22 33; do
    run="outputs/p3b/formal-$method-$seed"
    .venv/bin/python -m cr3_lab.rl.train_reach "$method" "$run" --steps 50000 --seed "$seed" >"$run.log" 2>&1
    .venv/bin/python -m cr3_lab.rl.evaluate_reach "$method" "$run/test.json" --checkpoint "$run/model.zip"
  done
done
.venv/bin/python -m cr3_lab.rl.plot_reach outputs/p3b
```

正式训练在同一主机并行运行。每个 run 的 `config.json` 记录参数与种子，`progress.jsonl` 记录每 5k 步的验证指标，`training.json` 记录交互步数、实耗和重载校验，`model.zip` 是最终推理 checkpoint，`test.json` 保存所有测试回合和按位置、端面拆分的指标。实耗包含周期验证，受并行 CPU 负载影响。最终验证点在最后一次策略更新后重算，曲线取该点。重载校验覆盖推理，尚未验证恢复优化器与 SAC replay buffer 后继续训练。

## 正式结果

每个 checkpoint 在 120 个 Test 回合评估。下表距离、步数、奖励与 IK 拒绝率都包含失败回合。各分组和种子的原始数据见 `outputs/p3b/`。

| 方法 / 种子 | 交互步数 | 成功 | 最终距离 mm | 平均步数 | 平均奖励 | IK 拒绝率 | 训练实耗 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | — | 0/120 | 81.30 | 150.00 | −0.756 | 0.006% | — |
| IK 启发式 | — | 120/120 | 7.28 | 50.00 | 1.115 | 0% | — |
| PPO 11 | 50,000 | 120/120 | 6.99 | 50.98 | 1.111 | 0% | 95.91 |
| PPO 22 | 50,000 | 120/120 | 7.52 | 50.00 | 1.113 | 0% | 86.65 |
| PPO 33 | 50,000 | 120/120 | 7.43 | 50.00 | 1.114 | 0% | 93.50 |
| SAC 11 | 50,000 | 0/120 | 144.46 | 150.00 | −1.071 | 0% | 147.32 |
| SAC 22 | 50,000 | 0/120 | 128.72 | 150.00 | −1.877 | 58.93% | 149.16 |
| SAC 33 | 50,000 | 0/120 | 144.88 | 150.00 | −1.073 | 0% | 153.35 |

每种模式和端面组合，PPO 三个种子均各 20/20，SAC 均各 0/20，IK 各 20/20，Random 各 0/20。`outputs/p3b/learning-curves.png` 展示验证均值和种子间范围。PPO 在 25k、25k、30k 步首次达到 100% 验证成功率，种子 11 在 30k 回落至 75%，35k 起三个种子均保持 100%。SAC 在 50k 内所有验证点均为 0%。以 8500 号大范围场景诊断，三个最终 SAC 策略初始 dz 均为正（0.608–0.646），方向远离较低的目标。种子 22 同时有较高的 IK 拒绝，其余两种子没有拒绝仍失败。结果显示该固定 SAC 配置未学到有效的确定性趋近策略，未进一步归因于单个超参数，也未扩展到 100k。

全部模型重载推理检查通过。自动化测试 **32 passed**。Gymnasium 对部分原始 Box 维度无界有提示。此实验以特权目标和当前工作区为限，不能外推至 Grasp & Lift 或视觉策略。等待 P3b 正式验收。

P3b 实现及实测报告提交为 `205ff89a5399cc1505f4c123671cf6b691a42c9b`。

# P3b-R1 SAC 收敛诊断

## 冻结 checkpoint 审查

沿用 P3b 的 Reach、三维动作、六维相对观测、物理、奖励与成功阈值。旧模型、训练和测试数据保留在 `outputs/p3b/`。本轮新增产物在 `outputs/p3b-r1/`。

- SB3 2.7.1，单环境、每 4 个交互步更新 1 次，预热 1000 步。50k 的实际更新次数为 (50,000 − 1,000)/4 = 12,250，三个 checkpoint 一致。日志最后一行常早于最终更新，次数以 checkpoint 为准。
- 原配置为 `ent_coef=auto`，初始 α=1，`target_entropy=-3`。1200 步日志 α≈0.986，终点 α=0.02578 / 0.02571 / 0.02579。49,800 步时 Actor loss 约 −19.4 至 −19.6，Critic loss 约 0.000129–0.000226。小 Critic loss 表示其软 Bellman 目标残差小，无法说明目标动作排序正确。
- 训练和评估使用同一 `MotorReachEnv`、固定缩放的 `RelativeReach` 和同一动作边界。训练额外包装了场景抽样和 NaN 检查；评估枚举场景。检查器使用独立短回合实测 replay 的 obs、action、reward、next obs、done 和 timeout。SB3 保存的是末帧观测；Reach 成功切断 bootstrap，超时继续 bootstrap。历史 checkpoint 未保存 replay，无法逐条恢复历史采样或确认成功转移首次出现的时刻。
- Validation 24 回合上，原 SAC 确定性和随机评估均为每种子 0/24。确定性平均距离 129.04–143.85 mm，随机为 138.13–141.90 mm。随机动作以 Torch 种子 1600 复现。
- large/8500 的初始相对位移约 (−18.7, +13.0, −79.2) mm。纯向下首步缩短 0.253 mm、奖励 −0.00374；纯向上增加 0.256 mm、奖励 −0.00628。IK/PPO 首步均向下，原 SAC 三个种子 dz 分别为 +0.646、+0.608、+0.645，距离均增加。三组 critic 在该状态对纯向上的 Q 评价均比纯向下高约 0.09–0.12。代表性 large 轨迹未见 dz 长期饱和（绝对值 >0.95）。这些证据定位到学习所得的价值与策略偏好，未发现环境动作反号。
- 初始 α 与毫级步进奖励的数值尺度相差很大，且低更新率使自动温度下降较慢。熵项在 SAC 的 actor 损失和 Bellman 目标中都有影响。这是有限对照的假设，无法从旧日志单独分离温度、探索覆盖和数据分布的因果贡献。

## 对照设计

A 只把每 4 步的更新次数从 1 提到 4（目标 49,000 次更新），保留自动温度。B 保留每 4 步 1 次更新，只将自动温度替换为固定 α=0.005，检验熵相对任务回报的权重。两个配置各先用种子 11、50k 步，使用 8500–8503 验证集做选择。原始 Test 未用于选择。R1 最终独立测试预留 9500–9519，每配置 120 回合。未改变环境基础奖励。

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cr3-p3b-mpl
mkdir -p outputs/p3b-r1
.venv/bin/python -m cr3_lab.rl.diagnose_sac outputs/p3b outputs/p3b-r1/diagnosis.json
for mode in updates temperature; do
  .venv/bin/python -m cr3_lab.rl.train_reach sac "outputs/p3b-r1/$mode-11" \
    --sac-control "$mode" --seed 11 --steps 50000 >"outputs/p3b-r1/$mode-11.log" 2>&1
done
.venv/bin/python -m pytest -q
```

`diagnosis.json` 记录固定、small、large 的逐步 XYZ 动作、相对位置、距离变化、奖励、IK 拒绝、初始 Q 排序及旧训练日志解析。自动化测试 **35 passed**，新增回放终止与超时语义检查。

固定低温组种子 11 在最终 Validation 达到 20/24，平均距离 10.38 mm、平均步数 86.25、平均回报 0.752、IK 拒绝率 0%，耗时含验证约 117.4 s。依据该验证结果，冻结其配置，扩展同一对照的种子 22 和 33。R1 Test 仅用于冻结后的比较。

```bash
for seed in 22 33; do
  run="outputs/p3b-r1/temperature-$seed"
  .venv/bin/python -m cr3_lab.rl.train_reach sac "$run" --sac-control temperature \
    --seed "$seed" --steps 50000 >"$run.log" 2>&1
done
mkdir -p outputs/p3b-r1/test
for seed in 11 22 33; do
  .venv/bin/python -m cr3_lab.rl.evaluate_reach sac "outputs/p3b-r1/test/temperature-$seed.json" \
    --checkpoint "outputs/p3b-r1/temperature-$seed/model.zip" --test-set r1
  for method in ppo sac; do
    .venv/bin/python -m cr3_lab.rl.evaluate_reach "$method" "outputs/p3b-r1/test/baseline-$method-$seed.json" \
      --checkpoint "outputs/p3b/formal-$method-$seed/model.zip" --test-set r1
  done
done
.venv/bin/python -m cr3_lab.rl.evaluate_reach ik outputs/p3b-r1/test/ik.json --test-set r1
```

## 结果与结论

| 50k 对照 / 种子 | Validation 成功 | 距离 mm | 步数 | 回报 | IK 拒绝 | 更新次数 | 最终 α | 实耗 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 更多更新 / 11 | 0/24 | 92.50 | 150.00 | −0.811 | 0% | 49,000 | 0.000555 | 246.39 |
| 固定低温 / 11 | 20/24 | 10.38 | 86.25 | 0.752 | 0% | 12,250 | 0.005 | 117.39 |
| 固定低温 / 22 | 24/24 | 7.45 | 72.67 | 1.001 | 0% | 12,250 | 0.005 | 122.48 |
| 固定低温 / 33 | 20/24 | 8.61 | 80.92 | 0.788 | 0% | 12,250 | 0.005 | 122.36 |

| 新独立 Test 9500–9519 | 成功 | 最终距离 mm | 平均步数 | 平均回报 | IK 拒绝 |
| --- | ---: | ---: | ---: | ---: | ---: |
| IK | 120/120 | 7.28 | 50.00 | 1.114 | 0% |
| PPO 原三种子 | 360/360 | 6.97–7.51 | 50–51 | 1.110–1.113 | 0% |
| SAC 原三种子 | 0/360 | 128.52–144.77 | 150 | −1.877 至 −1.072 | 0–58.97% |
| SAC 固定低温 / 11 | 118/120 | 7.65 | 75.17 | 0.969 | 0% |
| SAC 固定低温 / 22 | 120/120 | 7.46 | 71.88 | 1.003 | 0% |
| SAC 固定低温 / 33 | 92/120 | 8.97 | 86.63 | 0.689 | 0% |

低温三种子合计 330/360。种子 11 在大范围同一 XY 的两种朝向均超时；种子 33 在 fixed 全成功，在 small、large 的两种朝向均各 13/20。高更新组未通过验证筛选。固定低温在本任务和预算下有效，但种子一致性和完成速度仍弱于 PPO 与 IK。仅提高更新次数未解决问题，且最终 α 更低仍无成功，说明温度变化的时序可能重要。该对照无法区分低初始 α 与取消自动温度各自的影响。未修改默认训练配置，未追加调参或训练预算。

训练实耗含周期验证，受并行 CPU 负载影响。新增 checkpoint 均通过推理重载一致性检查。已完成两组限定对照，等待正式验收。

P3b-R1 实现及实测报告提交为 `9593ade42a167fae2fa82e5f8f80f6c3cf32a6ad`。

# P3c-1 Grasp & Lift 环境与基础验证

`rl.grasp_lift.MotorGraspLiftEnv` 提供 Gymnasium reset、step、Box spaces 和 info。默认固定 XY、凸起朝 +Y、从 `grasp_ready` 出发。可选 small、large 和双朝向，沿用已验收场景及种子采样。相同 seed 可复位到一致状态。可选 `start="pregrasp"` 属于简化课程，复位时仅用 IK 初始化机械臂，再沉降和检查接触。专家基线表使用完整的 `grasp_ready` 起点。工件只在复位时设置状态，模型唯一等式约束为两指联动。

| 观测索引 | 内容 |
| --- | --- |
| 0–11 | 关节角 rad、速度 rad/s，各 6 维 |
| 12–14 | 物体抓取点相对 TCP 的 World XYZ，m，特权 |
| 15–20 | 物体中心 World XYZ，m，以及凸起法向，特权 |
| 21–26 | 物体 World 线速度 m/s、角速度 rad/s，特权 |
| 27 | 主体中心相对静置高度，m，特权 |
| 28–29 | 双指滑轨位置，m |
| 30 | 实际发送的归一化闭合目标 |
| 31–36 | 双指接触、桌面支承、保持进度、曾确认抓取、失握计数比例、回合进度。物体接触及相关判据属特权 |
| 37–39 | 命令 TCP World XYZ，m |

观测为 40 维 float32。索引 30–36 范围 [0,1]，其余 Box 无有限上下限。动作是 4 维 float32，范围 [−1,1]。前三维每轴最多 1.5 mm，沿用 IK 及控制器，末端固定朝下。第四维经 (a+1)/2 映射到闭合目标，并应用每周期 0.05 的渐进限幅。每步 10 个 2 ms 物理步，50 Hz。默认 400 步（8 s），依据完整专家试跑最长 359 步留出余量。

## 奖励与终止

距离 d 是实际 TCP 到 MotorExpert 质心抓取点的距离。无合格双侧接触时，势函数为 `−5 min(d, 0.3)`。双侧接触且 d≤35 mm 时，势函数为 `1 + 10 clip(h, 0, 0.10) + 0.5×保持进度`，有桌面支承时去掉高度项。每步回报为 `势函数差 − 0.005 − 0.02×IK拒绝 + 3×成功`。接触与高度使用势差，持续挤压不重复计分。异常终止步直接记 −5。

抓取确认需要双侧指垫接触、脱离桌面且高于静置中心至少 20 mm。成功要求同时接触、离桌且抬升 ≥100 mm，线速度 ≤0.015 m/s、角速度 ≤0.15 rad/s，连续保持 20 步（0.40 s）。确认抓取后连续五步失去携带条件判失握。意外接触（包括偏离抓取区的早期指垫接触）、仿真告警或非有限观测终止。达到步数上限则截断。接触量由 MuJoCo 活动约束读取，每控制周期采样。

## 实测与复现

专家复用 MotorExpert 的质心抓取、预抓取与垂直抬升参数，经 RL action 接口执行，物理和验收与随机策略一致。种子 10000–10004，每档双朝向各五回合。

| 档位 | 专家成功 | 首次双侧接触步 | 确认离桌抓取步 | 成功步 | 峰值抬升 | 峰值指垫法向力和 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 10/10 | 175.0 | 221.0 | 341.0 | 约 138 mm | 27.67 N |
| small | 10/10 | 177.3 | 223.3 | 343.3 | 约 138 mm | 27.66 N |
| large | 10/10 | 184.4 | 230.6 | 350.4 | 约 138 mm | 27.66 N |

表中步数与峰值均为组均值。Gymnasium 和 SB3 环境检查通过，Gymnasium 对无界 Box 给出提示。PPO/SAC 各运行 512 环境步，更新计数分别为 4 / 112，策略参数有限，VecCheckNan 未发现异常。Smoke 仅验证交互与更新链路。

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cr3-p3b-mpl
mkdir -p outputs/p3c-1
.venv/bin/python -m cr3_lab.rl.check_grasp_lift --output outputs/p3c-1/results.json >outputs/p3c-1/check.log 2>&1
.venv/bin/python -m pytest -q
```

自动化测试覆盖双起点、真实抬升、开爪失握、下降碰撞、超时、reset 一致性和接近奖励方向。采样结果与完整 SB3 日志留在忽略目录。

## 后续设计

`rollout(..., on_transition=...)` 可将专家 RL 动作对应的 observation、action、reward、next observation、terminated 和 truncated 交给后续记录器。SAC 默认可从空 replay buffer 开始随机探索。专家经验预填是后续独立对照，会改变采样分布。行为克隆需要单独的监督目标。本轮未实现预填。

后续建议先在固定工位区分完整起点与预抓取课程，报告分阶段成功、碰撞和失握，再扩展随机位置及双朝向。当前特权接触信号依赖仿真，50 Hz 指垫和碰撞采样可能漏检短时接触。本阶段未开展正式训练。

最终 400 步配置下，随机策略 0/30，全部超时且未确认抬升。自动化测试 **39 passed**，含历史功能回归。等待正式验收。

P3c-1 实现及实测报告提交为 `47eb6e39a61aad4449b0d26b6153b58f08d1aba9`。

# P3c-2 固定工位 Grasp & Lift 课程

场景沿用 52 × 76 mm 电机主体、原凸起、夹爪及物理参数。训练只用 fixed、凸起朝 +Y、预抓取起点。未改变 40 维特权观测、4 维动作和物理成功阈值。训练包装器固定放大几何量，索引 12–17、37–39 乘 10，27–29 乘 20，其余为 1。未从物体状态构造策略动作。只有独立 MotorExpert 基线使用质心路点，预抓取课程在复位时用 IK 初始化机械臂。没有专家 replay 或 BC。

原奖励用未折扣势差，而训练的 gamma 为 0.99。进入高势然后立即退出会留下正的折扣势差。现使用 `0.99 Φ(next) − Φ(now) − 0.005 − 0.02×IK拒绝 + 3×成功`。无合格接触时 `Φ=0.5/(1+10d)`，有合格双侧接触时沿用原高度和保持势函数。异常终止用 `−5−Φ(now)` 清零末态势贡献。测试验证接触循环、重复持有和异常终止不能凭势差产生正折扣净收益。超时作为 Gymnasium 截断，有限预算的无条件回报仍可能受末态势影响。成功仍需要双侧接触、离桌、抬升 100 mm 并连续稳定保持 0.4 s。抓取、抬升、保持分别报告 20 mm 已确认携带、100 mm 携带、最终成功。仅双侧接触另计。

场景 Train 11000–11099，Validation 12000–12003，Test 13000–13009。固定几何下独立种子主要改变轴向滚转，不能据此声称位置泛化。PPO/SAC 同用 64×64 MLP、gamma 0.99、学习率 0.0003、400 步上限。PPO rollout 400、batch 100、5 epoch。SAC 空 buffer，容量 100k，预热 500，batch 128，每 4 步更新一次，固定 α=0.005。先各 2k Smoke，再各 20k×2 种子。保留完整日志、未筛选的训练终点以及逐回合探索事件。

MotorExpert 在独立 Test 的预抓取和完整起点各 10/10，平均 240 和 341 步，均小于 400 步。随机策略均 0/10，预抓取起点 8 次异常接触、2 次超时，平均 122.8 步、回报 −5.299；完整起点全部超时，平均回报 −2.774。随机接触失败未产生高回报。2k Smoke 的 PPO/SAC 各完成八个探索回合，PPO 一次双侧接触，SAC 零次，均无确认抬升。Smoke 的早期奖励候选和结果另存于 `outputs/p3c-2/`，最终奖励的所有对照使用 `outputs/p3c-2/final/`。

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cr3-p3b-mpl
out=outputs/p3c-2/final
mkdir -p "$out"
.venv/bin/python -m pytest -q
for m in expert random; do
  .venv/bin/python -m cr3_lab.rl.train_grasp_lift "$m" "$out/$m.json"
done
for m in ppo sac; do
  .venv/bin/python -m cr3_lab.rl.train_grasp_lift "$m" "$out/smoke-$m" --steps 2000 >"$out/smoke-$m.log" 2>&1
  for s in 11 22; do
    run="$out/$m-$s"
    .venv/bin/python -m cr3_lab.rl.train_grasp_lift "$m" "$run" --steps 20000 --seed "$s" >"$run.log" 2>&1
    .venv/bin/python -m cr3_lab.rl.train_grasp_lift "$m" "$run/test.json" --checkpoint "$run/model.zip"
  done
done
.venv/bin/python -m cr3_lab.rl.plot_grasp_lift "$out"
```

## 20k 正式对照结果

| 方法 / 种子 | 探索已结束回合 | 双侧接触 | 确认抓取 / 100 mm / 保持 | 碰撞 | 失握 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PPO 11 | 55 | 1 | 0 / 0 / 0 | 8 (14.5%) | 0 |
| PPO 22 | 59 | 1 | 0 / 0 / 0 | 12 (20.3%) | 0 |
| SAC 11 | 53 | 0 | 0 / 0 / 0 | 5 (9.4%) | 0 |
| SAC 22 | 51 | 1 | 0 / 0 / 0 | 2 (3.9%) | 0 |

四个最终 checkpoint 在独立 Test 中，预抓取课程与完整起点分别各 0/10（每算法每起点合计 0/20）。全数 400 步超时，双侧接触、确认抓取、100 mm 抬升和稳定保持均为零，失握、碰撞和 IK 拒绝也均为零。完整起点为零样本迁移评估。下表为两个起点的平均回报。

| 方法 / 种子 | 预抓取 | 完整起点 |
| --- | ---: | ---: |
| MotorExpert | 0.967 | 0.380 |
| Random | −5.299 | −2.774 |
| PPO 11 | −2.939 | −2.677 |
| PPO 22 | −2.888 | −2.671 |
| SAC 11 | −3.089 | −2.695 |
| SAC 22 | −3.059 | −2.708 |

在 Validation seed 12000 的动作诊断中，四个策略首步 dz 都为正（+0.215 至 +1），全回合最近抓取距离仍约 45 mm，PPO 保持张开，SAC 最大闭合指令约 0.41–0.44。大量超时与少量碰撞、极稀有双侧接触和零携带经验共同表明，这个预算下策略未探索到有效抓取序列。专家结果验证 400 步足以执行已知合法轨迹，尚无成功学习轨迹可验证学习策略需要的时长。没有证据足以单独归因于某个奖励项。固定课程未稳定成功，停止在本预算，未开展 small/large 和反向学习评估。后续若验收允许，专家 replay 可作为独立、等预算对照，优先检验成功转移稀缺这一假设。

产物在 `outputs/p3c-2/final/`，包括 `validation-curves.png`、`action-diagnostics.json`、独立 Test、每次训练配置、完整日志、验证时序、探索事件及模型。**40 项自动化测试通过**。沿用 P3b 依赖，未新增模型或许可。等待正式验收。

P3c-2 实现及实测报告提交为 `c8d379bc447d1db271db15c5a8d0f9205199d025`。

# P3c-2-R1 专家经验辅助 Grasp & Lift

沿用 P3c-2 的固定位置、凸起朝 +Y、Pre-Grasp 起点、标准 52 × 76 mm 零件、物理、奖励和成功条件。旧 PPO/SAC 实验保存在 `outputs/p3c-2/final/`。本轮只对照一种方法，即在 SAC 开始在线交互之前将专家 transition 写入普通 Replay Buffer，由 SAC 按原有规则采样。未做 BC、专家单独加权或额外离线梯度更新。在线预热仍为 500 步，固定 α=0.005，训练场景与 Validation/Test 沿用原划分。

专家数据由 MotorExpert 质心路点经 Gymnasium Action 接口采集，训练种子池内 11000–11007 共 8/8 成功，每回合 240 步，合计 **1,920 个示范环境步**，采集实耗约 1.20 s。NPZ 保存原始 observation、归一化 action、reward、next observation、terminated、truncated 及回合种子与偏移。8 个末帧均为成功终止，零个截断。完整动作重放逐步核对了缩放观测、后继观测、奖励、边界与终止信号，各项一致。写入 Replay Buffer 时只缩放两侧观测，动作已处在 SAC 的 [−1,1] 范围。真实终止阻断 bootstrap，导入逻辑将单独的截断标记映射为 SB3 timeout。

同一数据集跨种子复用。每个模型使用 1,920 条专家数据及 20,000 个独立在线环境步。每个种子的样本预算为 21,920 条采集 transition，跨种子唯一采集总计 41,920 步。生成数据时完整回放额外执行 1,920 步，训练前每次重新验证也各执行 1,920 步。这些审计步不进入训练、也不计入模型交互预算。验证和测试 rollout 同样不进入 Replay Buffer。

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cr3-p3b-mpl
out=outputs/p3c-2-r1
mkdir -p "$out"
.venv/bin/python -m cr3_lab.rl.expert_replay "$out/demos.npz"
for seed in 11 22; do
  run="$out/sac-$seed"
  .venv/bin/python -m cr3_lab.rl.train_grasp_lift sac "$run" --steps 20000 \
    --seed "$seed" --expert-data "$out/demos.npz" >"$run.log" 2>&1
  .venv/bin/python -m cr3_lab.rl.train_grasp_lift sac "$run/test.json" \
    --start pregrasp --checkpoint "$run/model.zip" >>"$run.log" 2>&1
done
.venv/bin/python -m pytest -q
```

## 配对结果

每行均为 20k 在线步。双侧接触、确认抓取（离桌 ≥20 mm）、100 mm 抬升和保持按已结束回合计数，每回合每项最多计一次。未结束尾段计入在线步数。

| 方法 / 种子 | 示范步 | 在线结束回合 | 双侧接触 | 确认抓取 / 抬升 / 保持 | 异常接触回合 | Test 保持 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO 11 | 0 | 55 | 1 | 0 / 0 / 0 | 8 | 0/10 |
| PPO 22 | 0 | 59 | 1 | 0 / 0 / 0 | 12 | 0/10 |
| SAC 空 Buffer 11 | 0 | 53 | 0 | 0 / 0 / 0 | 5 | 0/10 |
| SAC 空 Buffer 22 | 0 | 51 | 1 | 0 / 0 / 0 | 2 | 0/10 |
| SAC 专家预填 11 | 1,920 | 62 | 0 | 0 / 0 / 0 | 15 | 0/10 |
| SAC 专家预填 22 | 1,920 | 67 | 1 | 0 / 0 / 0 | 20 | 0/10 |

专家预填双侧接触为 1/129（0.78%），空 Buffer SAC 为 1/104（0.96%），未观察到改善。异常接触从 7/104 升至 35/129。两组均无确认抓取，因而无失握记录。专家预填在最终 Validation 各 0/4，Test 各 0/10，全部 400 步超时且无双侧接触、抬升、异常接触或 IK 拒绝。两个 Test 平均回报为 −2.901 和 −2.951，空 Buffer 对应 −3.089 和 −3.059。专家预填各 4,875 次更新，含周期验证训练耗时 55.81 和 49.80 s，策略参数均有限。

当前方案未改善有效探索，按预算停止。普通均匀 Replay 中示范占比由在线学习开始时约 79% 降至末尾约 8.8%，且八条示范来自固定工位同一控制路线。实验未隔离采样稀释、轨迹多样性及优化行为的各自影响。保留两个失败模型和日志。数据与逐回合结果位于 `outputs/p3c-2-r1/`，未将权重、数据集或日志纳入 Git。全量 **41 项测试通过**。等待验收。

P3c-2-R1 实现及实测报告提交为 `49c8df491902722110e435cd160d987ddea5d699`。

# P3c-3 Behavior Cloning 抓取可行性

沿用标准 52 × 76 mm 零件、固定 XY、凸起朝 +Y 和 Pre-Grasp 起点。复用 P3c-2-R1 的 8 条 MotorExpert 成功示范，共 1,920 步，逐步回放校验后按整条轨迹切分。Train 为 11000–11005（1,440 步），Validation 为 11006–11007（480 步），闭环 Test 为 13000–13009。独立种子改变轴向滚转，工位和初始任务几何固定。BC 不消耗在线训练步，闭环评估另计。

40 维观测包含抓取点相对 TCP、夹爪位置和指令、双侧接触、桌面支承、物体高度/速度及保持进度。这些信息能区分多数物理阶段。专家还使用驻留计时，观测中的回合进度可充当时间代理。以训练集现有缩放后的 L2 距离 ≤0.01、归一化动作最大分量差 ≥0.5 为阈值，排除相同步号的样本对。完整观测有 360/3,240 对冲突，删除时钟后有 1,044/6,192 对。典型冲突在初始驻留切换到下探，观测近似相同，dz 从 0 跳到 −0.8。计数包含跨轨迹复现的同类切换，不能视作独立事件。

## 监督学习与闭环实测

MLP 为两层各 64 单元、Tanh 隐层与四维 Tanh 输出，复用原缩放及 [−1,1] 动作。Adam 学习率 0.003、batch 128、MSE、400 epoch、训练种子 11。仅用 Validation MSE 选 checkpoint。另做一个删除观测索引 36（回合进度）的对照。

| 输入 | 选中 epoch | Train MAE (dx, dy, dz, 夹爪) | Validation MAE | Validation RMSE |
| --- | ---: | --- | --- | --- |
| 完整 | 397 | 0.00162, 0.00199, 0.03264, 0.01545 | 0.00162, 0.00199, 0.03264, 0.01545 | 0.00208, 0.00272, 0.08300, 0.11886 |
| 无时钟 | 376 | 0.00069, 0.00112, 0.03206, 0.01848 | 0.00069, 0.00112, 0.03206, 0.01848 | 0.00101, 0.00161, 0.08364, 0.12181 |

闭环每步用策略预测动作并调用真实 Gymnasium step。100 mm 抬升与连续稳定保持判据沿用原环境。

| Pre-Grasp Test | 双侧接触 | 确认抓取 | 100 mm 抬升 | 稳定保持 | 碰撞 | 平均步数 | 平均峰值抬升 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MotorExpert | 10/10 | 10/10 | 10/10 | 10/10 | 0 | 240 | 138.20 mm |
| BC 完整 | 10/10 | 10/10 | 10/10 | 10/10 | 0 | 210 | 133.11 mm |
| BC 无时钟 | 10/10 | 10/10 | 10/10 | 10/10 | 0 | 207 | 134.09 mm |

测试未出现闭环失败、失握或 IK 拒绝。动作预测在 dz 和夹爪维度误差较大，与阶段切换的相近状态异标签一致。当前闭环结果显示这些误差在固定课程下未累积至失败。未验证完整待命起点、位置随机化、初始双朝向或扰动恢复。固定工位下训练和验证轨迹高度相似，低监督误差不能证明更广的场景适用性。无 BC→SAC 微调，物理和任务条件保持原样。

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cr3-p3b-mpl
mkdir -p outputs/p3c-3
.venv/bin/python -m cr3_lab.rl.behavior_clone outputs/p3c-3/run \
  --demos outputs/p3c-2-r1/demos.npz >outputs/p3c-3/run.log 2>&1
.venv/bin/python -m pytest -q
```

模型、逐 epoch 抽样误差、近邻冲突和逐回合指标位于 `outputs/p3c-3/run/`。沿用 PyTorch CPU 与现有依赖。自动化测试 **44 passed**，覆盖历史回归、动作范围和观测近邻诊断。等待正式验收。

P3c-3 实现及实测报告提交为 `e47b3fcc33b34fad78f2d69a75e90ec06dd05b11`。

# P3c-4 冻结 BC 的位置扰动闭环验证

复用 P3c-3 两个 BC checkpoint 和八条示范。零件为 52 × 76 mm、凸起朝 +Y。只在 reset 偏移电机初始 XY。Pre-Grasp 仍对准标称零件位置，所有方法以同一初态、物理、动作伺服、判据和 400 步上限运行。±2、±5、±10 mm 均指每轴幅度，测试四个轴向边中点和四角。固定点和每个偏移位置使用种子 14000–14009，种子只改变轴向滚转。共 250 个配对场景，三方法共 750 个回合。

Open-Loop Replay 取示范种子 11000 的完整 240 步归一化动作序列，按回合步号播放。末尾保持当前 TCP 目标和最终闭合指令。回放不读取物体观测或重新规划抓取，底层 IK 和关节位置伺服仍与 BC 相同。冻结 BC 每步读取同样的 Gymnasium 特权观测并输出动作。

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cr3-p3b-mpl
mkdir -p outputs/p3c-4
.venv/bin/python -m cr3_lab.rl.bc_robustness \
  outputs/p3c-2-r1/demos.npz outputs/p3c-3/run outputs/p3c-4/results.json \
  >outputs/p3c-4/run.log 2>&1
.venv/bin/python -m pytest -q
```

八条轨迹均为 240 步。按 28 个轨迹对比较对应时刻，共 6,720 个观测和动作对。动作全部逐分量完全相同。观测原始最大分量差约 4.66e-10，轨迹划分隔开了种子，但几乎没有改变输入与预测标签。因此 Train/Validation MAE 高度相似，无法用这一划分证明几何泛化。

其中 61.98% 的缩放观测逐分量相同，100% 的对应时刻观测对在缩放 L2 距离 0.001 内，最大为 4.66e-10。相同动作比例为 100%。

| 策略 | 每轴偏移 | 双侧接触 / 确认抓取 / 100 mm / 稳定保持 | 碰撞 / 失握 | 平均回合步数 |
| --- | --- | --- | --- | ---: |
| 回放 | 0 | 10/10 / 10/10 / 10/10 / 10/10 | 0 / 0 | 240.0 |
| 回放 | ±2、±5、±10 mm，各档 | 80/80 / 80/80 / 80/80 / 80/80 | 0 / 0 | 240.0 |
| BC 完整 | 0 | 10/10 / 10/10 / 10/10 / 10/10 | 0 / 0 | 210.0 |
| BC 完整 | ±2、±5 mm，各档 | 80/80 / 80/80 / 80/80 / 80/80 | 0 / 0 | 210.0 |
| BC 完整 | ±10 mm | 40/80 / 20/80 / 20/80 / 20/80 | 60 / 0 | 106.1 |
| BC 无时钟 | 0 | 10/10 / 10/10 / 10/10 / 10/10 | 0 / 0 | 207.0 |
| BC 无时钟 | ±2 mm | 80/80 / 80/80 / 80/80 / 80/80 | 0 / 0 | 200.9 |
| BC 无时钟 | ±5 mm | 50/80 / 50/80 / 50/80 / 50/80 | 0 / 0 | 271.4 |
| BC 无时钟 | ±10 mm | 30/80 / 20/80 / 20/80 / 20/80 | 60 / 0 | 113.9 |

平均步数含碰撞早停和超时。±5 mm 无时钟模型在三个 −Y 偏移点共 30 次超时。±10 mm 两模型的失败均为指垫碰桌或抓取区外碰工件。开环回放在所有偏移均成功，说明固定 Pre-Grasp 合理且该扰动域内标称动作已足够。BC 会因观测变化采取不同动作，但测试未验证这种反馈能纠错。八条轨迹的观测覆盖极窄，偏移超出示范分布；现有数据无法分离阶段切换和闭环误差各自的贡献。

暂缓 BC→SAC。当前工位的开环成功率已到上限，继续用它比较难以验证反馈收益。若后续需要区分反馈能力，应先找出具备可行抓取轨迹、但标称回放无法完成的起点或扰动，并留出独立场景。此次复用 1,920 个示范步及原 BC 的 400 epoch，新增训练或 SAC 在线交互均为 0。示范校验和闭环评估为独立仿真开销。

全量 **46 项自动化测试通过**，新增固定待命姿态与偏移边界、对齐重复率检查。逐回合数据 `outputs/p3c-4/results.json`，日志 `outputs/p3c-4/run.log`。原 checkpoint 和历史实验保留。等待正式验收。

P3c-4 实现及实测报告提交为 `82e63ce943bf0ba20a249261ce312173fa547786`。

# P3c-5 随机工位 Grasp & Lift 配对基准

使用固定 `grasp_ready` 待命姿态，沿用标准电机主体直径 52 mm、长 76 mm，含凸起总质量 198 g。模型、夹爪、物理、成功判据和 400 步上限沿用环境原配置。双指接触、离桌 ≥20 mm、双指携带 ≥100 mm、连续稳定保持 0.40 s 分别记为四个里程碑。

在 fixed、+Y、种子 15000 上通过环境实际执行并录制完整专家动作。开环组按步号播放记录，末尾保持 TCP 命令和闭合目标，仍使用两组共有的底层 IK 和伺服。动态 Oracle Expert 每回合根据沉降后物体质心重算抓取与抬升路点。它在回合起点适应零件位置，实验未考查搬运过程中的在线重规划。所有配对逐项核对相同的初始观测向量。两个方法都从标称待命姿态出发。

开发种子 15100–15102，独立测试种子 15200–15219。开发集 fixed/small 双朝向均各 3/3，large 的专家和回放分别各 3/3 和 1/3，因此沿用标准位置条件。每种朝向复用相同种子的 XY，计数为朝向回合数。

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cr3-p3b-mpl
mkdir -p outputs/p3c-5
for split in dev test; do
  .venv/bin/python -m cr3_lab.rl.grasp_benchmark "outputs/p3c-5/$split" --split "$split" \
    >"outputs/p3c-5/$split.log" 2>&1
done
.venv/bin/python -m pytest -q
```

## 独立配对结果

参考轨迹为 341 步，保存在 `reference-actions.npy`。下表每行 20 个配对场景。四项依次为双侧接触、确认抓取、100 mm 抬升及稳定保持。步数为全部回合均值，含提前终止。成功回放均为 341 步。

| 模式 / 凸起方向 | Oracle 四项计数 | 回放四项计数 | 回放碰撞 | 平均步数 Oracle / 回放 |
| --- | --- | --- | ---: | ---: |
| fixed / +Y | 20/20/20/20 | 20/20/20/20 | 0 | 341.0 / 341.0 |
| fixed / −Y | 20/20/20/20 | 20/20/20/20 | 0 | 341.0 / 341.0 |
| small / +Y | 20/20/20/20 | 20/20/20/20 | 0 | 344.7 / 341.0 |
| small / −Y | 20/20/20/20 | 20/20/20/20 | 0 | 343.8 / 341.0 |
| large / +Y | 20/20/20/20 | 8/8/8/8 | 12 | 353.25 / 209.05 |
| large / −Y | 20/20/20/20 | 7/7/7/7 | 13 | 352.45 / 198.35 |

Oracle 120/120，回放 95/120。Oracle 无碰撞，两组无失握。25 个仅 Oracle 成功的配对涉及 13 个 large 场景种子；回放均在双侧接触前因 `pad / motor outside grasp region` 终止。例：+Y、种子 15206，初始 XY 为 (0.43431, 0.11443) m，回放第 124 步碰撞，Oracle 第 361 步完成保持。完整配对、失败阶段、接触、抬升及逐回合步数在 `outputs/p3c-5/test/results.json`。

## 后续决策

1. 固定与 ±10 mm 内，本次回放足以完成任务。±25 mm 内有可信且可复现的差异，当前物体位置可用于调整下探与夹持路线。
2. 差异发生在接近期间，配对初态和执行接口逐项一致。专家能在原成功条件内完成，结果支持位置适应需求。实验只比较一条标称开环路线，未证明所有开环路线都必然失败。
3. 若继续研究抓取学习，值得在 large 场景采集覆盖不同 XY、双朝向和失败邻域的成功轨迹，并以独立 XY 留出验证。先用 BC 对比开环和 Oracle，评估是否学到重规划行为。当前证据不足以直接进入 BC→SAC。若项目优先整理，现有配对基准也可独立交付。

保留冻结 BC 与所有历史结果。未训练模型、未增加扰动。**47 项自动化测试通过**，新增完整轨迹录制、开环无专家查询及不同工位共用固定待命点检查。完整日志在 `outputs/p3c-5/{dev,test}.log`。等待正式验收。

P3c-5 实现及实测报告提交为 `a482a06412b6fc0bcbe09ac7ca6e4b5ee8a2d2e2`。

# P3c-6 多样化专家数据与位置条件化 BC

沿用 P3c-5 的 52 × 76 mm 电机、固定 `grasp_ready`、4 维动作、400 步预算、物理及成功条件。每回合由动态 Oracle 专家从当前物体质心计算路点，通过 Gymnasium step 执行。按 1 µm 分箱检查实际复位 XY 在三集合间无重复；同一位置的两个朝向配对。所有样本按预定种子一次采集，代码保留失败轨迹和结果，仅成功示范参与监督拟合。

| 划分 | 场景种子 | 不同 XY | 双朝向回合 | 专家监督步数 | 距任一边界 ≤5 mm 的 XY |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | 16000–16063 | 64 | 128 | 45,059 | 21 |
| Validation | 16100–16115 | 16 | 32 | 11,260 | 7 |
| Test | 16200–16219 | 20 | 40 | 0 | 见逐回合结果 |

Train 偏移 X [−23.45, +24.75] mm、Y [−23.17, +23.47] mm，Validation 偏移 X [−20.96, +24.83] mm、Y [−21.49, +23.99] mm。160 条示范全部通过双指接触、确认抓取、100 mm 携带和稳定保持，均包含横移接近、下探、闭合和抬升动作。训练轨迹按相同时刻对齐前 339 步，8,128 个轨迹对无完全相同的动作序列。逐步相同动作比例为 86.55%，包含共享的下探、夹持及抬升段。对应时刻物体位置最大跨度 X 49.18、Y 47.12 mm。以每 16 步加动作跳变点采样，3,664 个状态中有 115 对缩放 L2 距离 ≤0.01，其中 77 对动作最大分量差 ≥0.5。这是诊断抽样，近邻冲突可能来自驻留和阶段切换。

复用 P3c-3 的 64×64 Tanh MLP、40 维特权观测和四维 Tanh 输出。保留工件真值、接触与回合进度，输入无专家路点、隐藏状态或未来动作。先应用原 SCALE，再用 Train 均值和标准差归一化（最小标准差 0.02，截断至 ±8）。Adam 0.003、batch 256、MSE、种子 11、最多 100 epoch，Validation 连续 15 epoch 无改善则停止。冻结第 27 epoch 的最低验证误差权重。

| 监督 MAE，归一化动作 | dx | dy | dz | 夹爪 |
| --- | ---: | ---: | ---: | ---: |
| Train | 0.01004 | 0.00712 | 0.01932 | 0.01270 |
| Validation | 0.01046 | 0.00702 | 0.01975 | 0.01269 |
| Train 动作跳变点 | 0.02413 | 0.02262 | 0.27537 | 0.24818 |
| Validation 动作跳变点 | 0.02575 | 0.02181 | 0.27951 | 0.25346 |

跳变点以相邻专家动作任一分量变化 ≥0.5 定义。逐维 RMSE、验证曲线及训练步数见 `training.json`。第一次采集的审计 JSON 遇到 NumPy 整型序列化错误，已修复并对同一批保存的示范重做审计，未重新筛选或采样。

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cr3-p3b-mpl
mkdir -p outputs/p3c-6
for phase in capture train test; do
  .venv/bin/python -m cr3_lab.rl.position_bc "$phase" outputs/p3c-6/run \
    >"outputs/p3c-6/$phase.log" 2>&1
done
.venv/bin/python -m pytest -q
```

## 冻结模型独立闭环

20 个新 XY、双朝向共 40 个配对。复用 P3c-5 的 341 步标称回放，三组每回合逐项核对初始观测一致。Test 偏移 X [−23.89, +23.14] mm、Y [−23.93, +23.55] mm。外缘定义为任一轴绝对偏移 ≥20 mm。

| 分组 | Oracle 保持 | 开环保持 | BC 保持 |
| --- | ---: | ---: | ---: |
| 全部 | 40/40 | 22/40 | 40/40 |
| +Y | 20/20 | 11/20 | 20/20 |
| −Y | 20/20 | 11/20 | 20/20 |
| 内部 | 24/24 | 20/24 | 24/24 |
| 外缘 | 16/16 | 2/16 | 16/16 |
| Oracle 成功且开环失败 | 18/18 | 0/18 | 18/18 |

每种方法的双侧接触、确认抓取、100 mm 抬升计数与保持计数相同。Oracle、开环、BC 平均回合步数分别为 351.7、241.3、328.9，开环均值含碰撞早停。开环 18 次异常接触，Oracle 与 BC 均无，三组无失握。开环失败集中在双侧接触前的抓取区外指垫接触，外缘贡献 14 次。逐回合 JSON 保存种子、XY、阶段、碰撞对象和各里程碑，便于复查配对。

结果支持 BC 在标称物理和完整待命起点下，能依据初始场景变化调整抓取行为。固定标称动作在同一批 XY 有明显差距。模型保留回合时钟，实验没有分离时钟对阶段进度的贡献，也未注入执行中扰动，因此不推断在线纠错或扰动恢复。结论限于单一物体和当前 ±25 mm 工位。未追加模型修订，未训练 PPO/SAC，也未开展 BC→SAC。

产物位于 `outputs/p3c-6/run/`，历史数据与 checkpoint 保留。**49 项自动化测试通过**，新增 XY 划分与覆盖、闭环策略接收当前观测检查。等待正式验收。

P3c-6 实现及实测报告提交为 `4ca6370379a176a7a090c406a2b4cccb9faa864c`。

# P3c-7 冻结 BC 挑战场景筛查

从 P3c-6 基线 `4ca6370379a176a7a090c406a2b4cccb9faa864c` 的 checkpoint 加载权重和输入缩放。实验仅在复位替换物体 XY 采样，保留标称 `grasp_ready`、既有控制与 400 步上限。两种方法在每次配对前逐项核对初始观测。动态 MotorExpert 按初态重算路径。冻结 BC 每步从原 40 维观测推理，使用原有特权观测接口。所有物理、默认场景和成功条件沿用基线。原始 checkpoint 和采集数据通过 SHA-256 核对。

| 每轴范围 | 开发种子 | 独立 Test 保留种子 |
| --- | --- | --- |
| ±25 mm | 17000–17019 | 17500–17539 |
| ±30 mm | 17100–17119 | 17600–17639 |
| ±35 mm | 17200–17219 | 17700–17739 |

每档 20 个均匀采样 XY，双朝向共 40 个配对，保留种子本轮不运行。用主体半径 26 mm 与轴向最大伸出 53 mm 检查桌面净空。最宽档的保守最小桌边净空为 X 119 mm、Y 22 mm，零件距离工位中线至少 2 mm。每档先跑四角、双朝向共 8 个完整专家回合，实测 IK 与碰撞，角点单列。

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cr3-p3b-mpl
.venv/bin/python outputs/p3c-7/benchmark.py >outputs/p3c-7/run.log 2>&1
.venv/bin/python -m pytest -q tests outputs/p3c-7/test_benchmark.py
```

## 开发结果与选择

| XY 范围 | 方法 | 双侧接触 | 确认抓取 | 100 mm | 保持 | 碰撞 / 失握 | 平均步数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ±25 mm | Oracle | 40 | 40 | 40 | 40 | 0 / 0 | 351.23 |
| ±25 mm | BC | 40 | 40 | 40 | 40 | 0 / 0 | 327.08 |
| ±30 mm | Oracle | 40 | 40 | 40 | 40 | 0 / 0 | 355.40 |
| ±30 mm | BC | 40 | 39 | 39 | 39 | 1 / 0 | 328.15 |
| ±35 mm | Oracle | 40 | 40 | 40 | 40 | 0 / 0 | 356.63 |
| ±35 mm | BC | 37 | 34 | 34 | 34 | 3 / 0 | 328.03 |

各行分母为 40。每档角点 Oracle 均 8/8，开发集 Oracle 无 IK 拒绝。±30 mm 的唯一失败为 17109 平面端，双侧接触后发生抓取区外接触。±35 mm 的六次失败来自三个 XY（17208、17211、17218），两种朝向各三次。三次碰撞均出现在双侧接触后、确认抓取前，环境记录为 `pad / motor outside grasp region`。三次超时发生在双侧接触前，各有 256、330、346 次 IK 拒绝。失败位置的初始 X 为 0.4830–0.4841 m，靠近 +X 边缘。配对专家均成功，说明这些初态存在可行轨迹。IK 拒绝记录反映实际执行，但尚未进一步分离位置不可达与单步关节增量限制。

选择 **±35 mm** 作为 P3c-8 候选。开发集 Oracle 100%、BC 85%，差 15 个百分点，且两种朝向都出现差异。值得验证在线微调能否改善这一缺口。改善幅度仍待实测，开发集也不能替代保留的独立 Test。当前差异来自初始位置外推，未注入执行中扰动。位置测试已达到筛选标准，本轮未改变物理参数或训练任何模型。

种子 17211、凸起朝 +Y 的 Oracle 成功与 BC 失败 GIF 分别为 `outputs/p3c-7/35-17211-oracle.gif` 和 `outputs/p3c-7/35-17211-bc.gif`。逐回合结果和日志位于同目录。

```bash
sha256sum -c outputs/p3c-7/frozen.sha256
MUJOCO_GL=egl MESA_SHADER_CACHE_DISABLE=true .venv/bin/python \
  outputs/p3c-7/benchmark.py --gif 35 17211 protrusion_facing_camera
```

**50 项自动化测试通过**，含实验采样、种子分离、质量与固定机械臂初态检查。等待正式验收。

P3c-7 实验实现及实测报告提交为 `e5eed716d0ed89440f81e64d10ef136c0278614f`。

# P3c-8 BC 失败机制与 RL 前置审计

从 P3c-7 实验提交 `e5eed716d0ed89440f81e64d10ef136c0278614f` 复用复位采样、模型加载和控制实现。只重放开发种子 17208、17211、17218，双朝向各一次，逐项核对 Oracle 与 BC 初态相同、结果及接触里程碑与 P3c-7 一致。保留的 Test 种子未运行，权重与示范 SHA-256 核对通过。未修改物理、奖励、阈值或数据，未训练任何模型。

## 配对诊断

| 种子 / 凸起 | Oracle 成功步 | BC 结果 / 步 | IK 未收敛 | 关节增量拒绝 | 推理输入有维度截断的步数 |
| --- | ---: | --- | ---: | ---: | ---: |
| 17208 / +Y | 365 | 超时 / 400 | 108 | 148 | 132 |
| 17208 / −Y | 365 | 抓取区外接触 / 244 | 0 | 0 | 21 |
| 17211 / +Y | 365 | 抓取区外接触 / 253 | 0 | 0 | 30 |
| 17211 / −Y | 369 | 抓取区外接触 / 238 | 0 | 0 | 38 |
| 17218 / +Y | 367 | 超时 / 400 | 324 | 6 | 380 |
| 17218 / −Y | 366 | 超时 / 400 | 340 | 6 | 374 |

**IK 超时。** 三个超时均未形成双侧接触。TCP 目标从 290 mm 待命高度上移，分别达到 Z 323、308、311 mm，从未下探至待命高度以下。首次拒绝分别在 113、34、33 步，均为求解成功但 J3 单步变化超过 0.03 rad（绝对值约 0.0370、0.03002、0.03072 rad）。随后多次迭代未达到 IK 残差阈值，共 772 次，另有 160 次单步增量拒绝。拒绝保持上次 TCP 和关节目标，夹爪命令仍执行。三次超时末段 TCP 离抓取点约 181–195 mm。数据证明当前 BC 指令和当前 IK 分支执行失败，单次求解器未收敛不证明目标在所有分支上都不可达。

**碰撞。** 三次失败都是已双侧接触、尚未确认抓取时指垫仍接触工件，但 TCP 距质心抓取点超过原阈值 35 mm。终止距离分别为 35.42、35.02、35.19 mm，TCP 世界 Z 约 162.4–162.6 mm；工件仍接近桌面静置高度。工件中心 X 相比首步向 +X 移动约 4.55、3.01、4.66 mm。最终指垫接触点位于主体，沿工件局部轴向约 −26.4 至 +36.3 mm，有些靠近主体端缘。例如 17208/−Y 的两侧点都在局部轴向 +35.8 mm 附近。事故记录是抓取几何偏离规则触发，未见指垫与桌面的终止接触。复用 P3c-7 的 17211/+Y 配对 GIF。

**分布外与动作。** 六个 BC 回合的初始物体 X（观测 15）和相对抓取 X（12）均超出训练逐维包络，部分 Y 也超出。六回合全程至少一维在包络外。推理前仍用原训练均值与标准差，原截断 ±8 不变。三次超时中截断最频繁的维度是右指滑轨位置（29），与未抓住物体时的闭合状态相符；碰撞组另见相对 Y 和物体角速度截断。Oracle 相同初态也有部分截断（每回合 0–18 步），包络超界及截断均为诊断相关证据。BC 的三个 TCP 动作分量无绝对值超过 0.95 的步，夹爪分量则有 236–399 步超过该阈值（正负都计）。尚未做消融，不能将失败单独归因于截断或夹爪饱和。

## BC→PPO/SAC 设计审计

- **SAC 直接迁移可行。** SB3 SAC 默认结构须显式改成与 BC 一致的 Tanh 64×64。复制两个隐藏层到 `latent_pi`、最后线性层到 `mu`，由 SAC 自带的 squashed Gaussian 提供唯一一次输出 Tanh。固定原输入变换。零权重、偏置 −3 的 log-std 头给预压缩标准差约 0.050，可作为待验证起点。1,523 条冻结观测上确定性动作最大分量误差为 **0**，没有执行优化。双 Critic 从随机权重初始化、target critic 拷贝初值，避免把动作拟合网络当价值网络。
- **PPO 直接迁移需自定义分布。** 原 SB3 PPO 的对角高斯以未压缩均值采样，环境侧裁剪与 BC 的 Tanh 不等价。可自定义一致的 squashed Gaussian（含变换后的对数概率），复制同样的隐藏层及均值头，价值网络独立随机初始化。此方案尚未实现或验证。
- **优先建议共同的残差执行接口。** 冻结 BC，令执行动作为 `clip(BC(obs) + ε·residual(obs), −1, 1)`，PPO 与 SAC 残差均值头零初始化，评估时零残差严格复现 BC。残差版本使用相同观测缩放、动作边界和残差上限。可先限定较小 XYZ 残差，夹爪由 BC 保持控制，再依据验证结果决定是否需要开放夹爪残差。碰撞组可能需要更早调整闭合或位置，小残差能否修复要实测。PPO/SAC 在残差空间计算概率和更新，避免对裁剪后的执行动作误算概率。
- **探索和 replay。** 初始小标准差，并同时记录采样残差、裁剪后的动作和 IK 实际执行。SAC replay 从空开始，预热时用冻结 BC 加有界残差探索收集在线数据；若使用 SB3 默认随机 warm-up，必须将随机量限制在同样的小残差接口内。存残差作为 critic action，另记执行动作，保存原始奖励、最终观测、终止和超时语义。与 P3c-2-R1 不同，起点已有 BC 能力，数据来自当前残差行为在线执行，未把静态示范直接预填或设为免费样本。固定低熵系数（例如已有配置 0.005）仅作为待验证起点，防止无界探索，当前未验证适合此任务。
- **保护初始能力。** 冻结 BC，存零步参考与权重，先验证确定性动作和小噪声扰动，短间隔评估开发失败组及独立的标称验证组。限制残差幅度与更新幅度，另设保留 BC 能力的停止判据，保留退化 checkpoint。独立 Test 只在方法和预算冻结后运行。RL 可行性目前限于接口与权重映射，未证明微调会改善。

## 复现

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cr3-p3b-mpl
.venv/bin/python outputs/p3c-8/diagnose.py >outputs/p3c-8/run.log
.venv/bin/python outputs/p3c-8/actor_audit.py
.venv/bin/python -m pytest -q tests outputs/p3c-8/test_diagnose.py
sha256sum -c outputs/p3c-7/frozen.sha256
```

12 份逐步 gzip JSONL 位于 `outputs/p3c-8/`，记录覆盖动作输入与实际 TCP 增量、候选与实际关节差、接触力、世界和物体局部坐标、位姿及阶段。观测和动作在 step 前采集，执行状态和接触在 step 后采集。自动化测试 **51 passed**（49 项主测试及 2 项诊断回归）。等待正式验收。

P3c-8 实现及实测报告提交为 `f3774536d8920af7f6540becf6bd6de879c412fe`。

# P3c-9 冻结 BC + Residual Policy 接口

基于 P3c-8 `f3774536d8920af7f6540becf6bd6de879c412fe`。实验代码位于 `outputs/p3c-9/`，复用 P3c-7 的复位适配器。40 维原始观测进入包装器，BC 路径执行一次原 SCALE、训练均值/标准差（训练下限 0.02）及 ±8 截断，再经原 Tanh 模型。BC 的推理使用 no-grad，权重不在 SAC 模型或优化器内。ResidualEnv 使用独立的四维归一化动作，执行 `clip(BC + scale × residual, −1, 1)`。开关独立控制 XYZ 和夹爪。记录 BC、残差、最终动作、应用后的 TCP 增量、等效 XYZ 动作、渐进闭合目标和 IK 状态。被拒绝时等效 XYZ 为零，夹爪仍按底层规则执行。

XYZ 每单位动作对应每控制周期 1.5 mm。默认 scale 0.5 对应每轴最多 0.75 mm/周期的叠加，约 37.5 mm/s 的名义差分上限。诊断上限 scale 1.0 可覆盖原步长量级，实际执行仍受动作裁剪、工作区和 IK 限制。夹爪默认关闭残差，额外探针才启用。各维 scale 支持 [0,1]，未扩大执行器本身的能力。这些范围是诊断配置，尚无训练结果支持选择。

## 零残差和接口审计

用 P3c-8 的 1,523 条同索引冻结观测，独立应用一次原变换并比较模型和包装器组合输出。动作逐分量相等，最大绝对误差为 0。闭环检查 fixed 标称双朝向 2、P3c-6 ±25 mm 原测试 40、P3c-7 ±35 mm 开发 40，共 82 个回合，六个已知失败包含在最后 40 个内。逐步观测、奖励、终止或截断标记和底层 info 相等。P3c-7 保留 Test 未运行。

SB3 SAC 只训练独立的残差 Actor 和 Critic。零初始化 Actor 均值头及 log-std 权重，log-std 偏置 −3 为接口试验配置，确定性残差逐分量为零。策略使用 SB3 squashed Gaussian 在残差空间取样及计算变换后的 log-prob，混合和裁剪发生在环境内。Critic 和 Replay Buffer 的 action 为残差，执行动作只在 info 中审计。单步超时测试核对保存原始 reward、源观测、末帧 next obs 和 timeout bootstrap；主测试已有成功终止的 replay 检查。未走优化器更新。以后训练需继续使用当前有效残差幅度和同一冻结 BC，换配置后不可直接混用经验。夹爪屏蔽时第四维没有环境效应，但仍贡献默认 SAC 熵；正式训练应先决定是保留该冗余维度并校准目标熵，还是屏蔽该维的概率与熵计算。PPO 自定义分布未实现。

独立 `TrainingChallenge` 以 P3c-7 相同采样器在每轴 ±35 mm 抽样，场景种子池 18000–18099，双朝向等概率。固定 `grasp_ready` 起点、400 步上限、原物理、奖励和成功条件。它区别于 P3c-2 的 Pre-Grasp 训练入口，拒绝池外种子（含 17700–17739）。未训练或评估任何新策略。

## 有界探针实测

探针使用原观测的相对质心抓取点，仅用于诊断。没有梯度更新，也没有改变成功阈值。

| P3c-8 失败组 | 零残差 | XYZ scale 0.5 | XYZ scale 1 |
| --- | --- | --- | --- |
| 三个 IK 超时 | 932 次拒绝 | 0 次拒绝，3 次异常接触 | 0 次拒绝，3 次碰桌 |
| 三个抓取区外接触 | 3 次抓取区外接触 | 2 次抓取区外接触，1 次超时 | 3 次抓取区外接触 |

对 IK 组，横向对准后下探解决了原目标方向问题，但探针在到达抓取点后继续下探。scale 0.5 的异常接触包括一次碰桌和两次抓取区外接触。对原碰撞组，横向修正保留 BC 的垂直与闭合时序，未产生成功。单独启用夹爪残差延迟闭合后，两次仍异常接触、一次超时且有一次关节步长拒绝。现有证据不能证明夹爪残差是必要条件。诊断只确认残差可改变失败机制，**未修复任何完整回合**。后续可先验证下探停止和闭合时序的协调，避免把 IK 拒绝下降视为抓取改善。

## 复现

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cr3-p3b-mpl
.venv/bin/python outputs/p3c-9/verify.py >outputs/p3c-9/verified.log 2>&1
.venv/bin/python -m pytest -q tests outputs/p3c-7/test_benchmark.py \
  outputs/p3c-8/test_diagnose.py outputs/p3c-9/test_residual.py
sha256sum -c outputs/p3c-7/frozen.sha256
```

**55 项自动化测试通过**。逐步探针记录及汇总保存在 `outputs/p3c-9/`。权重和数据集 SHA-256 与冻结清单一致，历史实验保留。等待验收。

P3c-9 实现及实测报告提交为 `04cb00fdbb37c92028904380cdd1c585af7f3468`。

# P3c-10 三维 Residual SAC Smoke

复用 P3c-9 和冻结 P3c-6 BC。每次启动与完成后核对模型及示范 SHA-256。BC 使用 no-grad 和原输入变换，独立于 SAC 参数。包装器暴露 40 维原观测及三维 `[-1,1]` 残差，内部补零并沿用原夹爪动作、限位及伺服。Actor/Critic 与 buffer 的动作均为三维，使用 SB3 的 squashed Gaussian 残差概率。最终裁剪动作、实际 TCP 目标增量、测量 TCP 增量和渐进闭合独立记录。测试覆盖自动熵目标 −3、零均值、输入与末帧回放语义、超时 bootstrap 和夹爪隔离。

两个种子各 5k 在线步。训练分布沿用 P3c-9 `TrainingChallenge`，每轴 ±35 mm，场景池 18000–18099，双向等概率，完整待命起点，400 步、原奖励、物理及判据。独立随机 Critic，target 初值拷贝，Actor 均值头归零，log-std 头偏置 −3，64×64 Tanh；初始 α=0.005，自动调节、目标熵 −3。空 buffer 容量 10k，前 500 步由初始化的 squashed Gaussian 采样，无均匀预热、专家预填或外加噪声。每 4 步更新一次，batch 128、LR 0.0003，其余 SB3 默认。每轴 scale 0.5，可分别配置。

六条冻结失败轨迹里，三个碰撞回合距抓取点 ≤65 mm 时有 148 个原 Z 下探步。其中 42 个可用最大 0.5 残差完全抵消，剩余仍可减速。三个 IK 超时回合始终没有进入该下探区域。首个拒绝候选和逐轴动作范围见 `calibration.json`。100k 次每轴模拟初始化采样，log-std −3 的绝对值 P50/P95/最大为约 0.0335/0.097/0.212–0.242，对应 0.025/0.073/0.159–0.181 mm 名义修正。−2 的离线对照约 0.091/0.259/0.531–0.570，对应 0.068/0.195/0.398–0.427 mm。该档只做采样校准，训练采用 −3。实际 TCP 增量另计，受 IK、伺服跟踪和 BC 动作共同影响。

每 1k 步保存模型、在线分布、损失、温度、样本熵及终止回合。阶段评估用相同初态配对 BC、确定性 SAC 和有种子随机 SAC，覆盖六个已知失败、名义位置 15000 双朝向与 P3c-7 开发随机位置 17200 双朝向。训练池与评估种子不重叠，预留 Test 17700–17739 未运行。评估采样隔离 Torch RNG。确认抓取与 20 mm 携带为同一环境判据，表中共用一栏。初次运行因逐步日志的末帧 NumPy 数组无法 JSON 序列化而中止，保留中断文件，修复记录器后从原种子重新执行。

## 结果与停止决定

每个种子完成 5,000 在线步及 1,125 次优化，保留 0–5k 六个 checkpoint、在线 Replay Buffer、逐步 JSONL 和评估轨迹。初始随机策略与零残差均保持 4/10 成功。下表为 5k 的 10 个配对回合，确认抓取即 20 mm 携带。

| 方法 | 双侧 | 确认 / 20 mm | 100 mm | 保持 | IK 未收敛 / 增量拒绝 | 碰撞 / 失握 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 零残差 BC | 7 | 4 | 4 | 4 | 772 / 160 | 3 / 0 |
| SAC 11 确定性 | 3 | 1 | 1 | 1 | 12 / 109 | 3 / 0 |
| SAC 11 随机 | 4 | 1 | 1 | 0 | 32 / 39 | 3 / 0 |
| SAC 22 确定性 | 7 | 4 | 4 | 4 | 643 / 159 | 3 / 0 |
| SAC 22 随机 | 5 | 3 | 3 | 0 | 237 / 136 | 3 / 1 |

最后三组碰撞均为抓取区外接触；SAC 22 在 1k 确定性出现六次指垫碰桌，虽无 IK 拒绝仍为 0/10 完成。表中增量拒绝对应整个关节向量 0.03 rad 门限；现有包装器未记录拒绝关节索引，不能将计数全部归为 J3。末轮确定性保持拆分（失败组六例 / 标称双朝向 / 开发随机双朝向）为种子 11 的 0/6、1/2、0/2 和种子 22 的 0/6、2/2、2/2。随机末轮均无保持；种子 22 在失败组有 1/6 确认并达到 100 mm。种子 22 在 1k 随机失败组 6/6 确认 20 mm、0/6 达到 100 mm，说明探索产生过中间经验，尚无稳定成功。两个种子均在 1k 和 2k 丢失标称成功。种子 11 在 3k 确定性短暂恢复 4/10，种子 22 在 5k 恢复到该基线，未观察到持续改善。

实际前 500 步逐轴绝对残差 P50/P95/最大分别为种子 11 的 [0.034,0.033,0.031] / [0.091,0.095,0.104] / [0.144,0.148,0.178]，种子 22 的 [0.035,0.036,0.033] / [0.108,0.104,0.095] / [0.156,0.172,0.152]。测量 TCP 绝对增量 P95 分别为 [0.063,0.076,1.240] 和 [1.186,1.198,1.344] mm，含 BC 位移。逐轴 P50/P95/最大、目标增量、饱和和裁剪、最小抓取距离和闭合时刻见 `summary.json`。最后 1k 各轴残差饱和比例约 5.3–6.3%。末轮 Actor/Critic loss 分别约 0.0985/0.000538 和 0.1353/0.001070，α 约 0.003520 和 0.003502。模型参数保持有限。

结论：三维残差和回放接口通过验证，冻结 BC 本身能力保持。在线更新在部分阶段获得携带经验并减少 IK 拒绝，但没有学出可靠的下探、闭合和抬升协调。达到预置 5k 后停止，未追加种子、探索档或训练预算。评估限于小型开发面板。

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cr3-p3b-mpl
.venv/bin/python outputs/p3c-10/calibrate.py
for seed in 11 22; do
  .venv/bin/python outputs/p3c-10/smoke.py "outputs/p3c-10/seed-$seed" --seed "$seed" \
    >"outputs/p3c-10/seed-$seed.log" 2>&1
done
.venv/bin/python outputs/p3c-10/summarize.py
.venv/bin/python outputs/p3c-10/plot.py
.venv/bin/python -m pytest -q tests outputs/p3c-7/test_benchmark.py outputs/p3c-8/test_diagnose.py \
  outputs/p3c-9/test_residual.py outputs/p3c-10/test_smoke.py
```

**56 项自动化测试通过**。日志、逐回合数据、checkpoint、buffer 和曲线在 `outputs/p3c-10/`。等待正式验收。

P3c-10 实现及实测报告提交为 `15fe7ab6d7aa3d31a764269a6cd66b460a4adb0b`。

# P3c-10.1 Residual SAC checkpoint 审计

本轮只加载 P3c-10（`15fe7ab6d7aa3d31a764269a6cd66b460a4adb0b`）的现有 SAC checkpoint、日志和最终 replay，未训练或更新权重。±35 mm 面板直接读取 P3c-7 JSON 中 20 个实际 XY 与双朝向的 40 份配对。另测标称 15000 双朝向，单列统计。全部采用确定性推理。保留 Test 17700–17739 未运行。模型、奖励、物理及成功条件保持原样。各 checkpoint 逐项核对初始观测，零步 SAC 与 BC 的分组结果相等。

## 训练审计方法

Actor 在同一批前 500 个在线观测上比较均值与 log-std。Critic 对各 checkpoint 对应时刻的 buffer 前缀重算 Q1/Q2、含熵的下一状态 target、TD 残差及分阶段分布。Target 重采样使用固定种子；这套离线统计与训练日志中的随机 batch 损失用途不同。固定前 128 条的梯度范数通过 autograd 重算，无 optimizer step。历史每一步梯度、batch 和 Adam 轨迹未保存，无法回溯实际梯度尖峰。Q 与软 bootstrap 目标不是独立真值，不能仅据其接近断言估值没有偏差。

最终 buffer 各 5,000 条，逐步核对保存残差与在线日志，终止索引和 timeout 区分一致。终止观测和原始 reward 复用已验证的 SB3 保存语义。样本阶段按当前观测分为互斥的接触前、双侧、确认携带、100 mm 和保持。确认携带在观测中为历史标记，阶段可能包含后续失握前的步。成功回合样本数与保持阶段数分开报告，未结束尾段计入交互数。

| 种子 / 前缀 | 接触前 | 双侧 | 确认 | 100 mm | 保持 | 已完成成功 / 终止 / 截断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 11 / 1k | 633 | 137 | 133 | 59 | 38 | 2 / 3 / 0 |
| 22 / 1k | 818 | 37 | 66 | 79 | 0 | 0 / 1 / 2 |
| 11 / 5k | 3480 | 555 | 868 | 59 | 38 | 2 / 7 / 7 |
| 22 / 5k | 4088 | 248 | 367 | 293 | 4 | 0 / 5 / 9 |

种子 11 成功轨迹包含 640 个步，最终仅 12.8% 的 buffer，成功回合在 1k 后未再增加。种子 22 没有成功回合。两个种子在早期更新前后探索分布快速移动。在固定观测上，log-std 从 −3 到 1k 时的各轴中位约 −0.66 至 −0.21，故标准差已从 0.050 升至约 0.52–0.81。1k 时确定性残差各轴中位，11 为 [−0.077, −0.655, +0.445]，22 为 [−0.580, −0.279, −0.433]。在每轴 scale 0.5 下，哪怕未饱和，这类持续偏置也足以改变整段路径。二者标称均在更新后的首个 1k 快照丢失能力。

离线 1k 梯度范数 actor/critic 分别为 11 的 0.214/0.0476、22 的 0.0857/0.0527，5k 为 0.0294/0.0390、0.0220/0.0184，均有限。5k Q1/Q2 中位数分别约 [−0.084,−0.083] 和 [−0.103,−0.101]，软 target 中位约 −0.097、−0.103。全前缀重算的 MSE 为 0.0329、0.0300，包含少数大终止回报。未观察到 Q 中位数爆炸，不能据此排除稀有阶段估值偏差。分阶段分布、动作幅度与裁剪见 `replay-*.json`。两种子早期分别表现为偏离抓取点和下探过度，与均值偏移、探索快速扩大、保持成功样本稀少相符。当前审计无法独立归因于熵、奖励权重或采样比例。

## 完整配对结果

每行分母 40；四项依次为双侧接触、确认抓取（同 20 mm 携带）、100 mm 抬升及稳定保持（同最终成功）。标称分母 2，失败组 6，其余开发场景 34，两个朝向各 20。

| checkpoint | 四阶段计数 | 失败组 / 其余成功 | 标称 | +Y / −Y | IK 未收敛 / 增量拒绝 | 碰桌 / 区外 / 失握 |
| --- | --- | --- | ---: | --- | --- | --- |
| BC | 37 / 34 / 34 / 34 | 0 / 34 | 2 | 17 / 17 | 772 / 160 | 0 / 3 / 0 |
| 11 / 0k | 37 / 34 / 34 / 34 | 0 / 34 | 2 | 17 / 17 | 772 / 160 | 0 / 3 / 0 |
| 11 / 1k | 0 / 0 / 0 / 0 | 0 / 0 | 0 | 0 / 0 | 914 / 1214 | 0 / 0 / 0 |
| 11 / 2k | 0 / 0 / 0 / 0 | 0 / 0 | 0 | 0 / 0 | 1442 / 599 | 0 / 0 / 0 |
| 11 / 3k | 35 / 33 / 33 / 25 | 0 / 25 | 2 | 15 / 10 | 521 / 49 | 0 / 5 / 0 |
| 11 / 4k | 0 / 0 / 0 / 0 | 0 / 0 | 0 | 0 / 0 | 0 / 0 | 40 / 0 / 0 |
| 11 / 5k | 25 / 12 / 12 / 11 | 0 / 11 | 1 | 6 / 5 | 12 / 135 | 0 / 14 / 0 |
| 22 / 0k | 37 / 34 / 34 / 34 | 0 / 34 | 2 | 17 / 17 | 772 / 160 | 0 / 3 / 0 |
| 22 / 1k | 25 / 13 / 6 / 0 | 0 / 0 | 0 | 0 / 0 | 0 / 0 | 27 / 0 / 0 |
| 22 / 2k | 0 / 0 / 0 / 0 | 0 / 0 | 0 | 0 / 0 | 226 / 4109 | 0 / 0 / 0 |
| 22 / 3k | 0 / 0 / 0 / 0 | 0 / 0 | 0 | 0 / 0 | 873 / 3357 | 0 / 0 / 0 |
| 22 / 4k | 0 / 0 / 0 / 0 | 0 / 0 | 0 | 0 / 0 | 1160 / 2625 | 0 / 0 / 0 |
| 22 / 5k | 37 / 33 / 33 / 33 | 0 / 33 | 2 | 17 / 16 | 643 / 159 | 0 / 4 / 0 |

没有任何 checkpoint 恢复 BC 失败配对。11/1k–2k 所有回合未双侧接触且超时，平均最近抓取距离由 BC 16.7 mm 增到 84–86 mm，符合偏离接近路径。22/1k 平均最近 2.9 mm、27 次碰桌，说明能接近却未协调下探停止。11/4k 零 IK 拒绝但 40 次碰桌，实际 TCP 平均路径长 186.4 mm（BC 297.8 mm），提前碰撞压缩了执行机会。11/5k IK 仅 147 次，却有 14 次抓取区外接触，平均最近距离 40.2 mm、平均路径 273.0 mm。22/5k 平均路径 303.5 mm、接近距离 21.7 mm，恢复接近原成功集合但仍丢一例。拒绝计数同时受动作方向、有效运动和回合长度影响。

**不建议当前配置直接进入 P3c-11。** 目前有局部接近或抬升能力变化，但无配对新增成功。成功经验稀少和首轮参数漂移同时可见。优先设计一个能限制探索增长和累计偏移、明确保留标称能力的短门槛实验，再决定是否追加训练。原奖励有明确保持成功项，但保持样本稀少；离线审计不足以证明应改奖励或采用某种回放方案。所有失败和快照均保留。

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cr3-p3b-mpl
out=outputs/p3c-10.1
.venv/bin/python "$out/audit.py" evaluate 0 >"$out/bc.log" 2>&1
for seed in 11 22; do
  for step in 0 1000 2000 3000 4000 5000; do
    .venv/bin/python "$out/audit.py" evaluate "$seed" --steps "$step"
  done >"$out/eval-$seed.log" 2>&1
  .venv/bin/python "$out/audit.py" replay "$seed" >"$out/replay-$seed.log" 2>&1
done
.venv/bin/python "$out/plot_audit.py"
.venv/bin/python -m pytest -q tests outputs/p3c-7/test_benchmark.py outputs/p3c-8/test_diagnose.py \
  outputs/p3c-9/test_residual.py outputs/p3c-10/test_smoke.py "$out/test_audit.py"
```

**58 项测试通过**。分组及逐回合数据、Actor/Critic 分位数与日志、阶段曲线和配对得失位于 `outputs/p3c-10.1/`。等待验收。

P3c-10.1 实现及实测报告提交为 `74f9858ffeae539344cf2f82d0c29689e7288d80`。

# P3c-10.2 残差训练设计审计

独立报告见 [P3c-10.2 技术报告](P3C_10_2.md)。离线审计提交为 `7cfafd13a69482a8402cbeb98f7ae2a43d6bb8dd`。

基于 P3c-10.1 `74f9858` 的实验，工作分支起点 `3d0c50a` 包含后续交接更新。本次读取原 SB3 2.7.1 源码、六个 SAC 快照、两个历史 buffer、历史在线日志和十二条 BC/Oracle 轨迹。仅做离线前向、梯度重算和日志汇总，未调用优化器、启动模拟回合或解封 Test。

## Actor 和温度

实际入口为 `outputs/p3c-10/smoke.py`。64×64 Tanh；三维独立高斯，均值和 log-std 共用特征主干，线性头分别零初始化和零权重、偏置 −3；无 gSDE。SB3 `sac/policies.py` 将状态相关 log-std 限制在 [−20,+2]，因此 −3 只是初值。重参数化采样 `u = μ + exp(log_std) ε`，执行残差 `r = tanh(u)`，密度含三维 Tanh Jacobian 修正。动作混合、工作区投影及 IK 均在环境一侧。

SB3 `sac/sac.py` 在每次梯度更新中先采样，再优化 `Lα = −log(α) mean(stopgrad(log π(r|s) + target_entropy))`，使用本次更新前已 detach 的 α 计算 `Lactor = E[α log π − min(Q1,Q2)]` 和软目标值。每四个环境步一次更新，500 步预热，1k 时约 125 次更新。目标熵 −3 是三维 squash 后联合微分熵的温度调节目标。令 H = −E log π，则温度对数梯度为 H + 3；当 H 低于 −3 时推动 α 上升，高于时推动 α 下降。Actor 熵项本身始终推动增熵，直至受数值边界、squash 或 Q 项制衡。

离线计算使用两份 buffer 各自前 500 个观测拼接成共同 1,000 样本，固定一组噪声。梯度为快照当前局部梯度，历史 mini-batch、Adam 动量和逐次因果贡献无法由这些快照复原。

| 种子/步数 | α | H (nat) | 各轴 log-std 中位 | 熵项损失 / 负 Q 损失 | 各轴残差绝对 P95 |
| --- | ---: | ---: | --- | --- | --- |
| 11 / 0 | .00500 | −4.679 | −3 / −3 / −3 | +.0234 / +.0923 | .098/.099/.100 |
| 22 / 0 | .00500 | −4.679 | −3 / −3 / −3 | +.0234 / +.1010 | .098/.099/.100 |
| 11 / 1k | .00499 | +1.147 | −.310/−.554/−.665 | −.00573 / +.1319 | .895/.935/.877 |
| 22 / 1k | .00497 | +1.487 | −.228/−.291/−.396 | −.00739 / +.1191 | .964/.915/.926 |
| 11 / 5k | .00352 | +1.494 | −.365/−.129/−.123 | −.00526 / +.0940 | .962/.949/.952 |
| 22 / 5k | .00350 | +1.965 | −.051/−.190/−.114 | −.00688 / +.1814 | .958/.930/.950 |

初值 log-std 头各轴偏置上的熵项梯度约 −.00497，Q 梯度绝对值均 ≤.000048。均值头偏置上，熵项绝对梯度 ≤.000018，Q 项为 .00373–.03474。梯度下降因此能同时扩大探索和改变确定性策略。以 11 的 Z 为例，1k 时均值偏置梯度为熵项 +.00366、Q 项 −.02428，局部方向相反；22 的 X log-std 偏置则为 −.00117、−.00215，两项同向。1k 时两种子各至少一轴的 pre-tanh 均值绝对 P95 达 .676；此时 α 仍接近初值。温度下降滞后于方差变化，共用主干使后续两头的变化可能相关。损失项数值不能代替梯度比较。已有证据支持早期熵驱动的方差放大和 Q 驱动的均值偏移。缺少逐更新记录，无法量化二者对闭环失败各自的因果贡献。

## 单步和整回合位移

XYZ 每轴动作 [−1,1] 对应 1.5 mm/步，50 Hz；历史 scale=(.5,.5,.5)，故残差最大单步指令差每轴 0.75 mm/步（37.5 mm/s 的指令变化率）。初始化记录的三轴绝对 P50 约 .025 mm，P95 .073 mm。下表为历史在线 5k 真实日志，各列为 XYZ，mm。实际 TCP 位移含整个 BC 命令、控制跟踪和动力学。它与同一回合反事实的“仅残差造成的位移”无法直接等同。

| 种子 | 混合后指令差 P50 | 混合后指令差 P95 | 实测 TCP 单步绝对 P50 / P95 | 任意 XYZ 混合裁剪 |
| --- | --- | --- | --- | ---: |
| 11 | .329/.335/.334 | .702/.706/.708 | .120/.115/.714 / .499/.550/1.372 | 12.98% |
| 22 | .383/.309/.326 | .711/.698/.709 | .108/.097/.386 / .712/.736/1.423 | 12.78% |

历史每轴裁剪率为 11 的 0.78/1.70/10.70%，22 的 1.40/1.38/10.36%（以在线日志为准）。按同一在线观测的混合动作减 BC 动作，只累加 IK 接受步，工作区投影前的有符号指令差绝对 P95 已达 66/93/110 mm 和 115/63/118 mm，最大达 94/128/153 mm 和 185/101/200 mm。该量用于衡量持续偏置的规模。相同初态两条真实 TCP 轨迹的差异须分别配对记录，按步及按事件对齐；因 BC 也会随观测变化，前者无法给后者提供硬上界。

六条失败轨迹的 XY 初态约距标称 (+33 至 +34, −34 至 +31) mm，Oracle 从同一待命姿态均能完成。三个异常接触在约 35 mm 的抓取点球面触发；当时距抓取点的 XY 分量绝对值仅 4.80–5.18 mm 和 .52–1.50 mm，Z 约 −34 mm（抓取点在 TCP 下方）。同一步 Oracle 抓取距离 6.54–6.56 mm，Oracle 在完整轨迹中曾进入 0.001 mm 内。三条 IK 超时首次拒绝时，与同一步 Oracle 的 Z 差约 7.7、11.8、114.9 mm。114.9 mm 含不同的轨迹时序，不能直接作为修正预算。P3c-9 的导向注入消除了 932 次拒绝，仍无法完成六个回合；P3c-10 的冻结轨迹标定中，scale=.5 在各撞击案例的 47/49/52 个近场下探步中，仅 14/11/17 步能完全抵消 BC 下探。因此增加横向幅度或禁止一切下探都缺少依据。

### 推荐控制约束（待验证）

- 残差 scale 从 (.5,.5,.5) 开始，夹爪沿用 BC，目标增量上限保持上述尺度。暂定 log-std [−4,−3]；零均值且 σ 取上限时，初始绝对修正 P50/P95 约 .025/.073 mm。已有数据仅校准了零均值探索，在线均值偏离后需重新实测分位数与裁剪。
- 首次离线/诊断候选的有符号累计预算 X/Y/Z 为 ±(6,3,12) mm。X/Y 比异常接触时最大约 5.18/1.50 mm 留少量裕度，Z 取两个早期 IK 配对偏差的较大量级。它是局部早期修正假设，无法覆盖晚期相位差，也没有证据保证可救回碰撞组。仅当受控探针给出成功轨迹且触及该限制时，才考虑另行预注册参数；禁止依据 Test 扩张。
- 对同一实时观测，分别计算 BC-only 与混合动作的工作区裁剪目标，其差为本步有效预算候选。将此差逐轴投影到剩余区间，再调用原 IK；IK 接受才更新累计账本。达到边界后抑制继续向外的分量，反向允许释放有符号预算。额外记录每轴绝对累计消耗，防止忽略反复往返；首轮只监测该量，尚无证据校准绝对预算。全过程须保留原动作、投影后动作、接受后目标增量和实测偏差。零残差及零预算必须复现原 BC。预算投影施加在环境中，策略密度和存储使用投影前残差。
- 有符号账本是环境状态。若进入实现，建议向 SAC 和 PPO 都显式提供三个归一化账本状态，RL 输入 43 维，冻结 BC 仍读取原始 40 维及既有预处理。若阶段验收要求 RL 输入也固定 40 维，需单独审议账本隐状态的部分可观测性或改用可从既有观测确定的限制；不能将隐藏累计限幅视为马尔可夫接口。
- L2 约束 `λ E[||tanh μ||²/3]` 只牵引均值，不能限制方差。动作空间 KL 锚到初始零残差窄高斯可同时约束均值和方差，但窄参考分布会使几何修正成本迅速上升。log-std 上限管住随机探索，仍须保护确定性路径。固定低 α 消除目标熵的拉升反馈，仍会奖励随机性。首轮可锁定 α=.005 以保持初始量级并隔离自动调节，配合方差上限和均值锚。建议 λ=.05 作为待验证初值，在单轴 μ=.1 时梯度约 .0033，达到离线 1k 较小均值偏置梯度量级；在 0 处梯度为 0，仍需回合保护和 checkpoint 门槛。上述组合用于可行性，不足以分离各项贡献。

## Buffer 和奖励

| 种子 | 双侧接触前 | 双侧未确认 / 确认未达 100 mm | 100 mm / 保持 | 完成回合成功比例 | 终止 / 超时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11 | 3480/5000 | 555 / 868 | 59 / 38 | 2/14 | 7 / 7 |
| 22 | 4088/5000 | 248 / 367 | 293 / 4 | 0/14 | 5 / 9 |

阶段基于当前观测且互斥，回合统计排除最后未完成片段。11 的 640 个来自成功回合的转移全在 1k 前产生，约占最终 buffer 12.8%，其中进入保持的当前状态只有 38 个。22 在 1k 时已有 79 条 100 mm 阶段数据却无保持。普通均匀采样自然以接触前更新为主。以 128 batch 的最终 buffer 比例估算，每批仅期望 0.97/0.10 条保持状态。P3c-10.1 的有限 Q 值及总体误差无法排除稀有阶段估值偏差，两个 Q 接近也不能证明其均校准。

现有 shaping 为 γΦ(s′) − Φ(s) − .005 − .02×IK拒绝 + 3×成功；失败为 −5 − Φ(s)。Φ 按靠近、双侧接触、离台高度（100 mm 封顶）与连续保持推进，连续 20 步稳定才成功。阶段信息存在，保持的真实样本弱覆盖，末端奖惩的信号传回早期动作仍困难。几何、闭合时序与持稳约束共同决定可行性，改变采样只可重加权已有样本。短实验先保留原奖励和均匀在线 buffer，对首次双侧、确认、离台 20 mm、100 mm、稳定保持分桶监测。若一个随后获批的实验已经产生可重复保持样本但均匀采样仍明显欠覆盖，再独立比较同一在线 buffer 的阶段分层和重要性加权，同时分别报告 Actor 与 Critic 的采样分布。此处不引入专家预填或新奖励。

## 两套短预算候选与停止门槛

两者共用冻结 BC、三维残差、预算投影和对照；场景池 18000–18099，±35 mm 双朝向，预注册相同场景抽取序列、固定待命起点、400 步、原物理、奖励、成功及超时定义。以 17200–17219 的 40 配对回合和标称双朝向作开发验证。另记随机策略行为，与确定性指标分开。17700–17739 留待冻结方案后的正式独立测试。本阶段以及建议短门槛内均不开放。

| 项目 | A 保守 Residual SAC（优先） | B Residual PPO（设计备选） |
| --- | --- | --- |
| 模型与初态 | 64×64 Tanh 残差 Actor，均值头全零，log-std −3；双 Q 独立于 Actor 及彼此，target 从在线 Critic 拷贝；空 buffer | 64×64 Tanh 均值网络，均值头全零且固定 log-std −3 起步；独立 value 网络 |
| 探索与熵 | squashed Gaussian，log-std [−4,−3]，固定 α=.005，无外部噪声 | 同样的初始 squashed Gaussian，固定标准差，entropy bonus=0 |
| 约束与候选超参 | 共用 scale、累计预算和均值 L2；Actor LR 1e−4，Critic LR 3e−4，每 4 步一次，batch 128，原 γ=.99 和 timeout bootstrap | 共用 scale、累计预算和均值 L2；LR 1e−4，γ=.99，GAE λ=.95；rollout 250、batch 50、最多 3 epochs，clip .1、target KL .01（建议触及即停当次 epoch） |
| 交互与风险 | 单种子建议 11，最多 1k 在线步：前 500 步只用初始化分布热身，后 500 步最多约 125 次梯度更新。均值锚可能仍不足，价值估计仍依赖覆盖 | 对比时同种子与 1k 在线步，前 500 步只采样作门槛检查，优化只消费其后的新鲜 on-policy 片段。短 rollout 对 300 步以上轨迹的优势估计依赖 bootstrap，样本利用率可能更低 |

PPO 要显式实现 Squashed Gaussian，建议保存 pre-tanh 样本 u 和旧对数概率，同时记录 r 及最终动作。新旧密度对同一 u 计算三轴联合 log-prob，减去稳定计算的 `sum(log(1−tanh(u)^2))`。同一映射的 Jacobian 在比率中理论抵消，但统一记录完整密度便于验证边界和熵。rollout 与 deterministic predict 分别用 tanh(sample) 与 tanh(μ)。禁止对环境裁剪动作反求 PPO 比率，也不能把普通 SB3 PPO 的默认非 squash 路径当成这个实现。PPO clip 和近似 KL 约束相邻策略，无法保证相对于 BC 的长期漂移；零均值锚、预算和回归门槛仍必需。SAC Replay 保存三维 squash 后、预算前的动作与 timeout 标记，对超时使用最终状态 bootstrap；PPO 同样处理 timeout，完整失败/成功终止不 bootstrap。算法使用同一入口和交互上限，单独报告更新和壁钟成本。

推荐顺序如下。

1. 先获批实现并验证约束。在不更新权重的情况下，验证投影零残差与 BC 完全配对等价、账本和超时语义，并在六个失败对中验证至少一个在预算内可稳定完成的有界诊断轨迹。已有探针全部失败，若新诊断仍失败，先停在接口可行性，避免把训练当作寻找可行动作的替代品。
2. 仅通过无训练门槛才建议 A 的 1k 单种子实验。0/250/500/750/1000 评估，持久保存 checkpoint、每轴残差及实际 TCP、累计预算触顶、拒绝原因、接触阶段与逐回合结果。若任一标称失败、BC 成功的 34 对丢失超过 1 对、配对异常接触总数增加，或出现 NaN/Inf，立即停止并保留退化 checkpoint。若连续检查出现整轴残差绝对值 ≥.95 的比例超过 5%，也停止；该门槛针对旧快照已出现的饱和水平。
3. 入选 checkpoint 须保住 2/2 标称和 34/34 原成功对，六个失败对至少恢复一个完整稳定保持，且无新增异常接触；优先按配对成功和碰撞选取，IK 拒绝、reward 仅作诊断。不满足则停在现预算，独立 Test 继续封存。B 仅作为另行获批的同协议对照。恢复一例只支持局部可行性判断，不能推断泛化。

## 复现与核验

```bash
MPLCONFIGDIR=/tmp/cr3-mpl PYTHONPATH=src .venv/bin/python outputs/p3c-10.2/audit.py > outputs/p3c-10.2/audit.json
sha256sum -c outputs/p3c-7/frozen.sha256
git diff --check
```

审计入口的自动断言校验六组配对、有限损失以及 0k 的零均值和 log-std −3；冻结 BC 和示范 SHA-256 校验通过。结果 JSON 是可复现的本机生成产物。完整模拟回归沿用 P3c-10.1 及交接的历史结果，本轮未重跑。

# P3c-10.3 残差安全约束实现与验证

独立报告见 [P3c-10.3 阶段报告](P3C_10_3.md)。

以 P3c-10.1 `74f9858ffeae539344cf2f82d0c29689e7288d80` 为实验基线。用户给出的 `3d0c50a` 为当前分支祖先，实际从含 P3c-10.2 同步报告的 `1884100` 开始工作。沿用 P3c-6 冻结 BC 和预处理、原有控制频率 50 Hz、MuJoCo 2 ms 步长、400 步上限、原物理、接触判据和奖励。

## 接口

- `src/cr3_lab/rl/residual_budget.py` 的 `BudgetedResidualEnv` 面向三维 `[-1,1]` 残差，XYZ scale 默认各 .5。夹爪由原 BC 输出驱动。BC 只收到原 40 维观测。RL 额外收到 3 个归一化有符号账本状态，边界为 ±1，对应世界轴 ±(6,3,12) mm。零预算轴的状态定义为 0。Reset 清零；terminated 与 truncated 返回本步更新后的账本。
- 使用同一个当前 TCP 目标、同一观测分别计算 BC-only 与混合动作的工作区裁剪目标。逐轴把两者之差限制在剩余账本区间，所得目标直接进入原 IK 和控制逻辑，避免再次从动作反算目标引入舍入差。原 `step` 共享相同的工作区和物理推进代码。
- IK 接受才记两目标之差。反向差可释放有符号额度，绝对累计额持续增加，仅记录。账本描述接受的指令相对于同观测 BC 的差值。实测 TCP 位移另行保存。
- `info.residual_budget` 保存 BC、原始残差、scale 后残差、混合和最终等效动作、两组工作区裁剪前后目标、预算投影目标、IK 状态和候选关节角、接受差、命令和实测 TCP 位移、TCP 位姿位置、两套累计账本。IK 拒绝时保留原控制器推进和夹爪行为。

## 等价性和物理诊断

82 个历史配对场景由标称双朝向 2 个、P3c-6 的历史 ±25 mm panel 40 个（16200–16219）和 P3c-7 manifest 的 ±35 mm 开发回合 40 个组成。对每个场景同时运行 BC 本体、默认预算零残差、零预算零残差。逐步核对原观测、BC/最终动作、TCP 目标、实测 TCP 位移、奖励、terminated/truncated 和底层 info。全部一致。±35 mm 保留 34/34 原成功对和原有失败结果，标称 2/2，无新增异常接触。单测另外覆盖零预算收到非零命令时的等价性。

观测反馈探针读取现有特权观测中的相对抓取点、抓取确认和物体高度。近场横向反馈将 TCP 拉向抓取点，横向对齐后按垂直距离下探，在抓取点附近减速，并在确认后达到 102 mm 时尝试停止继续上移。夹爪沿用 BC 闭合时序。此探针无需未来轨迹，部署到同类仿真需要已有特权观测。Oracle 参考探针按回合时刻读取 P3c-8 离线 Oracle TCP 目标，只用于机制对照，保留 BC 夹爪及相同预算。它是借助已知轨迹的探针，其结果不是数学最优上界。

| 种子 / 朝向 | 原 BC | 反馈步数 | 双侧 / 确认 / 100 mm 首步 | 稳定保持 | Oracle 参考 |
| --- | --- | ---: | --- | ---: | --- |
| 17208 / 凸起 | IK 超时 | 332 | 162 / 212 / 279 | 20 | 成功 |
| 17208 / 平面 | 抓取区外接触 | 333 | 162 / 213 / 280 | 20 | 成功 |
| 17211 / 凸起 | 抓取区外接触 | 341 | 163 / 222 / 288 | 20 | 成功 |
| 17211 / 平面 | 抓取区外接触 | 338 | 168 / 219 / 286 | 20 | 成功 |
| 17218 / 凸起 | IK 超时 | 353 | 171 / 234 / 301 | 20 | 超时 |
| 17218 / 平面 | IK 超时 | 343 | 168 / 224 / 291 | 20 | 超时 |

反馈探针六回合均无 IK 拒绝、异常接触或失握，最高抬升 138.0–139.6 mm。有符号 X/Z 均触边，Y 仅 17218 两朝向触边。Oracle 参考 17218 凸起拒绝 176 次（7 单步关节增量、169 未收敛），平面拒绝 181 次（135 单步增量、46 未收敛），均未接触。两者使用相同的预算，闭环反馈与按时刻跟踪对自身状态偏差的处理不同。六条反馈成功轨迹绝对累计消耗的逐轴最大值为 7.37/4.54/15.92 mm。它们说明绝对消耗高于有符号边界仍可正常完成。若另行获批训练，可将 7.4/4.6/16.0 mm 作为已知失败回合成功轨迹的审计参照，超出时保存并审查轨迹。样本不足以把参照值定为全场景停止阈值，本轮运行器对绝对消耗只有监测。

## 复现与产物

```bash
export PYTHONPATH=src MPLCONFIGDIR=/tmp/cr3-mpl MUJOCO_GL=egl MESA_SHADER_CACHE_DISABLE=true
out=outputs/p3c-10.3/verified
.venv/bin/python outputs/p3c-10.3/validate.py --output "$out" --render
.venv/bin/python outputs/p3c-10.3/plot.py "$out"
.venv/bin/python -m pytest -q tests outputs/p3c-7/test_benchmark.py outputs/p3c-8/test_diagnose.py \
  outputs/p3c-9/test_residual.py outputs/p3c-10/test_smoke.py outputs/p3c-10.1/test_audit.py
sha256sum -c outputs/p3c-7/frozen.sha256
```

逐步对照为 `parity.jsonl.gz`，各探针另有独立 JSONL.GZ 与 GIF，逐回合指标为 `results.json`，非零反馈回归为 `feedback-regression.json`，图为 `budget.png`，均位于上述输出目录。自动化回归 66 项通过，覆盖预算正负边界、反向释放、两类 IK 拒绝、工作区裁剪、reset、终止和超时账本、零预算及零残差。冻结 checkpoint、示范 SHA-256 校验通过。独立 Test 17700–17739 保持封存。未训练 SAC/PPO 或更新模型。

## BC 保护与验收结论

同一个非零观测反馈探针额外通过原成功集 34/34 和标称 2/2，均无新增异常接触。最终共在 42 个指定场景验证反馈诊断，全数稳定抓取。这些是设计所用的开发条件。三个种子的失败配对贡献了验证可行性的证据，独立泛化仍待验证。对全部 42 条成功反馈轨迹，绝对消耗逐轴最大为 8.195/4.836/21.070 mm。若后续训练另获批准，建议将向上取整的 8.2/4.9/21.1 mm 作为首轮短门槛验证的**暂停审查阈值**，按回合独立计算，保留 checkpoint 与轨迹供审查。它们不进入步进接口，不限制绝对预算，不能视为独立分布的安全界限。实际接受目标差与两套账本已用逐步轨迹复算，43 维观测末三项逐步核对通过。

本轮已验证至少一条合规稳定抓取轨迹，同时保住冻结 BC 原有能力。等待项目负责人验收后再决定下一步。

# P3c-10.4 Feedback Teacher 示范采集

独立报告见 [P3c-10.4 阶段报告](P3C_10_4.md)。

基线为已验收的 P3c-10.3 `7509c5d`。沿用冻结 BC、40 维输入及原预处理、43 维 RL 观测、三维 XYZ 残差、scale 各 0.5、有符号预算 ±(6,3,12) mm、BC 夹爪、grasp_ready、400 步上限和原物理、奖励、成功判据。

## Teacher 依赖审计

采集脚本直接导入 P3c-10.3 `validate.py` 的 `feedback`。它直接读取 43 维观测中的 `[12:15]`（抓取点相对 TCP 的世界坐标差，m）、`[27]`（物体离初始高度，m）、`[34]`（抓取确认）。另外依赖冻结 BC 从前 40 维经既有 SCALE、训练统计和 ±8 截断生成的基础 XYZ 动作。该动作可从同一观测确定性复算。Teacher 无内部状态，不读取预算以外的隐藏控制信息，不调用物理状态 API，也不调用依赖历史 Oracle 目标的 `oracle_probe`。预算由既有环境根据动作、43 维观测中公开的账本和底层 IK 执行。原观测依然含仿真真值，本接口限定于具有同等观测的环境。

标签是 Teacher 计算并传入三维残差接口的归一化原始动作。scale、预算投影、IK 是否接受及实测 TCP 位移分别记录。训练者从当前观测预测该标签时不应使用事后反馈列。

## 数据格式和划分

按种子 `18000–18079` 采集 Train 80 回合，`18080–18099` 留出 Validation 20 回合。每个种子一个 ±35 mm 工位，两种朝向按种子奇偶交替，Train 各 40、Validation 各 10。P3c-7 开发集 17200–17219、标称双朝向和独立 Test 17700–17739 保持原用途，本轮采集不使用这些场景。

数据目录 `outputs/p3c-10.4/data/` 每回合一个 gzip JSONL。每行保存动作前的 `observation_40`、`budget_state_3`、`observation_rl_43` 以及动作后的 `next_observation_rl_43`。调试阶段按动作前的观测划分为 approach、near、closing、bilateral、lift、hold，优先识别稳定保持、确认与接触。两份账本、实际 TCP 位移、IK 状态及完整动作链位于 `info.residual_budget`，属于动作后数据。 `episode_success` 为回合级结果，成功和失败全保留。 `accepted_target_delta_xyz` 为接受目标相对 BC-only 工作区目标的差值，单位 m。`budget_projected_action_3` 为预算投影后的等效 XYZ 动作。Manifest 记录种子、划分、朝向、设定和沉降后的 XY、里程碑、结果、文件和 SHA-256。原始数据与 SHA256SUMS 存本机。

## 复现

```bash
export PYTHONPATH=src MPLCONFIGDIR=/tmp/cr3-mpl
out=outputs/p3c-10.4/data
sha256sum -c outputs/p3c-7/frozen.sha256
.venv/bin/python outputs/p3c-10.4/dataset.py collect "$out"
.venv/bin/python outputs/p3c-10.4/dataset.py replay "$out"
(cd "$out" && sha256sum -c SHA256SUMS)
.venv/bin/python -m pytest -q tests outputs/p3c-7/test_benchmark.py outputs/p3c-8/test_diagnose.py \
  outputs/p3c-9/test_residual.py outputs/p3c-10/test_smoke.py outputs/p3c-10.1/test_audit.py \
  outputs/p3c-10.4/test_dataset.py
```

采集命令要求新目录，避免覆盖既有轨迹。回放先核对每个文件的 SHA-256，再使用保存的残差从相同种子和工位复位逐步执行，并对照当前 Teacher 重新计算标签。审计保存为 `audit.json`。本轮未训练模型。

## 实测与质量审计

100 个不同设定 XY 均成功，Train 80/80、Validation 20/20，凸起和平面朝相机各 50/50。以标称点为原点，X 采样 −30.47 至 +34.66 mm，Y −34.37 至 +34.82 mm。按四象限 (++、+−、−+、−−) 成功为 33/33、28/28、21/21、18/18。两轴偏移绝对值最大值达到 25 mm 的外圈 47/47，内圈 53/53。回合长度中位/P95/最大为 334/356/366 步。33,679 个转移逐行精确回放一致，全部 IK 接受。无失败、失握、异常接触或超时，因此本批次没有失败轨迹可归因，也没有正负样本对照来估计失效边界。Validation 是采集池留出集，该结果不代表独立 Test。

| 动作前阶段 | 样本数 | 全零残差 | 原始残差绝对值 P95（XYZ） |
| --- | ---: | ---: | --- |
| 接近 | 9,627 | 585 | 0 / 0 / .218 |
| 近场对准 | 5,532 | 0 | .137 / .049 / .563 |
| 闭合 | 493 | 2 | .111 / .095 / 1 |
| 双侧接触、未确认 | 6,470 | 0 | 1 / .140 / 1 |
| 抬升 | 9,657 | 6,834 | 0 / 0 / 1 |
| 保持 | 1,900 | 0 | 0 / 0 / .051 |

- 全零 7,421 步（22.03%），非零 26,258 步（77.97%）。抬升阶段全零约 70.8%，接近阶段 XY 全零，双侧接触阶段 X 的绝对中位为 1。普通均匀 MSE 可能偏向长阶段，且在饱和轴上拟合未产生物理位移的请求。后续若获批训练，建议先比较均匀与按阶段/非零分桶采样，保持 Validation 原分布并分别报告轴向误差与闭环结果。
- 原始残差绝对 P50 为 0/0/.0817，P95 为 1/.0936/1。达到 ±1 的步数为 5,905/0/4,943。接受目标差绝对 P95 为 .0629/.0679/.1914 mm，理论无裁剪单步上限每轴约 .75 mm。残差为非零且本轴接受差小于 1e−12 m 的步数为 5,268/109/7,666。
- 步后有符号预算触及任一正负边界的各轴计数为 16,925/3,189/14,755，占 50.25%/9.47%/43.81%。绝对累计消耗逐回合 P50 为 7.743/3.970/17.218 mm，P95 为 8.181/4.564/20.230 mm，最大为 8.252/4.881/21.178 mm。与 P3c-10.3 的审计参照接近，实测少量超出，现阶段不据此设置新限制。

69 项自动化测试通过。100 个 gzip JSONL 和 manifest 的 SHA-256 全部验证通过，冻结 BC 与示范 SHA-256 校验通过。数据约 25 MB，原始回合留在本机。manifest、SHA256SUMS、审计小结同时保存在实验脚本目录以供 Git 追踪。等待验收。

# P3c-10.5 Residual Teacher 蒸馏与 SAC 门槛

独立报告见 [P3c-10.5 阶段报告](P3C_10_5.md)。

本轮在干净的 `feat/gripper` 分支从 `c1eb59184b6141ed8a680be64705b99837247a5c` 开始。该提交包含 P3c-10.4 采集、回放代码和测试，以及 manifest、SHA256SUMS 与审计。P3c-10.3 的反馈可行性提交为 `7509c5d8c082aeb12f8a583e811c1dd73b74fb8f`。原始 BC、示范文件与全部 Teacher 轨迹校验通过。

## 输入、标签与训练

数据来自 `outputs/p3c-10.4/data/`，Train 18000–18079 共 26,940 条，Validation 18080–18099 共 6,739 条。采集消耗共 33,679 个物理环境步。蒸馏入口仅读取当前 43 维观测、三维原始残差标签和阶段类别。阶段只用于指标及一次可选的加权采样。43 维观测含 MuJoCo 特权状态。动作后 info、未来观测和历史 Oracle 目标不进入网络。

此前逐步校验的标签审计显示，7,421/33,679 个标签全零（22.03%），±1 饱和计数 XYZ 为 5,905/0/4,943。非零请求在本轴未产生接受目标差（绝对值 <1e−12 m）的计数为 5,268/109/7,666，说明原始残差幅度与有效物理修正不能直接等同。接受目标差 P95 为 .0629/.0679/.1914 mm。阶段样本数依次为接近 9,627、近场 5,532、闭合 493、双侧未确认 6,470、抬升 9,657、保持 1,900。抬升全零约 70.8%，双侧阶段 X 残差绝对中位为 1。完整分位数在 P3c-10.4 的审计 JSON。

模型为 43→64→64→3，两层 Tanh 隐层与 Tanh 输出。输入按 Train 均值和总体标准差标准化，标准差下限 .001。原 BC 及预处理不变。训练固定种子 11，Adam LR .001，batch 256，MSE，最多 100 epoch，连续 15 epoch 验证 MSE 无改善则提前停止。每个训练配置仅按留出 Validation MSE 选 checkpoint。±35 mm 开发集和标称双朝向仅用于冻结模型闭环门槛。沿用原夹爪、XYZ scale .5、±(6,3,12) mm 账本、物理、奖励、起点和 400 步限制。

均匀采样训练 100 epoch，最佳 epoch 99。Train/Validation MSE 为 .001843/.002269，XYZ MAE 分别为 .01421/.00750/.03742 和 .01523/.00804/.04152。验证预测绝对值 ≥.95 的比例为 17.48%/0%/12.02%。训练流程耗时 11.46 s（包括数据读取与验证）。各阶段完整误差与每 epoch 曲线记录在 `uniform/training.json`、`uniform/curve.csv`。

一次修订按阶段采样，每个样本权重为其所属阶段计数的倒数，有放回抽取与均匀版相同的每 epoch 样本数。其他设置相同，仍以原分布的 Validation MSE 选模型。训练 100 epoch，最佳 95，耗时 11.47 s。Train/Validation MSE 为 .002225/.002641，XYZ MAE 为 .01361/.00695/.03612 和 .01433/.00756/.04046。验证预测饱和率 17.50%/0%/11.28%。下表给出留出集各阶段 XYZ MAE，完整训练指标见对应 JSON。

| 阶段 | 均匀采样 | 阶段均衡 |
| --- | --- | --- |
| 接近 | .0123/.0085/.0526 | .0116/.0066/.0402 |
| 近场 | .0248/.0096/.0484 | .0181/.0069/.0458 |
| 闭合 | .0269/.0234/.0754 | .0156/.0123/.0636 |
| 双侧未确认 | .0143/.0116/.0342 | .0170/.0120/.0375 |
| 抬升 | .0141/.0044/.0359 | .0123/.0061/.0449 |
| 保持 | .0083/.0037/.0110 | .0183/.0056/.0083 |

## 闭环配对与 SAC 门槛

42 个配对回合含实际 P3c-7 manifest 的 40 个 ±35 mm 开发回合和标称双向 2 回合。逐项核对各方法初态。Frozen BC 通过零残差等价接口执行，逐回合结果与历史 manifest 核对。阶段均衡版重用均匀实验的 BC/Teacher 配对参考并核对初观测。稳定保持需满足原成功判据，连续 20 步。

| 策略 | 双侧 / 确认 / 100 mm / 稳定 | 保留 34 对 | 标称 | 修复 6 对 | 异常接触 | IK 拒绝 | 平均步数 |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| Frozen BC | 39 / 36 / 36 / 36 | 34/34 | 2/2 | 0/6 | 3 | 932 | 327.55 |
| Feedback Teacher | 42 / 42 / 42 / 42 | 34/34 | 2/2 | 6/6 | 0 | 0 | 333.45 |
| Residual BC 均匀 | 42 / 42 / 42 / 41 | 33/34 | 2/2 | 6/6 | 0 | 0 | 331.62 |
| Residual BC 阶段均衡 | 41 / 40 / 40 / 40 | 33/34 | 2/2 | 5/6 | 1 | 224 | 330.48 |

原六个失败配对为 17208、17211、17218 各两种朝向。Teacher 和均匀版全部恢复，阶段均衡仅 17218 平面失败，超时前 8 次关节增量拒绝和 216 次未收敛，未抬升。该版另外在原成功的 17201 平面出现新增抓取区外接触。

均匀版失去原成功的 17207 平面回合。其在第 156 步双侧接触、第 241 步确认、第 307 步达到 100 mm，最高 135.8 mm，400 步终点仅连续保持 6 步。无异常接触或 IK 拒绝。该回合绝对累计消耗为 30.18/6.11/46.70 mm。账本未超有符号预算，长期往返仍可累积。该症状支持闭环保持不足，尚无法单独归因于阶段边界或监督误差。阶段均衡降低部分稀有阶段离线误差，仍产生闭环退化，未验证为修复手段。

| 策略 | 闭环预测饱和 XYZ（≥.95） | 非零请求产生有效接受差 XYZ | 绝对累计 XYZ 中位 / 最大（mm） |
| --- | --- | --- | --- |
| Teacher | 16.9/0/15.3% | 59.1/99.5/69.2% | 7.59/4.15/16.54 / 8.19/4.84/21.07 |
| 均匀 | 13.7/0/11.5% | 64.9/95.1/64.3% | 8.08/5.64/18.43 / 30.31/8.12/46.70 |
| 阶段均衡 | 15.9/0/11.7% | 77.5/96.7/64.6% | 10.68/4.61/17.33 / 19.90/7.87/44.08 |

**阶段 A 能力保护门槛未通过，阶段 B 未执行。** 两版均有原 BC 成功配对退化，阶段均衡还有新增碰撞。均匀版 41/42 说明蒸馏策略在开发面板能完成大多数任务，但不足以作为此次 SAC 初始化。未创建或训练 SAC 模型，新增在线 RL 交互为 0，因而没有可归因于 RL 的收益。独立 Test 17700–17739 继续封存。

## 复现与产物

```bash
export PYTHONPATH=src MPLCONFIGDIR=/tmp/cr3-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=outputs/p3c-10.5
sha256sum -c outputs/p3c-7/frozen.sha256
(cd outputs/p3c-10.4/data && sha256sum -c --quiet SHA256SUMS)
.venv/bin/python "$out/distill.py" uniform
.venv/bin/python "$out/distill.py" phase_balanced
.venv/bin/python "$out/summarize.py"
.venv/bin/python -m pytest -q tests outputs/p3c-7/test_benchmark.py outputs/p3c-8/test_diagnose.py \
  outputs/p3c-9/test_residual.py outputs/p3c-10/test_smoke.py outputs/p3c-10.1/test_audit.py \
  outputs/p3c-10.4/test_dataset.py "$out/test_distill.py"
(cd "$out" && sha256sum -c --quiet SHA256SUMS)
```

已有产物目录受保护。复跑训练时两个命令均传入相同且新的 `--root`，汇总时传入该目录作为位置参数。72 项自动化测试通过。实验目录中两配置各留有 `model.pt`、`training.json`、`curve.csv`、`evaluation.json`、逐步 `rollouts.jsonl.gz`。目录根包含训练日志、测试日志、`summary.json`、`validation-loss.png` 和 `SHA256SUMS`。冻结数据及模型留在原路径。均匀模型 SHA-256 为 `27456f45c93961e012087528581472d02c65eca15ed47bd48c6d59191f123be0`，阶段均衡模型为 `d6b55df170d4f15b103f3a07b23e69fd73fdccfba17056571740a7a54d36e53b`。原数据 manifest SHA-256 为 `aae7252424677fa0dcc6467e3d9e74cc68a10ddb2c0dc45d303947d903d9347f`。等待正式验收。

# P3c-10.6 Residual BC 闭环退化诊断与增益门槛

独立报告见 [P3c-10.6 阶段报告](P3C_10_6.md)。

基线为 `97210808e663e9c2d9c4c1a6d690cc731bc1d05c`。复用原 Uniform 与 Phase Balanced 权重、冻结原 BC、43 维输入、三维归一化残差、夹爪 BC 控制、原物理与奖励、400 步上限和原成功判据。闭环使用实际 P3c-7 ±35 mm manifest，开发 40 对与标称 2 对，独立 Test 17700–17739 封存。

## 逐步复现

三个指定平面朝向场景各运行 Frozen BC、Teacher、Uniform、Phase Balanced。每组四种方法初观测逐项相同。逐步 JSONL 保存动作前后 43 维观测、相同动作前状态上的 Teacher 标签、执行的残差、BC 原始动作、增益、两套预算、投影目标、IK、实测 TCP、物体位置和速度（观测）、接触和抬升及保持里程碑、奖励、阶段。三个异常结局均复现。

| 场景 | Frozen BC | Teacher | Uniform | Phase Balanced |
| --- | --- | --- | --- | --- |
| 17207 平面 | 成功 | 成功 | 超时，终点保持 6 步 | 成功 |
| 17218 平面 | IK 超时 | 成功 | 成功 | IK 超时 |
| 17201 平面 | 成功 | 成功 | 成功 | 抓取区外接触 |

### 17207 平面

第 1 步同一个输入下 Uniform 输出约 (−.0011,+.0137,−.0772)，Teacher 为零，产生 Z 目标差 −.058 mm，实际 TCP 差约 .010 mm。这是首次预测和执行分叉，早于任何接触。第 22 步首次达到 1.00 mm TCP 差，Uniform Z 绝对累计消费为 2.54 mm。第 156 步 Uniform 首次双侧接触，Teacher 在 158 步。第 163 步物体位置首次相差至少 1 mm。接触后的阶段时序进一步拉开，Teacher 在 223/289 步确认/抬升到 100 mm，Uniform 为 241/307 步。

第 320 步首次保持计数分叉。Teacher 的物体抬升约 134.6 mm，Uniform 约 114.7 mm，TCP 相差 19.45 mm。Uniform 的 Z 输出为 −.369，同一学生状态上 Teacher 请求 +.306，阶段转换后的修正方向不一致。Uniform 全回合最高连续保持 10 步，终点 6 步。无 IK 拒绝或异常接触。绝对累计消耗为 30.18/6.11/46.70 mm，持续往返与保持失败并存。有符号边界始终有效。首步预测误差、运动偏差、接触及后续保持偏差按此顺序发生。单个配对不能确定某一帧的预测误差足以造成最终失败。

### 17218 与 17201 平面

- 17218 的 Phase Balanced 在第 27 步就与 Teacher 的 TCP 相差 ≥1 mm。Teacher 第 168 步双侧接触时，两者 TCP 已相差约 181.9 mm。Phase Balanced 在第 170 步首次关节增量拒绝，此时距抓取点约 183.3 mm，Y/Z 有符号预算已到负边界。最终 8 次关节步长拒绝、216 次未收敛，未发生双侧接触。路径分叉先于首次拒绝。
- 17201 的 Phase Balanced 第 11 步即相差 ≥1 mm。Teacher 第 155 步双侧接触时 TCP 相差 16.6 mm；Phase Balanced 第 170 步双侧接触，尚未确认抓取，于第 193 步在相对抓取点距离约 35.024 mm 处触发指垫抓取区外接触。终止步 IK 接受。该失败发生在接触几何/时序上，早期偏差并未表现为 IK 拒绝。

## 增益扫描及能力保护

增益乘在 Uniform 网络的 Tanh 输出之后、原环境 scale=.5 和投影之前。有符号预算仍为 ±(6,3,12) mm。以 P3c-10.5 保存的 Frozen BC 和 Teacher 作为完整开发面板的配对参考，每一回合核对初态一致。三场景诊断则重新运行所有方法。

| 增益 γ | 双侧 / 确认 / 100 mm / 稳定 | 保留原成功 | 标称 | 修复原失败 | 新增异常接触 | IK 拒绝 | 平均步数 |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| .25 | 42 / 42 / 42 / 42 | 34/34 | 2/2 | 6/6 | 0 | 21 | 332.69 |
| .50 | 42 / 42 / 42 / 42 | 34/34 | 2/2 | 6/6 | 0 | 2 | 339.02 |
| .75 | 42 / 42 / 42 / 42 | 34/34 | 2/2 | 6/6 | 0 | 0 | 337.33 |
| 1.00 | 42 / 42 / 42 / 41 | 33/34 | 2/2 | 6/6 | 0 | 0 | 331.62 |

Frozen BC 和 Teacher 分别为 36/42 和 42/42。按验收要求选择最小通过增益 **γ=.25**，即沿用 Uniform 权重并冻结推理配置。17207 平面在第 229 步确认、第 296 步达到 100 mm、第 332 步累计连续保持 20 步；绝对累计消耗为 6.85/1.12/12.89 mm。三档缩放均无执行接口动作达到 .95 饱和阈值，这是缩放的直接结果。γ=.25 的 42 条轨迹最大绝对累计消耗为 15.39/2.26/15.84 mm，全部通过原物理成功判据。有符号预算及保护判据与原环境相同。缩放版在开发数据上通过初始化门槛；这批场景已用于选择增益，独立泛化未验证。

按门槛跳过 DAgger，未采集聚合数据、更新任何权重或训练 SAC/PPO。此次神经网络改进来自动作增益配置，未验证任何在线 RL 收益。

## 复现与验证

```bash
export PYTHONPATH=src MPLCONFIGDIR=/tmp/cr3-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=outputs/p3c-10.6
.venv/bin/python "$out/experiment.py" "$out/reproduction"
.venv/bin/python "$out/analyze.py" "$out/reproduction"
.venv/bin/python -m pytest -q tests outputs/p3c-7/test_benchmark.py outputs/p3c-8/test_diagnose.py \
  outputs/p3c-9/test_residual.py outputs/p3c-10/test_smoke.py outputs/p3c-10.1/test_audit.py \
  outputs/p3c-10.4/test_dataset.py outputs/p3c-10.5/test_distill.py "$out/test_experiment.py"
(cd "$out" && sha256sum -c --quiet SHA256SUMS)
```

75 项自动化测试通过。冻结原 BC、Teacher 数据、原 Residual 模型和新产物 SHA-256 校验通过。所选模型路径与哈希见 `outputs/p3c-10.6/selected.json`。完整关联回合及逐步日志在 `outputs/p3c-10.6/run/`，结果摘要、测试日志和校验和在其父目录。等待正式验收。

# P3c-10.7 Conservative Residual SAC Smoke

独立报告见 [P3c-10.7 阶段报告](P3C_10_7.md)。

在 `feat/gripper@ae30993b` 开始。采用 P3c-10.6 Uniform checkpoint 与 γ=.25，43 维输入、三维动作，原环境 scale=.5 和 ±(6,3,12) mm 有符号预算。冻结原 BC 与夹爪、奖励、物理、任务成功条件和 400 步回合上限。独立 Test 17700–17739 封存。

## 初始化和配置

SB3 2.7.1 SAC Actor 使用 64×64 Tanh 隐层及复制的线性均值头，分布执行一次 Tanh。使用 Residual BC 的标准化，冻结原 BC 走原预处理路径。增益在环境包装器内生效，Replay 和双 Q 使用增益前分布动作，log-prob 保持在该空间。参考均值冻结，actor loss 附加 .05×动作均值平方偏差，参照初始化教师蒸馏策略。Critic 独立随机初始化并复制到 target。配置为 seed=11、固定 α=.005、log-std 初值 −3 及范围 [−4,−3]、Actor/Critic LR=.0001/.0003、折扣 .99、τ=.005、batch 128、空 buffer 容量 2,000、前 500 步以初始化策略的 squashed Gaussian 采样。只从 18000–18079 抽场景。设置逐步回调以实现 checkpoint，预热后仅每 4 步实际更新一次。

在历史开发与标称 42 回合，冻结原 BC 36/42、Teacher 42/42、γ=.25 Residual BC 42/42（均步 332.69，21 次 IK 拒绝，无碰撞）。另冻结留出 18080–18099 共 20 回合，每种朝向 10 个，Residual BC 20/20。零次优化的 SAC 在全 62 回合逐步动作严格相等，解压后 JSONL 完全一致。完成步数、100 mm 抬升步数及 IK 的配对变化均为 0。

## 在线停止与结论

预注册监测绝对累计 XYZ 超过 20/5/22 mm、非有限状态，及增益前动作绝对值 ≥.95 在最近 50 步的任一轴比例超过 50%。每 250 步另做确定性成对保护检查。种子 11 首个回合 18010 凸起朝向在第 **211 步**触发 X 饱和保护（52%），未完成一个完整在线回合。此时绝对累计消耗约 6.60/1.77/7.60 mm。执行侧乘增益后没有饱和；对同一访问状态复算的确定性 X 饱和反而为 54%。因此这是初始策略在该状态序列中的概率空间饱和，未观察到梯度驱动漂移。实际优化 **0 次**，Actor/Critic loss 和 α 更新曲线无新增点。250/500/750/1000 均未到达。没有可归因于 SAC 更新的效率、成功率或泛化收益。

保留首次浮点等价失败日志、在线停止日志及无效原始停止 zip。修复 SB3 写入流序列化后，从零步权重重建等价的 211 步分析快照，使用原动作精确重放 211 步，重建分析用 buffer。未重跑在线训练。详见 [技术调试记录](P3C_10_7_DEBUG.md)。

## 复现与产物

```bash
export PYTHONPATH=src MPLCONFIGDIR=/tmp/cr3-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=outputs/p3c-10.7
.venv/bin/python "$out/conservative_sac.py" "$out/reproduction"
.venv/bin/python "$out/report.py" "$out/run"
.venv/bin/python "$out/replay_stop.py" "$out/run"
.venv/bin/python -m pytest -q tests outputs/p3c-7/test_benchmark.py outputs/p3c-8/test_diagnose.py \
  outputs/p3c-9/test_residual.py outputs/p3c-10/test_smoke.py outputs/p3c-10.1/test_audit.py \
  outputs/p3c-10.4/test_dataset.py outputs/p3c-10.5/test_distill.py \
  outputs/p3c-10.6/test_experiment.py "$out/test_conservative_sac.py"
(cd "$out" && sha256sum -c --quiet SHA256SUMS)
sha256sum -c outputs/p3c-7/frozen.sha256
(cd outputs/p3c-10.4/data && sha256sum -c --quiet SHA256SUMS)
```

77 项自动化测试通过，冻结 BC、数据和模型/日志校验通过。有效 checkpoint、重放、基线配对、逐步日志与曲线均在 `outputs/p3c-10.7/run/`；校验和及小型摘要位于父目录。教师采集先前耗费 33,679 步，区别于本轮 211 个 SAC 在线物理步（确定性评估和诊断回放另计）。等待验收。

# P3c-10.7-R1 Conservative Residual SAC Smoke

独立报告见 [P3c-10.7-R1 阶段报告](P3C_10_7_R1.md)。

从干净的 `feat/gripper@9adaa76d75ebed651935b0b8e8a092629ed324dc` 开始，核对原 SAC 代码、报告、序列化修复和交接记录。P3c-10.7 的 211 步停机保留为独立历史实验。R1 使用新目录 `outputs/p3c-10.7-r1/run/`。冻结 Uniform 初始化与原 BC，保持 43 维输入、三维策略动作、Gain=.25、scale=.5、±(6,3,12) mm 有符号预算、BC 夹爪、原仿真、奖励和 400 步回合。训练只用 18000–18079，留出验证为 18080–18099，历史开发及标称为 42 个回合，独立 Test 17700–17739 封存。

## 预注册监测和配置

对同一在线动作前观测计算随机采样、当前确定性 Actor 和冻结初始化 Actor。使用相同的实时 TCP 目标、BC 动作与账本副本，计算三者在混合、工作区与预算投影后的候选目标。确定性候选差是同状态命令差，实际 IK 接受与实测位移由环境对采样动作记录。候选计算不运行影子 IK 或推进物理。随机动作、确定性动作、参考、执行增益后动作和接受修正分开保存。

离线精确复放历史 211 步，随机相对初始 Actor 的 XYZ 差 P95 最大 .091，投影差 P95 最大 .0171 mm，观测最大分别约 .148/.0278 mm。零更新的确定性相对偏差为 0。训练前冻结相对漂移门槛：实际更新后，连续 50 步窗口内同轴同时超过绝对差 P95 .25 与投影差 P95 .046875 mm，在下一确定性配对检查停止。第二阈值来自 .25 × Gain .25 × scale .5 × 1.5 mm。历史饱和 ≥.95 单独报告。绝对累计 20/5/22 mm 是经验审查线，在线仅记录，确定性配对越线停机。任一原成功退化、新增接触、模拟故障触发配对停止。训练探索中的普通失败由环境结束回合，违规动作、非有限状态、账本或保存重载不一致按硬故障处理。

SB3 2.7.1、seed 11，Actor/Critic 64×64，复制原标准化和线性均值头，分布执行一次 Tanh，增益位于环境端。固定 α=.005，log-std 初值 −3、范围 [−4,−3]，Actor/Critic LR .0001/.0003，γ=.99、τ=.005、batch 128、空 Replay 容量 2000。前 500 步在初始策略周围采样，随后每 4 步一次更新，预注册最多 1,000 步。Actor 对冻结初始策略的均值动作 MSE 锚权重 .05。Replay 存增益前分布动作。Checkpoint 在转移入库后保存并重载，逐项验证模型、两套 optimizer、Replay 内容及冻结参考。每 250 步评估固定面板。详见 [调试记录](P3C_10_7_R1_DEBUG.md)。

## 结果

0/250/500 步在 62 个回合的逐步动作精确匹配初始化，Actor/Critic 参数完全一致。内容核对时发现 R1 基线压缩日志有单行 `?bs` 键，按 `obs` 识别后 0/250/500 的 JSON 逐行相同。原产物保留。实际优化自 504 步开始，到 748 步共 62 次。750 步 Actor/Critic 最大参数变化 .00671/.01749，冻结参考不变。首末 Actor loss .11537/.11751，Critic loss .02218/.000342，α 始终 .005，采样批次记录的首末 log-std 三轴均为 −3。更新记录见 `updates.jsonl` 与训练图。

| 方法或快照 | 历史含标称 | 留出验证 | IK 拒绝（历史/留出） | 异常接触（历史/留出） |
| --- | --- | --- | --- | --- |
| Frozen BC | 36/42 | 18/20 | 932/0 | 3/2 |
| Teacher | 42/42 | 20/20 | 0/0 | 0/0 |
| Residual BC γ=.25 | 42/42 | 20/20 | 21/0 | 0/0 |
| SAC 0/250/500 各自 | 42/42 | 20/20 | 21/0 | 0/0 |
| SAC 750 | 42/42 | 20/20 | 15/0 | 0/0 |

750 步相对冻结初始化的任务完成步数配对变化如下，负值表示更快。

| 场景 | 更快 / 更慢 / 持平 | 均值 / 中位 / P95 | 100 mm 首步变化均值 / 中位 / P95 |
| --- | --- | --- | --- |
| 历史及标称 | 35 / 6 / 1 | −4.55 / −5 / +3 | −4.50 / −4 / −1 |
| 留出 | 17 / 2 / 1 | −4.45 / −5 / +3.30 | −5.50 / −5 / −1 |
| 历史凸起 / 平面 | 15/5/1；20/1/0 | −1.52/−2/+3；−7.57/−7/−1 | −2.52/−3/−1；−6.48/−7/−2 |
| 留出凸起 / 平面 | 7/2/1；10/0/0 | −2.10/−2.5/+6.30；−6.80/−6/−3.35 | −3.80/−3/−1；−7.20/−5.5/−2.35 |

没有配对新增 IK 拒绝。历史面板绝对累计 XYZ 的均值/中位/P95 为 7.16/6.95/7.79、3.85/3.83/4.90、12.13/12.02/13.78 mm。留出分别为 7.03/6.87/7.71、3.65/3.35/4.95、12.16/12.23/14.03 mm。在线暖启动确定性偏差为零。暖启动后原动作 XYZ 最大偏差 .098/.210/.259，投影目标差 XYZ P95 .0177/.0384/.0225 mm。无滚动窗口触发相对偏移组合阈值。采样饱和由暖启动的 9.2/0/5.0% 变为之后的 19.6/0/10.8%，不同阶段在线状态分布也变化，不能直接解释为退化。两段均无增益后接口饱和。

**按预注册规则在 750 步停止。** 17208、17211、18093 的平面朝向回合仍成功，但 Y 绝对累计分别 5.577、5.391 和 5.702 mm，超过经验审查线。1000 步未执行。更新后在两个面板均观察到平均加快约 4.5 步的效率信号，但由于未满足全部保护条件，没有验证出可接受的 SAC 净收益。暂不追加实验。

在线 750 物理步，完成两个成功探索回合（18010 凸起、18063），末回合未完成，无在线探索失败。冻结初始化和各 checkpoint 的面板评估共 102,885 步，留出 BC/Teacher 额外比较 13,268 步，历史离线动作复放校准 211 步。在线含门槛评估 85.53 s，训练入口总 139.56 s，额外比较另计。Teacher 数据历史采集成本 33,679 步。

## 复现及交付

```bash
export PYTHONPATH=src MPLCONFIGDIR=/tmp/cr3-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=outputs/p3c-10.7-r1
# 历史实验已执行一次，下列训练命令仅用于独立复现。
.venv/bin/python "$out/r1.py" "$out/reproduction"
.venv/bin/python "$out/analyze.py" "$out/reproduction"
.venv/bin/python -m pytest -q tests outputs/p3c-7/test_benchmark.py outputs/p3c-8/test_diagnose.py \
  outputs/p3c-9/test_residual.py outputs/p3c-10/test_smoke.py outputs/p3c-10.1/test_audit.py \
  outputs/p3c-10.4/test_dataset.py outputs/p3c-10.5/test_distill.py outputs/p3c-10.6/test_experiment.py \
  outputs/p3c-10.7/test_conservative_sac.py "$out/test_r1.py"
(cd "$out" && sha256sum -c --quiet SHA256SUMS)
```

79 项自动化测试通过。新产物和历史冻结哈希校验通过。模型、optimizer、Replay、逐步在线与确定性评估日志、分组 JSON 与曲线位于 `outputs/p3c-10.7-r1/run/`，配置和小型摘要入 Git。原阶段历史保留，独立 Test 继续封存，等待验收。

# P4a RGB Vision + Learned Grasp & Lift Integration

独立报告见 [P4a 阶段报告](P4A.md)。

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

# P4b 仿真示范采集与 LeRobot v3 导出

独立报告见 [P4b 阶段报告](P4B.md)。

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

# P4c LeRobot 动作片段训练与闭环验证

独立报告见 [P4c 阶段报告](P4C.md)。

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

# P5a Residual PPO Smoke 与冻结 SAC 对照

独立报告见 [P5a 阶段报告](P5A.md)。

从 feat/gripper `fcf5dcb1e07c7c1cde50aa30c72397d410f18e88` 开始。main 为 `75106355459af2f283e21593734ca172655162a4`，含已验收 P4a/P4b，P4c 仍在功能分支。保留原未追踪 `docs/开发共识.md`、CR3 备份、冻结模型与历史数据。已核对 P4c、SAC R1、BC 和备份 SHA-256。O6 未操作。

## P4c 离线调试

以实际权重和部署 `predict` 检查 1,284 个来自 LeRobot 的验证帧，同像素与状态下实际预处理和直接网络反标准化的输出逐项一致。J6 MAE .009268 rad，夹爪 .017070，复现历史离线指标。J6 以 Release .01327、Reorient .01310、Grasp .01148 rad 偏差较高。夹爪反标准化范围 [−.0721,1.0333]，6,420 项中 2,668 越界，多数在物理边界附近，实际控制前有记录的限幅。

用固定失败回合 000/001 的原始专家 RGB 与状态重算模型输出，前 30 步 J6 最大误差 .0157/.0155 rad、夹爪 .0163/.0165。历史学生自执行状态上对应值为 .1880/.1739 rad 与 .5961/.7262，差异在接触前已经出现。验证集重用解码缓存，专家失败对照用仿真原图，两条路径的图像编码来源有差异。未发现确定的归一化/反标准化接口错误；剩余可能包括小动作偏差与自执行状态偏移的反馈、编码差异及低分辨率视觉。没有复训或更改 P4c 产物。

## PPO 接口及保护

基于 SB3 2.7.1，43→64→64→3 Tanh 隐层，复制 Residual BC 标准化、隐藏层和线性均值，确定性输出仅一次 Tanh。三轴状态无关 log_std 初值 −3、执行 clamp [−4,−3]。有界重参数采样及 log-prob 采用 SB3 的 SquashedDiagGaussian，保存增益前动作并对存储动作统一逆变换、减去 Tanh Jacobian，更新前新旧比为 1。环境外层只乘 Gain=.25，继续原 scale=.5、±(6,3,12) mm、冻结 BC 夹爪、奖励和 400 步判据。时序 50 Hz。训练不使用额外 actor anchor，冻结参考用于核对初始化和同观测偏移。

固定 seed 11、750 步上限、250 步 rollout、125 minibatch、最多 2 epoch、LR 1e−4、clip .1、target KL .01、γ .99、GAE λ .95、熵权重 0、value 权重 .5、梯度裁剪 .5。仅 18000–18079 参与交互。62 个检查场景复用 R1 的历史及标称 42 和留出 20。复用账本硬检查、配对异常接触/成功保护、20/5/22 mm XYZ 绝对累计审查线，另监测同轴连续 50 步 P95 原动作偏移 .25 和投影差 .046875 mm 的组合。监督离线诊断、确定性评估和渲染不计训练步。

## 实测及调试

预检首次因带 Normalize 的 Sequential 切片参数编号不同而拒绝加载，发生在交互与优化前。修正为逐层复制，通过采样范围、数值 log-prob、概率比和载入测试。零更新 62 个回合逐步动作严格相同，回合结果与冻结基线一致，Actor/Critic 参数变化均 0。第一次绘图因复用的 plant_for 导入位置错误退出，修正后重新生成产物，未重复训练。

250 步完成第一次更新。优化器实际走了 1 个 minibatch，后续 minibatch 触发 SB3 的 KL 早停；记录 Approx KL .05891、clip fraction .356、policy gradient loss .06439、value loss .01855、entropy loss 5.776、有效 log_std 约 −3.00010。Actor/Critic 最大参数变化 .0001000/.0000994。0 和 250 的权重与 optimizer 都重载比对通过。

| 策略 | 历史及标称 | 留出 | 成功完成步数均值 / P50 / P95（历史） | 留出均值 / P50 / P95 | IK 历史 / 留出 | 异常接触 |
| --- | --- | --- | --- | --- | --- | --- |
| Residual BC / PPO 0 | 42/42 | 20/20 | 332.69 / 331 / 356.8 | 333.00 / 332.5 / 339.7 | 21 / 0 | 0 |
| 冻结 SAC 750 | 42/42 | 20/20 | 328.14 / 324 / 346.7 | 328.55 / 327 / 342.3 | 15 / 0 | 0 |
| PPO 250 | 40/42 | 20/20 | 330.18 / 330.5 / 339.1 | 332.90 / 332.5 / 340.6 | 657 / 0 | 0 |

PPO 历史共同成功的 40 个配对平均仅快 .25 步，留出快 .10 步。确认与 100 mm 首次抬升（历史成功回合均值）BC 为 227.12/294.31 步，SAC 223.21/289.81，PPO 225.55/292.63；分位数、两面板累计消耗及配对行保存在 JSON。PPO 历史绝对累计 XYZ 最大 6.91/1.75/15.83 mm，SAC 9.87/5.58/14.16 mm。SAC 在全 62 面板三个平面回合超 5 mm Y 线而停机，PPO 未因累计线停止。

PPO 仅完成 250 步、1 次梯度更新，在线含检查耗时 27.64 s。SAC 历史为 750 步、62 次更新、在线含检查 85.53 s。PPO 两个回归均为 17218，凸起/平面各有 350/307 次 IK 拒绝，未建立双侧接触，400 步超时。训练期间只有更新前的一个未结束探索片段。更新后 Actor 在该片段同一观测上的原动作差 P95=.0094/.0048/.0450，虽然幅度小，闭环可达性和后续观察已改变；不能单独判定某轴是根因。按配对保护停止，500/750 不运行。独立 Test 17700–17739 继续封存。

这是同环境的一次探索性对照。PPO 尚无合格的在线收益，SAC 的效率信号也跨越了既定审查线。正式比较需另行批准，预先冻结两算法评价门槛、分离等交互预算和更新次数、多种子复现，并在选定后保留独立评估。此轮不重启训练。

## 复现和产物

```bash
export PYTHONPATH=src MPLCONFIGDIR=/tmp/cr3-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MUJOCO_GL=egl MESA_SHADER_CACHE_DISABLE=true
out=outputs/p5a
.venv/bin/python "$out/vision_audit.py" "$out/reproduction-vision-audit.json"
.venv/bin/python -m pytest -q tests "$out/test_ppo_dist.py"
.venv/bin/python "$out/ppo.py" "$out/reproduction"
.venv/bin/python "$out/report.py" "$out/reproduction" --media
(cd "$out" && sha256sum -c --quiet SHA256SUMS)
```

86 项历史与本轮自动化测试通过。模型、在线、逐步确定性和训练日志在 `outputs/p5a/run/`。配对演示选择 17208 平面成功回合，三种策略同种子；保留 17218 两朝向失败轨迹。没有改视觉、SAC 或默认控制器。等待验收。

# P5a-R1 PPO 首次更新诊断与低学习率复验

独立报告见 [P5a-R1 阶段报告](P5A_R1.md)。

基线 feat/gripper `f429d1834953b1bebf82d09d51f086933f6d7e08`，main `75106355459af2f283e21593734ca172655162a4`。核验原 PPO 快照、超参、日志、冻结 BC/Residual BC、SAC R1 750 和交接文档。原历史数据未覆盖。O6、独立 Test 17700–17739 保持封存或原状。工作树的未追踪用户文件 `docs/开发共识.md` 保留。

## 不训练的诊断

针对 17218 双朝向重跑 Residual BC、PPO 0、PPO 250。相同种子、初态逐项一致，BC 与 PPO 0 整个日志一致。逐步附件保存动作前后 43 维状态、原 BC 动作、增益前后残差、工作区和预算后的 IK 目标、候选关节和拒绝、TCP、机体及物体状态、接触、奖励、里程碑。真实不同轨迹的 TCP 差按步骤报告；策略动作差另在同一当前观测计算。

| 原 PPO 250 对 PPO 0 | 首动作和有效目标差 | 首次 IK 拒绝 | TCP 首次相差至少 1 mm | 结局 |
| --- | ---: | --- | ---: | --- |
| 17218 凸起 | 第 1 步，实测 TCP 差 .000067 mm | 38 步，关节步长，TCP 差 .0234 mm | 56 步 | 无双侧接触，400 步超时，350 次拒绝 |
| 17218 平面 | 第 1 步，实测 TCP 差 .000063 mm | 37 步，关节步长，TCP 差 .0392 mm | 62 步 | 无双侧接触，400 步超时，307 次拒绝 |

首次拒绝早于毫米级轨迹分叉。后续相互依赖的动作、状态与 IK 效应不能由逐帧差值唯一归因。报告脚本保留同状态旧/新动作及全部候选目标。

旧 250 在线步包含一个未结束回合，0 完成成功、0 任务终止奖励。动作前阶段计数为接近 108、近场 58、闭合 4、双侧未确认 64、抬升 16、保持 0。样本覆盖部分抓取和初期抬升，对完整任务和保持的直接监督不足。按历史同状态、同存储采样动作复算，原 PPO 250 对 0 的概率比中位 .967、P95 1.634、最大 5.297。全 rollout 近似 KL .10341、解析 Gaussian KL 均值 .10433，clip .700。训练日志 .05891/.356 是历史 minibatch 早停聚合量，口径不同。完整采样动作都在开区间内，动作与执行接口增益前日志一致；同一动作按逐样本与 batch 重算旧 log-prob 最大差 1.96e−5。未发现重复 Tanh、旧概率不一致、归一化或检查点载入错误。原 rollout 没有超时，从 SB3 2.7.1 路径和合成测试核对超时 bootstrap，且 GAE 不串回合。

标准差约 .049787→.049782，更新主要改变均值。Z 轴 pre-Tanh 均值偏移 P95 为 0.952 个旧标准差，后 Tanh 的同状态 XYZ 动作偏移 P95 为 .00942/.00478/.04498。方差较小时，原单位尺度下不大的均值变化也能形成较高 KL。GAE 均值 −.1123、范围 [−.2802,.2617]，价值变化均值 −.00924。0 点全 rollout 梯度范数 15.92，属于离线重算，不是历史 125 样本 minibatch 梯度日志；原裁剪上限 .5，optimizer 状态记录一步。

## 单因素复验

训练接口与保护不变，仅把学习率 `1e-4` 改为 `1e-5`。种子 11、750 上限、250 rollout、125 minibatch、2 epoch、clip .1、target KL .01、γ .99、GAE λ .95、log-std −3、原增益 .25 和预算。配置文件逐项比对。仍用原 Train、验证和开发配对分区。0 更新恢复 42/42 和 20/20。250 时完成 4 次 minibatch 更新、2 epoch，实际参数变动 Actor/Critic 最大 3.99e−5/3.97e−5。Approx KL .001331、clip .054。权重及优化器状态保存重载完全一致。在线含评估 27.71 s。

| 方法 | 交互 / 更新 | 历史及标称 | 留出 | 历史成功步数均值/P50/P95 | IK 历史/留出 | 接触 |
| --- | --- | --- | --- | --- | --- | --- |
| Residual BC | 0 / 0 | 42/42 | 20/20 | 332.69/331/356.8 | 21/0 | 0 |
| SAC 750 冻结 | 750 / 62 | 42/42 | 20/20 | 328.14/324/346.7 | 15/0 | 0 |
| PPO 原 250 | 250 / 1 | 40/42 | 20/20 | 330.18/330.5/339.1 | 657/0 | 0 |
| PPO R1 250 | 250 / 4 | 41/42 | 20/20 | 331.71/331/352.0 | 312/0 | 0 |

R1 17218 凸起恢复成功（378 步、9 次拒绝），平面仍超时（303 次拒绝、无双侧接触）。历史共同成功回合相较初始化平均慢 .098 步；留出平均快 .10 步。R1 整体最大累计 XYZ 12.23/2.03/16.30 mm，无新增接触或绝对审查越线。**250 步因平面场景能力退化停止**，未运行 500/750。低 LR 降低了 KL 和总体 IK 拒绝，仍未满足能力保护，未验证可接受的 PPO 收益。冻结 SAC 原 Y 审查线停机仍有效。对照预算和更新不同，不能给出算法排名。

## P4c 归档

沿用 [P5a 的离线审计](P5A.md)。未发现确定的部署反标准化错误；学生闭环前 30 步 J6 最大误差 .174–.188 rad，专家状态约 .016 rad。结果支持状态分布偏移，仍无法排除编码图像与仿真原图差异及其他原因。未训练视觉策略。

## 产物与复现

```bash
export PYTHONPATH=src MPLCONFIGDIR=/tmp/cr3-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MUJOCO_GL=egl MESA_SHADER_CACHE_DISABLE=true
out=outputs/p5a-r1
.venv/bin/python "$out/ppo_update_audit.py" "$out/repro-diagnostic"
# 原实验只运行一次。此命令用于独立复现。
.venv/bin/python outputs/p5a/ppo.py "$out/repro-run" --learning-rate 0.00001
.venv/bin/python outputs/p5a/report.py "$out/repro-run" --media --media-seed 17218
.venv/bin/python "$out/plots.py" "$out/repro-plots"
.venv/bin/python -m pytest -q tests outputs/p5a/test_ppo_dist.py "$out/test_audit_r1.py"
(cd "$out" && sha256sum -c --quiet SHA256SUMS)
```

87 项项目与历史测试通过。逐步诊断、同场景 TCP/IK/累计图、KL 更新曲线、快照及原始日志位于 `outputs/p5a-r1/`。17218 平面失败配对 GIF 位于 `run/`，17208 成功配对位于 `media-pass/`。详见[独立调试记录](P5A_R1_DEBUG.md)。未追加训练，等待验收。

# P5a-R2 Residual PPO 与 SAC 等交互预算对照

独立报告见 [P5a-R2 阶段报告](P5A_R2.md)。

起点为 `feat/gripper@12433e763081b40aa26352439f14f47452373064`，本机 `main@7510635`。冻结 BC、Uniform 残差 checkpoint、P5a/P5a-R1 与 P3c-10.7-R1 校验通过。保留本机忽略资产及未追踪的 `docs/开发共识.md`。O6、历史训练和独立 Test 17700–17739 保持原样。

## 预注册协议

六组固定实验，PPO/SAC 各 seed 11、22、33，每组上限 20,000 在线步。每个 seed 使用同一份按回合索引预生成的场景与朝向序列，种子只在 18000–18079 内。由于回合长短不同，相同步数使用的序列前缀长度可不同。所有检查点使用完全相同的 20 回合验证和 42 回合历史/标称诊断。零更新全 62 回合逐步动作严格一致，评分逐项相同。历史面板只用于诊断。

共享冻结 BC 和夹爪、43 维特权观测、Residual BC 输入标准化、64×64 Tanh 隐层、三维 Squashed Gaussian 动作、初始 log-std −3、范围 [−4,−3]、Gain=.25、XYZ scale=.5、原工作区和 ±(6,3,12) mm 有符号预算。原 50 Hz / 400 步任务物理、奖励和成功判据保持。对两算法的 actor 目标加入相同 `0.05 × mean((tanh(μ) − a_init)²)`，在增益前确定性动作空间按训练批次取均值。比较的是包含不同优化目标、批次和熵处理的完整训练方案。

- PPO 采用原 Squashed Gaussian 和 SB3 GAE/超时 bootstrap，重用记录的旧 log-prob。每 1,000 步 rollout，125 minibatch，至多 2 epoch，LR 1e−5，clip .1，target KL .01，γ .99，GAE λ .95，梯度上限 .5，value 权重 .5，entropy 权重 0。Critic 为标量 V，log-std 为状态无关参数。
- SAC 用历史保守 actor/critic 更新式，独立双 Q，target 软更新 .005，γ .99，actor/critic LR 1e−4/3e−4，batch 128，固定 α=.005，状态相关 log-std，500 步初始化高斯探索，之后每 4 步一次。Replay 由历史 2,000 增至 24,000，容纳本轮全部在线经验。Critic 和 Replay 使用增益前动作。
- 0、1k、2k、5k、10k、20k 保存和重载；策略退化、IK 拒绝或普通任务失败仅记录。原 20/5/22 mm 仍作为部署审查线。仿真数值、动作接口、账本、模型或快照硬故障方可中断。
- 预先冻结的候选规则见 `protocol.json`。对更新过的模型，两面板同保留原成功、无异常接触、绝对消耗不越线才 eligible；优先验证成功数、成功回合平均步数、较早步数。无合格者保留最后模型并按验证成功数、较早步数列研究候选。独立 Test 继续封存。

## 实验结果

六组均完成 20,000 在线步，总计 120,000 步。每组完成全部六个检查点及 62 回合确定性评估，合计 2,232 次评估回合。PPO 每组 320 次 minibatch 更新，SAC 每组 4,875 次双 Q/Actor 更新轮次。六组无硬故障。PPO 每组总耗时约 238–242 s，扣除评估和保存约 53–55 s；SAC 分别约 275–285 s 和 86–92 s。六个进程并发使用 CPU，时间含资源争用。

## 复现与产物

```bash
export PYTHONPATH=src MPLCONFIGDIR=/tmp/cr3-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MUJOCO_GL=egl MESA_SHADER_CACHE_DISABLE=true
out=outputs/p5a-r2
# 训练入口拒绝覆盖现有目录。以下为独立训练复现示例。
.venv/bin/python "$out/research.py" ppo 11 "$out/reproduction-ppo-11"
.venv/bin/python "$out/analyze_r2.py"
.venv/bin/python "$out/media_r2.py"
.venv/bin/python -m pytest -q tests "$out/test_research.py" outputs/p5a/test_ppo_dist.py
(cd "$out" && sha256sum -c --quiet SHA256SUMS)
```

各 seed 独立目录包含 `scenario-order.json`、在线与优化日志、逐回合及逐步确定性评估、模型与状态文件。SAC 额外保存 Replay。汇总表/区间、曲线、配对 GIF 和候选 checkpoint 路径见实验根目录。完整轨迹与权重留本机。自动化回归实际运行 88 项，全通过。

## 配对指标与候选

下表每格为历史及标称成功数 / 验证成功数，分母固定 42 / 20。

| 组 | 0 | 1k | 2k | 5k | 10k | 20k | 入选步数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PPO 11 | 42/20 | 42/20 | 42/20 | 41/20 | 42/20 | 42/20 | 2k |
| PPO 22 | 42/20 | 41/20 | 42/20 | 42/20 | 42/20 | 42/20 | 5k |
| PPO 33 | 42/20 | 41/20 | 41/20 | 42/20 | 42/20 | 42/20 | 20k |
| SAC 11 | 42/20 | 42/20 | 42/20 | 41/20 | 36/19 | 40/20 | 2k |
| SAC 22 | 42/20 | 36/16 | 40/20 | 42/20 | 40/20 | 40/20 | 5k |
| SAC 33 | 42/20 | 41/20 | 41/20 | 42/20 | 40/20 | 41/20 | 5k |

六个入选均符合预注册面板审查，尚待独立检验。20k 时 PPO 三组都合格，SAC 三组都因历史成功退化失去资格，另有累计越线。冻结 BC 为 42/42 和 20/20，历史 IK 拒绝为 21。末次检查按共同成功样本计算完成步数差，负值更快；失败样本仍计入成功率及所有回合平均长度，三种统计分列在 JSON。

| 组 | 成功回合配对差 历史 / 验证（步） | 末次历史 IK 拒绝 | 末次历史 XYZ 绝对累计最大（mm） | 在线完成 成功 / 超时 |
| --- | --- | ---: | --- | --- |
| PPO 11 | +.76 / −.45 | 23 | 12.81 / 2.23 / 15.85 | 60 / 0 |
| PPO 22 | +1.43 / +1.15 | 18 | 13.21 / 2.37 / 15.97 | 59 / 1 |
| PPO 33 | −3.81 / −4.20 | 19 | 11.81 / 2.06 / 15.88 | 60 / 0 |
| SAC 11 | −13.60 / −13.90 | 657 | 12.66 / 8.07 / 16.72 | 57 / 2 |
| SAC 22 | +.53 / +.35 | 635 | 17.78 / 7.09 / 17.22 | 59 / 1 |
| SAC 33 | −12.80 / −8.15 | 308 | 16.30 / 3.69 / 23.88 | 56 / 4 |

末次留出 IK 拒绝均为 0，历史面板无异常接触。SAC 11/22 在留出 Y 最大达 8.25/7.91 mm。末次五个 SAC 历史失败均在双侧接触前超时。SAC 在 1k 就出现明显种子间差异，seed 22 在 5k 恢复、10k 再退化。PPO 也出现早期单配对退化，三组在 10k 恢复。PPO 首轮 1,000 步有 3/2/3 个成功回合，观测覆盖接触、抬升与保持，区别于原 250 步 smoke 的无成功、无保持 rollout。Replay 24k 容量无覆盖写入，在线阶段计数即其进料分布。SAC 末次累计约 1,064–1,121 个保持观测。

末段 10k–20k 同状态增益前动作偏移，PPO 各轴 P95 最大为 .021，SAC 最高约 1.14，主要差值在 Z。在线访问分布随策略而变，该对比只用于观察偏移。SAC 两组共同成功回合更快，同时丢失历史成功并增加 IK 拒绝，显示执行效率与保留性之间的实测取舍。单纯回报或共同成功子集速度不能证明整体稳定改善。三种种子、相同交互数的结论限于所冻结的两套优化配置。

代表性演示配对使用 17218 平面朝向。残差 BC、PPO seed 11@2k、SAC seed 11@2k 分别在 377、374、366 步成功。视频覆盖整段并标注研究/资格状态，五步采一帧；完整逐控制轨迹另存。独立 Test 仍封存，未延长训练或追加选参。

# P5a-R3 独立 Test 与 RL 研究冻结

独立报告见 [P5a-R3 阶段报告](P5A_R3.md)。

以 `feat/gripper@68013997ff513bac027243a2b4c4687281b71c32` 为起点。main 为 `7510635`，O6 为 `1465436`，两个工作树均未操作。保留未追踪的 `docs/开发共识.md`。核对 P5a-R2 六组原训练日志、配置、选模记录及完整资产 SHA-256，也核对冻结 BC 和原始示范哈希。

## 揭封与评估

揭封前提交 `10b44c0` 固定七个策略及原场景。Residual BC 使用原 Uniform、Gain=.25。PPO 种子 11/22/33 为 2k/5k/20k，SAC 为 2k/5k/5k，均按 P5a-R2 原资格和排序。冻结清单记录每个权重 SHA-256、共享标准化、原 BC、选择协议与零更新参考。

原 P3c-7 manifest 保存 35 mm 测试种子 17700–17739。用其原 `sample_xy` 复原 40 个实际 XY，每个两朝向。保持原 50 Hz、400 步、预算、物理、奖励及成功判据。各策略对每场景核对初始完整 MuJoCo 积分状态及 43 维观测的 SHA-256。只执行确定性推理。每回合保存动作前后观测、参考动作、物理里程碑、IK、TCP 与累计账本。单独保存仿真积分状态供图像重建。历史学习曲线与独立结果分图保留。

## 复现与文件

正式入口以新目录写入，通过 `started.json` 记录冻结清单与测试 manifest 哈希。实际正式命令如下，已执行一次。

```bash
export PYTHONPATH=src MPLCONFIGDIR=/tmp/cr3-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MUJOCO_GL=egl MESA_SHADER_CACHE_DISABLE=true
out=outputs/p5a-r3
.venv/bin/python "$out/final_test.py" freeze
.venv/bin/python "$out/final_test.py" run
.venv/bin/python "$out/report_final.py"
.venv/bin/python -m pytest -q tests "$out/test_final.py"
(cd "$out" && sha256sum -c --quiet SHA256SUMS)
```

正式数据位于 `run/`，包含七个逐回合 JSON、压缩控制轨迹及实际状态快照。分组统计、逐回合 CSV、候选、图和媒体选样记录位于实验根目录。90 项自动化测试通过，含历史回归、场景还原、候选哈希及计时指标测试。详见 [技术记录](P5A_R3_DEBUG.md)。

## 独立 Test 结果

七个策略各 80 回合，共 560 回合、183,746 个控制步。80 个场景的初态哈希在七策略间逐项一致。下表成功项同时达到双侧接触、抓取确认、100 mm 抬升及连续 20 步稳定保持。两次失败均未完成任何上述里程碑。

| 策略 | 成功 | 成功步数均值 / P50 / P95 | 共同成功对相对 BC（步） | IK 未收敛 / 关节步长拒绝 | XYZ 绝对累计最大（mm） | 独立审查 |
| --- | ---: | --- | ---: | --- | --- | --- |
| Residual BC | 80/80 | 331.29 / 332 / 338.05 | 0 | 0 / 0 | 7.39 / 1.59 / 16.09 | 通过 |
| PPO 11 @2k | 80/80 | 330.78 / 331 / 338.00 | −.51 | 0 / 0 | 7.41 / 1.62 / 16.07 | 通过 |
| PPO 22 @5k | 80/80 | 331.41 / 332 / 339.00 | +.13 | 0 / 0 | 7.35 / 1.55 / 16.15 | 通过 |
| PPO 33 @20k | 80/80 | 327.41 / 327.5 / 335.10 | −3.88 | 0 / 0 | 7.91 / 1.54 / 16.24 | 通过 |
| SAC 11 @2k | 79/80 | 321.52 / 321 / 334.00 | −9.71 | 203 / 5 | 7.35 / 3.40 / 15.01 | 原成功配对退化 |
| SAC 22 @5k | 80/80 | 328.94 / 327.5 / 347.05 | −2.35 | 0 / 0 | 14.87 / 3.77 / 13.77 | 通过 |
| SAC 33 @5k | 79/80 | 323.54 / 322 / 340.00 | −7.68 | 294 / 7 | 7.55 / 4.21 / 14.31 | 原成功配对退化 |

七策略均无异常接触、无有符号预算违规、无 20/5/22 mm 绝对累计审查越线。两次超时均为同一 XY 的 17721 平面朝向，SAC 11 和 33 都停在闭合阶段，分别 208/301 次拒绝。两次均为 400 步，含失败的平均长度分别为 322.50/324.50 步。CSV 保留逐回合数据，JSON 另列按朝向的里程碑、配对快慢、阶段步数、回报、TCP 路径与同状态参考动作差。

三种子独立成功数均值与范围为 PPO 80/80（80–80），SAC 79.33/80（79–80）。共同成功平均变化 PPO −1.42 步（−3.88 至 +.13），SAC −6.58 步（−9.71 至 −2.35）。PPO 的三个候选保留 BC 能力，其中 seed 33 有最明显的合格 PPO 速度信号。SAC seed 22 在保留成功率、零 IK 拒绝和零碰撞下平均快 2.35 步，X/Y 峰值绝对累计明显增加但仍在经验线内。其 95 分位完成步数较 BC 高约 9 步。另两个 SAC 的速度收益仅适用于 79 个共同成功回合，伴随真实任务退化与更高 IK 代价。原始同状态动作偏移的逐回合 P95 均值各轴最大 PPO 约 .027、SAC 约 .364，说明候选执行对参考策略的偏离幅度不同。该指标不能单独定位物理失败机制。

开发与验证资格在三个 PPO 与一个 SAC 候选上得到独立支持。SAC 的两个失效集中于同一配对场景，符合场景敏感性的解释，不能据此断言是某个单轴、奖励项或算法机制所致。PPO 的在线利用率与更新频次不同，SAC 复用 replay 且熵、方差参数化不同；三种子结果是该仿真工位的有限证据。通过独立审查的是 Residual BC、全部三个 PPO 候选与 SAC seed 22，资格限于现有仿真协议。未依据 Test 重选模型，也未验证实机或纯视觉部署。

媒体在统计完成后选取：固定 BC/PPO 11/SAC 11 三方首个共同成功；BC 成功而 RL 失败的最大 IK 拒绝案例；同一固定三方在其余共同成功案例中 TCP 路径长度差最大的场景。具体种子、方向和模型见 `media-selection.json`。全程用已保存的真轨迹状态生成。研究冻结，未追加训练、测试、发布或进入 P5b。


## P5b-1 开源工程与复现

[独立阶段报告](P5B_1.md)

### Git 与产物审计

本地发布候选分支为 `codex/p5b-1-release`，从 `feat/gripper@4ff1eec5584a54a232a2b4477397b556efd224d8` 创建。核对时 `main@75106355459af2f283e21593734ca172655162a4`，O6 为 `feat/o6@1465436`。原主线的未追踪用户文档 `docs/开发共识.md` 原位保留。未合并或改写原分支。P4c 至 P5a-R3 的阶段提交依次为 `fcf5dcb`、`f429d18`、`12433e7`、`6801399`、`10b44c0` 和 `4ff1eec`。历史运行入口已在 Git 跟踪，部分放在 `outputs/`。历史训练、测试和视频数据继续置于本机忽略目录。P5a-R2、R3 的历史 SHA-256 清单检查通过。

### 工程与复现入口

`scripts/demo.py` 提供 RGB Expert、RGB + Residual BC、双相机数采、LeRobot v3 导出及读回、50 Hz 物理回放、七策略开发配对、视觉 BC 失败诊断和归档图表及 GIF。当前 RGB + Residual BC 为混合式完整抓放，抓取阶段使用 43 维仿真特权状态。PPO/SAC 独立研究任务为 Grasp & Lift。LeRobot 的第 5 个控制周期对应下一帧，按动作前时间戳对齐，保留所有 50 Hz 动作。新增可选 RGB Expert GIF 与固定位置单一划分导出，现有默认控制、物理及评分未改动。

[英文 README](../README.md) · [两级复现命令](REPRODUCTION.md) · [系统与数据接口](PLATFORM.md) · [资产与许可](ASSETS_AND_LICENSES.md) · [SHA-256 资产清单](../configs/assets.json) · [技术调试记录](P5B_1_DEBUG.md)

### 实测

| 层 | 实际运行与结果 |
| --- | --- |
| 干净基础环境 | 独立临时源码目录、全新 Python 3.12.3 venv、重新获取固定版 CR3 源码。通过 SHA-256 检查，从仓库之外执行 `python /path/to/checkout/scripts/demo.py rgb OUTPUT`，固定双朝向 2/2，生成完整 GIF。未挂载开发工作树或旧权重。 |
| 学习型完整任务 | 独立发布工作树从经过哈希校验的本地资产恢复，`demo.py learned OUTPUT` 固定双朝向 2/2，生成完整 GIF。 |
| 采集、读取与回放 | `demo.py collect RAW --episodes 2` 得到 2/2。原始控制分别 1,073、1,065 步逐步复现。LeRobot 0.7.0 读取 428 对 600×400 RGB 帧、20 维 state 和 7 维 action，10 FPS，另保存回放 GIF。六回合扩展运行 6/6，实际 Loader 读回 Train 849 对、Validation 427 对。相同收集接口记录学习型双朝向 2/2，逐步重放 1,013 与 1,009 个控制周期。LeRobot 使用已有独立环境。 |
| 冻结研究策略 | `demo.py research OUTPUT`，从历史开发面板复制的 17218 双朝向，两回合 × 七策略，14/14。使用归档内的同一策略字节。 |
| 视觉 BC | `demo.py vision-bc OUTPUT --manifest RAW`，加载 P4c 私有原始场景和模型，复现 0/8，保存真实失败轨迹及固定双朝向演示。 |
| 图表与视频 | `demo.py figures OUTPUT` 重建训练与独立评估图表。 `demo.py render-archive OUTPUT --archive RUN` 从已保存物理状态重建 17700 三方成功、17721 平面 SAC 失败、17719 TCP 路径对照。 |
| 自动化测试 | `PYTHONPATH=src MUJOCO_GL=egl python -m pytest -q tests` 64/64。加入 14 个阶段测试模块后，私有历史输入未恢复时 87 通过、4 缺件失败。补齐后 91/91。 |

本地实测产物保存在开发根目录的 `outputs/p5b-1-validation/`，该目录含 `SHA256SUMS`。完整抓放 GIF 位于 `expert/` 和 `learned/`，数采回放位于 `collection-replay.gif`，研究对照位于 `archive/`，图表位于 `figures/`。数据、权重和渲染媒体留在本机。

### 冻结研究结果与待决事项

P5a-R3 的独立 Test 保留原 560 回合。Residual BC 80/80，PPO 11/22/33 均为 80/80，SAC 11/22/33 为 79/80、80/80、79/80。PPO 33 共同成功平均少 3.88 步。SAC 22 平均少 2.35 步，P95 多 9 步，X/Y 峰值累计修正较高。SAC 11、33 在共同成功配对平均少 9.71、7.68 步，但都在 17721 平面朝向退化，分别 208、301 次 IK 拒绝。历史测试未重新执行，训练未启动。全部候选见[原始逐回合数据](../outputs/p5a-r3/episodes.csv)。

项目代码许可证尚待负责人选择。CR3 包的 BSD 变体以及权重、数据和渲染媒体的公开分发许可尚待确认。Quick Start 可自行下载明确版本的模型输入。公开用户在没有获准资产时仍可运行 RGB Expert、生成自己的演示和重建已有统计图表，精确复现私有权重轨迹需要授权副本。阶段提交以本发布候选分支 Git 记录为准。等待验收。


## P5b-2 GitHub 展示、视频与技术博客

[独立阶段报告](P5B_2.md)

从 `codex/p5b-1-release@f5eb6074ff92d059e873ca7dd47c3981c04d524e` 建立本地 `codex/p5b-2-presentation`。核对时 `feat/gripper@4ff1eec`、`main@7510635`、O6 `1465436`。原工作树用户文件 `docs/开发共识.md` SHA-256 保持 `e627f9aef6643738258b8ed637d3a6655ff7355207adc6377556b42d45d045b5`。

- 英文 README 保留干净环境 Quick Start，增加 Mermaid 架构、数据示例、复现分层及全部六个 RL 候选和 BC 数据。
- 本地保留的中英文博客初稿区分已观察结果、状态分布偏移推断和未来方案。
- [许可审计](RELEASE_AUDIT.md)记录代码许可证、SB3 衍生逻辑、CR3 模型、权重、数据及视频的负责人决策。公开可读统计和静态数值图表已纳入候选。
- [媒体索引](MEDIA.md)对应本地 `outputs/p5b-2-review/final-videos/`，内含 A 完整版 83.10 s、A 短版 14.28 s、B 三策略配对 6.60 s、C 数采回放 21.50 s、D 17721 SAC 超时 6.90 s。五个 H.264 文件逐帧解码成功。README Hero 使用媒体索引，正式视频入口待分发许可确认。
- [学习曲线](figures/learning.png)、[独立结果](figures/independent.png)、[PPO KL / Clip 与 SAC 漂移](figures/optimization.png)、[视觉 BC 离线与闭环](figures/vision-bc.png)由冻结输入生成。R3 CSV 已追踪，560 条、252,721 字节，SHA-256 与原冻结账目一致。

### 实测命令

```bash
python scripts/check_release.py
python scripts/test_release.py public-core
python scripts/test_release.py optional-research --output outputs/research-verification
python scripts/test_release.py historical-full
python scripts/presentation.py charts outputs/presentation-charts --archive /path/to/historical/outputs
/path/to/lerobot-env/bin/python scripts/presentation.py videos outputs/presentation-videos --archive /path/to/p5b-1-validation
```

| 层 | 结果 |
| --- | --- |
| Public Core | 无 Git 元数据、无私有模型的干净源码副本及新基础依赖环境，13/13。CR3 从固定上游下载件恢复。CI 工作流已入候选，未在 GitHub 执行。 |
| Optional Research | 14/14 测试，另执行历史开发双朝向 7 策略 × 2，14/14。 |
| Historical Full | 补齐授权历史数据后 92/92。 |
| 完整性 | R2/R3 原始 SHA-256 清单通过，冻结 CSV 正好覆盖七策略与 80 个场景配对。 |

逐项日志、视频帧数和本地 SHA-256 清单保存在 `outputs/p5b-2-review/`。分支提交以本地 Git 记录为准。P5a-R3 的 560 个独立回合保持冻结，没有新增训练或正式独立评估。公开推送、媒体发布和模型上传等待负责人审阅。[调试详情](P5B_2_DEBUG.md)


## P5b-3 最终发布审计与候选冻结

[独立阶段报告](P5B_3.md)

### Git 审计

最终本地候选为 `codex/p5b-3-final`，父候选为 `codex/p5b-2-presentation@e5d180149b03ebb2ae471f53a5532362920795bb`。提交 ID 以审阅包 `freeze.json` 和该分支 Git 引用为准。

| 工作树 | 审计起点 | Git 状态 |
| --- | --- | --- |
| `codex/p5b-2-presentation` | `e5d1801` | 干净 |
| `codex/p5b-1-release` | `f5eb607` | 干净 |
| `feat/gripper` | `4ff1eec` | 仅用户未追踪文件 `docs/开发共识.md` |
| `main` | `7510635` | 干净 |
| `feat/o6` | `1465436` | 干净 |

用户文档留在原工作树，SHA-256 为 `e627f9aef6643738258b8ed637d3a6655ff7355207adc6377556b42d45d045b5`。最终候选包含 P0 至 P5b-2 必需代码、文档和冻结统计，以及本轮审计。它从 `b962a1d` 与 main 分叉。隔离克隆用 `git merge-tree --write-tree` 预演，返回 0、无冲突，合成树与候选树一致。main 与 O6 工作树和引用未变动。P5a-R2/R3 历史文件与原冻结账目一致。

### Experiment numbers

| 阶段和面板 | 来源及计数方法 | 核对值 |
| --- | --- | --- |
| P3c-5 large，±25 mm，20 个种子 15200–15219，各双朝向 | 本地 `outputs/p3c-5/test/results.json` 中 `pairs` 过滤 `profile=large`，各方法累计 `is_success`。完整试验另含 fixed 和 small | 动态专家 **40/40**，固定开环 **15/40** |
| P3c-6 Test，±25 mm，20 个新种子 16200–16219，各双朝向 | 本地 `outputs/p3c-6/run/test.json` 全部 `pairs`。沿用原 341 步参考动作，按回合稳定完成计数 | 动态专家 **40/40**，固定开环 **22/40**，位置条件化 BC **40/40** |
| P4a 完整任务，fixed 19000 与 large 19001–19010，各双朝向 | [配对场景](../outputs/p4a/manifest.json)和[分组结果](../outputs/p4a/summary.json)，按方法累计标称 2 和随机 20 回合 | RGB Expert 与 RGB + Learned Grasp 各 **22/22** |
| P4c 视觉 BC，RGB Expert 示范与留出 Test | 本地 `outputs/p4c/dataset/manifest.json` 合计 38 回合与 8,187 有效片段。验证 MSE 来自 `train/training.json` 的 1,284 个 Validation 片段，闭环取 `eval/results.json` 与[追踪汇总](../outputs/p4c/summary.json) | 示范 **38/38**，最佳验证归一化 MSE **0.001862**，闭环 **0/8** |
| P5a-R2 历史与验证 | [固定协议](../outputs/p5a-r2/protocol.json)和[学习记录](../outputs/p5a-r2/summary.json)。三个种子、两算法，每组 20,000 在线步，历史面板 42/检查点、验证 20/检查点 | **120,000** 在线步，曲线来自历史与验证面板 |
| P5a-R3 独立 Test，±35 mm，17700–17739 双朝向 | [冻结场景](../outputs/p5a-r3/test_manifest.json)和[逐回合 CSV](../outputs/p5a-r3/episodes.csv)，40 XY × 2 朝向 × 7 策略。核对场景初态哈希，统计稳定成功。共同成功速度按相同 scene 上的策略步数减 BC 步数 | **560** 回合。BC **80/80**，PPO **80/80、80/80、80/80**，SAC **79/80、80/80、79/80** |

R3 原 CSV 重算的配对步数为 PPO −0.5125、+0.125、−3.875，SAC −9.70886、−2.35、−7.68354。SAC 11/33 各由 79 对成功回合计算，其余策略由 80 对计算。两次失败都发生在 17721 平面朝向，IK 拒绝为 208 和 301。独立统计保留完整失败记录，与 R2 学习曲线独立绘制。四张统计图从冻结源重建后与候选 PNG 逐字节一致。[R3 详细报告](P5A_R3.md)

P3c-5 与 P3c-6 的原始证据继续留本地，SHA-256 分别为 `414ea076024c6638712eaa87abfce16f16ba489b95335e6f2fea300ed5972bb6` 和 `b69af3e876e434f3fa96b35286a06d0252cf8a8f48d852961b767b15343a2c44`。双语博客保留两个已核实的数字和各自面板。

### 公开文件与权利清单

| 类别 | 来源及权利 | 发布状态和负责人动作 |
| --- | --- | --- |
| 自研 Python、配置、文档、脚本 | 项目自研，Git 尚无项目级授权文本 | 确认权利归属并决定代码和文档许可证 |
| 第三方源码与衍生逻辑 | SB3 2.7.1 安装包声明 MIT，候选含 PPO/SAC 的本地更新实现，依赖由包管理器安装 | 核查相关实现的衍生关系，落实必要的版权与许可声明。额外分发 wheel 或镜像需另查其依赖 |
| CR3 Xacro/Mesh 和转换模型 | Dobot ROS2 V4 固定提交，根文本 MIT，模型包 `<license>BSD</license>` 变体未定 | 上游文件本地下载。确认具体文件条款前保持模型与派生文件本地 |
| 自建两指夹爪 | 项目原创 `assets/robots/gripper/parallel.xml`，无引入外部网格 | 随自研源码一并确认权利及许可证 |
| BC/PPO/SAC 权重 | 本项目仿真中训练，冻权重哈希已记录 | 暂留本地。负责人逐项确认可分发范围及适用许可 |
| LeRobot v3 数据 | 本项目仿真动作、状态及双 RGB；LeRobot 软件安装包声明 Apache-2.0 | 格式可复现。归档数据含模型渲染，继续留本地直至授权 |
| GIF/MP4 | 原 MuJoCo 轨迹渲染及项目字幕，包括 CR3 几何 | 本地审阅。授权后用外部媒体地址接入 README Hero |
| 实验 JSON/CSV、数值 PNG | 本项目统计，R3 CSV 已追踪，无模型权重或原始 RGB | 可以作为候选统计材料，负责人批准随文档公开 |

历史与当前文件扫描覆盖候选与 main 的可达 Git 对象，按常见 token、私钥格式、字面秘密赋值、带凭证 URL、用户路径、内部地址、二进制后缀和大小检查。未发现明确凭证、私有研究数据包、下载模型、checkpoint 或视频进入候选历史。最大 blob 为 252,721 字节的 R3 CSV；二进制为四个数值 PNG。历史 `file://` 唯一命中为 CR3 上游包资源前缀。Git 作者/提交者元数据含姓名及非 noreply 邮箱，公开历史前须经负责人确认。若不授权暴露现有元数据，保留本地历史，另以经批准的快照初始化公开仓库。模式检索覆盖有限，日志不保存匹配原文。[调试记录](P5B_3_DEBUG.md)

### 文档和视频

跟踪链接检查覆盖 README、Docs、数值图、R3 CSV、R2 协议以及演示脚本。README 与平台图均由本地浏览器中的 Mermaid 解析为 `flowchart-v2`。GitHub 托管页面与 CI 尚待负责人启用后验证。README Quick Start 在仓库之外执行，命令参数与当前脚本一致。

| 成品 | 完整解码与内容 |
| --- | --- |
| A 完整 / 短版 | 1,385 / 238 帧，83.10 / 14.28 s。完整含 Expert 与 Learned 双朝向。短版为 Expert 3× |
| B 同场景比较 | 66 帧，6.60 s。17700 凸起，BC 325、PPO 11@2k 325、SAC 11@2k 312 步，三者成功 |
| C 数采回放 | 215 帧，21.50 s。两路 RGB、实际控制游标及解码时间对齐。overview 首帧黑屏，下一采样正常 |
| D 失败示例 | 69 帧，6.90 s。17721 平面，BC 和 PPO 11@2k 各 336 步成功，SAC 33@5k 400 步超时、301 次 IK 拒绝 |

五段均为 H.264，逐帧解码且 PTS 严格递增。对照场景原始初态哈希一致，结束较早的策略保持末帧，字幕步数与冻结 CSV 一致。抽检首中尾未见关键字幕裁切。P5b-2 视频 SHA-256 全部通过。视频保留在原私有媒体目录，README 使用[媒体索引](MEDIA.md)。

### 复现验收

- Public Core 使用新 Python 3.12 venv、锁定依赖、重新下载固定 CR3 源并校验模型。干净源码归档在仓库外运行，测试 **13/13**，RGB Expert 双朝向 **2/2**。
- Optional Research 恢复授权的冻结权重并校验哈希，**14/14** 测试，历史开发 panel 17218 双朝向、七策略 **14/14**，另外重建全部四张图。
- Historical Full 恢复授权的旧面板、Teacher 数据和训练日志，**92/92**。无授权私有输入时明确报缺失。
- `python scripts/check_release.py` 检查已追踪链接和冻结 CSV；`python scripts/test_release.py public-core`、`optional-research --output outputs/dev-verification`、`historical-full` 分别对应三个测试层。公开复现细节见[复现文档](REPRODUCTION.md)。本地 `outputs/p5b-3-audit/` 存日志、`freeze.json`、全候选文件清单和 SHA-256。未训练、未重跑正式独立 Test。

### 审批后的合并与公开方案

默认建议以负责人审阅后的本地 `main` 作为公开入口。保留 `feat/gripper` 和 `feat/o6` 的原开发工作树、三个候选工作树以及本地实验数据。O6 是独立历史开发线。镜像范围只选通过审查的主线，数据、权重、视频、备份保持本地，授权后可按独立条款提供外部链接。

需先确认 **项目许可证、适用第三方声明、Git 身份元数据是否公开、统计材料发布批准，以及任何外链资产的分发许可**。批准后由负责人执行，基线变化时重新预演。下列命令是给负责人的操作建议。

```bash
# 本地 main 工作树，先确认 clean 且 main 仍是已审计版本
git status --short --branch
git rev-parse main
git rev-parse codex/p5b-3-final
git merge-tree --write-tree main codex/p5b-3-final
git merge --no-ff codex/p5b-3-final
python scripts/check_release.py
# 配置获准的公开 remote 后，仅发布已审查的主线
git push --dry-run github main:main
git push github main:main
```

本轮未执行合并、远程推送或任何上传。最终候选等待审阅。

最终候选快照移除了 AI 开发约定、两份 Codex 任务指令和内部交接文档。面向读者的项目计划、复现指南、研究报告及发布审计保留。以上内部文档仍可在旧提交查看。如果公开仓库也要排除这些文档的历史版本，应在审批后从已审阅快照初始化公开仓库。本地开发历史保留。

### P5b-3 展示内容修订

README 明示冻结 PPO/SAC 开发评估与图表复建入口并嵌入研究图表。候选快照移除两份博客初稿及公开链接。数采回放使用独立 MuJoCo 状态初始化 0.00 s 的 overview，实际 1,073 个控制周期逐步匹配原状态，修复版 H.264 保持 215 帧与 10 FPS。视频和初稿保留本地。见[展示内容修订](P5B_3_MEDIA.md)。
