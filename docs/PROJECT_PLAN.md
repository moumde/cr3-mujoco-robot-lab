# CR3 MuJoCo 工业零件定向操作与 RL：项目方案 v0.1

**性质**：方案与验收标准，尚未执行。**目标机器**：Ubuntu 24.04 + RTX 5060 8 GB；计算方案以单机为基础，低维 PPO/SAC 可优先 CPU 训练。

## 1. 定位与非目标

**目标**：在 MuJoCo 创建可验证的 CR3 + 两指夹爪仿真；完成凸起/平面端面识别下的定向抓放；对 PPO/SAC 在随机位置的任务泛化能力和样本效率进行可复现的比较；独立探索 CR3 + O6 物理建模。

**不作为第一版承诺**：高保真数字孪生、真实机器人部署、VLA 大模型训练、RL 优于 IK、完整 O6 灵巧操作 RL、通用多机器人训练框架、大型 GUI。

## 2. 任务的准确几何定义

零件使用可公开发布的简化参数化几何体：横放圆柱，局部轴向一端带凸起，另一端平整。**以相机朝向为参考定义二分类** `protrusion_facing_camera` / `flat_facing_camera`，不要与之前讨论过的左右朝向标签混用。

目标：不论初始端面朝向，抓起后放置到目标区域，让凸起端面法向量指向机器人一侧。物体位置（x,y）可随机，但第一版物体始终以规定的水平姿态摆放，避免把任意 6D 姿态估计作为前置条件。

检验：物体中心位置误差、端面法向量与目标方向夹角、稳定落台、抓取/抬升事件与物理碰撞；将最终成功定义为上述条件的合取，阈值写入配置并公开。

**关键可行性关卡**：先用两指夹爪验证横放圆柱是否可侧夹、抬升、绕桌面法线旋转约 180°（必要时）再放置；若夹爪/手腕可达性、碰撞或端面结构不允许一次抓取完成，需调整抓取几何或明确引入 regrasp，不得靠假设直接跳过。

## 3. 双线并行的依赖顺序

先建立**公共 CR3 核心契约 P0**（模型、关节命名/顺序、法兰/TCP、坐标系、6D rad 关节状态与位置控制、固定仿真 dt 和控制频率）。在 P0 冻结之前，O6 Agent 只做模型审计/离线实验，不修改公共 CR3 模型。

### 主线 A：CR3 + 两指夹爪

| 阶段 | 核心工作 | 验收与输出 |
| --- | --- | --- |
| P0 | 官方 CR3 URDF/Mesh → MuJoCo；关节、TCP、执行器、夹爪；基本单元测试 | Headless 加载正常；关节限位/方向正确；运动学 FK 与独立基准（如 URDF FK）在多组关节角下对齐；基础运动可录屏 |
| P1a | 简化电机几何、桌面、目标；Oracle pose；专家抓取阶段机 + IK/规划 | 固定位置两种端面朝向完整抓放；记录位置和朝向误差、成功失败原因 |
| P1b | Oracle pose 下的随机物体位置，每一回合重新规划 | 固定种子、多位置的成功率报告；必须证明 IK 并非仅适用固定位置 |
| P2 | 独立 RGB 识别凸起/平面，单 RGB+已知桌面平面或 RGB-D 估计位置；相机标定 | 正反分类准确率、XY 定位误差、Vision+IK 完整成功率；与 Oracle+IK 分开汇报 |
| P3 | Gymnasium Reach + Grasp/Lift，低维状态 PPO/SAC | `reset/step` 契约、奖励单测、训练/测试种子分离、训练/评估脚本、实测曲线 |
| P4 | 随机位置定向操作 RL 实验（风险较高，可分阶段完成） | 从零 PPO/SAC 与 Oracle/视觉传统规划在**相同的任务难度与输入条件**下评估；不承诺 RL 优于 IK |
| P5 可选 | IK + Residual PPO/SAC，添加定位噪声/摩擦/质量扰动 | 与固定规划、传统反馈修正、残差 RL 三者比较；明确残差施加坐标系、幅度限制 |

P3 最好先用真实的训练记录证明 Reach 能收敛，再增加接触复杂度；P4 不应阻塞一个包含 P0–P3 的可展示仓库发布。

### 支线 B：CR3 + O6

| 阶段 | 核心工作 | 验收与输出 |
| --- | --- | --- |
| O0 | 审计官方 O6 URDF、linker-sim 等资源；核对右手/版本/许可证 | 资源清单，主动/从动关节及 mesh、碰撞、执行器覆盖情况，开源再分发决定 |
| O1 | 在稳定的 CR3 法兰接口上拼接 O6 | 结构正确，可加载，法兰/指尖坐标系正确，静态自碰撞可解释 |
| O2 | O6 六维控制到仿真关节的映射、联动和限位 | 单指逐关节扫掠、关节方向和范围记录，控制接口与真机的**定义**对应（不宣称动态一致） |
| O3 | 基础接触与抓取验证 | 可复现的接触日志；记录穿透、滑落、抖动等问题；若不能稳定抓取，如实说明 |
| O4 可选 | 与主线同一目标零件的操作测试 | 在 O3 通过后接入，不能为它修改主线的固定策略假设 |

## 4. 软件架构（仅是目标，不要一次性生成全部空文件）

```text
cr3-mujoco-lab/
├── README.md
├── pyproject.toml
├── docs/
│   ├── PROJECT_PLAN.md
│   ├── gripper_status.md
│   └── o6_status.md
├── assets/
│   ├── robots/
│   │   ├── cr3/                 # 许可核实后加入；否则记录下载脚本
│   │   ├── gripper/
│   │   └── o6/
│   └── objects/
│       └── motor_part.xml
├── configs/
├── src/cr3_lab/
│   ├── core/                   # 机器人状态、坐标约定、公共模型加载
│   ├── gripper/                # 两指夹爪执行器/控制
│   ├── o6/                     # O6 支线专属
│   ├── tasks/                  # 定向抓放环境与成功判定
│   ├── planning/               # grasp planner、IK、轨迹状态机
│   ├── perception/             # RGB 分类与位置估计
│   ├── rl/                     # Gymnasium + PPO/SAC 实验
│   └── evaluation/             # 结果、视频、统计
├── scripts/
├── tests/
└── outputs/                    # gitignore
```

### 公共接口约定

- `JointState`: 6D CR3 关节角 rad + 速度 rad/s，关节顺序固定。
- `Pose`: world/base/object/camera 等 frame 明确；MuJoCo 原生 quat 使用 `wxyz`，向其他库交换要显式转换。
- `RobotCommand`: 控制类型（关节目标/增量/TCP 目标）、单位与控制频率显式写入，不暗中混合不同控制模式。
- `EndEffector`: 用 `open/close` 或低维协同命令来抽象抓取，不假定 O6 与两指夹爪具有相同底层自由度。
- `Env`: Gymnasium `reset(seed)` 返回 observation/info，`step(action)` 返回 observation/reward/terminated/truncated/info。
- 物理计算固定 `model.opt.timestep`，策略控制频率独立，报告真实环境步数与 wall-clock 时间。
- 观测模式 `oracle` / `vision` 必须分开命名，禁止不显式披露的 ground-truth 泄漏到视觉实验。

## 5. RL 研究设计

**基础策略**：低维状态 MLP PPO/SAC，从零在线交互；不需要 IK 数据集。动作可从 `ΔTCP xyz + gripper continuous` 起步，严格定义运动学 IK、动作范围与限速。抓取前可以先研究 Reach，再研究 Grasp/Lift，最后才考虑完整定向放置。

**残差策略**：独立于上述训练，名义动作由 IK/专家实时提供，PPO/SAC 学习有限幅度的 ΔTCP 或 Δgrip。需要与传统反馈控制比较，不把“RL 必须胜过 IK”作为成功标准。

**数据划分**：随机初始 XY 训练/同分布测试/偏移分布测试；不同随机种子；物体朝向类别均衡；测试时不能把训练脚本当作评估脚本。

**指标**：完整任务成功率、正反识别精度、位置误差、方向角误差、抓取/抬升成功率、样本效率、环境交互步数、实际训练时间、接触异常。训练曲线必须是运行获得，失败结果同样保留。

**框架**：初期 SB3 作为容易调试的 Gymnasium PPO/SAC 基线。保留 RLinf 调研文档与未来适配点；未经独立跑通 RLinf 现有 MuJoCo 示例前不将其设为主线硬依赖。RTX 5060 8GB 不承担完整 VLA 在线 RL 训练。

## 6. 开源与求职作品展示

优先完成：运行一条命令展示 CR3 正反定向抓放；一条命令评估多场景成功率；一条命令训练并评估 Reach PPO/SAC；公开机器人模型适配/FK 验证报告；录制视频/动图；写明模型来源与许可证。

- 公开自建的简单 motor-part 几何，不上传任职公司原有生产零件 CAD、训练数据或权重。
- 用户能在没有真实 CR3/O6、没有大 GPU 的机器上运行 Oracle/规划基线。
- README 中明确：哪些阶段已实现、哪些为计划、哪些经过统计测试。
- 用客观可复现报告代替声称“工业级数字孪生”或“RL 必然优于传统算法”。

## 7. 两个 Codex 的集成门槛

1. 公共 P0 首先完成并集成到 `main`；O6 Agent 在此期间可以只做 O0 独立资源审计。
2. 单个 Agent 只修改自己的路径；公共契约变更先提出，等集成者合并。
3. 每次阶段结束提交 Commit，并写 `docs/gripper_status.md` 或 `docs/o6_status.md`；测试必须可复现。
4. 集成者只在两个 Agent 停止编辑相关文件后从 `main` 合并；运行测试，确认通过后再进入下一阶段。
5. 两个 Worktree 的虚拟环境和 `outputs/` 独立。只共享 Git 追踪的源码、配置、文档、轻量测试资产。不要让两个高负载训练进程同时争抢 GPU/内存。

## 8. 第一轮官方参考资源（执行前仍需核对具体文件与许可证）

- Dobot ROS2 V4 / CR3 描述模型：`https://github.com/Dobot-Arm/DOBOT_6Axis_ROS2_V4`（`cra_description`、`cr3_moveit`）；可仅提取 URDF/Mesh，不必安装 ROS2 Humble。
- LinkerHand URDF / O6：`https://github.com/linker-bot/linkerhand-urdf`（O6 左右手结构、官方 Apache-2.0 仓库许可；检查实际子文件）。
- LinkerBot MuJoCo 仿真：`https://github.com/linker-bot/linker-sim`（检查 O6 是否有可以直接复用的模型、控制器、许可证）。
- Git worktree 官方命令说明：`https://git-scm.com/docs/git-worktree`。
- Codex Worktrees 说明：`https://learn.chatgpt.com/docs/environments/git-worktrees`。
