# P5b-3 展示内容修订

README 展开即可看到训练学习曲线与 R3 独立测试结果，并可展开 PPO KL / Clip Fraction 和 SAC 漂移图。RL 部分给出复建图表和加载七个冻结策略的命令，历史训练配置链接至 R2 协议。完整抓放 MP4 仍在本机审阅包，公开内嵌需要媒体分发授权。

博客初稿从公开候选快照和 README 移除。旧本地提交保留两份初稿。公开时应使用经批准的干净快照，使博客和内部开发文档的旧版本不随历史发布。

数采视频在 0.00 s 的 overview 黑屏来自 MuJoCo 积分状态恢复后尚未初始化的渲染字段。回放首帧改用独立状态调用 `mj_forward`，实际执行动作的状态继续逐控制周期核对。原始日志和原 P5b-2 视频保留。

```bash
python scripts/demo.py replay /path/to/raw --episode episode-000 --gif outputs/replay-fixed.gif
```

修订的本地 GIF 与 MP4 位于 `outputs/p5b-3-audit/replay-fix/`。1,073 个原始控制动作的积分状态逐步匹配，视频 215 帧以 10 FPS 完整解码，首帧时间保留 0.00 s，overview 可见。
