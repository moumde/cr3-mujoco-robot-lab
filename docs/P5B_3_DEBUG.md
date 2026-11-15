# P5b-3 核查记录

- 初次在受限工作树运行 `merge-tree --write-tree` 时 Git 无法写对象。改用隔离临时克隆，分别预演 P5b-2 和最终候选。Git 对象、引用及工作树的变更仅发生在临时克隆。
- 逐回合筛选 P3c-5 的 `profile=large` 与 P3c-6 Test 的全部场景，校验种子不重叠以及每个种子的双朝向配对。按方法统计 `is_success`，得到 40/40 对 15/40 及 40/40 对 22/40。双语叙述补齐面板标识。
- README 和平台文档的 Mermaid 使用本地浏览器运行 Mermaid `parse()`，两个返回 `flowchart-v2`。受限环境中的 Chrome 首次启动因 crashpad socket 权限失败。离线浏览器解析通过，GitHub 托管页面尚无实测。
- 历史路径规则仅命中 CR3 转换器内上游资源的 `file://$(find cra_description)/` 前缀。该字符串是包内资源 URI。Git 作者和提交者的姓名与邮箱属于历史元数据，审查包记录是否存在，不复制其值。
- 视频逐帧解码，标注与冻结 CSV 对齐。C 的物理回放首个 0.00 s 样本中 overview 尚未显示场景，同时刻双 RGB 和曲线数据仍可读，随后画面正常。保留原 P5b-2 媒体及其 SHA-256。
