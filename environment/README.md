# 环境证据

Git 中保留 `runtime-versions.json` 主要版本摘要及 `bootstrap-summary.json` 初始化验收结论。以下详细快照留在本地并被忽略；它们不是仓库克隆后自带的文件，在其他机器需重新检查生成。原始文件没有删除。

- `doctor.local.json`：当前工具环境可见的解释器、包版本、设备和GPU查询。
- `installed-packages.local.json`：元数据得到的包名/版本快照，不含源地址或凭据；不是可跨平台直接安装的依赖锁。
- `lerobot-cli-check.local.json`：训练CLI帮助/导入结果，不代表训练已执行。
- `lerobot-train-help.txt` / `lerobot-train-help.stderr.txt`：实际帮助和TorchCodec兼容警告原文。
- `video-smoke.local.json`：合成4帧H.264的PyAV读写结果，不是相机或正式数据验证。
- `bootstrap-summary.json`：本次初始化验收摘要。
- `train-preview.json`：当前项目配置生成的训练预览，缺设备/数据/统计证据，不能当作已运行记录。

包版本可读取、CLI可导入、GPU可用、模型可加载、硬件可执行是不同验收项。当前仅前两项及小范围PyAV读写通过，实际GPU/模型/硬件另测。

后续详细运行报告优先放在 `outputs/`；将必要版本与验收结论检查后归档。版本管理约定见 [Git 版本管理](../docs/Git版本管理.md)。
