# LeRobot退出清理补丁

`0001-best-effort-so101-shutdown.patch`针对LeRobot提交`d6039b8376a3ee099a6f0f64c1bb8599ca743ae3`（本机0.6.2），修复ISSUE-019的清理短路，并增加原始电机写入错误诊断。变更仅包含总线退出、SOFollower断开、teleoperate连接/退出保护及一个标准库清理函数；不包含模型代码或权重。

**本机已应用，不必重复执行。** 原有`policies/act/modeling_act.py`改动保持。下一次使用conda `lerobot`环境运行原遥操作命令会加载此补丁。硬件验证仍待用户现场完成，见[调试报告](../../docs/SO101退出锁力与异常清理修复_2026-10-17.md)。

在其他匹配的运行时复现时，先核对版本、工作区修改并结束共享控制/采集进程，再做检查；检查通过后才应用，不能覆盖不匹配源码：

```bash
git -C /path/to/lerobot apply --check /path/to/pi05-so101-washstand/patches/lerobot/0001-best-effort-so101-shutdown.patch
git -C /path/to/lerobot apply /path/to/pi05-so101-washstand/patches/lerobot/0001-best-effort-so101-shutdown.patch
```

用Python 3.12+运行离线故障注入，读取给定运行时源码，不导入硬件模块、不打开设备：

```bash
python /path/to/pi05-so101-washstand/patches/lerobot/verify_shutdown.py /path/to/lerobot
```

16项测试涵盖单/多电机故障、清缓冲/关串口失败、相机故障和残留线程、部分连接、Ctrl+C及清理中中断、主错误保留、显示器失败，以及原始错误码/真实重试次数。此验证独立于普通项目测试，需要显式指定运行时；它不能证明真实卸力或native abort已修复。

保留补丁便于外部运行时版本追踪；不要以`git reset`或替换整个外部目录的方式清理用户已有ACT改动。后续升级LeRobot时重新审查及测试，不能只依据包版本号认定补丁仍有效。录制入口的顶层收尾另需验收。

## 录制入口补充：0002

随后为用户单条red_left试采准备了`0002-record-independent-cleanup.patch`，本机也已应用。它依赖0001的`run_cleanup`，将record的构造/连接纳入异常保护，独立处理硬件与数据收尾，零帧dataset也finalize，清理失败不上传。现有模型、动作和编码配置不变。

其他匹配运行时先应用0001，再检查并应用0002：

```bash
git -C /path/to/lerobot apply --check /path/to/pi05-so101-washstand/patches/lerobot/0002-record-independent-cleanup.patch
git -C /path/to/lerobot apply /path/to/pi05-so101-washstand/patches/lerobot/0002-record-independent-cleanup.patch
python /path/to/pi05-so101-washstand/patches/lerobot/verify_record_cleanup.py /path/to/lerobot
```

11项录制故障注入在应用前、应用后和补丁重建副本中通过，包含部分连接/零帧收尾、单条正常保存、finalize失败仍卸力、原错及异常上下文保留、显示器/通知失败、失败不上传。测试是虚拟设备和数据集，不代表真实数据或实际卸力。管理器首次编码仍先于外层硬件清理，结束录制前需支撑从臂。完整操作见[单条试采](../../docs/单条红牙刷左杯试采.md)。
