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
