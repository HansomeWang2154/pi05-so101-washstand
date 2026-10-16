# 部署接入位置

计划复用LeRobot async_inference.policy_server与robot_client。当前项目仅提供 `washstand plan server` 和 `washstand plan client` 的参数预览，尚未验证本项目的远程预测或真机执行。

G2将核验权重/处理器加载、top/wrist映射、六维动作语义和端到端延迟；本地负责固定频率执行、停止、限幅和断网处理。具体设备与单位确认后再建立实际启动脚本，不创建伪控制器。
