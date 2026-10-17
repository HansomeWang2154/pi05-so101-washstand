# 可选仿真接入位置

已实现MuJoCo+SO101单环境牙刷入杯原型：两自由牙刷、两自由空腔杯、top/wrist RGB、五臂关节+夹爪目标、状态机专家、独立校验、控制回放及LeRobot导出。尺寸为nominal，未测量真实装置；仿真弧度数据尚未对齐真机驱动。

SO101资产固定来源及许可证见 `washstand/assets/so101/PROVENANCE.json`。物体不使用绑定、牵引或额外执行器；训练候选需通过真实接触、释放后2秒稳定、仅控制序列回放和数据重读。

2026-10-17用户明确要求现在实施，仿真原型已启动；原计划G2之后再开始的顺序据此调整，真机阶段没有因此标记完成。当前实现与证据见 [仿真实施记录](../docs/仿真实施记录.md)。

## 运行入口

在项目根目录使用项目环境。需要MuJoCo、NumPy、SciPy、PyAV；导出另需固定版本LeRobot。普通项目CLI不会自动加载仿真依赖。

已验证的Python3.12仿真依赖版本记录在 `washstand/requirements.txt`，没有自动安装或替换原有环境。标准库模式的主工具测试仍可运行，缺少NumPy/PyAV时跳过对应可选检查。

```bash
make sim-test

# 新目录：已存在时拒绝覆盖；四种指令各一次。
MUJOCO_GL=egl .venv/bin/python -m sim.washstand.generate \
  --output data/sim/raw/my_run --attempts 4 --seed 100

# 从同一初态只重放保存的控制目标，独立重新判定。
MUJOCO_GL=egl .venv/bin/python -m sim.washstand.replay \
  data/sim/raw/my_run/attempt-0000-red_left

# 仅导出物理/回放/数据检查均合格且双视频齐全的轨迹，不上传。
HF_DATASETS_CACHE="$PWD/outputs/hf_datasets_cache" .venv/bin/python -m sim.washstand.export_lerobot \
  --raw-root data/sim/raw/my_run --output data/sim/lerobot/my_run

HF_DATASETS_CACHE="$PWD/outputs/hf_datasets_cache" .venv/bin/python -m sim.washstand.inspect_dataset \
  --dataset data/sim/lerobot/my_run --output outputs/sim/my_run_inspection
```

可用 `--attempts 20 --jitter-m 0.002`做小范围位置变化；这仍是有限工作区的原型，不代表任意位置/朝向鲁棒。`--no-images`仅作快速物理调试，不能导出为带视觉的训练候选。

录制中断后，在相同参数后追加 `--resume`。`--attempts`仍表示整个批次的总槽位数；已完成的成功或失败记录均跳过，未完成目录保留到批次内的 `interrupted/`，然后从该槽位重新录制。参数或有效配置不一致时拒绝继续；同一批次禁止并发录制。默认不加此参数时继续拒绝已有目录。

```bash
MUJOCO_GL=egl .venv/bin/python -m sim.washstand.generate \
  --output data/sim/raw/my_jitter_run --attempts 20 --seed 200 \
  --jitter-m 0.002 --resume
```

原始录制提交和后续校验源码哈希分别记录。仅修改检查器后重验，不会把原始录制提交改成新提交；跨代码修复恢复的批次可包含多个录制提交，需查看每条metadata/source_manifest。

本工具会话检测到Mesa llvmpipe软件渲染，因此默认关闭阴影/反射提速；320×240原型图像和渲染设置写入元数据。这不是4060硬件性能实测。真实两路640×480设置应在单位/相机对齐阶段一起核验。

`data/sim/` 和 `outputs/sim/` 被Git忽略，需另行保存。LeRobot导出标记 `robot_type=so101_sim`、`real_contract_verified=false`、`training_eligible=false`，表示尚不能直接拼接真实数据用于迁移训练。

- [环境与专家数据生成设计](../docs/仿真环境与专家数据生成方案.md)：原始设计；实际模块与设计差异以实施记录和代码为准。
- [可选仿真与联合训练](../docs/可选仿真与联合训练.md)：启动条件、时间范围与 sim+real 对照原则。

原型v1实际结果：20条、12,177帧、5组布局；完整物理回放/导出/新进程检查通过。分布、16/4分组清单、统计限制与演示位置见 [数据卡](../docs/数据卡_仿真原型v1.md)。
