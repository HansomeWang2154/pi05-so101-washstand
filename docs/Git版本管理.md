# Git 版本管理

更新：2026-10-16。项目使用独立仓库，仓库根目录为 `pi05-so101-washstand/`，默认分支 `main`。当前只建立本地版本历史，未配置远程仓库。

## 跟踪范围

| 内容 | 管理方式 |
|---|---|
| 项目代码、测试、构建入口 | Git 跟踪 |
| 公共配置、设备配置示例、数据清单示例 | Git 跟踪 |
| 规划、进度、问题、实验模板 | Git 跟踪，与实现同时更新 |
| 环境主要版本、验收摘要 | 检查后归档到 `environment/` 并跟踪 |
| 本机 `configs/local.json`、凭据、虚拟环境 | 本地保留，Git 忽略 |
| 本机检查详情、帮助输出与命令预览 | 本地保留，Git 忽略；摘要另存 |
| 视频、数据集、模型权重、训练输出 | 独立存储；用版本、清单与哈希关联提交 |
| 外部 LeRobot 源码 | 在其自己的仓库管理，本项目记录对应提交 |

详细忽略规则见 [`.gitignore`](../.gitignore)。忽略本机快照不会删除文件；因此本地问题记录中的原始证据仍可查看，其他机器需重新生成。共享环境信息见 [环境说明](../environment/README.md) 与 [版本摘要](../environment/runtime-versions.json)。版本摘要是已观测信息，不是完整依赖锁或 GPU 可用证明。

## 日常提交约定

完成一个可独立检查的修改，就进行一次提交；代码、相应配置、文档和必要测试一起提交。提交前查看改动和暂存内容，仅纳入当前任务涉及的文件。

```bash
git status --short
git diff
make test
make check
git add <本次修改的文件>
git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "feat: describe the completed change"
git status --short
```

占位符需要替换为实际文件和说明。`make check` 的结构合法与设备/数据是否填齐是两件事；日志继续保留实际未完成项。

提交标题使用 `feat:`、`fix:`、`docs:`、`test:`、`chore:` 等前缀，描述具体结果，例如 `docs: define simulation expert validation`。避免把若干互不相关的阶段压在一次提交中，也不只写“更新代码”。

`main` 保留通过当前检查、可解释的节点。较大功能可从 main 建立短期分支，例如 `feat/sim-scene`、`feat/expert-recorder`；完成检查后合入。小型独立文档修改可直接提交 main。修复已有问题使用新提交，保留原因和历史。

每轮结束报告实际提交号、检查结果和仍未完成的内容。若存在本轮之外的修改，保留它们，并说明工作区状态；不为了显示干净而提交或清除无关改动。

## 本地与云端如何对齐

训练前先提交本轮代码，记录：

```text
project_commit / project_branch / working_tree_dirty
lerobot_commit / runtime_versions
base_model_revision
dataset_version / split_manifest / statistics_source
effective_config / seed / output_location
```

云端使用相同项目提交和 LeRobot 提交，再同步对应的数据版本与有效配置。本机 `local.json` 不进入代码历史；实验配置导出时只保存复现需要的设置，凭据独立管理。

远程仓库建立后，通过 fetch/checkout 对齐代码；在此之前可用 Git bundle 传送本地提交历史。不要把整个 `.venv`、模型缓存或数据目录当作代码同步。数据清单与统计文件需要按实验归档，忽略大文件不等于已经做了数据备份。

云端修改也应形成可追溯提交并带回本地。每次正式实验绑定确定的提交，工作区有改动时先说明并处理，再开始计费训练，避免只记录分支名却无法还原当时代码。

## 远程与备份状态

当前 `git remote -v` 为空。本地提交可回溯代码，但仍只保存在本机；远程地址、账户与可见性确定后再接入并验证推送。数据、模型和运行日志的持久化安排见 [本地云端协作](环境与本地云端协作.md)。
