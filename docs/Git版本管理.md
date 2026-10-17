# Git 版本管理

更新：2026-10-17。项目使用独立仓库，仓库根目录为 `pi05-so101-washstand/`，默认分支 `main`，跟踪 `origin/main`。公开远程仓库为 [HansomeWang2154/pi05-so101-washstand](https://github.com/HansomeWang2154/pi05-so101-washstand)。

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

通过已建立的 `origin` fetch/checkout 对齐代码；离线场景可用 Git bundle 传送本地提交历史。不要把整个 `.venv`、模型缓存或数据目录当作代码同步。数据清单与统计文件需要按实验归档，忽略大文件不等于已经做了数据备份。

云端修改也应形成可追溯提交并带回本地。每次正式实验绑定确定的提交，工作区有改动时先说明并处理，再开始计费训练，避免只记录分支名却无法还原当时代码。

## 远程与备份状态

用户2026-10-17明确选择公开仓库。已核对账号为 `HansomeWang2154`，创建同名项目仓库并设置：

```text
origin https://github.com/HansomeWang2154/pi05-so101-washstand.git
main -> origin/main
```

首次推送保留原有14次提交，基线为 `local-revision-014`；远端 `refs/heads/main` 与该提交一致，GitHub返回可见性 `PUBLIC`、默认分支 `main`、仓库非空。同步状态文档随后单独提交并继续推送；最终提交核对证据保存在本机忽略的 `outputs/github-sync/`。

本轮使用已授权的官方 GitHub CLI，通过命令级凭据助手完成HTTPS推送，没有修改全局Git凭据助手。新终端或其他机器推送前仍需完成相应GitHub登录及Git认证；连接器能读取账号不代表本机Git已经登录，设备代码过期需重新生成，见ISSUE-021。

已跟踪文件没有真实数据、视频、权重或 `configs/local.json`；原有 `SO-ARM101开源6轴机械臂/` 资料目录保持未跟踪。公开Git仓库是代码和已审查文档的远端副本，原始数据、模型及运行日志仍需独立持久化，见 [本地云端协作](环境与本地云端协作.md)。
