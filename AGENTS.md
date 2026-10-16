# Project guidance

- Goal: pi05_base → SO101 real demonstrations → expert-only fine-tuning → remote inference/local execution → honest real-robot evaluation.
- Read README.md, docs/项目总体规划.md and docs/进度看板.md first. Keep existing plans and experiment evidence; do not overwrite unrelated older VLA projects.
- Local preparation tools must run with the Python standard library. LeRobot is an external runtime; do not vendor model code or heavy weights here.
- Keep machine-specific devices in ignored configs/local.json. Never guess serial/camera IDs or claim calibration/model/CUDA checks passed without evidence.
- `plan` generates previews, not execution. Add real launch paths only in the corresponding implementation stage with environment validation.
- After meaningful work, update project_status.json, docs/进度看板.md and docs/开发日志.md using actual results. Do not turn planned tasks into completed tasks merely because files exist.
- Record new errors, recurring pitfalls and resolved project questions in docs/问题记录.md with a stable ISSUE ID, evidence, cause certainty, attempted remedies, verification and remaining work. Distinguish workarounds from root-cause fixes; update the original entry when a problem recurs.
- Keep complete episodes/groups together when splitting datasets. Training subset selection alone does not guarantee train-only normalization statistics.
- No simulated/old RoboTwin result may be reported as this project's real SO101 result. Optional MuJoCo work must not displace the real baseline or its 40 trials.
- Run appropriate offline checks via `make test` and `make check` before reporting completion. Avoid importing hardware/model packages for ordinary tool tests.
- User preference: continuously manage this project with Git. Commit each completed, coherent change after appropriate checks; stage only files belonging to that change and preserve unrelated local modifications. Report the commit and any remaining changes. Follow docs/Git版本管理.md.
- Keep local environment reports, credentials, datasets, videos and weights out of ordinary commits. Preserve reviewed environment/version summaries and experiment-to-commit references. Use only an explicitly established remote; do not invent a publishing destination.
