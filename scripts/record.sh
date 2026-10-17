#!/usr/bin/env bash
set -euo pipefail

# 先在终端执行：conda activate lerobot
# 默认试采1条；例如 NUM_EPISODES=20 bash scripts/record.sh 可改为20条。
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${TASK:-Put the red toothbrush into the left cup.}"
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-45}"
RESET_TIME_S="${RESET_TIME_S:-15}"
SESSION_ID="${SESSION_ID:-real_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_red_left_$$}"
DATASET_ROOT="$PROJECT_ROOT/data/real/$SESSION_ID"
LOG_DIR="$PROJECT_ROOT/outputs/real/$SESSION_ID"
MODE="${1:-record}"

case "$MODE" in
  record|--dry-run|--check-only) ;;
  -h|--help)
    echo "Usage: bash $0 [--dry-run|--check-only]"
    echo "Settings: TASK, NUM_EPISODES, EPISODE_TIME_S, RESET_TIME_S, SESSION_ID"
    exit 0 ;;
  *) echo "Unknown option: $MODE" >&2; exit 2 ;;
esac
[[ "$#" -le 1 ]] || { echo "Too many arguments" >&2; exit 2; }
command -v python >/dev/null || { echo "Run conda activate lerobot first (python not found)" >&2; exit 2; }
[[ "$SESSION_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid SESSION_ID" >&2; exit 2; }
for value in "$NUM_EPISODES" "$EPISODE_TIME_S"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Counts and durations must be positive integers" >&2; exit 2; }
done
[[ "$RESET_TIME_S" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "RESET_TIME_S must be a nonnegative integer" >&2; exit 2; }

# 设备身份只从本机配置读取，不把这台电脑的USB序列号写进公共脚本。
mapfile -t CONFIG < <(python - "$PROJECT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
base = json.loads((root / "configs/project.json").read_text())
local = json.loads((root / "configs/local.json").read_text())
robot = base["robot"] | local.get("robot", {})
cameras = {name: base["cameras"][name] | local.get("cameras", {}).get(name, {}) for name in ("top", "wrist")}
values = [robot["type"], robot["port"], robot["id"], robot["leader_type"], robot["leader_port"], robot["leader_id"]]
for camera in cameras.values():
    if camera.get("index_or_path") is None or camera.get("fps") != 30:
        raise SystemExit("Fill known camera paths and 30fps settings in configs/local.json first")
if any(not isinstance(value, str) or not value or "\n" in value for value in values):
    raise SystemExit("Fill known robot ports and IDs in configs/local.json first")
if values[1] == values[4]:
    raise SystemExit("Leader and follower must use different ports")
for value in values:
    print(value)
print(json.dumps(cameras, ensure_ascii=False))
for camera in cameras.values():
    print(camera["index_or_path"])
PY
)
[[ "${#CONFIG[@]}" -eq 9 ]] || { echo "Could not load local robot/camera configuration" >&2; exit 2; }

CALIBRATION_ROOT="${HF_LEROBOT_CALIBRATION:-${HF_LEROBOT_HOME:-${HF_HOME:-$HOME/.cache/huggingface}/lerobot}/calibration}"
FOLLOWER_CALIBRATION_DIR="${FOLLOWER_CALIBRATION_DIR:-$CALIBRATION_ROOT/robots/so_follower}"
LEADER_CALIBRATION_DIR="${LEADER_CALIBRATION_DIR:-$CALIBRATION_ROOT/teleoperators/so_leader}"

# 当前安装版本使用 scripts.lerobot_record；录制频率由 dataset.fps 设置。
# 关闭实时Rerun显示以绕过已发生的同步发送阻塞；top/wrist仍会写入视频。
RECORD_COMMAND=(
  python -u -m lerobot.scripts.lerobot_record
  "--robot.type=${CONFIG[0]}"
  "--robot.port=${CONFIG[1]}"
  "--robot.id=${CONFIG[2]}"
  "--robot.calibration_dir=$FOLLOWER_CALIBRATION_DIR"
  --robot.use_degrees=true
  --robot.disable_torque_on_disconnect=true
  "--robot.cameras=${CONFIG[6]}"
  "--teleop.type=${CONFIG[3]}"
  "--teleop.port=${CONFIG[4]}"
  "--teleop.id=${CONFIG[5]}"
  "--teleop.calibration_dir=$LEADER_CALIBRATION_DIR"
  --teleop.use_degrees=true
  --display_data=false
  --play_sounds=false
  "--dataset.repo_id=local/$SESSION_ID"
  "--dataset.root=$DATASET_ROOT"
  --dataset.no_stamp=true
  --dataset.push_to_hub=false
  --dataset.fps=30
  "--dataset.num_episodes=$NUM_EPISODES"
  "--dataset.episode_time_s=$EPISODE_TIME_S"
  "--dataset.reset_time_s=$RESET_TIME_S"
  --dataset.video=true
  --dataset.streaming_encoding=false
  --dataset.video_encoding_batch_size=1
  --dataset.rgb_encoder.vcodec=h264
  "--dataset.single_task=$TASK"
)

if [[ "$MODE" == "--dry-run" ]]; then
  # 仅输出完整命令JSON，不检查/打开设备，不创建数据目录。
  python - "${RECORD_COMMAND[@]}" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1:], ensure_ascii=False, indent=2))
PY
  exit 0
fi

[[ ! -e "$DATASET_ROOT" && ! -e "$LOG_DIR" ]] || { echo "Session already exists; choose a new SESSION_ID" >&2; exit 2; }
for file in "$FOLLOWER_CALIBRATION_DIR/${CONFIG[2]}.json" "$LEADER_CALIBRATION_DIR/${CONFIG[5]}.json"; do
  [[ -r "$file" ]] || { echo "Calibration file not readable: $file" >&2; exit 2; }
done
command -v fuser >/dev/null || { echo "Install/use fuser for occupancy checks first" >&2; exit 2; }
for device in "${CONFIG[1]}" "${CONFIG[4]}" "${CONFIG[7]}" "${CONFIG[8]}"; do
  [[ -c "$device" && -r "$device" && -w "$device" ]] || {
    echo "Device missing or inaccessible: $device" >&2
    echo "Check connections and current dialout group membership, then activate lerobot again." >&2
    exit 2
  }
  if fuser -s "$device" 2>/dev/null; then
    echo "Device is busy; stop the existing control/camera process: $device" >&2
    exit 2
  fi
done
python - <<'PY'
from pathlib import Path
import importlib.util

if importlib.util.find_spec("lerobot") is None:
    raise SystemExit("Activate the lerobot conda environment first")
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        args = (proc / "cmdline").read_bytes().split(b"\0")
        comm = (proc / "comm").read_text().strip()
    except OSError:
        continue
    if comm.startswith("python") and any(arg in (b"lerobot.scripts.lerobot_record", b"lerobot.scripts.lerobot_teleoperate") for arg in args):
        raise SystemExit("Another recording/teleoperation process is running")
PY
echo "Metadata checks passed; occupancy checks cover visible processes only. Close other control/preview programs."
if [[ "$MODE" == "--check-only" ]]; then
  echo "No recorder started and no device opened."
  exit 0
fi
[[ -t 0 ]] || { echo "Run in an interactive terminal so q can save and exit" >&2; exit 2; }

mkdir -p -- "$LOG_DIR"
python - "$LOG_DIR/command.local.json" "${RECORD_COMMAND[@]}" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"command": sys.argv[2:], "status": "launch_requested"}, ensure_ascii=False, indent=2) + "\n")
PY
echo "Task: $TASK"
echo "Dataset: $DATASET_ROOT"
echo "Recording starts after connection. Support the follower before q; wait for saving and torque release."
echo "The duration is checked between iterations; neither the timer nor q is an emergency stop."
set +e
"${RECORD_COMMAND[@]}" 2>&1 | tee "$LOG_DIR/record.log"
statuses=("${PIPESTATUS[@]}")
set -e
printf '{"recorder_exit":%d,"log_exit":%d}\n' "${statuses[0]}" "${statuses[1]}" > "$LOG_DIR/exit.local.json"
if [[ "${statuses[0]}" -ne 0 ]]; then
  exit "${statuses[0]}"
fi
exit "${statuses[1]}"
