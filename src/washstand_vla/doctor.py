"""Read-only environment inventory without importing torch or opening devices."""

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def inspect_environment(root: Path, check_gpu: bool = False) -> dict:
    packages = {}
    for name in ("lerobot", "torch", "torchvision", "transformers", "mujoco", "numpy", "pyarrow", "av", "grpcio"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = {"checked": False, "available": None, "detail": "Use --gpu for an nvidia-smi probe"}
    if check_gpu:
        gpu["checked"] = True
        executable = shutil.which("nvidia-smi")
        if executable is None:
            gpu.update(available=False, detail="nvidia-smi not found in this execution environment")
        else:
            try:
                run = subprocess.run([executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=10, check=False)
                gpu.update(available=run.returncode == 0, returncode=run.returncode,
                           detail=(run.stdout or run.stderr).strip()[:2000])
            except (OSError, subprocess.TimeoutExpired) as exc:
                gpu.update(available=False, detail=str(exc))
    warnings = []
    if sys.version_info < (3, 12):
        warnings.append("Project tools work on Python>=3.10; the selected LeRobot 0.6.2 candidate needs Python>=3.12")
    if packages["lerobot"] is None:
        warnings.append("LeRobot is not installed in the current interpreter")
    if check_gpu and not gpu["available"]:
        warnings.append("GPU probe failed in this tool environment; this alone does not diagnose the physical laptop")
    usage = shutil.disk_usage(root)
    return {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "project_root": str(root.resolve()),
            "python_executable": sys.executable, "python_version": platform.python_version(),
            "platform": platform.platform(), "packages": packages,
            "commands": {k: shutil.which(k) for k in ("git", "uv", "ffmpeg", "lerobot-train", "lerobot-record")},
            "device_paths": {"serial": sorted(str(p) for p in Path('/dev').glob('ttyACM*')),
                             "video": sorted(str(p) for p in Path('/dev').glob('video*'))},
            "disk_free_gib": round(usage.free / 2**30, 2), "gpu": gpu, "warnings": warnings,
            "not_verified": ["camera capture", "serial connection", "robot calibration", "torch CUDA execution", "pi05 weights"]}
