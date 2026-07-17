"""Endpoints for reading real training/evaluation metrics from outputs."""
import csv
import json
import subprocess
import sys
import threading
import time
import re
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

DETECTION_DIR = Path("outputs/detection")
SKIN_SEGMENTATION_DIR = Path("outputs/segmentation/skin")
CHEST_SEGMENTATION_DIR = Path("outputs/segmentation/chest_xray")
PIPELINE_DIR = Path("outputs/pipeline")
LEGACY_LOGS_DIR = Path("outputs/logs")

EVALUATIONS = {
    "itobos-detection": {
        "label": "Multidomain checkpoint on iToBoS test",
        "total": 8481,
        "checkpoint": "outputs/detection/checkpoints/best_detection_multidomain.pth",
        "legacy_checkpoint": "outputs/checkpoints/best_detection_multidomain.pth",
        "args": ["--model", "detection", "--config", "configs/detection_itobos_config.yaml", "--checkpoint", "outputs/detection/checkpoints/best_detection_multidomain.pth", "--split", "test", "--output-csv", "outputs/detection/metrics/itobos_detection_test_per_image.csv"],
    },
    "multidomain-isic-detection": {
        "label": "Multidomain checkpoint on ISIC test",
        "total": 390,
        "checkpoint": "outputs/detection/checkpoints/best_detection_multidomain.pth",
        "legacy_checkpoint": "outputs/checkpoints/best_detection_multidomain.pth",
        "args": ["--model", "detection", "--config", "configs/detection_multidomain_isic_eval_config.yaml", "--checkpoint", "outputs/detection/checkpoints/best_detection_multidomain.pth", "--split", "test", "--output-csv", "outputs/detection/metrics/multidomain_on_isic_test_per_image.csv"],
    },
    "detection": {
        "label": "ISIC checkpoint on ISIC test",
        "total": 390,
        "args": ["--model", "detection", "--config", "configs/detection_config.yaml", "--checkpoint", "outputs/detection/checkpoints/best_detection.pth", "--split", "test", "--output-csv", "outputs/detection/metrics/isic2018_detection_test_per_image.csv"],
    },
    "isic-segmentation": {
        "label": "Phân đoạn ISIC",
        "total": 390,
        "args": ["--model", "segmentation", "--config", "configs/segmentation_config.yaml", "--checkpoint", "outputs/segmentation/skin/checkpoints/best_segmentation.pth", "--split", "test", "--output-csv", "outputs/segmentation/skin/metrics/isic2018_test_per_image.csv"],
    },
    "chest-xray-segmentation": {
        "label": "Phân đoạn X-ray",
        "total": 22,
        "args": ["--model", "segmentation", "--config", "configs/chest_xray_segmentation_config.yaml", "--checkpoint", "outputs/segmentation/chest_xray/checkpoints/best_chest_xray_segmentation.pth", "--split", "test", "--output-csv", "outputs/segmentation/chest_xray/metrics/chest_xray_test_per_image.csv"],
    },
    "isic-pipeline": {
        "label": "Full pipeline ISIC",
        "total": 390,
        "script": "scripts/evaluate_pipeline.py",
        "args": ["--split", "test", "--output-csv", "outputs/pipeline/metrics/isic2018_pipeline_test_per_image.csv"],
    },
    "isic-unet-all": {
        "label": "U-Net diagnostic toàn bộ ISIC",
        "total": 2594,
        "args": ["--model", "segmentation", "--config", "configs/segmentation_unet_baseline_config.yaml", "--checkpoint", "outputs/segmentation/skin/checkpoints/best_segmentation_unet_baseline.pth", "--split", "all", "--output-csv", "outputs/segmentation/skin/metrics/isic2018_unet_baseline_all_2594_per_image.csv"],
    },
}
_evaluation_lock = threading.Lock()
_evaluation_state = {"running": False, "kind": None, "message": "", "started_at": None}


def _run_evaluation(kind: str, limit: int):
    spec = EVALUATIONS[kind]
    args = list(spec["args"])
    checkpoint = _spec_checkpoint(spec)
    if checkpoint and "--checkpoint" in args:
        args[args.index("--checkpoint") + 1] = str(checkpoint)
    command = [sys.executable, spec.get("script", "scripts/evaluate.py"), *args, "--max-images", str(limit)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, cwd=Path.cwd())
        message = result.stdout.strip() if result.returncode == 0 else (result.stderr.strip() or result.stdout.strip())
        with _evaluation_lock:
            _evaluation_state.update(running=False, success=result.returncode == 0, message=message, finished_at=int(time.time() * 1000))
    except Exception as exc:
        with _evaluation_lock:
            _evaluation_state.update(running=False, success=False, message=str(exc), finished_at=int(time.time() * 1000))


def _read_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    # Allow the UI to distinguish evaluation runs with identical metric values.
    data["updated_at"] = path.stat().st_mtime_ns // 1_000_000
    return data


def _read_json_for_dataset(path: Path, expected_dataset: str):
    """Reject stale or mislabelled artifacts from another evaluation domain."""
    data = _read_json(path)
    if data is None or data.get("dataset") != expected_dataset:
        return None
    return data


def _read_history(*paths):
    """Read the first CSV that exists AND contains at least one data row.

    This prevents a header-only file (e.g. from a failed/interrupted training
    run) from shadowing a legacy file that has real data.
    """
    best = None
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        with open(p, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        result = {
            "path": str(p),
            "epochs": len(rows),
            "last": rows[-1] if rows else None,
            "rows": rows,
        }
        if rows:
            return result
        # Remember the first header-only file as fallback
        if best is None:
            best = result
    return best


def _first_existing(*paths):
    return next((Path(path) for path in paths if Path(path).exists()), Path(paths[0]))


def _spec_checkpoint(spec):
    if not spec.get("checkpoint"):
        return None
    return _first_existing(spec["checkpoint"], spec.get("legacy_checkpoint", spec["checkpoint"]))


def _multidomain_training_status():
    log_path = _first_existing(
        DETECTION_DIR / "logs/multidomain_train.stderr.log",
        LEGACY_LOGS_DIR / "multidomain_train.stderr.log",
    )
    checkpoint = _first_existing(
        DETECTION_DIR / "checkpoints/best_detection_multidomain.pth",
        "outputs/checkpoints/best_detection_multidomain.pth",
    )
    progress = None
    failed = False
    log_age_seconds = None
    if log_path.exists():
        log_age_seconds = max(0, time.time() - log_path.stat().st_mtime)
        with open(log_path, "rb") as handle:
            handle.seek(max(0, log_path.stat().st_size - 65536))
            tail = handle.read().decode("utf-8", errors="replace")
        matches = re.findall(r"Epoch\s+\d+/\d+:[^\r\n]*", tail)
        progress = matches[-1].strip() if matches else None
        failed = "Traceback (most recent call last)" in tail
    running = bool(log_path.exists() and log_age_seconds < 120 and not failed)
    return {
        "label": "Detection đa miền Clinical + ISIC",
        "status": "running" if running else "failed" if failed else "ready" if checkpoint.exists() else "idle",
        "progress": progress,
        "checkpoint_ready": checkpoint.exists(),
        "checkpoint": str(checkpoint),
        "config": "configs/detection_multidomain_config.yaml",
        "command": "python scripts/train.py --config configs/detection_multidomain_config.yaml",
        "train_images": 10288,
        "clinical_test_images": 8481,
        "confidence_threshold": 0.8,
        "log": str(log_path),
    }


@router.get("/metrics")
async def metrics():
    """Return metrics generated by scripts/evaluate.py and training history CSVs."""
    return {
        "summaries": {
            "chest_xray_segmentation": _read_json(
                CHEST_SEGMENTATION_DIR / "metrics/chest_xray_test_per_image.summary.json"
            ),
            "isic2018_segmentation": _read_json(
                SKIN_SEGMENTATION_DIR / "metrics/isic2018_test_per_image.summary.json"
            ),
            "isic2018_detection": _read_json_for_dataset(
                DETECTION_DIR / "metrics/isic2018_detection_test_per_image.summary.json", "isic2018"
            ),
            "multidomain_isic_detection": _read_json_for_dataset(
                DETECTION_DIR / "metrics/multidomain_on_isic_test_per_image.summary.json", "isic2018"
            ),
            "itobos_detection": _read_json_for_dataset(
                DETECTION_DIR / "metrics/itobos_detection_test_per_image.summary.json", "itobos_clinical"
            ),
            "isic2018_pipeline": _read_json(
                PIPELINE_DIR / "metrics/isic2018_pipeline_test_per_image.summary.json"
            ),
            "isic2018_unet_baseline": _read_json(
                SKIN_SEGMENTATION_DIR / "metrics/isic2018_unet_baseline_test_per_image.summary.json"
            ),
            "isic2018_unet_all_diagnostic": _read_json(
                SKIN_SEGMENTATION_DIR / "metrics/isic2018_unet_baseline_all_2594_per_image.summary.json"
            ),
        },
        "training_history": {
            "detection": _read_history(
                DETECTION_DIR / "logs/best_detection_training_history.csv"
            ),
            "detection_multidomain": _read_history(
                DETECTION_DIR / "logs/best_detection_multidomain_training_history.csv",
                LEGACY_LOGS_DIR / "best_detection_multidomain_training_history.csv",
            ),
            "isic2018_segmentation": _read_history(
                SKIN_SEGMENTATION_DIR / "logs/best_segmentation_training_history.csv"
            ),
            "chest_xray_segmentation": _read_history(
                CHEST_SEGMENTATION_DIR / "logs/best_chest_xray_segmentation_training_history.csv"
            ),
        },
        "training_jobs": {
            "detection_multidomain": _multidomain_training_status(),
        },
    }


@router.get("/metrics/evaluation-status")
async def evaluation_status():
    with _evaluation_lock:
        return {**_evaluation_state, "options": {
            key: {
                "label": value["label"],
                "total": value["total"],
                "ready": not value.get("checkpoint") or _spec_checkpoint(value).exists(),
            }
            for key, value in EVALUATIONS.items()
        }}


@router.post("/metrics/evaluate/{kind}", status_code=202)
async def start_evaluation(kind: str, limit: int | None = Query(default=None, ge=1)):
    if kind not in EVALUATIONS:
        raise HTTPException(status_code=404, detail="Loại đánh giá không hợp lệ")
    spec = EVALUATIONS[kind]
    if spec.get("checkpoint") and not _spec_checkpoint(spec).exists():
        raise HTTPException(
            status_code=409,
            detail="Checkpoint đa miền chưa sẵn sàng; hãy đợi train hoàn tất epoch đầu tiên.",
        )
    image_limit = spec["total"] if limit is None else min(limit, spec["total"])
    with _evaluation_lock:
        if _evaluation_state["running"]:
            raise HTTPException(status_code=409, detail="Một lượt đánh giá khác đang chạy")
        _evaluation_state.update(
            running=True,
            kind=kind,
            label=spec["label"],
            limit=image_limit,
            success=None,
            message="",
            started_at=int(time.time() * 1000),
            finished_at=None,
        )
    threading.Thread(target=_run_evaluation, args=(kind, image_limit), daemon=True).start()
    return {**_evaluation_state}
