"""Move artifacts written by a legacy active job into the function-based layout."""
import argparse
import ctypes
import os
import shutil
import time
from pathlib import Path


MOVES = {
    Path("outputs/checkpoints/best_detection_multidomain.pth"):
        Path("outputs/detection/checkpoints/best_detection_multidomain.pth"),
    Path("outputs/logs/best_detection_multidomain_training_history.csv"):
        Path("outputs/detection/logs/best_detection_multidomain_training_history.csv"),
    Path("outputs/logs/multidomain_train.stdout.log"):
        Path("outputs/detection/logs/multidomain_train.stdout.log"),
    Path("outputs/logs/multidomain_train.stderr.log"):
        Path("outputs/detection/logs/multidomain_train.stderr.log"),
}


def pid_exists(pid: int) -> bool:
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, pid
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def finalize():
    for source, destination in MOVES.items():
        if not source.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        shutil.move(str(source), str(destination))
        print(f"moved {source} -> {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    while args.wait_pid and pid_exists(args.wait_pid):
        time.sleep(max(5, args.poll_seconds))
    time.sleep(5)
    finalize()
