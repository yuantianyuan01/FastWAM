"""File-backed task queue shared by persistent LIBERO evaluation workers."""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def task_queue_lock(lock_file: Path):
    """Serialize queue updates on filesystems where flock is unavailable."""
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = lock_file.with_name(f"{lock_file.name}.lockdir")
    while True:
        try:
            os.mkdir(lock_dir)
            break
        except FileExistsError:
            time.sleep(0.05)
    try:
        yield
    finally:
        os.rmdir(lock_dir)


def write_worker_status(status_dir: Path, worker_id: str, status: str, message: str) -> None:
    status_dir.mkdir(parents=True, exist_ok=True)
    status_file = status_dir / f"worker_{worker_id}.json"
    temp_file = status_dir / f".{status_file.name}.{os.getpid()}.tmp"
    temp_file.write_text(
        json.dumps(
            {
                "worker_id": worker_id,
                "status": status,
                "message": message,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(temp_file, status_file)


def read_worker_status(status_dir: Path, worker_id: str) -> dict | None:
    status_file = status_dir / f"worker_{worker_id}.json"
    if not status_file.exists():
        return None
    return json.loads(status_file.read_text(encoding="utf-8"))


def pop_task(
    pending_file: Path,
    lock_file: Path,
    status_dir: Path,
    worker_id: str,
) -> tuple[str, int] | None:
    with task_queue_lock(lock_file):
        lines = pending_file.read_text(encoding="utf-8").splitlines()
        while lines:
            raw_task = lines.pop(0).strip()
            pending_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            if raw_task:
                suite_name, task_id = raw_task.split(",", 1)
                write_worker_status(status_dir, worker_id, "running", raw_task)
                return suite_name, int(task_id)
        write_worker_status(status_dir, worker_id, "idle", "waiting for tasks")
    return None


def requeue_task(pending_file: Path, lock_file: Path, task: tuple[str, int]) -> bool:
    raw_task = f"{task[0]},{task[1]}"
    with task_queue_lock(lock_file):
        lines = pending_file.read_text(encoding="utf-8").splitlines()
        if raw_task in [line.strip() for line in lines]:
            return False
        lines.append(raw_task)
        pending_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def pending_task_count(pending_file: Path, lock_file: Path) -> int:
    with task_queue_lock(lock_file):
        return sum(1 for line in pending_file.read_text(encoding="utf-8").splitlines() if line.strip())
