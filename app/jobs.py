"""异步作业管理：线程池执行，支持进度查询、取消；取消不落半成品。"""
from __future__ import annotations

import concurrent.futures
import threading

from . import storage

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
_running = {}
_running_lock = threading.Lock()


def submit(job_id: str, kind: str, fn):
    """fn(progress_cb, cancel_cb) -> result dict。"""
    def progress_cb(p):
        storage.update_job(job_id, progress=int(p), status="running")

    def cancel_cb():
        return storage.is_cancel_requested(job_id)

    def task():
        with _running_lock:
            _running[job_id] = threading.current_thread()
        try:
            if cancel_cb():
                storage.update_job(job_id, status="cancelled", progress=0)
                return
            storage.update_job(job_id, status="running", progress=1)
            result = fn(progress_cb, cancel_cb)
            storage.update_job(job_id, status="succeeded", progress=100,
                               result=result)
        except JobCancelled:
            storage.update_job(job_id, status="cancelled")
        except Exception as e:  # noqa: BLE001
            storage.update_job(job_id, status="failed",
                               error={"type": type(e).__name__, "message": str(e)})
        finally:
            with _running_lock:
                _running.pop(job_id, None)

    fut = _executor.submit(task)
    return fut


class JobCancelled(Exception):
    pass


def cancel(job_id: str) -> bool:
    ok = storage.request_cancel(job_id)
    return ok
