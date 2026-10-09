"""Generic worker queue for callable inference tasks."""

import queue
import threading
from typing import Any, Callable, Optional, Tuple

from VehicleDetectionTracker.logging_utils import log as shared_log


class InferenceQueue:
    def __init__(self, num_workers: int = 6, log_func: Optional[Callable[[str], None]] = None):
        self.task_queue = queue.Queue()
        self.num_workers = num_workers
        self.workers = []
        self.running = True
        self._log = log_func or shared_log
        self._start_workers()
        self._log(f"[INFERENCE_QUEUE] Initialized with {num_workers} worker threads")

    def _start_workers(self) -> None:
        for index in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=f"InferenceWorker-{index}",
            )
            worker.start()
            self.workers.append(worker)
            self._log(
                f"[INFERENCE_QUEUE] Started worker thread {index}: {worker.name} (daemon=True)"
            )

    def _worker_loop(self) -> None:
        worker_name = threading.current_thread().name
        self._log(f"[INFERENCE_QUEUE] {worker_name} started")
        while self.running:
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None:
                    self._log(f"[INFERENCE_QUEUE] {worker_name} received shutdown signal")
                    self.task_queue.task_done()
                    break
                task_func, task_args, callback = task
                try:
                    self._log(f"[INFERENCE_QUEUE] {worker_name} executing task...")
                    result = task_func(*task_args)
                    self._log(
                        f"[INFERENCE_QUEUE] {worker_name} task executed, "
                        f"result={'None' if result is None else 'plate_detected'}"
                    )
                    if callback:
                        self._log(f"[INFERENCE_QUEUE] {worker_name} calling callback...")
                        callback(result)
                        self._log(f"[INFERENCE_QUEUE] {worker_name} callback completed")
                except Exception as error:
                    import traceback

                    self._log(f"[INFERENCE_QUEUE] ⚠ {worker_name} Error in worker task: {error}")
                    self._log(f"[INFERENCE_QUEUE] {worker_name} Traceback: {traceback.format_exc()}")
                    if callback:
                        try:
                            callback(None)
                        except Exception as callback_error:
                            self._log(
                                f"[INFERENCE_QUEUE] ⚠ {worker_name} Callback error: {callback_error}"
                            )
                finally:
                    self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as error:
                self._log(f"[INFERENCE_QUEUE] ⚠ {worker_name} Worker error: {error}")
        self._log(f"[INFERENCE_QUEUE] {worker_name} stopped")

    def submit_task(
        self,
        task_func: Callable[..., Any],
        task_args: Tuple[Any, ...],
        callback: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self.task_queue.put((task_func, task_args, callback))

    def wait_for_all_tasks(self, timeout: Optional[float] = None) -> bool:
        try:
            import sys

            self._log(
                f"[INFERENCE_QUEUE] Waiting for {self.task_queue.qsize()} tasks to complete "
                "(this blocks until all tasks are processed)..."
            )
            sys.stdout.flush()
            self.task_queue.join()
            self._log("[INFERENCE_QUEUE] ✓ Queue.join() completed - all tasks are done")
            return True
        except Exception as error:
            self._log(f"[INFERENCE_QUEUE] ⚠ Error waiting for tasks: {error}")
            return False

    def shutdown(self) -> None:
        self._log("[INFERENCE_QUEUE] Starting graceful shutdown...")
        self._log("[INFERENCE_QUEUE] Step 1: Waiting for all queued tasks to complete...")
        self.wait_for_all_tasks()
        self._log("[INFERENCE_QUEUE] Step 2: Signaling workers to stop...")
        self.running = False
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        self._log("[INFERENCE_QUEUE] Shutdown initiated - daemon workers will exit with main thread")
        self._log("[INFERENCE_QUEUE] ✓ Shutdown complete")