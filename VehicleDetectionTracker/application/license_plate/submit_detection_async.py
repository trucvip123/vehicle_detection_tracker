"""Use case for queueing a detection callable without depending on queue details."""

from typing import Any, Callable, Optional, Protocol, Tuple


class TaskQueue(Protocol):
    def submit_task(
        self,
        task_func: Callable[..., Any],
        task_args: Tuple[Any, ...],
        callback: Optional[Callable[[Any], None]] = None,
    ) -> None:
        ...


class SubmitLicensePlateDetectionUseCase:
    def execute(
        self,
        task_queue: TaskQueue,
        task_func: Callable[..., Any],
        task_args: Tuple[Any, ...],
        callback: Optional[Callable[[Any], None]] = None,
    ) -> None:
        task_queue.submit_task(task_func, task_args, callback)