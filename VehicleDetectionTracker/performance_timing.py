"""
Performance timing utilities for tracking function execution times
"""

import time
import functools
from typing import Callable, Any
from datetime import datetime


def time_function(log_func: Callable[[str], None] = print):
    """
    Decorator to measure and log function execution time.
    
    Args:
        log_func: Logging function to use (default: print)
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            func_name = func.__name__
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                log_func(f"[TIMING] {func_name}: {elapsed_ms:.2f}ms")
        
        return wrapper
    return decorator


def time_block(block_name: str, log_func: Callable[[str], None] = print):
    """
    Context manager for timing a code block.
    
    Args:
        block_name: Name of the block being timed
        log_func: Logging function to use
    
    Usage:
        with time_block("detecting vehicles", log):
            results = model.track(...)
    """
    class TimingContext:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed_ms = (time.perf_counter() - self.start) * 1000
            log_func(f"[TIMING] {block_name}: {elapsed_ms:.2f}ms")
    
    return TimingContext()


class PerformanceMonitor:
    """
    Tracks performance metrics for multiple operations.
    """
    def __init__(self):
        self.timings = {}
    
    def start_timer(self, operation_name: str):
        """Start timing an operation."""
        self.timings[operation_name] = {'start': time.perf_counter()}
    
    def end_timer(self, operation_name: str) -> float:
        """End timing and return elapsed milliseconds."""
        if operation_name not in self.timings:
            return 0
        
        elapsed = (time.perf_counter() - self.timings[operation_name]['start']) * 1000
        self.timings[operation_name]['elapsed'] = elapsed
        return elapsed
    
    def get_report(self) -> str:
        """Get formatted report of all timings."""
        report = "\n[PERFORMANCE REPORT]\n"
        for op_name, data in sorted(self.timings.items(), key=lambda x: x[1].get('elapsed', 0), reverse=True):
            elapsed = data.get('elapsed', 0)
            report += f"  {op_name}: {elapsed:.2f}ms\n"
        return report
    
    def reset(self):
        """Reset all timings."""
        self.timings = {}


# Global performance monitor instance
_perf_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    return _perf_monitor
