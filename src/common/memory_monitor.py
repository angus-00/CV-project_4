"""GPU Memory Monitor with Emergency Brake

Monitors GPU memory usage and triggers emergency stop if threshold exceeded.
"""

import torch
import threading
import time
import sys
from typing import Optional, Callable


class MemoryMonitor:
    """GPU memory monitor with emergency brake"""
    
    def __init__(self, threshold_gb: float = 15.5, check_interval: float = 1.0):
        """
        Args:
            threshold_gb: Memory threshold in GB (default 15.5GB)
            check_interval: Check interval in seconds (default 1.0s)
        """
        self.threshold_gb = threshold_gb
        self.threshold_bytes = threshold_gb * 1024**3
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None
        self.emergency_callback = None
        
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, memory monitor disabled")
            self.enabled = False
        else:
            self.enabled = True
            self.device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            print(f"Memory Monitor initialized:")
            print(f"  Total GPU memory: {total_memory / 1024**3:.2f}GB")
            print(f"  Emergency threshold: {threshold_gb:.2f}GB")
            print(f"  Check interval: {check_interval}s")
    
    def start(self, emergency_callback: Optional[Callable] = None):
        """Start monitoring
        
        Args:
            emergency_callback: Optional callback function to call before exit
        """
        if not self.enabled:
            return
        
        if self.monitoring:
            print("Memory monitor already running")
            return
        
        self.emergency_callback = emergency_callback
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Memory monitor started")
    
    def stop(self):
        """Stop monitoring"""
        if not self.enabled or not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("Memory monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                allocated = torch.cuda.memory_allocated(self.device)
                reserved = torch.cuda.memory_reserved(self.device)
                
                # Use reserved memory as it's more accurate for OOM prediction
                current_gb = reserved / 1024**3
                
                if reserved > self.threshold_bytes:
                    self._trigger_emergency_brake(current_gb)
                    break
                
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Memory monitor error: {e}")
                break
    
    def _trigger_emergency_brake(self, current_gb: float):
        """Trigger emergency stop"""
        print("\n" + "="*70)
        print("🚨 EMERGENCY BRAKE TRIGGERED 🚨")
        print("="*70)
        print(f"GPU memory exceeded threshold!")
        print(f"  Current usage: {current_gb:.2f}GB")
        print(f"  Threshold: {self.threshold_gb:.2f}GB")
        print(f"  Stopping execution to prevent OOM crash...")
        print("="*70 + "\n")
        
        # Call emergency callback if provided
        if self.emergency_callback:
            try:
                print("Executing emergency cleanup...")
                self.emergency_callback()
            except Exception as e:
                print(f"Emergency callback error: {e}")
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Force exit
        print("Exiting program...")
        sys.exit(1)
    
    def get_memory_info(self) -> dict:
        """Get current memory usage info"""
        if not self.enabled:
            return {}
        
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        total = torch.cuda.get_device_properties(self.device).total_memory
        
        return {
            'allocated_gb': allocated / 1024**3,
            'reserved_gb': reserved / 1024**3,
            'total_gb': total / 1024**3,
            'free_gb': (total - reserved) / 1024**3,
            'usage_percent': (reserved / total) * 100
        }
    
    def print_memory_info(self):
        """Print current memory usage"""
        if not self.enabled:
            return
        
        info = self.get_memory_info()
        print(f"GPU Memory: {info['reserved_gb']:.2f}GB / {info['total_gb']:.2f}GB "
              f"({info['usage_percent']:.1f}%)")


# Global monitor instance
_global_monitor: Optional[MemoryMonitor] = None


def start_memory_monitor(threshold_gb: float = 15.5, 
                         check_interval: float = 1.0,
                         emergency_callback: Optional[Callable] = None):
    """Start global memory monitor
    
    Args:
        threshold_gb: Memory threshold in GB
        check_interval: Check interval in seconds
        emergency_callback: Optional callback before exit
    """
    global _global_monitor
    
    if _global_monitor is None:
        _global_monitor = MemoryMonitor(threshold_gb, check_interval)
    
    _global_monitor.start(emergency_callback)


def stop_memory_monitor():
    """Stop global memory monitor"""
    global _global_monitor
    
    if _global_monitor:
        _global_monitor.stop()


def get_memory_monitor() -> Optional[MemoryMonitor]:
    """Get global monitor instance"""
    return _global_monitor
