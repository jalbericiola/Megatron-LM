# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SOL (Speed of Light) Estimator integration for Megatron-RL.

This module provides the glue code to integrate the sol_estimator module
with Megatron-RL training. It provides:
    - initialize_sol(): Set up SOL tracking on the model
    - sol_nvtx_range(): Context manager for phase tracking with NVTX + Megatron timers
    - log_training_sol(): Log SOL metrics to TensorBoard/WandB
    - clear_sol_captures(): Clear captured data between iterations
"""

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Add sol_estimator to path if needed
_sol_estimator_path = Path(__file__).parent.parent.parent.parent.parent / "sol_estimator"
if _sol_estimator_path.exists() and str(_sol_estimator_path.parent) not in sys.path:
    sys.path.insert(0, str(_sol_estimator_path.parent))

try:
    from sol_estimator import (
        LayerSOLHooks,
        PhaseTimer,
        RooflineAnalyzer,
        get_current_device_spec,
        CUDAGraphTracker,
        OptimizerTracker,
    )
    SOL_AVAILABLE = True
except ImportError as e:
    SOL_AVAILABLE = False
    _import_error = str(e)

from megatron.training.utils import get_nvtx_range, print_rank_0
from megatron.training.global_vars import get_timers

logger = logging.getLogger(__name__)


# Global state for SOL tracking
class _SOLState:
    """Global state container for SOL tracking."""
    enabled: bool = False
    initialized: bool = False
    layer_hooks: Optional[Any] = None  # LayerSOLHooks
    phase_timer: Optional[Any] = None  # PhaseTimer
    cuda_graph_tracker: Optional[Any] = None  # CUDAGraphTracker
    optimizer_tracker: Optional[Any] = None  # OptimizerTracker
    roofline: Optional[Any] = None  # RooflineAnalyzer
    device_spec: Optional[Any] = None
    report_interval: int = 100
    current_iteration: int = 0
    use_megatron_timers: bool = True  # Whether to also use Megatron timers
    timer_log_level: int = 1  # Log level for Megatron timers
    # Track metrics per iteration
    iteration_metrics: Dict[str, Any] = {}
    
    @classmethod
    def reset(cls):
        """Reset all state."""
        cls.enabled = False
        cls.initialized = False
        if cls.layer_hooks is not None:
            try:
                cls.layer_hooks.remove()
            except Exception:
                pass
        cls.layer_hooks = None
        cls.phase_timer = None
        cls.cuda_graph_tracker = None
        cls.optimizer_tracker = None
        cls.roofline = None
        cls.device_spec = None
        cls.iteration_metrics = {}


_state = _SOLState()


def is_sol_enabled() -> bool:
    """Check if SOL tracking is enabled."""
    return _state.enabled and _state.initialized


def initialize_sol(model: torch.nn.Module, args: Any) -> bool:
    """Initialize SOL tracking on the model.
    
    Args:
        model: The PyTorch model to track (can be wrapped in list)
        args: Megatron args containing SOL configuration
        
    Returns:
        True if initialization succeeded, False otherwise
    """
    if not SOL_AVAILABLE:
        if getattr(args, 'rl_enable_sol_tracking', False):
            print_rank_0(f"[SOL] Warning: sol_estimator not available: {_import_error}")
        return False
    
    if not getattr(args, 'rl_enable_sol_tracking', False):
        return False
    
    if _state.initialized:
        # Already initialized, just return
        return True
    
    try:
        # Unwrap model if needed
        if isinstance(model, (list, tuple)):
            model = model[0]
        if hasattr(model, 'module'):
            model = model.module
        if hasattr(model, 'module'):  # Double wrapped (e.g., DDP)
            model = model.module
            
        # Get device spec
        _state.device_spec = get_current_device_spec()
        
        # Initialize roofline analyzer
        _state.roofline = RooflineAnalyzer(device_spec=_state.device_spec)
        
        # Initialize layer hooks with timing enabled
        _state.layer_hooks = LayerSOLHooks(
            timing=True,
            track_backward=getattr(args, 'rl_sol_track_backward', True),
        )
        _state.layer_hooks.register(model)
        
        # Initialize phase timer
        _state.phase_timer = PhaseTimer()
        
        # Initialize CUDA graph tracker
        _state.cuda_graph_tracker = CUDAGraphTracker()
        
        # Initialize optimizer tracker
        _state.optimizer_tracker = OptimizerTracker()
        
        # Store config
        _state.report_interval = getattr(args, 'rl_sol_report_interval', 100)
        _state.use_megatron_timers = getattr(args, 'rl_sol_use_megatron_timers', True)
        _state.timer_log_level = getattr(args, 'rl_sol_timer_log_level', 1)
        _state.enabled = True
        _state.initialized = True
        
        print_rank_0(f"[SOL] Initialized SOL tracking on device: {_state.device_spec.name if _state.device_spec else 'unknown'}")
        print_rank_0(f"[SOL] Megatron timer integration: {'enabled' if _state.use_megatron_timers else 'disabled'}")
        return True
        
    except Exception as e:
        logger.warning(f"[SOL] Failed to initialize: {e}")
        _state.reset()
        return False


def cleanup_sol():
    """Clean up SOL tracking resources."""
    _state.reset()
    print_rank_0("[SOL] Cleaned up SOL tracking")


@contextmanager
def sol_nvtx_range(name: str, log_level: int = 1):
    """Context manager that combines NVTX range with SOL phase tracking and Megatron timers.
    
    Args:
        name: Name of the phase/range
        log_level: Log level for Megatron timer (0=always, 1=default, 2=verbose)
        
    Yields:
        None
        
    Example:
        with sol_nvtx_range("rl/train/forward"):
            logits = model(tokens)
    """
    nvtx_range = get_nvtx_range()
    timers = None
    use_timer = False
    
    # Try to get Megatron timers
    try:
        timers = get_timers()
        # Use timers if SOL is enabled with Megatron timer integration, OR always for basic NVTX+timer
        use_timer = timers is not None
    except Exception:
        pass
    
    # Start SOL phase tracking if enabled
    if is_sol_enabled() and _state.phase_timer is not None:
        _state.phase_timer.push_phase(name)
        
        # Set current phase on layer hooks for op tagging
        if _state.layer_hooks is not None:
            _state.layer_hooks.set_phase(name)
    
    # Start Megatron timer
    if use_timer:
        try:
            timers(name, log_level=log_level).start()
        except Exception:
            use_timer = False
    
    try:
        with nvtx_range(name):
            yield
    finally:
        # Stop Megatron timer
        if use_timer:
            try:
                timers(name).stop()
            except Exception:
                pass
        
        # Stop SOL phase tracking
        if is_sol_enabled() and _state.phase_timer is not None:
            _state.phase_timer.pop_phase()
            
            # Clear phase on layer hooks
            if _state.layer_hooks is not None:
                _state.layer_hooks.set_phase(None)


def clear_sol_captures():
    """Clear captured SOL data for the next iteration."""
    if not is_sol_enabled():
        return
        
    if _state.layer_hooks is not None:
        _state.layer_hooks.clear()
        
    if _state.phase_timer is not None:
        _state.phase_timer.clear()
        
    _state.iteration_metrics = {}


def get_sol_metrics() -> Dict[str, Any]:
    """Get current SOL metrics.
    
    Returns:
        Dictionary with SOL analysis results
    """
    if not is_sol_enabled():
        return {}
    
    metrics = {}
    
    try:
        # Synchronize GPU to ensure timing is accurate
        if _state.layer_hooks is not None:
            _state.layer_hooks.synchronize()
        
        # Get layer hooks analysis
        if _state.layer_hooks is not None:
            analysis = _state.layer_hooks.analyze()
            if analysis:
                metrics['layer_hooks'] = {
                    'total_ops': analysis.get('total_ops', 0),
                    'total_flops': analysis.get('total_flops', 0),
                    'total_memory_bytes': analysis.get('total_memory_bytes', 0),
                    'estimated_time_us': analysis.get('estimated_time_us', 0),
                    'measured_time_us': analysis.get('measured_time_us', 0),
                    'by_layer_type': analysis.get('by_layer_type', {}),
                    'by_phase': analysis.get('by_phase', {}),
                }
                
                # Calculate efficiency if we have both estimated and measured
                if analysis.get('estimated_time_us', 0) > 0 and analysis.get('measured_time_us', 0) > 0:
                    metrics['sol_efficiency'] = analysis['estimated_time_us'] / analysis['measured_time_us']
        
        # Get phase timer summary
        if _state.phase_timer is not None:
            phase_summary = _state.phase_timer.get_summary()
            if phase_summary:
                metrics['phase_times'] = phase_summary
        
        # Get CUDA graph tracker info
        if _state.cuda_graph_tracker is not None:
            graph_summary = _state.cuda_graph_tracker.get_summary()
            if graph_summary:
                metrics['cuda_graphs'] = graph_summary
        
        # Get optimizer tracker info
        if _state.optimizer_tracker is not None:
            opt_summary = _state.optimizer_tracker.get_summary()
            if opt_summary:
                metrics['optimizer'] = opt_summary
                
    except Exception as e:
        logger.warning(f"[SOL] Error getting metrics: {e}")
        
    return metrics


def log_training_sol(
    iteration: int,
    tb_writer: Optional[Any] = None,
    wandb_writer: Optional[Any] = None,
    clear: bool = True,
) -> Dict[str, Any]:
    """Log SOL metrics to TensorBoard and/or WandB.
    
    Args:
        iteration: Current training iteration
        tb_writer: TensorBoard SummaryWriter (optional)
        wandb_writer: WandB run object (optional)
        clear: Whether to clear captures after logging
        
    Returns:
        Dictionary of logged metrics
    """
    if not is_sol_enabled():
        return {}
    
    _state.current_iteration = iteration
    metrics = get_sol_metrics()
    
    if not metrics:
        return {}
    
    logged_metrics = {}
    
    try:
        # Flatten metrics for logging
        if 'layer_hooks' in metrics:
            lh = metrics['layer_hooks']
            logged_metrics['sol/total_ops'] = lh.get('total_ops', 0)
            logged_metrics['sol/total_tflops'] = lh.get('total_flops', 0) / 1e12
            logged_metrics['sol/total_memory_gb'] = lh.get('total_memory_bytes', 0) / 1e9
            logged_metrics['sol/estimated_time_ms'] = lh.get('estimated_time_us', 0) / 1e3
            logged_metrics['sol/measured_time_ms'] = lh.get('measured_time_us', 0) / 1e3
            
            if 'sol_efficiency' in metrics:
                logged_metrics['sol/efficiency'] = metrics['sol_efficiency']
            
            # Log by phase
            by_phase = lh.get('by_phase', {})
            for phase, phase_data in by_phase.items():
                phase_key = phase.replace('/', '_')
                if isinstance(phase_data, dict):
                    logged_metrics[f'sol/phase/{phase_key}/ops'] = phase_data.get('count', 0)
                    logged_metrics[f'sol/phase/{phase_key}/time_ms'] = phase_data.get('total_time_us', 0) / 1e3
        
        # Log phase wall times
        if 'phase_times' in metrics:
            pt = metrics['phase_times']
            by_phase = pt.get('by_phase', {})
            for phase, phase_data in by_phase.items():
                phase_key = phase.replace('/', '_')
                if isinstance(phase_data, dict):
                    logged_metrics[f'sol/wall/{phase_key}/time_ms'] = phase_data.get('total_wall_time_us', 0) / 1e3
        
        # Log to TensorBoard
        if tb_writer is not None:
            for key, value in logged_metrics.items():
                if isinstance(value, (int, float)):
                    tb_writer.add_scalar(key, value, iteration)
        
        # Log to WandB
        if wandb_writer is not None:
            wandb_writer.log(logged_metrics, step=iteration)
        
        # Print report at intervals
        should_print = (
            _state.report_interval > 0 and 
            iteration % _state.report_interval == 0
        )
        if should_print:
            _print_sol_report(iteration, metrics)
            
    except Exception as e:
        logger.warning(f"[SOL] Error logging metrics: {e}")
    
    if clear:
        clear_sol_captures()
    
    return logged_metrics


def _print_sol_report(iteration: int, metrics: Dict[str, Any]):
    """Print a human-readable SOL report."""
    try:
        lines = []
        lines.append("=" * 80)
        lines.append(f"SOL Analysis Report - Iteration {iteration}")
        lines.append("=" * 80)
        
        if _state.device_spec:
            lines.append(f"Device: {_state.device_spec.name}")
        
        if 'layer_hooks' in metrics:
            lh = metrics['layer_hooks']
            lines.append("")
            lines.append("Summary:")
            lines.append(f"  Total Ops: {lh.get('total_ops', 0):,}")
            lines.append(f"  Total TFLOPs: {lh.get('total_flops', 0) / 1e12:.2f}")
            lines.append(f"  Total Memory: {lh.get('total_memory_bytes', 0) / 1e9:.2f} GB")
            lines.append(f"  Estimated Time: {lh.get('estimated_time_us', 0) / 1e3:.2f} ms")
            lines.append(f"  Measured Time: {lh.get('measured_time_us', 0) / 1e3:.2f} ms")
            
            if 'sol_efficiency' in metrics:
                lines.append(f"  SOL Efficiency: {metrics['sol_efficiency'] * 100:.1f}%")
            
            # By phase breakdown
            by_phase = lh.get('by_phase', {})
            if by_phase:
                lines.append("")
                lines.append("By Phase:")
                for phase, data in sorted(by_phase.items()):
                    if isinstance(data, dict):
                        count = data.get('count', 0)
                        time_ms = data.get('total_time_us', 0) / 1e3
                        lines.append(f"  {phase:40s} {count:6d} ops, {time_ms:8.2f} ms")
            
            # By layer type breakdown
            by_type = lh.get('by_layer_type', {})
            if by_type:
                lines.append("")
                lines.append("By Layer Type:")
                for layer_type, data in sorted(by_type.items(), key=lambda x: -x[1].get('total_time_us', 0) if isinstance(x[1], dict) else 0):
                    if isinstance(data, dict):
                        count = data.get('count', 0)
                        time_ms = data.get('total_time_us', 0) / 1e3
                        pct = data.get('time_pct', 0)
                        lines.append(f"  {layer_type:40s} {count:6d} ops, {time_ms:8.2f} ms ({pct:5.1f}%)")
        
        # Phase wall times
        if 'phase_times' in metrics:
            pt = metrics['phase_times']
            by_phase = pt.get('by_phase', {})
            if by_phase:
                lines.append("")
                lines.append("Wall Clock Time by Phase:")
                total_ms = pt.get('top_level_wall_time_us', 0) / 1e3
                for phase, data in sorted(by_phase.items()):
                    if isinstance(data, dict) and data.get('is_top_level', False):
                        time_ms = data.get('total_wall_time_us', 0) / 1e3
                        pct = (time_ms / total_ms * 100) if total_ms > 0 else 0
                        lines.append(f"  {phase:40s} {time_ms:8.2f} ms ({pct:5.1f}%)")
        
        lines.append("=" * 80)
        
        print_rank_0("\n".join(lines))
        
    except Exception as e:
        logger.warning(f"[SOL] Error printing report: {e}")


def track_optimizer_step(optimizer: Any):
    """Track an optimizer step for SOL analysis.
    
    Args:
        optimizer: The optimizer being stepped
    """
    if not is_sol_enabled() or _state.optimizer_tracker is None:
        return
        
    try:
        _state.optimizer_tracker.track_step(optimizer)
    except Exception as e:
        logger.warning(f"[SOL] Error tracking optimizer step: {e}")
