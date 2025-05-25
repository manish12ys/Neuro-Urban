"""
GPU utilities for NeuroUrban system.
Handles GPU detection, optimization, and memory management.
"""

import logging
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
import psutil
import gc

logger = logging.getLogger(__name__)

class GPUManager:
    """Manages GPU resources and optimization for NeuroUrban."""
    
    def __init__(self, config):
        """
        Initialize GPU manager.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.device = None
        self.gpu_info = {}
        self.mixed_precision_enabled = False
        
        self._initialize_gpu()
    
    def _initialize_gpu(self):
        """Initialize GPU settings and detect available hardware."""
        logger.info("🔧 Initializing GPU configuration...")
        
        # Check CUDA availability
        if torch.cuda.is_available() and self.config.model.use_gpu:
            self._setup_cuda()
        else:
            self._setup_cpu()
        
        # Enable optimizations
        self._enable_optimizations()
        
        # Log configuration
        self._log_hardware_info()
    
    def _setup_cuda(self):
        """Setup CUDA GPU configuration."""
        try:
            # Determine device
            if self.config.model.gpu_device == "auto":
                # Select GPU with most free memory
                self.device = self._select_best_gpu()
            else:
                self.device = torch.device(self.config.model.gpu_device)
            
            # Set as default device
            torch.cuda.set_device(self.device)
            
            # Get GPU information
            self.gpu_info = self._get_gpu_info()
            
            # Enable mixed precision if supported
            if self.config.model.mixed_precision and self._supports_mixed_precision():
                self.mixed_precision_enabled = True
                logger.info("✅ Mixed precision training enabled")
            
            logger.info(f"✅ GPU initialized: {self.device}")
            
        except Exception as e:
            logger.warning(f"⚠️ GPU setup failed: {str(e)}, falling back to CPU")
            self._setup_cpu()
    
    def _setup_cpu(self):
        """Setup CPU configuration."""
        self.device = torch.device("cpu")
        
        # Optimize CPU performance
        torch.set_num_threads(psutil.cpu_count(logical=False))
        
        # Disable mixed precision for CPU
        self.mixed_precision_enabled = False
        
        logger.info(f"✅ CPU initialized with {torch.get_num_threads()} threads")
    
    def _select_best_gpu(self) -> torch.device:
        """Select GPU with most available memory."""
        if not torch.cuda.is_available():
            return torch.device("cpu")
        
        best_gpu = 0
        max_free_memory = 0
        
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = i
        
        return torch.device(f"cuda:{best_gpu}")
    
    def _get_gpu_info(self) -> Dict:
        """Get detailed GPU information."""
        if not torch.cuda.is_available():
            return {}
        
        device_id = self.device.index if self.device.index is not None else 0
        props = torch.cuda.get_device_properties(device_id)
        
        return {
            "name": props.name,
            "total_memory": props.total_memory / (1024**3),  # GB
            "major": props.major,
            "minor": props.minor,
            "multi_processor_count": props.multi_processor_count,
            "max_threads_per_block": props.max_threads_per_block,
            "max_shared_memory_per_block": props.max_shared_memory_per_block
        }
    
    def _supports_mixed_precision(self) -> bool:
        """Check if GPU supports mixed precision (FP16)."""
        if not torch.cuda.is_available():
            return False
        
        # Check for Tensor Cores (Volta architecture and newer)
        device_id = self.device.index if self.device.index is not None else 0
        props = torch.cuda.get_device_properties(device_id)
        
        # Volta (7.0), Turing (7.5), Ampere (8.0+)
        return props.major >= 7
    
    def _enable_optimizations(self):
        """Enable various PyTorch optimizations."""
        try:
            # Enable cuDNN benchmark for consistent input sizes
            if torch.cuda.is_available() and self.config.model.enable_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                logger.info("✅ cuDNN benchmark enabled")
            
            # Enable cuDNN deterministic for reproducibility (optional)
            # torch.backends.cudnn.deterministic = True
            
            # Set memory allocation strategy
            if torch.cuda.is_available():
                # Use memory pool for faster allocation
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.warning(f"⚠️ Some optimizations failed: {str(e)}")
    
    def _log_hardware_info(self):
        """Log detailed hardware information."""
        logger.info("🖥️ Hardware Configuration:")
        logger.info(f"  Device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"  CUDA Version: {torch.version.cuda}")
            logger.info(f"  cuDNN Version: {torch.backends.cudnn.version()}")
            logger.info(f"  Available GPUs: {torch.cuda.device_count()}")
            
            if self.gpu_info:
                info = self.gpu_info
                logger.info(f"  GPU: {info['name']}")
                logger.info(f"  Memory: {info['total_memory']:.1f} GB")
                logger.info(f"  Compute Capability: {info['major']}.{info['minor']}")
                logger.info(f"  Multiprocessors: {info['multi_processor_count']}")
        
        logger.info(f"  CPU Threads: {torch.get_num_threads()}")
        logger.info(f"  Mixed Precision: {self.mixed_precision_enabled}")
    
    def get_device(self) -> torch.device:
        """Get the current device."""
        return self.device
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available and being used."""
        return self.device.type == "cuda"
    
    def get_memory_info(self) -> Dict:
        """Get current memory usage information."""
        if not torch.cuda.is_available():
            return {"device": "cpu", "memory_used": 0, "memory_total": 0}
        
        device_id = self.device.index if self.device.index is not None else 0
        
        return {
            "device": str(self.device),
            "memory_used": torch.cuda.memory_allocated(device_id) / (1024**3),  # GB
            "memory_reserved": torch.cuda.memory_reserved(device_id) / (1024**3),  # GB
            "memory_total": torch.cuda.get_device_properties(device_id).total_memory / (1024**3)  # GB
        }
    
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for current hardware."""
        try:
            # Move model to device
            model = model.to(self.device)
            
            # Compile model for PyTorch 2.0+ (significant speedup)
            if self.config.model.compile_models and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model)
                    logger.info("✅ Model compiled with torch.compile")
                except Exception as e:
                    logger.warning(f"⚠️ Model compilation failed: {str(e)}")
            
            # Enable mixed precision wrapper if needed
            if self.mixed_precision_enabled:
                # This will be handled by the training loop with autocast
                pass
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Model optimization failed: {str(e)}")
            return model
    
    def create_dataloader(self, dataset, batch_size: int, shuffle: bool = True, **kwargs):
        """Create optimized DataLoader."""
        from torch.utils.data import DataLoader
        
        # GPU optimizations
        if self.is_gpu_available():
            kwargs.update({
                'num_workers': self.config.model.num_workers,
                'pin_memory': self.config.model.pin_memory,
                'persistent_workers': True if self.config.model.num_workers > 0 else False
            })
        else:
            # CPU optimizations
            kwargs.update({
                'num_workers': min(4, psutil.cpu_count(logical=False)),
                'pin_memory': False
            })
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )
    
    def clear_cache(self):
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Python garbage collection
        gc.collect()
        
        logger.info("🧹 GPU cache and memory cleared")
    
    def get_optimal_batch_size(self, model: torch.nn.Module, input_shape: Tuple, max_batch_size: int = 128) -> int:
        """Automatically determine optimal batch size for current GPU."""
        if not self.is_gpu_available():
            return min(32, max_batch_size)  # Conservative for CPU
        
        try:
            model.eval()
            optimal_batch_size = 1
            
            # Binary search for optimal batch size
            low, high = 1, max_batch_size
            
            while low <= high:
                mid = (low + high) // 2
                
                try:
                    # Test batch size
                    dummy_input = torch.randn(mid, *input_shape).to(self.device)
                    
                    with torch.no_grad():
                        _ = model(dummy_input)
                    
                    optimal_batch_size = mid
                    low = mid + 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        high = mid - 1
                        torch.cuda.empty_cache()
                    else:
                        raise e
                
                finally:
                    del dummy_input
                    torch.cuda.empty_cache()
            
            logger.info(f"🎯 Optimal batch size determined: {optimal_batch_size}")
            return optimal_batch_size
            
        except Exception as e:
            logger.warning(f"⚠️ Batch size optimization failed: {str(e)}")
            return min(16, max_batch_size)  # Safe fallback
    
    def profile_model(self, model: torch.nn.Module, input_shape: Tuple, num_iterations: int = 100):
        """Profile model performance."""
        if not self.is_gpu_available():
            logger.info("⚠️ Profiling skipped - GPU not available")
            return
        
        try:
            model.eval()
            dummy_input = torch.randn(1, *input_shape).to(self.device)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            torch.cuda.synchronize()
            
            # Profile
            import time
            start_time = time.time()
            
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_iterations * 1000  # ms
            throughput = 1000 / avg_time  # FPS
            
            logger.info(f"📊 Model Performance:")
            logger.info(f"  Average inference time: {avg_time:.2f} ms")
            logger.info(f"  Throughput: {throughput:.1f} FPS")
            
        except Exception as e:
            logger.error(f"❌ Profiling failed: {str(e)}")

# Global GPU manager instance
_gpu_manager = None

def get_gpu_manager(config=None):
    """Get global GPU manager instance."""
    global _gpu_manager
    if _gpu_manager is None and config is not None:
        _gpu_manager = GPUManager(config)
    return _gpu_manager

def setup_gpu_environment(config):
    """Setup global GPU environment."""
    global _gpu_manager
    _gpu_manager = GPUManager(config)
    return _gpu_manager
