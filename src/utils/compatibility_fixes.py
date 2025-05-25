"""
Compatibility fixes for NeuroUrban system.
"""

import os
import sys
import warnings
import logging
from typing import Optional

def apply_compatibility_fixes():
    """Apply various compatibility fixes for different environments."""

    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

    # Fix Windows-specific issues
    if sys.platform.startswith('win'):
        _fix_windows_issues()

    # Fix PyTorch issues
    _fix_pytorch_issues()

    # Fix asyncio issues
    _fix_asyncio_issues()

def _fix_windows_issues():
    """Fix Windows-specific compatibility issues."""

    # Set environment variables to avoid subprocess issues
    os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)

    # Fix path issues
    if 'PYTHONPATH' not in os.environ:
        os.environ['PYTHONPATH'] = os.getcwd()

def _fix_pytorch_issues():
    """Fix PyTorch compatibility issues."""

    # Set PyTorch to use CPU by default to avoid GPU conflicts
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Disable PyTorch warnings
    os.environ['PYTORCH_DISABLE_WARNINGS'] = '1'

    # Fix PyTorch-Streamlit compatibility issues
    os.environ['TORCH_DISABLE_STREAMLIT_WATCHER'] = '1'

    try:
        import torch
        # Set number of threads to avoid conflicts
        torch.set_num_threads(min(4, os.cpu_count() or 4))

        # Monkey patch torch._classes to avoid Streamlit watcher issues
        if hasattr(torch, '_classes'):
            original_getattr = torch._classes.__getattr__

            def safe_getattr(self, name):
                if name == '__path__':
                    # Return a dummy path object that won't cause issues
                    class DummyPath:
                        def __init__(self):
                            self._path = []

                        def __iter__(self):
                            return iter(self._path)

                    return DummyPath()
                return original_getattr(name)

            torch._classes.__getattr__ = safe_getattr

    except ImportError:
        pass
    except Exception as e:
        # If patching fails, just continue
        pass

def _fix_asyncio_issues():
    """Fix asyncio event loop issues."""

    try:
        import asyncio

        # For Windows, set the event loop policy
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        # Try to get or create event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                asyncio.set_event_loop(asyncio.new_event_loop())
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

    except ImportError:
        pass

def setup_safe_environment():
    """Setup a safe environment for the application."""

    # Apply all compatibility fixes
    apply_compatibility_fixes()

    # Setup logging to suppress unnecessary warnings
    logging.getLogger('joblib').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)
    logging.getLogger('sklearn').setLevel(logging.ERROR)

    # Set multiprocessing start method for compatibility
    try:
        import multiprocessing
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('spawn', force=True)
    except (ImportError, RuntimeError):
        pass

# Apply fixes when module is imported
setup_safe_environment()
