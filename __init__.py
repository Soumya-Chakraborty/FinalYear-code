"""
Raga AI Package Initialization

This package provides tools for analyzing Indian classical music performances,
specifically for detecting errors in the Yaman raga.
"""

__version__ = "1.0.0"
__author__ = "Raga AI Development Team"
__email__ = "info@raga-ai.example.com"

# Import main functionality.
# Use relative import when package context is available, otherwise fall back to absolute import
try:
    from .main import main  # works when package is imported as a package
except (ImportError, SystemError):
    # Fallback for test runners that import modules as top-level scripts
    from main import main

__all__ = ['main']