# ref: 727ca08c2e05c1b88f998b5f7a2f3abd988134d6
import importlib
import pytest

MODULES = [
    "main",
    "module_1_ear",
    "module_2_translator",
    "module_3_judge",
]

@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    # Ensure module can be imported
    mod = importlib.import_module(module_name)
    assert mod is not None

def test_main_has_key_functions():
    mod = importlib.import_module("main")
    # Check for the presence of the main functions used by the workflows
    for fn in ("extract_advanced_features", "analyze_performance", "segment_notes"):
        assert hasattr(mod, fn), f"main.py missing required function: {fn}"

# Also test the original imports to ensure dependencies work
def test_core_dependencies():
    """Test that core dependencies can be imported"""
    import numpy as np
    import librosa
    import torch
    import torchcrepe
    import sounddevice
    import matplotlib.pyplot as plt
    import scipy.io.wavfile as wav

    # Basic functionality tests
    arr = np.array([1, 2, 3])
    assert len(arr) == 3

    # If we reach this point, all imports succeeded
    assert True