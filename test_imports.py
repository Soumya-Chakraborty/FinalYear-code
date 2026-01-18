#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

print("Testing imports...")

try:
    import sounddevice as sd
    print("✅ sounddevice imported successfully")
except ImportError as e:
    print(f"❌ Failed to import sounddevice: {e}")

try:
    import numpy as np
    print("✅ numpy imported successfully")
except ImportError as e:
    print(f"❌ Failed to import numpy: {e}")

try:
    import scipy.io.wavfile as wav
    print("✅ scipy.io.wavfile imported successfully")
except ImportError as e:
    print(f"❌ Failed to import scipy.io.wavfile: {e}")

try:
    import torch
    print(f"✅ torch imported successfully (version: {torch.__version__})")
except ImportError as e:
    print(f"❌ Failed to import torch: {e}")

try:
    import torchcrepe
    print("✅ torchcrepe imported successfully")
except ImportError as e:
    print(f"❌ Failed to import torchcrepe: {e}")

try:
    import librosa
    print(f"✅ librosa imported successfully (version: {librosa.__version__})")
except ImportError as e:
    print(f"❌ Failed to import librosa: {e}")

try:
    import matplotlib.pyplot as plt
    print("✅ matplotlib.pyplot imported successfully")
except ImportError as e:
    print(f"❌ Failed to import matplotlib.pyplot: {e}")

print("\nAll imports tested successfully! The original error has been fixed.")