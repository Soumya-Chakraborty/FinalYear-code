# Requirements Documentation

## System Requirements

### Hardware Requirements
- Computer with microphone input (for live audio capture)
- Minimum 4GB RAM (recommended 8GB+ for optimal performance)
- Sound card capable of 16kHz audio sampling
- At least 500MB free disk space

### Software Requirements
- Python 3.8 or higher
- Operating System: Windows, macOS, or Linux
- Compatible with both CPU and GPU (CUDA) environments

## Python Dependencies

The system requires the following Python packages:

### Core Dependencies
- `numpy>=1.21.0` - Numerical computing
- `librosa>=0.9.0` - Audio signal processing
- `scipy>=1.7.0` - Scientific computing
- `torch>=2.0.0` - Deep learning framework
- `torchaudio>=2.0.0` - Audio processing for PyTorch
- `torchcrepe>=0.0.17` - Pitch estimation
- `sounddevice>=0.4.6` - Audio I/O
- `matplotlib>=3.5.0` - Visualization

### Additional Dependencies
- `scikit-learn>=1.0.0` - Machine learning utilities
- `soundfile>=0.10.0` - Audio file I/O

## Installation

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install numpy librosa scipy torch torchaudio torchcrepe sounddevice matplotlib scikit-learn soundfile
```

## Audio Specifications

### Input Requirements
- **Sample Rate**: 16,000 Hz (automatically resampled if needed)
- **Channels**: Mono (single channel)
- **Bit Depth**: 16-bit or 24-bit
- **Format**: WAV, MP3, FLAC, or other common audio formats supported by librosa

### Recommended Audio Quality
- **Signal-to-Noise Ratio**: >30dB recommended
- **Dynamic Range**: 60dB or greater
- **Frequency Response**: 20Hz to 5kHz (covers human vocal range)

## Performance Requirements

### Real-time Processing
- **Latency**: <100ms for real-time feedback (hardware dependent)
- **CPU Usage**: Up to 50% during active processing
- **Memory Usage**: ~500MB during operation

### Accuracy Requirements
- **Minimum Signal Level**: -30dBFS for reliable pitch detection
- **Steady Tone Duration**: >0.15 seconds for reliable note segmentation
- **Pitch Range**: C3 (130Hz) to C6 (1046Hz) for optimal detection

## Environmental Requirements

### Acoustic Environment
- **Background Noise**: <30dB SPL recommended
- **Reverberation Time**: <0.5 seconds preferred
- **Room Treatment**: Moderate acoustic treatment to minimize reflections

### Operating Conditions
- **Temperature**: 15°C to 30°C
- **Humidity**: 30% to 70% non-condensing
- **Power Supply**: Stable electrical supply for consistent audio performance

## Compatibility Matrix

| Component | Supported Versions | Notes |
|-----------|-------------------|-------|
| Python | 3.8 - 3.14 | Tested up to Python 3.14 |
| PyTorch | 2.0+ | CUDA support available |
| Librosa | 0.9+ | Backward compatible |
| NumPy | 1.21+ | Required for tensor operations |

## Optional Components

### GPU Acceleration
- **CUDA**: 11.7+ for GPU acceleration
- **GPU Memory**: 2GB+ VRAM recommended
- **Compute Capability**: 3.5+ for optimal performance

### Additional Audio Formats
- **FFmpeg**: For extended audio format support
- **SoX**: Alternative audio processing backend

## Troubleshooting Common Issues

### Audio Input Problems
- Check microphone permissions in OS settings
- Verify audio driver compatibility
- Ensure sample rate is 16kHz or allow automatic resampling

### Performance Issues
- Reduce concurrent applications during processing
- Ensure sufficient RAM availability
- Consider using CPU instead of GPU if experiencing memory issues

### Installation Issues
- Use virtual environment to avoid conflicts
- Install PyTorch with appropriate CUDA version
- Check for system-specific installation requirements