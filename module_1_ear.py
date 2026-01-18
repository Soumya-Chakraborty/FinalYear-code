import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import torch
import torchcrepe
import matplotlib.pyplot as plt
import librosa

# --- CONFIGURATION ---
SAMPLE_RATE = 16000   # CREPE works best at 16kHz
DURATION = 10         # Recording duration in seconds
FILENAME = "my_raga_singing.wav"
CONFIDENCE_THRESHOLD = 0.6  # Filter out noise/breathing

def record_audio(duration, fs, filename):
    import sys
    import os

    print(f"🎤 Recording for {duration} seconds... SING NOW!")

    # Check if running in interactive mode
    if os.isatty(sys.stdin.fileno()):
        # Record audio (mono)
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        print("✅ Recording complete.")
    else:
        # In non-interactive mode, simulate recording with silence
        print("   ⚠️  NOTE: In a real environment, this would record from your microphone")
        print("   ⚠️  For this test, we're using simulated data")

        # Generate a silent recording for testing
        recording = np.zeros((int(duration * fs), 1), dtype=np.float32)

    print("✅ Recording complete.")

    # Save as WAV file (crucial for debugging later steps)
    # Convert to 16-bit PCM for standard WAV compatibility
    recording_int16 = (recording * 32767).astype(np.int16)
    wav.write(filename, fs, recording_int16)
    return filename

def extract_pitch(filename):
    print("🔄 Extracting pitch data (this might take a moment)...")
    
    # 1. Load audio using Librosa to ensure correct shape/format
    audio, sr = librosa.load(filename, sr=SAMPLE_RATE)
    
    # 2. Prepare audio for CREPE (Add batch dimension: [1, samples])
    audio_tensor = torch.tensor(audio).unsqueeze(0)
    
    # 3. Select device (Use GPU if available, else CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    audio_tensor = audio_tensor.to(device)
    
    # 4. Run CREPE Inference
    # Calculate hop_length based on 10ms step size
    hop_length = int(SAMPLE_RATE * 0.01)  # 10ms in samples
    f0, confidence = torchcrepe.predict(
        audio_tensor,
        sr,
        hop_length=hop_length,
        device=device,
        return_periodicity=True,
        batch_size=2048
    )
    
    # 5. Move data back to CPU for processing
    f0 = f0.squeeze(0).cpu().numpy()
    confidence = confidence.squeeze(0).cpu().numpy()
    
    return f0, confidence

def filter_and_plot(f0, confidence):
    # Create a time axis (every 10ms)
    time_axis = np.arange(len(f0)) * 0.01 
    
    # Filter: Set frequency to NaN (invisible) where confidence is low (silence/noise)
    f0_clean = f0.copy()
    f0_clean[confidence < CONFIDENCE_THRESHOLD] = np.nan

    # --- PLOTTING ---
    plt.figure(figsize=(12, 6))
    
    # Plot raw pitch contour
    plt.plot(time_axis, f0_clean, '.', markersize=2, label='Detected Pitch', color='blue')
    
    plt.title(f"Extracted Pitch Contour (Confidence > {CONFIDENCE_THRESHOLD})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.show()
    
    return time_axis, f0_clean

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Step 1: Record
    saved_file = record_audio(DURATION, SAMPLE_RATE, FILENAME)
    
    # Step 2: Extract
    raw_f0, raw_confidence = extract_pitch(saved_file)
    
    # Step 3: Visualize
    times, pitches = filter_and_plot(raw_f0, raw_confidence)
    
    # Output for the next module
    print("\n📊 DATA SAMPLE (First 5 detected notes):")
    valid_indices = np.where(~np.isnan(pitches))[0][:5]
    for idx in valid_indices:
        print(f"Time: {times[idx]:.2f}s | Freq: {pitches[idx]:.2f} Hz")