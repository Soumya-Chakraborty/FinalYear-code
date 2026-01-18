import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import torch
import torchcrepe
import librosa
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURATION
# ==========================================
SAMPLE_RATE = 16000
CALIBRATION_DURATION = 4  # Seconds to sing "Sa"
PERFORMANCE_DURATION = 15 # Seconds to sing the Raga
CONFIDENCE_THRESHOLD = 0.6

# Raga Yaman Definition
RAGA_RULES = {
    "name": "Yaman",
    # 0=Sa, 2=Re, 4=Ga, 6=Tivra Ma, 7=Pa, 9=Dha, 11=Ni
    "allowed_swaras": [0, 2, 4, 6, 7, 9, 11],
    "forbidden_sequences": [
        [0, 5], # Sa -> Shuddha Ma
        [0, 6], # Sa -> Tivra Ma (Direct jump forbidden in strict grammar)
    ]
}

SWARA_NAMES = {
    0: "Sa", 1: "re", 2: "Re", 3: "ga", 4: "Ga", 5: "ma",
    6: "Ma", 7: "Pa", 8: "dha", 9: "Dha", 10: "ni", 11: "Ni"
}

# ==========================================
# MODULE 1: THE EAR (Recording & Extraction)
# ==========================================
def record_audio(duration, filename, prompt):
    print(f"\n🎤 {prompt}")
    print(f"   (Recording for {duration} seconds...)")
    
    # In a real scenario, this would record audio from microphone
    # For testing purposes, we'll simulate with a silent recording
    print("   ⚠️  NOTE: In a real environment, this would record from your microphone")
    print("   ⚠️  For this test, we're using simulated data")
    
    # Generate a silent recording for testing
    recording = np.zeros((int(duration * SAMPLE_RATE), 1), dtype=np.float32)
    
    # Save as WAV
    recording_int16 = (recording * 32767).astype(np.int16)
    wav.write(filename, SAMPLE_RATE, recording_int16)
    return filename

def extract_pitch(filename):
    print("🔄 Processing audio (Running CREPE)...")
    audio, sr = librosa.load(filename, sr=SAMPLE_RATE)

    # Prepare for GPU/CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)

    # Run Inference
    # Calculate hop_length based on 10ms step size
    hop_length = int(SAMPLE_RATE * 0.01)  # 10ms in samples
    f0, confidence = torchcrepe.predict(
        audio_tensor, sr, hop_length=hop_length, return_periodicity=True, device=device, batch_size=2048
    )

    return f0.squeeze(0).cpu().numpy(), confidence.squeeze(0).cpu().numpy()

# ==========================================
# MODULE 2: THE TRANSLATOR (Hz -> Swara)
# ==========================================
def get_user_sa(f0, confidence):
    # Filter out silence/noise
    valid_f0 = f0[confidence > CONFIDENCE_THRESHOLD]
    if len(valid_f0) == 0:
        print("⚠️  Warning: Could not detect any clear pitch during calibration!")
        print("   Using a default Sa frequency of 138.59 Hz")
        return 138.59  # Default Sa frequency

    # Take the median pitch as "Sa" (robust against outliers)
    user_sa = np.median(valid_f0)
    print(f"🎹 Calibrated Sa: {user_sa:.2f} Hz")
    return user_sa

def hz_to_swara_indices(f0, sa_hz):
    # Avoid log(0)
    f0_safe = np.nan_to_num(f0, nan=sa_hz)

    # Convert to Cents
    cents = 1200 * np.log2(f0_safe / sa_hz)
    norm_cents = cents % 1200

    # Quantize to nearest 100 cents (0-11 index)
    swara_indices = np.round(norm_cents / 100).astype(int) % 12
    return swara_indices, cents

# ==========================================
# MODULE 3: THE JUDGE (Logic & Reporting)
# ==========================================
def segment_notes(times, swara_indices, confidence):
    events = []
    if len(swara_indices) == 0: return events

    curr_note = swara_indices[0]
    start_time = times[0]

    # Minimum duration to count as a note (0.15s)
    min_duration = 0.15

    for i in range(1, len(swara_indices)):
        # If confidence is low, treat it as a break
        if confidence[i] < CONFIDENCE_THRESHOLD:
            continue

        note = swara_indices[i]
        t = times[i]

        if note != curr_note:
            if (times[i-1] - start_time) > min_duration:
                events.append({
                    "note_idx": curr_note,
                    "note_name": SWARA_NAMES[curr_note],
                    "start": start_time,
                    "end": times[i-1]
                })
            curr_note = note
            start_time = t

    return events

def analyze_performance(events, rules):
    errors = []

    # 1. Anya Swara Check
    for e in events:
        if e["note_idx"] not in rules["allowed_swaras"]:
            errors.append({
                "time": e["start"],
                "msg": f"ANYA SWARA: You sang '{e['note_name']}' (Forbidden in {rules['name']})",
                "duration": e["end"] - e["start"]
            })

    # 2. Transition Check
    for i in range(len(events) - 1):
        n1 = events[i]["note_idx"]
        n2 = events[i+1]["note_idx"]

        if [n1, n2] in rules["forbidden_sequences"]:
            errors.append({
                "time": events[i]["end"],
                "msg": f"BAD TRANSITION: {SWARA_NAMES[n1]} -> {SWARA_NAMES[n2]} is forbidden.",
                "duration": 0
            })

    return errors

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("="*50)
    print("      RAGA AI: YAMAN ERROR DETECTOR      ")
    print("="*50)

    # --- STEP 1: CALIBRATION ---
    print("\n🎤 Starting Calibration (Singing a long, steady 'Sa')...")
    calib_file = record_audio(CALIBRATION_DURATION, "calib_sa.wav", "Sing Saaaa...")
    
    # For testing purposes, let's generate some mock pitch data for calibration
    # This simulates what would happen if someone actually sang a steady "Sa"
    print("🔄 Processing audio (Running CREPE)...")
    print("   ⚠️  For this test, using simulated pitch data")
    
    # Simulate calibration pitch data (steady Sa around 138.59 Hz)
    calib_duration_samples = int(CALIBRATION_DURATION * SAMPLE_RATE)
    hop_length = int(SAMPLE_RATE * 0.01)  # 10ms hop length
    num_frames = calib_duration_samples // hop_length
    
    # Generate mock pitch data centered around Sa frequency
    sa_frequency = 138.59
    sa_f0 = np.full(num_frames, sa_frequency) + np.random.normal(0, 1, num_frames)  # Add slight variation
    sa_conf = np.full(num_frames, 0.8)  # High confidence
    
    USER_SA = get_user_sa(sa_f0, sa_conf)

    # --- STEP 2: PERFORMANCE ---
    print(f"\n🎤 Starting Performance (Singing Raga {RAGA_RULES['name']})...")
    perf_file = record_audio(PERFORMANCE_DURATION, "performance.wav", "Singing...")

    # For testing, let's generate mock performance data
    print("🔄 Processing audio (Running CREPE)...")
    print("   ⚠️  For this test, using simulated performance data")
    
    # Simulate performance pitch data with some intentional errors
    perf_duration_samples = int(PERFORMANCE_DURATION * SAMPLE_RATE)
    num_perf_frames = perf_duration_samples // hop_length
    
    # Create mock performance data with some valid and invalid notes
    # Using frequencies corresponding to different swaras in Yaman
    # Corrected frequencies to ensure accurate conversion back to indices
    swara_frequencies = {
        0: 138.59,   # Sa
        1: 146.83,   # re (komal Re - forbidden in Yaman)
        2: 155.56,   # Re (shuddha Re - allowed in Yaman)
        3: 164.81,   # ga (komal Ga - forbidden in Yaman)
        4: 174.61,   # Ga (shuddha Ga - allowed in Yaman)
        5: 185.00,   # ma (shuddha Ma - forbidden in Yaman)
        6: 196.00,   # Ma (Tivra Ma - allowed in Yaman)
        7: 207.65,   # Pa (allowed in Yaman)
        8: 220.00,   # dha (komal Dha - forbidden in Yaman)
        9: 233.07,   # Dha (shuddha Dha - allowed in Yaman) - corrected to avoid octave confusion
        10: 246.94,  # ni (komal Ni - forbidden in Yaman) - corrected to avoid octave confusion
        11: 261.63   # Ni (shuddha Ni - allowed in Yaman) - corrected to avoid octave confusion
    }

    # Create a sequence with some valid notes and some errors
    # Valid in Yaman: [0, 2, 4, 6, 7, 9, 11]
    # Invalid in Yaman: [1, 3, 5, 8, 10] - komal Re, komal Ga, shuddha Ma, komal Dha, komal Ni
    # Forbidden transitions: [[0, 5], [0, 6]] - Sa to shuddha Ma, Sa to Tivra Ma
    mock_sequence = [0, 2, 4, 7, 9, 11, 0, 1, 7, 0, 6, 0, 5]  # Includes forbidden komal Re (index 1), forbidden transition [0, 6], and forbidden transition [0, 5]
    
    perf_f0 = np.array([])
    perf_conf = np.array([])
    
    frames_per_note = num_perf_frames // len(mock_sequence)
    
    for i, note_idx in enumerate(mock_sequence):
        freq = swara_frequencies[note_idx]
        start_frame = i * frames_per_note
        end_frame = (i + 1) * frames_per_note if i < len(mock_sequence) - 1 else num_perf_frames
        
        note_f0 = np.full(end_frame - start_frame, freq) + np.random.normal(0, 0.5, end_frame - start_frame)
        note_conf = np.full(end_frame - start_frame, 0.75)
        
        perf_f0 = np.concatenate([perf_f0, note_f0])
        perf_conf = np.concatenate([perf_conf, note_conf])
    
    times = np.arange(len(perf_f0)) * 0.01

    # Convert to Swaras
    indices, cents_stream = hz_to_swara_indices(perf_f0, USER_SA)

    # --- STEP 4: ANALYSIS ---
    # Segment into distinct notes
    note_events = segment_notes(times, indices, perf_conf)

    # Check for errors
    errors = analyze_performance(note_events, RAGA_RULES)

    # --- STEP 5: REPORTING ---
    print("\n" + "="*50)
    print("         🏁 FINAL ANALYSIS REPORT 🏁         ")
    print("="*50)

    if not errors:
        print("\n🎉 Perfect! No grammatical errors detected in this clip.")
    else:
        print(f"\n⚠️ Found {len(errors)} potential errors:\n")
        for i, err in enumerate(errors, 1):
            print(f"{i}. At {err['time']:.2f}s: {err['msg']}")

    # --- VISUALIZATION ---
    print("\n📊 Generating visualization...")
    plt.figure(figsize=(14, 6))

    # Plot Pitch Track
    # Mask low confidence for cleaner plot
    mask = perf_conf > CONFIDENCE_THRESHOLD
    plt.plot(times[mask], cents_stream[mask], color='gray', alpha=0.6, label='Your Pitch')

    # Plot Errors
    for err in errors:
        plt.axvspan(err['time'], err['time'] + (err['duration'] or 0.5), color='red', alpha=0.3)
        plt.text(err['time'], 600, "ERROR", color='red', rotation=90, verticalalignment='center')

    plt.yticks([i*100 for i in range(12)], [SWARA_NAMES[i] for i in range(12)])
    plt.grid(axis='y', linestyle='--')
    plt.title(f"Performance Analysis: {RAGA_RULES['name']} (Red Zones = Errors)")
    plt.xlabel("Time (s)")
    plt.ylabel("Swara")
    plt.legend()
    plt.tight_layout()
    plt.savefig('performance_analysis.png')
    print("   ✅ Visualization saved as 'performance_analysis.png'")
    # plt.show()  # Commented out for non-interactive environments

if __name__ == "__main__":
    main()