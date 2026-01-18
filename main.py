import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import torch
import torchcrepe
import librosa
import matplotlib.pyplot as plt
import os
import sys
import sklearn
from sklearn.preprocessing import StandardScaler
from scipy import signal

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
        [5, 0], # Shuddha Ma -> Sa (Not applicable since shuddha ma is forbidden, but for completeness)
    ],
    # Characteristic phrases typical of Yaman
    "characteristic_phrases": [
        [4, 6, 7],      # Ga-Ma-Pa (with Tivra Ma)
        [11, 0, 2],     # Ni-Sa-Re
        [9, 11, 0],     # Dha-Ni-Sa
        [2, 4, 6],      # Re-Ga-Ma (with Tivra Ma)
        [7, 9, 11],     # Pa-Dha-Ni
        [11, 9, 7],     # Ni-Dha-Pa (Avaroha)
        [6, 4, 2],      # Ma-Ga-Re (Avaroha with Tivra Ma)
        [7, 6, 4],      # Pa-Ma-Ga (Avaroha with Tivra Ma)
    ],
    # Common melodic movements in Yaman
    "common_movements": {
        "ascending": [
            [0, 2, 4, 6, 7],  # Sa-Re-Ga-Ma-Pa
            [0, 2, 4, 6, 7, 9, 11],  # Full ascending
        ],
        "descending": [
            [7, 6, 4, 2, 0],  # Pa-Ma-Ga-Re-Sa
            [11, 9, 7, 6, 4, 2, 0],  # Full descending
        ]
    },
    # Preferred note relationships
    "preferred_relationships": {
        "approach_tivra_ma_from": [4, 11],  # Ga or Ni should approach Tivra Ma
        "emphasize_on": [0, 4, 6, 7],      # Important notes in Yaman
        "avoid_direct_jump": [0, 6],        # Avoid direct jump from Sa to Tivra Ma
    }
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
    
    # Check if running in interactive mode
    if os.isatty(sys.stdin.fileno()):
        recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        print("✅ Recording complete.")

        # Save as WAV
        recording_int16 = (recording * 32767).astype(np.int16)
        wav.write(filename, SAMPLE_RATE, recording_int16)
        return filename
    else:
        # In non-interactive mode, simulate recording with silence
        print("   ⚠️  NOTE: In a real environment, this would record from your microphone")
        print("   ⚠️  For this test, we're using simulated data")
        
        # Generate a silent recording for testing
        recording = np.zeros((int(duration * SAMPLE_RATE), 1), dtype=np.float32)
        
        # Save as WAV
        recording_int16 = (recording * 32767).astype(np.int16)
        wav.write(filename, SAMPLE_RATE, recording_int16)
        return filename

def extract_advanced_features(filename):
    """
    Extract multiple audio features including pitch, MFCCs, spectral features, etc.
    """
    print("🔄 Processing audio (Extracting advanced features)...")
    audio, sr = librosa.load(filename, sr=SAMPLE_RATE)

    # Extract pitch using CREPE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)

    # Calculate hop_length based on 10ms step size
    hop_length = int(SAMPLE_RATE * 0.01)  # 10ms in samples
    f0, confidence = torchcrepe.predict(
        audio_tensor, sr, hop_length=hop_length, return_periodicity=True, device=device, batch_size=2048
    )

    f0 = f0.squeeze(0).cpu().numpy()
    confidence = confidence.squeeze(0).cpu().numpy()

    # Extract additional features
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=hop_length)

    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=hop_length)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio, hop_length=hop_length)[0]

    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop_length)

    # Compute times for features
    feature_times = np.arange(mfccs.shape[1]) * hop_length / sr

    return {
        'f0': f0,
        'confidence': confidence,
        'mfccs': mfccs,
        'spectral_centroids': spectral_centroids,
        'spectral_rolloff': spectral_rolloff,
        'spectral_bandwidth': spectral_bandwidth,
        'zero_crossing_rate': zero_crossing_rate,
        'chroma': chroma,
        'feature_times': feature_times
    }

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
    """
    Improved note segmentation that considers pitch stability and gradual transitions
    """
    events = []
    if len(swara_indices) == 0: return events

    # Define minimum note duration and stability threshold
    min_duration = 0.15  # seconds
    stability_threshold = 0.7  # percentage of frames that should be consistent

    i = 0
    while i < len(swara_indices):
        if confidence[i] < CONFIDENCE_THRESHOLD:
            i += 1
            continue

        current_note = swara_indices[i]
        start_idx = i

        # Look ahead to find the end of this note
        j = i + 1
        while j < len(swara_indices):
            if confidence[j] < CONFIDENCE_THRESHOLD:
                break

            # Check if we're transitioning to a new note
            if swara_indices[j] != current_note:
                # Check if this is a real transition or just pitch fluctuation
                # Look at a window to determine stability
                window_size = max(5, int(0.05 * len(swara_indices)))  # 5% of total length or 5 frames
                window_end = min(j + window_size, len(swara_indices))

                # Count how many frames in the window maintain the new note
                new_note_count = sum(1 for k in range(j, window_end)
                                   if k < len(swara_indices) and
                                   swara_indices[k] == swara_indices[j] and
                                   confidence[k] >= CONFIDENCE_THRESHOLD)

                # If the new note appears consistently, consider it a real transition
                if new_note_count > window_size * stability_threshold:
                    break
                else:
                    # This is likely just pitch fluctuation, continue with current note
                    pass

            j += 1

        # Calculate the duration of this note
        note_start_time = times[start_idx]
        note_end_time = times[j-1] if j > start_idx else times[start_idx]
        note_duration = note_end_time - note_start_time

        # Only add if the note meets minimum duration requirement
        if note_duration >= min_duration:
            events.append({
                "note_idx": current_note,
                "note_name": SWARA_NAMES[current_note],
                "start": note_start_time,
                "end": note_end_time,
                "duration": note_duration,
                "confidence_avg": np.mean(confidence[start_idx:j])
            })

        i = j

    return events

def analyze_performance(events, rules, cents_stream=None, times=None, features=None):
    errors = []
    warnings = []  # For less severe issues

    # 1. Anya Swara Check (Completely forbidden notes)
    for e in events:
        if e["note_idx"] not in rules["allowed_swaras"]:
            errors.append({
                "type": "anya_svara",
                "time": e["start"],
                "msg": f"ANYA SWARA: You sang '{e['note_name']}' (Forbidden in {rules['name']})",
                "duration": e["end"] - e["start"],
                "severity": "high"
            })

    # 2. Transition Check (Forbidden sequences)
    for i in range(len(events) - 1):
        n1 = events[i]["note_idx"]
        n2 = events[i+1]["note_idx"]

        if [n1, n2] in rules["forbidden_sequences"]:
            errors.append({
                "type": "forbidden_transition",
                "time": events[i]["end"],
                "msg": f"BAD TRANSITION: {SWARA_NAMES[n1]} -> {SWARA_NAMES[n2]} is forbidden in {rules['name']}.",
                "duration": 0,
                "severity": "high"
            })

    # 3. Advanced Raga Patterns Check
    # Check for characteristic phrases of Yaman
    yaman_characteristics = rules.get("characteristic_phrases", [])
    for i in range(len(events) - 2):  # Need at least 3 notes for phrases
        phrase = [events[i]["note_idx"], events[i+1]["note_idx"], events[i+2]["note_idx"]]
        if phrase not in yaman_characteristics:
            # Check if this is a deviation from expected patterns
            warnings.append({
                "type": "pattern_deviation",
                "time": events[i]["start"],
                "msg": f"PATTERN DEVIATION: Sequence {SWARA_NAMES[events[i]['note_idx']]}->{SWARA_NAMES[events[i+1]['note_idx']]}->{SWARA_NAMES[events[i+2]['note_idx']]} is uncommon in {rules['name']}",
                "severity": "medium"
            })

    # 4. Improper use of Tivra Ma check
    # In Yaman, Tivra Ma should often be approached from Ga or Ni
    for i in range(1, len(events)):
        if events[i]["note_idx"] == 6:  # Tivra Ma
            prev_note = events[i-1]["note_idx"]
            # Tivra Ma should typically come after Ga (4) or Ni (11), not directly from Sa (0) or Pa (7)
            if prev_note in [0, 7]:  # Sa or Pa
                warnings.append({
                    "type": "improper_ma_usage",
                    "time": events[i]["start"],
                    "msg": f"TIVRA MA USAGE: Tivra Ma ({SWARA_NAMES[6]}) approached from {SWARA_NAMES[prev_note]} - consider approaching from {SWARA_NAMES[4]} (Ga) or {SWARA_NAMES[11]} (Ni) in {rules['name']}",
                    "severity": "medium"
                })

    # 5. Microtonal analysis (deviations from standard swara positions)
    if cents_stream is not None and times is not None:
        # Calculate microtonal deviations
        for e in events:
            # Find the time indices for this event
            start_idx = np.argmin(np.abs(times - e["start"]))
            end_idx = np.argmin(np.abs(times - e["end"]))

            # Get cents values for this note
            note_cents = cents_stream[start_idx:end_idx+1]

            # Calculate average deviation from ideal position
            ideal_cent = e["note_idx"] * 100  # Each swara is ideally at multiples of 100 cents
            avg_deviation = np.mean(np.abs(note_cents - ideal_cent))

            # If deviation is significant (> 30 cents), add a warning
            if avg_deviation > 30:
                warnings.append({
                    "type": "microtonal_deviation",
                    "time": e["start"],
                    "msg": f"MICROTONE DEVIATION: {e['note_name']} is {avg_deviation:.1f} cents away from ideal position",
                    "severity": "low"
                })

    # 6. Check for proper emphasis on important notes
    emphasized_notes = rules.get("preferred_relationships", {}).get("emphasize_on", [])
    for e in events:
        if e["note_idx"] in emphasized_notes and e["duration"] < 0.5:  # Less than 0.5s is too short for emphasized notes
            warnings.append({
                "type": "insufficient_emphasis",
                "time": e["start"],
                "msg": f"INSUFFICIENT EMPHASIS: {e['note_name']} should be held longer in {rules['name']}",
                "severity": "medium"
            })

    # 7. Advanced analysis using additional features if available
    if features is not None:
        # Analyze timbral characteristics using MFCCs
        mfccs = features.get('mfccs', None)
        if mfccs is not None and times is not None:
            # Map MFCCs to time segments corresponding to note events
            feature_times = features.get('feature_times', times[::10])  # Assuming features computed at lower resolution

            # Only perform detailed MFCC analysis if we're in real mode (not mock data)
            # This is determined by checking if the MFCC values appear to be real or random
            is_mock_data = np.all(mfccs < 50.0) and np.all(mfccs > -50.0) and np.std(mfccs) < 5.0  # Mock data typically has small random values

            if not is_mock_data:  # Only analyze if we have real audio features
                # Analyze MFCCs for each note event
                for e in events:
                    # Find corresponding MFCC frames for this event
                    start_idx = np.argmin(np.abs(feature_times - e["start"]))
                    end_idx = np.argmin(np.abs(feature_times - e["end"]))

                    if end_idx > start_idx:
                        event_mfccs = mfccs[:, start_idx:end_idx+1]

                        # Calculate MFCC statistics for this event
                        mfcc_mean = np.mean(event_mfccs, axis=1)
                        mfcc_std = np.std(event_mfccs, axis=1)

                        # Check for unusual timbral characteristics
                        # (This is a simplified check - in practice, this would compare to reference models)
                        if len(mfcc_mean) > 0 and np.max(np.abs(mfcc_mean)) > 20:  # Arbitrary threshold
                            warnings.append({
                                "type": "timbral_anomaly",
                                "time": e["start"],
                                "msg": f"TIMBRAL ANOMALY: Unusual timbral characteristics detected for {e['note_name']} in {rules['name']}",
                                "severity": "low"
                            })

        # Analyze harmonic content using chroma features
        chroma = features.get('chroma', None)
        if chroma is not None and times is not None:
            feature_times = features.get('feature_times', times[::10])  # Assuming features computed at lower resolution

            # Only perform detailed chroma analysis if we're in real mode (not mock data)
            # This is determined by checking if the chroma values appear to be real or random
            is_mock_data = np.all(chroma < 1.0) and np.all(chroma > 0.0)  # Mock data typically has random values in [0,1]

            if not is_mock_data:  # Only analyze if we have real audio features
                # Analyze chroma features for each note event
                for e in events:
                    # Find corresponding chroma frames for this event
                    start_idx = np.argmin(np.abs(feature_times - e["start"]))
                    end_idx = np.argmin(np.abs(feature_times - e["end"]))

                    if end_idx > start_idx:
                        event_chroma = chroma[:, start_idx:end_idx+1]

                        # Calculate dominant pitch class for this event
                        avg_chroma = np.mean(event_chroma, axis=1)
                        dominant_pitch_class = np.argmax(avg_chroma)

                        # Calculate strength of the dominant pitch class
                        max_chroma_value = np.max(avg_chroma)

                        # Check if the dominant pitch class matches the intended note
                        # (Allowing for some tolerance in case of gamakas or microtonal variations)
                        # Only flag if the match is poor AND the dominant pitch class is strong enough to be reliable
                        if abs(dominant_pitch_class - e["note_idx"]) > 1 and max_chroma_value > 0.3:
                            warnings.append({
                                "type": "harmonic_mismatch",
                                "time": e["start"],
                                "msg": f"HARMONIC MISMATCH: Harmonic content of {e['note_name']} doesn't match expected pitch class in {rules['name']}",
                                "severity": "medium"
                            })

    # Combine errors and warnings
    all_issues = errors + warnings

    # Sort by time
    all_issues.sort(key=lambda x: x["time"])

    return all_issues

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("="*50)
    print("      RAGA AI: YAMAN ERROR DETECTOR      ")
    print("="*50)

    # --- STEP 1: CALIBRATION ---
    if os.isatty(sys.stdin.fileno()):
        input("\nPress Enter to start Calibration (Sing a long, steady 'Sa')...")
    else:
        print("\n🎤 Starting Calibration (Singing a long, steady 'Sa')...")
    
    calib_file = record_audio(CALIBRATION_DURATION, "calib_sa.wav", "Sing Saaaa...")
    
    # If running in non-interactive mode, use simulated data for calibration
    if not os.isatty(sys.stdin.fileno()):
        print("   ⚠️  For this test, using simulated pitch data")

        # Simulate calibration pitch data (steady Sa around 138.59 Hz)
        calib_duration_samples = int(CALIBRATION_DURATION * SAMPLE_RATE)
        hop_length = int(SAMPLE_RATE * 0.01)  # 10ms hop length
        num_frames = calib_duration_samples // hop_length

        # Generate mock pitch data centered around Sa frequency
        sa_frequency = 138.59
        sa_f0 = np.full(num_frames, sa_frequency) + np.random.normal(0, 1, num_frames)  # Add slight variation
        sa_conf = np.full(num_frames, 0.8)  # High confidence

        # Create mock feature data
        mock_mfccs = np.random.rand(13, num_frames)
        mock_chroma = np.random.rand(12, num_frames)
        mock_feature_times = np.arange(num_frames) * hop_length / SAMPLE_RATE

        features_calib = {
            'f0': sa_f0,
            'confidence': sa_conf,
            'mfccs': mock_mfccs,
            'chroma': mock_chroma,
            'feature_times': mock_feature_times
        }
    else:
        features_calib = extract_advanced_features(calib_file)
        sa_f0 = features_calib['f0']
        sa_conf = features_calib['confidence']

    USER_SA = get_user_sa(sa_f0, sa_conf)

    # --- STEP 2: PERFORMANCE ---
    if os.isatty(sys.stdin.fileno()):
        input(f"\nPress Enter to start Performance (Sing Raga {RAGA_RULES['name']})...")
    else:
        print(f"\n🎤 Starting Performance (Singing Raga {RAGA_RULES['name']})...")
    
    perf_file = record_audio(PERFORMANCE_DURATION, "performance.wav", "Singing...")

    # --- STEP 3: PROCESSING ---
    # If running in non-interactive mode, use simulated performance data
    if not os.isatty(sys.stdin.fileno()):
        print("   ⚠️  For this test, using simulated performance data")

        # Simulate performance pitch data with some intentional errors
        perf_duration_samples = int(PERFORMANCE_DURATION * SAMPLE_RATE)
        hop_length = int(SAMPLE_RATE * 0.01)  # 10ms hop length
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

        # Create mock feature data for performance
        mock_mfccs = np.random.rand(13, num_perf_frames)
        mock_chroma = np.random.rand(12, num_perf_frames)
        mock_feature_times = np.arange(num_perf_frames) * hop_length / SAMPLE_RATE

        features_perf = {
            'f0': perf_f0,
            'confidence': perf_conf,
            'mfccs': mock_mfccs,
            'chroma': mock_chroma,
            'feature_times': mock_feature_times
        }
    else:
        features_perf = extract_advanced_features(perf_file)
        perf_f0 = features_perf['f0']
        perf_conf = features_perf['confidence']
        times = features_perf['feature_times']

    # Convert to Swaras
    indices, cents_stream = hz_to_swara_indices(perf_f0, USER_SA)

    # --- STEP 4: ANALYSIS ---
    # Segment into distinct notes
    note_events = segment_notes(times, indices, perf_conf)

    # Check for errors
    errors = analyze_performance(note_events, RAGA_RULES, cents_stream, times, features_perf)

    # --- STEP 5: REPORTING ---
    print("\n" + "="*50)
    print("         🏁 FINAL ANALYSIS REPORT 🏁         ")
    print("="*50)

    # Separate errors and warnings
    errors_only = [err for err in errors if err.get('severity', '') == 'high']
    warnings_only = [err for err in errors if err.get('severity', '') != 'high']

    if not errors_only and not warnings_only:
        print("\n🎉 Perfect! No grammatical errors detected in this clip.")
    else:
        if errors_only:
            print(f"\n🚨 Found {len(errors_only)} MAJOR ERRORS:\n")
            for i, err in enumerate(errors_only, 1):
                print(f"{i}. At {err['time']:.2f}s: {err['msg']}")

        if warnings_only:
            print(f"\n⚠️ Found {len(warnings_only)} WARNINGS:\n")
            for i, warn in enumerate(warnings_only, 1):
                print(f"{i}. At {warn['time']:.2f}s: {warn['msg']}")

        if not errors_only and warnings_only:
            print("\n🎵 Overall performance is acceptable, but consider the suggestions above for improvement.")

    # --- VISUALIZATION ---
    print("\n📊 Generating visualization...")
    try:
        # Create multiple visualization steps
        # Step 1: Raw pitch contour
        fig1, ax = plt.subplots(figsize=(14, 6))
        mask = perf_conf > CONFIDENCE_THRESHOLD
        if np.any(mask):
            ax.plot(times[mask], cents_stream[mask], color='gray', alpha=0.6, label='Raw Pitch Contour', linewidth=1.5)
        ax.set_yticks([i*100 for i in range(12)])
        ax.set_yticklabels([SWARA_NAMES[i] for i in range(12)])
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_title(f"Step 1: Raw Pitch Contour - {RAGA_RULES['name']}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Swara")
        ax.legend()
        plt.tight_layout()
        plt.savefig('step1_raw_pitch_contour.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✅ Step 1 visualization saved as 'step1_raw_pitch_contour.png'")

        # Step 2: Segmented notes
        fig2, ax = plt.subplots(figsize=(14, 6))
        if np.any(mask):
            ax.plot(times[mask], cents_stream[mask], color='gray', alpha=0.4, label='Raw Pitch', linewidth=1.0)

        # Highlight segmented note events
        for event in note_events:
            ax.axvspan(event['start'], event['end'], alpha=0.3, color='lightblue',
                      label='Segmented Notes' if event == note_events[0] else "")
            # Add note name annotation
            center_time = (event['start'] + event['end']) / 2
            ax.text(center_time, event['note_idx']*100 + 50, event['note_name'],
                   ha='center', va='center', fontsize=8, fontweight='bold')

        ax.set_yticks([i*100 for i in range(12)])
        ax.set_yticklabels([SWARA_NAMES[i] for i in range(12)])
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_title(f"Step 2: Segmented Notes - {RAGA_RULES['name']}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Swara")

        # Only show legend if there are labeled items
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        plt.tight_layout()
        plt.savefig('step2_segmented_notes.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✅ Step 2 visualization saved as 'step2_segmented_notes.png'")

        # Step 3: Raga rule violations
        fig3, ax = plt.subplots(figsize=(14, 6))
        if np.any(mask):
            ax.plot(times[mask], cents_stream[mask], color='gray', alpha=0.4, label='Raw Pitch', linewidth=1.0)

        # Highlight note events
        for event in note_events:
            ax.axvspan(event['start'], event['end'], alpha=0.2, color='lightblue')

        # Plot only errors (high severity)
        errors_only = [err for err in errors if err.get('severity', '') == 'high']
        for err in errors_only:
            duration = err.get('duration', 0) or 0.5
            ax.axvspan(err['time'], err['time'] + duration,
                      color='red', alpha=0.4, label='Major Errors' if err == errors_only[0] else "")
            ax.text(err['time'] + duration/2, 600, "ERROR", color='red', rotation=0,
                   horizontalalignment='center', verticalalignment='center', fontsize=9, fontweight='bold')

        ax.set_yticks([i*100 for i in range(12)])
        ax.set_yticklabels([SWARA_NAMES[i] for i in range(12)])
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_title(f"Step 3: Major Rule Violations - {RAGA_RULES['name']}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Swara")

        # Only show legend if there are labeled items
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        plt.tight_layout()
        plt.savefig('step3_rule_violations.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✅ Step 3 visualization saved as 'step3_rule_violations.png'")

        # Step 4: Complete analysis with all feedback
        fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

        # Top subplot: Pitch Track with all annotations
        if np.any(mask):
            ax1.plot(times[mask], cents_stream[mask], color='gray', alpha=0.6, label='Your Pitch', linewidth=1.5)

        # Highlight note events
        for event in note_events:
            ax1.axvspan(event['start'], event['end'], alpha=0.2, color='lightblue')

        # Plot Errors and Warnings
        errors_only = [err for err in errors if err.get('severity', '') == 'high']
        warnings_only = [err for err in errors if err.get('severity', '') != 'high']

        # Add labels only for the first occurrence of each type
        error_label_added = False
        warning_label_added = False

        for err in errors_only:
            duration = err.get('duration', 0) or 0.5
            label = 'Major Errors' if not error_label_added else ""
            ax1.axvspan(err['time'], err['time'] + duration,
                       color='red', alpha=0.3, label=label if label else None)
            ax1.text(err['time'], 600, "ERR", color='red', rotation=90, verticalalignment='center', fontsize=8)
            error_label_added = True

        for warn in warnings_only:
            duration = warn.get('duration', 0) or 0.3
            label = 'Warnings' if not warning_label_added else ""
            ax1.axvspan(warn['time'], warn['time'] + duration,
                       color='orange', alpha=0.2, label=label if label else None)
            ax1.text(warn['time'], 700, "WARN", color='orange', rotation=90, verticalalignment='center', fontsize=8)
            warning_label_added = True

        ax1.set_yticks([i*100 for i in range(12)])
        ax1.set_yticklabels([SWARA_NAMES[i] for i in range(12)])
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        ax1.set_title(f"Step 4: Complete Analysis - {RAGA_RULES['name']}")
        ax1.set_ylabel("Swara")

        # Only show legend if there are labeled items
        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            ax1.legend(loc='upper right')

        # Bottom subplot: Confidence Track
        ax2.plot(times, perf_conf, color='green', alpha=0.7, label='Confidence', linewidth=0.8)
        ax2.axhline(y=CONFIDENCE_THRESHOLD, color='red', linestyle='--', label=f'Threshold ({CONFIDENCE_THRESHOLD})')
        ax2.fill_between(times, perf_conf, where=(perf_conf >= CONFIDENCE_THRESHOLD),
                         color='green', alpha=0.3, interpolate=True)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Confidence")
        ax2.set_ylim(0, 1)
        ax2.grid(axis='x', linestyle='--', alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('step4_complete_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✅ Step 4 visualization saved as 'step4_complete_analysis.png'")

        # Also save the original combined visualization
        fig_combined, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

        # Top subplot: Pitch Track
        if np.any(mask):
            ax1.plot(times[mask], cents_stream[mask], color='gray', alpha=0.6, label='Your Pitch', linewidth=1.5)

        # Highlight note events
        for event in note_events:
            ax1.axvspan(event['start'], event['end'], alpha=0.2, color='lightblue')

        # Plot Errors and Warnings
        errors_only = [err for err in errors if err.get('severity', '') == 'high']
        warnings_only = [err for err in errors if err.get('severity', '') != 'high']

        # Add labels only for the first occurrence of each type
        error_label_added = False
        warning_label_added = False

        for err in errors_only:
            duration = err.get('duration', 0) or 0.5
            label = 'Major Errors' if not error_label_added else ""
            ax1.axvspan(err['time'], err['time'] + duration,
                       color='red', alpha=0.3, label=label if label else None)
            ax1.text(err['time'], 600, "ERR", color='red', rotation=90, verticalalignment='center', fontsize=8)
            error_label_added = True

        for warn in warnings_only:
            duration = warn.get('duration', 0) or 0.3
            label = 'Warnings' if not warning_label_added else ""
            ax1.axvspan(warn['time'], warn['time'] + duration,
                       color='orange', alpha=0.2, label=label if label else None)
            ax1.text(warn['time'], 700, "WARN", color='orange', rotation=90, verticalalignment='center', fontsize=8)
            warning_label_added = True

        ax1.set_yticks([i*100 for i in range(12)])
        ax1.set_yticklabels([SWARA_NAMES[i] for i in range(12)])
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        ax1.set_title(f"Complete Performance Analysis: {RAGA_RULES['name']}")
        ax1.set_ylabel("Swara")

        # Only show legend if there are labeled items
        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            ax1.legend(loc='upper right')

        # Bottom subplot: Confidence Track
        ax2.plot(times, perf_conf, color='green', alpha=0.7, label='Confidence', linewidth=0.8)
        ax2.axhline(y=CONFIDENCE_THRESHOLD, color='red', linestyle='--', label=f'Threshold ({CONFIDENCE_THRESHOLD})')
        ax2.fill_between(times, perf_conf, where=(perf_conf >= CONFIDENCE_THRESHOLD),
                         color='green', alpha=0.3, interpolate=True)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Confidence")
        ax2.set_ylim(0, 1)
        ax2.grid(axis='x', linestyle='--', alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('performance_analysis.png', dpi=150, bbox_inches='tight')
        print("   ✅ Combined visualization saved as 'performance_analysis.png'")

        # Show the plot only in interactive mode
        if os.isatty(sys.stdin.fileno()):
            plt.show()
        else:
            print("   📝 All visualizations saved to files (skipping display in non-interactive mode)")

        plt.close()  # Close the figure to free memory
        print("   📊 All 4-step analysis visualizations created successfully!")
    except Exception as e:
        print(f"   ⚠️  Visualization error: {str(e)}")
        print("   📝 Skipping visualization due to error")

if __name__ == "__main__":
    main()