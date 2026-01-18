import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# In a real app, this would be detected automatically from the first 2 seconds of singing.
# For now, let's hardcode a common Sa.
# Male typical: ~140Hz (C#3), Female typical: ~220Hz (A3)
USER_SA_HZ = 138.59  # C#3 (Example Tonic)

# The Standard 12 Swaras of Hindustani Music
SWARA_MAP = {
    0:  "Sa",
    1:  "re",  # Komal Re
    2:  "Re",  # Shuddha Re
    3:  "ga",  # Komal Ga
    4:  "Ga",  # Shuddha Ga
    5:  "ma",  # Shuddha Ma
    6:  "Ma",  # Tivra Ma
    7:  "Pa",
    8:  "dha", # Komal Dha
    9:  "Dha", # Shuddha Dha
    10: "ni",  # Komal Ni
    11: "Ni"   # Shuddha Ni
}

def hz_to_cents(frequency_array, sa_hz):
    """
    Converts a stream of Hz values to Cents relative to Sa.
    """
    # Avoid log(0) errors by replacing 0 or NaN with Sa (or handling them separately)
    safe_freq = np.nan_to_num(frequency_array, nan=sa_hz)
    
    # Calculate Cents
    cents = 1200 * np.log2(safe_freq / sa_hz)
    
    # Handle the fact that you might sing below Sa (Mandra Saptak) or above (Taar Saptak)
    # We normalize everything to one octave (0-1200) for note identification, 
    # but keep the original for graphing pitch accuracy.
    normalized_cents = cents % 1200
    
    return cents, normalized_cents

def quantize_to_swara(normalized_cents_array):
    """
    Maps cents to the nearest Swara index (0-11).
    Example: 190 cents -> Index 2 (Re), 210 cents -> Index 2 (Re)
    """
    # Each semitone is 100 cents apart.
    # We round to the nearest 100 to find the 'intended' note.
    # / 100 -> round -> cast to int -> modulo 12 (loops back after Ni)
    swara_indices = np.round(normalized_cents_array / 100).astype(int) % 12
    return swara_indices

def visualize_swaras(times, cents, swara_indices):
    plt.figure(figsize=(14, 6))
    
    # 1. Plot the actual singing path (Red Line)
    # We only plot valid pitches (not the NaNs from silence)
    mask = ~np.isnan(cents)
    plt.plot(times[mask], cents[mask], color='gray', alpha=0.5, label='Actual Singing (Micro-tones)')
    
    # 2. Plot the "Detected Notes" (Blue Dots)
    # We multiply indices by 100 to place them on the Cents Y-axis
    detected_pitch_centers = swara_indices[mask] * 100
    plt.scatter(times[mask], detected_pitch_centers, s=5, color='blue', label='Detected Swara Center')

    # Formatting
    plt.yticks(
        ticks=[i*100 for i in range(12)], 
        labels=[SWARA_MAP[i] for i in range(12)]
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Swara")
    plt.title("Visualizing Your Singing vs. The Notes Detected")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

# --- INTEGRATING WITH MODULE 1 ---
# (This simulates the data passing from the previous step)
# In a real run, you would import variables from module_1_ear.
if __name__ == "__main__":
    # 1. Mock Data (Replace this with real `pitches` from Module 1)
    # Let's pretend you sang: Sa -> Re -> Ga -> Sa
    mock_times = np.linspace(0, 4, 400) # 4 seconds
    # Create frequencies that slide smoothly: 138Hz -> 155Hz -> 174Hz -> 138Hz
    mock_pitches = np.concatenate([
        np.full(100, 138.59),           # Sa
        np.linspace(138.59, 155.5, 50), # Slide Sa->Re
        np.full(50, 155.56),            # Re
        np.full(100, 174.61),           # Ga
        np.linspace(174.61, 138.59, 100)# Slide Ga->Sa
    ])
    
    # 2. Convert to Cents
    raw_cents, norm_cents = hz_to_cents(mock_pitches, USER_SA_HZ)
    
    # 3. Identify Swaras
    detected_swaras = quantize_to_swara(norm_cents)
    
    # 4. Print Translation
    print("📝 Translated Singing Stream (Sample):")
    # Print every 20th data point to avoid spamming screen
    for i in range(0, len(detected_swaras), 20):
        note_name = SWARA_MAP[detected_swaras[i]]
        print(f"Time {mock_times[i]:.2f}s : {note_name} ({norm_cents[i]:.0f} cents)")

    # 5. Visualize
    visualize_swaras(mock_times, raw_cents, detected_swaras)