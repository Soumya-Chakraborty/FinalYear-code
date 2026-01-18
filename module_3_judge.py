import numpy as np

# --- RAGA DEFINITION (The "Law") ---
# You can swap this dictionary for Bhairav, Bhoopali, etc.
RAGA_YAMAN = {
    "name": "Yaman",
    # Valid: Sa(0), Re(2), Ga(4), Tivra Ma(6), Pa(7), Dha(9), Ni(11)
    "allowed_swaras": [0, 2, 4, 6, 7, 9, 11],
    
    # Specific sequences that are strictly forbidden
    "forbidden_sequences": [
        # Format: [From_Note, To_Note]
        # Example: In Yaman, you rarely jump straight from Sa to Tivra Ma
        [0, 6], 
        # Example: In strict Yaman, ascent often avoids Sa -> Ga directly (usually Sa-Ni-Re-Ga)
        # We can add more here.
    ]
}

SWARA_MAP = {
    0: "Sa", 1: "re", 2: "Re", 3: "ga", 4: "Ga", 5: "ma", 
    6: "Ma", 7: "Pa", 8: "dha", 9: "Dha", 10: "ni", 11: "Ni"
}

def segment_notes(times, swara_indices):
    """
    Condenses a stream of 1000 detection points into a clean list of 'Note Events'.
    Example: [Sa, Sa, Sa, Re, Re] -> [('Sa', start, end), ('Re', start, end)]
    """
    events = []
    
    if len(swara_indices) == 0:
        return events

    current_note = swara_indices[0]
    start_time = times[0]
    
    for i in range(1, len(swara_indices)):
        note = swara_indices[i]
        time = times[i]
        
        # If the note changes, save the previous event and start a new one
        if note != current_note:
            # Only save if the note was held long enough (e.g., > 0.1s) to be real
            # (Filtering out transient glitches)
            if time - start_time > 0.15: 
                events.append({
                    "note_idx": current_note,
                    "note_name": SWARA_MAP[current_note],
                    "start": start_time,
                    "end": times[i-1]
                })
            
            current_note = note
            start_time = time
            
    # Append the final note
    events.append({
        "note_idx": current_note,
        "note_name": SWARA_MAP[current_note],
        "start": start_time,
        "end": times[-1]
    })
    
    return events

def check_raga_errors(note_events, raga_rules):
    """
    Scans the list of Note Events for violations.
    """
    errors = []
    
    # 1. Check for ANYA SWARA (Forbidden Notes)
    for event in note_events:
        if event["note_idx"] not in raga_rules["allowed_swaras"]:
            errors.append({
                "type": "Anya Swara (Forbidden Note)",
                "detail": f"You sang {event['note_name']} (Index {event['note_idx']}). Yaman only allows Tivra Ma (Ma), not Shuddha ma.",
                "timestamp": f"{event['start']:.2f}s - {event['end']:.2f}s"
            })

    # 2. Check for FORBIDDEN TRANSITIONS
    for i in range(len(note_events) - 1):
        current_note = note_events[i]["note_idx"]
        next_note = note_events[i+1]["note_idx"]
        
        transition = [current_note, next_note]
        
        if transition in raga_rules["forbidden_sequences"]:
            errors.append({
                "type": "Forbidden Transition",
                "detail": f"You moved from {SWARA_MAP[current_note]} directly to {SWARA_MAP[next_note]}.",
                "timestamp": f"{note_events[i]['end']:.2f}s"
            })
            
    return errors

# --- TESTING THE JUDGE ---
if __name__ == "__main__":
    # Mock Data: Time and Swara Indices (0=Sa, 5=ma, 6=Ma, etc.)
    # Scenario: Singer sings Sa -> Shuddha Ma (Error) -> Pa -> Tivra Ma
    mock_times = np.arange(0, 4.0, 0.01) # 4 seconds of audio
    
    # Create a stream of indices simulating the singer
    # 0-1s: Sa (0)
    # 1-2s: Shuddha Ma (5) <-- ERROR in Yaman!
    # 2-3s: Pa (7)
    # 3-4s: Sa (0) -> Tivra Ma (6) <-- ERROR (Forbidden jump)
    
    mock_indices = np.concatenate([
        np.full(100, 0), # Sa
        np.full(100, 5), # Shuddha Ma (Wrong!)
        np.full(100, 7), # Pa
        np.full(50, 0),  # Sa...
        np.full(50, 6)   # ...Direct jump to Tivra Ma (Wrong!)
    ])
    
    print(f"🔎 Analyzing Raga: {RAGA_YAMAN['name']}...")
    
    # 1. Segment the raw stream into events
    detected_events = segment_notes(mock_times, mock_indices)
    print(f"🎵 Detected {len(detected_events)} distinct phrases.")
    
    # 2. Run the Judge
    report = check_raga_errors(detected_events, RAGA_YAMAN)
    
    # 3. Print the Verdict
    print("\n" + "="*40)
    print("       RAGA ANALYSIS REPORT       ")
    print("="*40)
    
    if not report:
        print("✅ Perfect! No errors detected.")
    else:
        for i, err in enumerate(report, 1):
            print(f"{i}. [{err['type']}]")
            print(f"   ⏱ Time: {err['timestamp']}")
            print(f"   ⚠️ {err['detail']}")
            print("-" * 20)