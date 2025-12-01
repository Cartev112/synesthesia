"""
Test script to verify EEG simulator 10-second state rotation timing.
"""

import time
import asyncio
from backend.pipeline.realtime_pipeline import RealtimePipeline

async def test_rotation_timing():
    """Test that simulator rotates through states every 10 seconds."""
    
    print("Testing EEG Simulator 10-Second State Rotation")
    print("=" * 60)
    print("Expected pattern: RELAX -> NEUTRAL -> FOCUS -> RELAX...")
    print("Monitoring for 35 seconds (3.5 full cycles)")
    print("=" * 60)
    
    # Track state changes
    state_history = []
    last_dominant_state = None
    
    async def on_brain_state(brain_state: dict):
        """Callback to track brain state changes."""
        nonlocal last_dominant_state
        
        # Determine dominant state
        focus = brain_state.get('focus', 0)
        relax = brain_state.get('relax', 0)
        neutral = brain_state.get('neutral', 0)
        
        max_val = max(focus, relax, neutral)
        if focus == max_val:
            dominant = "FOCUS"
        elif relax == max_val:
            dominant = "RELAX"
        else:
            dominant = "NEUTRAL"
        
        # Log state changes
        if dominant != last_dominant_state:
            timestamp = time.time()
            state_history.append({
                'time': timestamp,
                'state': dominant,
                'focus': focus,
                'relax': relax,
                'neutral': neutral
            })
            last_dominant_state = dominant
    
    # Create pipeline
    pipeline = RealtimePipeline(
        sampling_rate=256,
        use_simulator=True,
        on_brain_state=on_brain_state
    )
    
    # Start pipeline
    await pipeline.start()
    
    # Monitor for 35 seconds
    start_time = time.time()
    try:
        while time.time() - start_time < 35:
            await asyncio.sleep(0.1)
    finally:
        await pipeline.stop()
    
    # Analyze results
    print("\n" + "=" * 60)
    print("STATE CHANGE HISTORY:")
    print("=" * 60)
    
    if len(state_history) == 0:
        print("❌ ERROR: No state changes detected!")
        return
    
    # Set reference time to first state change
    reference_time = state_history[0]['time']
    
    for i, entry in enumerate(state_history):
        elapsed = entry['time'] - reference_time
        state = entry['state']
        focus = entry['focus'] * 100
        relax = entry['relax'] * 100
        neutral = entry['neutral'] * 100
        
        print(f"\n{i+1}. {state:8s} at {elapsed:5.1f}s")
        print(f"   Focus: {focus:5.1f}% | Relax: {relax:5.1f}% | Neutral: {neutral:5.1f}%")
        
        if i > 0:
            time_since_last = entry['time'] - state_history[i-1]['time']
            print(f"   (Δt = {time_since_last:.1f}s from previous)")
    
    # Verify timing
    print("\n" + "=" * 60)
    print("TIMING VERIFICATION:")
    print("=" * 60)
    
    expected_states = ["RELAX", "NEUTRAL", "FOCUS", "RELAX", "NEUTRAL", "FOCUS"]
    
    # Check state sequence
    actual_states = [entry['state'] for entry in state_history]
    print(f"\nExpected sequence: {' -> '.join(expected_states[:len(actual_states)])}")
    print(f"Actual sequence:   {' -> '.join(actual_states)}")
    
    sequence_match = all(
        actual == expected 
        for actual, expected in zip(actual_states, expected_states)
    )
    
    if sequence_match:
        print("✓ State sequence is CORRECT")
    else:
        print("✗ State sequence is INCORRECT")
    
    # Check timing (should be ~10 seconds between changes)
    print("\nTiming between state changes:")
    timing_ok = True
    for i in range(1, len(state_history)):
        time_diff = state_history[i]['time'] - state_history[i-1]['time']
        # Allow 1 second tolerance (9-11 seconds)
        if 9.0 <= time_diff <= 11.0:
            status = "✓"
        else:
            status = "✗"
            timing_ok = False
        print(f"  {status} Change {i}: {time_diff:.1f}s")
    
    if timing_ok:
        print("\n✓ All timing intervals are within tolerance (9-11s)")
    else:
        print("\n✗ Some timing intervals are outside tolerance")
    
    # Overall result
    print("\n" + "=" * 60)
    if sequence_match and timing_ok:
        print("✅ TEST PASSED: State rotation is working correctly!")
    else:
        print("❌ TEST FAILED: Issues detected with state rotation")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_rotation_timing())
