"""
Test script to verify EEG simulator state switching.
"""

import time
from backend.eeg.simulator import StreamingEEGSimulator
from backend.signal_processing import SignalPreprocessor, FeatureExtractor, CircularBuffer

def test_state_switching():
    """Test that simulator states produce distinct brain state values."""
    
    # Initialize components
    simulator = StreamingEEGSimulator(n_channels=8, sampling_rate=256)
    preprocessor = SignalPreprocessor(sampling_rate=256)
    extractor = FeatureExtractor(sampling_rate=256)
    buffer = CircularBuffer(n_channels=8, buffer_duration=2.0, sampling_rate=256)
    
    # Disable auto-variation for controlled testing
    simulator.auto_vary_states = False
    simulator.start_stream()
    
    states_to_test = ["neutral", "focus", "relax"]
    
    print("Testing EEG Simulator State Switching")
    print("=" * 60)
    
    for state in states_to_test:
        print(f"\n{state.upper()} State:")
        print("-" * 40)
        
        # Set state and wait for transition
        simulator.set_mental_state(state, intensity=1.0, transition_time=0.5)
        time.sleep(1.0)
        
        # Collect 1 second of data (recreate buffer to clear it)
        buffer = CircularBuffer(n_channels=8, buffer_duration=2.0, sampling_rate=256)
        for _ in range(8):  # 8 chunks of 32 samples = 256 samples = 1 second
            raw_data = simulator.read_samples(32)
            processed = preprocessor.process_chunk(raw_data)
            buffer.append(processed)
        
        # Extract features
        feature_window = buffer.get_latest(duration=1.0)
        features = extractor.extract_features(feature_window)
        
        # Also check what band powers the simulator is using
        sim_band_powers = simulator._get_current_band_powers()
        print(f"  Simulator Band Powers:")
        for band, power in sim_band_powers.items():
            print(f"    {band}: {power:.3f}")
        print(f"\n  Extracted Band Powers:")
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            print(f"    {band}: {features[f'{band}_power']:.3f}")
        
        # Apply normalization (same as pipeline - empirically calibrated)
        import numpy as np
        focus_raw = features['focus_metric']
        relax_raw = features['relax_metric']
        
        # Empirical baselines from measured values
        focus_baseline = 0.30
        relax_baseline = 2.5
        focus_range = 0.23
        relax_range = 1.5
        
        focus_deviation = (focus_raw - focus_baseline) / focus_range
        relax_deviation = (relax_raw - relax_baseline) / relax_range
        
        focus_score = 1.0 / (1.0 + np.exp(-focus_deviation * 3.0))
        relax_score = 1.0 / (1.0 + np.exp(-relax_deviation * 3.0))
        neutral_score = np.exp(-(focus_deviation**2 + relax_deviation**2))
        
        total = focus_score + relax_score + neutral_score
        focus_norm = focus_score / total
        relax_norm = relax_score / total
        neutral_norm = neutral_score / total
        
        # Display results
        print(f"  Raw Metrics:")
        print(f"    focus_metric:  {focus_raw:.3f}")
        print(f"    relax_metric:  {relax_raw:.3f}")
        print(f"\n  Normalized Brain State:")
        print(f"    focus:   {focus_norm:.3f} ({focus_norm*100:.1f}%)")
        print(f"    relax:   {relax_norm:.3f} ({relax_norm*100:.1f}%)")
        print(f"    neutral: {neutral_norm:.3f} ({neutral_norm*100:.1f}%)")
        
        # Determine dominant state
        max_val = max(focus_norm, relax_norm, neutral_norm)
        if focus_norm == max_val:
            dominant = "FOCUS"
        elif relax_norm == max_val:
            dominant = "RELAX"
        else:
            dominant = "NEUTRAL"
        
        print(f"\n  Dominant State: {dominant}")
        
        # Verify correct state is dominant
        expected_dominant = state.upper()
        if dominant == expected_dominant:
            print(f"  ✓ PASS: {dominant} is correctly dominant")
        else:
            print(f"  ✗ FAIL: Expected {expected_dominant}, got {dominant}")
    
    simulator.stop_stream()
    print("\n" + "=" * 60)
    print("Test complete!")

if __name__ == "__main__":
    test_state_switching()
