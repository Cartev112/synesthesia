"""
Test script to verify Phase 1 Week 1 setup.

Run this to check that all components are working:
    python scripts/test_setup.py
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from backend.core.config import settings
        print(f"✓ Config loaded (env: {settings.env})")
        
        from backend.core.logging import get_logger
        print("✓ Logging configured")
        
        from backend.eeg.simulator import EEGSimulator, StreamingEEGSimulator
        print("✓ EEG simulator imported")
        
        from backend.signal_processing import CircularBuffer, SignalPreprocessor, FeatureExtractor
        print("✓ Signal processing modules imported")
        
        from backend.data import User, Calibration, Session, SessionEvent
        print("✓ Database models imported")
        
        from backend.api.main import app
        print("✓ FastAPI app imported")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_eeg_simulator():
    """Test EEG simulator."""
    print("\nTesting EEG simulator...")
    
    try:
        from backend.eeg.simulator import EEGSimulator
        
        # Create simulator
        sim = EEGSimulator(n_channels=8, sampling_rate=256)
        
        # Generate sample
        sample = sim.generate_sample()
        assert sample.shape == (8,), f"Expected shape (8,), got {sample.shape}"
        print(f"✓ Generated single sample: {sample.shape}")
        
        # Generate chunk
        chunk = sim.generate_chunk(duration=1.0)
        assert chunk.shape == (256, 8), f"Expected shape (256, 8), got {chunk.shape}"
        print(f"✓ Generated 1s chunk: {chunk.shape}")
        
        # Test state changes
        sim.set_mental_state("focus", intensity=0.8)
        print("✓ Mental state set to 'focus'")
        
        # Generate with new state
        focused_chunk = sim.generate_chunk(duration=0.5)
        print(f"✓ Generated focused state chunk: {focused_chunk.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ EEG simulator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_signal_processing():
    """Test signal processing pipeline."""
    print("\nTesting signal processing...")
    
    try:
        import numpy as np
        from backend.eeg.simulator import EEGSimulator
        from backend.signal_processing import CircularBuffer, SignalPreprocessor, FeatureExtractor
        
        # Create simulator
        sim = EEGSimulator(n_channels=8, sampling_rate=256)
        data = sim.generate_chunk(duration=2.0)
        
        # Test buffer
        buffer = CircularBuffer(n_channels=8, buffer_duration=5.0, sampling_rate=256)
        buffer.append(data)
        latest = buffer.get_latest(duration=1.0)
        print(f"✓ Circular buffer: stored {data.shape[0]} samples, retrieved {latest.shape[0]}")
        
        # Test preprocessor
        preprocessor = SignalPreprocessor(sampling_rate=256)
        filtered = preprocessor.process_chunk(data)
        assert filtered.shape == data.shape
        print(f"✓ Preprocessor: filtered {filtered.shape[0]} samples")
        
        # Test feature extractor
        extractor = FeatureExtractor(sampling_rate=256, window_size=1.0)
        features = extractor.extract_features(data)
        print(f"✓ Feature extractor: extracted {len(features)} features")
        print(f"  - Focus metric: {features['focus_metric']:.3f}")
        print(f"  - Relax metric: {features['relax_metric']:.3f}")
        print(f"  - Alpha power: {features['alpha_power']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Signal processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration...")
    
    try:
        from backend.core.config import settings
        
        print(f"✓ App name: {settings.app_name}")
        print(f"✓ Version: {settings.app_version}")
        print(f"✓ Environment: {settings.env}")
        print(f"✓ EEG device: {settings.eeg_device_type}")
        print(f"✓ Sampling rate: {settings.eeg_sampling_rate} Hz")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Synesthesia BCI - Phase 1 Week 1 Setup Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_configuration()))
    results.append(("EEG Simulator", test_eeg_simulator()))
    results.append(("Signal Processing", test_signal_processing()))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! Setup is complete.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())



