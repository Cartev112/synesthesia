# OpenMuse Integration Guide

## Overview

OpenMuse is a Python library for interfacing with the Muse S Athena EEG headband. This guide covers programmatic integration for streaming applications.

## Table of Contents

1. [Installation](#installation)
2. [Core Architecture](#core-architecture)
3. [Device Discovery](#device-discovery)
4. [LSL Streaming](#lsl-streaming)
5. [Data Processing](#data-processing)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)

---

## Installation

```bash
pip install https://github.com/DominiqueMakowski/OpenMuse/zipball/main
```

### Dependencies

- **bleak**: Bluetooth Low Energy (BLE) communication
- **mne-lsl**: Lab Streaming Layer for real-time data streaming
- **numpy**: Numerical operations
- **pandas**: Data manipulation
- **scipy**: Signal processing
- **vispy**: Real-time visualization (optional)

---

## Core Architecture

### Module Structure

```
OpenMuse/
├── find.py          # Device discovery
├── muse.py          # Device communication & control
├── decode.py        # Raw data parsing & decoding
├── stream.py        # LSL streaming implementation
├── process.py       # Signal processing utilities
├── view.py          # Real-time visualization
└── utils.py         # Helper functions
```

### Data Flow

```
Muse Device (BLE)
    ↓
Raw BLE Packets
    ↓
parse_message() → Decode packets
    ↓
make_timestamps() → Add timestamps
    ↓
LSL Streams (EEG, ACCGYRO, OPTICS, BATTERY)
    ↓
Your Application
```

---

## Device Discovery

### Finding Devices

```python
from OpenMuse import find_devices

# Scan for Muse devices (10 second timeout)
devices = find_devices(timeout=10, verbose=True)

# Extract device info
for device in devices:
    print(f"Name: {device['name']}")
    print(f"Address: {device['address']}")
```

### Manual Device Selection

```python
def select_device():
    """Interactive device selection"""
    devices = find_devices(timeout=10, verbose=False)
    
    if not devices:
        raise RuntimeError("No Muse devices found")
    
    if len(devices) == 1:
        return devices[0]['address']
    
    print("Multiple devices found:")
    for i, dev in enumerate(devices):
        print(f"{i+1}. {dev['name']} ({dev['address']})")
    
    choice = int(input("Select device: ")) - 1
    return devices[choice]['address']

# Usage
address = select_device()
```

---

## LSL Streaming

### Basic Streaming

```python
from OpenMuse import stream

# Stream with all channels (preset p1041)
stream(
    address="XX:XX:XX:XX:XX:XX",  # Your device MAC
    preset="p1041",                # All channels enabled
    duration=None,                 # Stream indefinitely
    record=False,                  # Don't save raw data
    verbose=True
)
```

### Streaming with Recording

```python
import asyncio
from datetime import datetime

# Option 1: Save to default timestamped file
stream(
    address="XX:XX:XX:XX:XX:XX",
    preset="p1041",
    record=True,  # Saves to rawdata_stream_TIMESTAMP.txt
    verbose=True
)

# Option 2: Custom filename
stream(
    address="XX:XX:XX:XX:XX:XX",
    preset="p1041",
    record="my_recording.txt",
    verbose=True
)
```

### Programmatic Streaming Control

```python
import asyncio
import threading
from OpenMuse.stream import _stream_async

class MuseStreamer:
    def __init__(self, address, preset="p1041"):
        self.address = address
        self.preset = preset
        self.loop = None
        self.thread = None
        self.task = None
        
    def start(self):
        """Start streaming in a background thread"""
        self.thread = threading.Thread(target=self._run_async)
        self.thread.daemon = True
        self.thread.start()
        
    def _run_async(self):
        """Run async streaming loop"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.task = self.loop.create_task(
                _stream_async(
                    address=self.address,
                    preset=self.preset,
                    duration=None,
                    raw_data_file=None,
                    verbose=True
                )
            )
            self.loop.run_until_complete(self.task)
        except asyncio.CancelledError:
            pass
        finally:
            self.loop.close()
    
    def stop(self):
        """Stop streaming gracefully"""
        if self.task and not self.task.done():
            self.loop.call_soon_threadsafe(self.task.cancel)
        if self.thread:
            self.thread.join(timeout=5.0)

# Usage
streamer = MuseStreamer("XX:XX:XX:XX:XX:XX")
streamer.start()

# ... your application runs ...

streamer.stop()
```

---

## Data Processing

### Consuming LSL Streams

```python
from mne_lsl.stream import StreamLSL
import numpy as np

class MuseDataConsumer:
    def __init__(self, buffer_size=10.0):
        self.buffer_size = buffer_size
        self.streams = {}
        
    def connect(self):
        """Connect to all available Muse streams"""
        stream_names = ["Muse_EEG", "Muse_ACCGYRO", "Muse_OPTICS", "Muse_BATTERY"]
        
        for name in stream_names:
            try:
                stream = StreamLSL(bufsize=self.buffer_size, name=name)
                stream.connect(timeout=5.0)
                self.streams[name] = stream
                print(f"Connected to {name}")
            except Exception as e:
                print(f"Could not connect to {name}: {e}")
    
    def get_eeg_data(self, window=1.0):
        """Get recent EEG data"""
        if "Muse_EEG" not in self.streams:
            return None, None
        
        stream = self.streams["Muse_EEG"]
        data, timestamps = stream.get_data(winsize=window)
        
        return data, timestamps  # Shape: (8 channels, n_samples)
    
    def get_motion_data(self, window=1.0):
        """Get recent accelerometer/gyroscope data"""
        if "Muse_ACCGYRO" not in self.streams:
            return None, None
        
        stream = self.streams["Muse_ACCGYRO"]
        data, timestamps = stream.get_data(winsize=window)
        
        return data, timestamps  # Shape: (6 channels, n_samples)
    
    def get_ppg_data(self, window=1.0):
        """Get recent optical (PPG) data"""
        if "Muse_OPTICS" not in self.streams:
            return None, None
        
        stream = self.streams["Muse_OPTICS"]
        data, timestamps = stream.get_data(winsize=window)
        
        return data, timestamps  # Shape: (16 channels, n_samples)
    
    def disconnect(self):
        """Disconnect all streams"""
        for stream in self.streams.values():
            stream.disconnect()

# Usage
consumer = MuseDataConsumer()
consumer.connect()

# Get 2 seconds of EEG data
eeg_data, eeg_times = consumer.get_eeg_data(window=2.0)
print(f"EEG shape: {eeg_data.shape}")  # (8, ~512)

consumer.disconnect()
```

### Channel Information

#### EEG Channels (256 Hz)
```python
EEG_CHANNELS = (
    "EEG_TP9",   # Left ear
    "EEG_AF7",   # Left forehead
    "EEG_AF8",   # Right forehead
    "EEG_TP10",  # Right ear
    "AUX_1",     # Auxiliary 1
    "AUX_2",     # Auxiliary 2
    "AUX_3",     # Auxiliary 3
    "AUX_4",     # Auxiliary 4
)
```

#### Motion Channels (52 Hz)
```python
ACCGYRO_CHANNELS = (
    "ACC_X",   # Accelerometer X-axis (±2g)
    "ACC_Y",   # Accelerometer Y-axis
    "ACC_Z",   # Accelerometer Z-axis
    "GYRO_X",  # Gyroscope X-axis (±250°/s)
    "GYRO_Y",  # Gyroscope Y-axis
    "GYRO_Z",  # Gyroscope Z-axis
)
```

#### Optical Channels (64 Hz)
```python
OPTICS_CHANNELS = (
    "OPTICS_LO_NIR",   # Left Outer - 730nm
    "OPTICS_RO_NIR",   # Right Outer - 730nm
    "OPTICS_LO_IR",    # Left Outer - 850nm
    "OPTICS_RO_IR",    # Right Outer - 850nm
    "OPTICS_LI_NIR",   # Left Inner - 730nm
    "OPTICS_RI_NIR",   # Right Inner - 730nm
    "OPTICS_LI_IR",    # Left Inner - 850nm
    "OPTICS_RI_IR",    # Right Inner - 850nm
    "OPTICS_LO_RED",   # Left Outer - 660nm
    "OPTICS_RO_RED",   # Right Outer - 660nm
    "OPTICS_LO_AMB",   # Left Outer - Ambient
    "OPTICS_RO_AMB",   # Right Outer - Ambient
    "OPTICS_LI_RED",   # Left Inner - 660nm
    "OPTICS_RI_RED",   # Right Inner - 660nm
    "OPTICS_LI_AMB",   # Left Inner - Ambient
    "OPTICS_RI_AMB",   # Right Inner - Ambient
)
```

---

## Advanced Features

### Signal Processing

#### EEG Preprocessing

```python
from OpenMuse.process import apply_butter_filter
import numpy as np

def preprocess_eeg(eeg_data, sampling_rate=256):
    """
    Apply bandpass filter to EEG data
    
    Args:
        eeg_data: Shape (n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Filtered data with same shape
    """
    # Bandpass filter: 0.5-50 Hz (typical EEG range)
    filtered = apply_butter_filter(
        eeg_data,
        cutoff=(0.5, 50.0),
        sampling_rate=sampling_rate,
        btype="band",
        order=4
    )
    
    return filtered

# Usage
consumer = MuseDataConsumer()
consumer.connect()

eeg_data, _ = consumer.get_eeg_data(window=5.0)
filtered_eeg = preprocess_eeg(eeg_data)
```

#### Motion Processing

```python
from OpenMuse.process import preprocess_accgyro
import pandas as pd

def process_motion(data, timestamps):
    """
    Extract motion features from accelerometer/gyroscope
    
    Returns:
        DataFrame with ACC_MAG, ACC_PITCH, ACC_ROLL, GYRO_MAG
    """
    # Convert to DataFrame
    df = pd.DataFrame(
        data.T,
        columns=["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"]
    )
    df["time"] = timestamps
    
    # Extract features
    features = preprocess_accgyro(df, sampling_rate=52, lp_cutoff=5.0)
    
    return features

# Usage
motion_data, motion_times = consumer.get_motion_data(window=10.0)
features = process_motion(motion_data, motion_times)
```

#### PPG Heart Rate Extraction

```python
from OpenMuse.process import preprocess_ppg
import pandas as pd

def extract_heart_rate(ppg_data, timestamps, sampling_rate=64):
    """
    Extract heart rate from PPG data
    
    Returns:
        BVP signal and quality metrics
    """
    # Convert to DataFrame with channel names
    df = pd.DataFrame(
        ppg_data.T,
        columns=[
            "OPTICS_LI_NIR", "OPTICS_RI_NIR", "OPTICS_LI_IR", "OPTICS_RI_IR",
            "OPTICS_LO_NIR", "OPTICS_RO_NIR", "OPTICS_LO_IR", "OPTICS_RO_IR",
            # ... add more channels as needed
        ]
    )
    df["time"] = timestamps
    
    # Extract BVP signal
    bvp, info = preprocess_ppg(
        df,
        sampling_rate=sampling_rate,
        hp_cutoff=0.5,
        lp_cutoff=8.0,
        verbose=True
    )
    
    return bvp, info

# Usage
ppg_data, ppg_times = consumer.get_ppg_data(window=30.0)
bvp_signal, quality_info = extract_heart_rate(ppg_data, ppg_times)
```

#### fNIRS Hemodynamics

```python
from OpenMuse.process import preprocess_fnirs
import pandas as pd

def extract_hemodynamics(optics_data, timestamps, sampling_rate=64):
    """
    Extract hemodynamic signals (HbO, HbR) from fNIRS data
    
    Returns:
        DataFrame with HbO, HbR, HbDiff, and SQI for each position
    """
    # Convert to DataFrame
    df = pd.DataFrame(optics_data.T, columns=OPTICS_CHANNELS[:optics_data.shape[0]])
    df["time"] = timestamps
    
    # Process fNIRS data
    hb_df, info = preprocess_fnirs(
        df,
        sampling_rate=sampling_rate,
        band_cutoff=(0.01, 0.1),  # Hemodynamic band
        separation=3.0,  # Source-detector separation in cm
        dpf=7.0,  # Differential pathlength factor
        verbose=True
    )
    
    return hb_df, info

# Usage
optics_data, optics_times = consumer.get_ppg_data(window=60.0)
hemodynamics, info = extract_hemodynamics(optics_data, optics_times)
```

### Real-Time Processing Pipeline

```python
import time
import numpy as np
from collections import deque

class RealtimeProcessor:
    def __init__(self, window_size=2.0, update_rate=10):
        self.consumer = MuseDataConsumer(buffer_size=window_size)
        self.window_size = window_size
        self.update_interval = 1.0 / update_rate
        self.running = False
        
        # Buffers for processed data
        self.eeg_buffer = deque(maxlen=1000)
        self.motion_buffer = deque(maxlen=1000)
        
    def start(self):
        """Start real-time processing"""
        self.consumer.connect()
        self.running = True
        
        while self.running:
            start_time = time.time()
            
            # Get and process EEG
            eeg_data, eeg_times = self.consumer.get_eeg_data(self.window_size)
            if eeg_data is not None and eeg_data.shape[1] > 0:
                processed_eeg = self.process_eeg(eeg_data, eeg_times)
                self.eeg_buffer.append(processed_eeg)
            
            # Get and process motion
            motion_data, motion_times = self.consumer.get_motion_data(self.window_size)
            if motion_data is not None and motion_data.shape[1] > 0:
                processed_motion = self.process_motion(motion_data, motion_times)
                self.motion_buffer.append(processed_motion)
            
            # Maintain update rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.update_interval - elapsed)
            time.sleep(sleep_time)
    
    def process_eeg(self, data, timestamps):
        """Process EEG data - implement your logic here"""
        # Example: compute band power
        from scipy import signal
        
        # Bandpass filter
        filtered = apply_butter_filter(data, (8, 12), 256, "band")
        
        # Compute average power in alpha band
        alpha_power = np.mean(filtered ** 2)
        
        return {
            "timestamp": timestamps[-1],
            "alpha_power": alpha_power,
            "channels": filtered[:, -1]  # Latest sample
        }
    
    def process_motion(self, data, timestamps):
        """Process motion data - implement your logic here"""
        # Example: detect movement
        acc_magnitude = np.sqrt(np.sum(data[:3, :] ** 2, axis=0))
        movement_detected = np.any(acc_magnitude > 1.5)
        
        return {
            "timestamp": timestamps[-1],
            "movement": movement_detected,
            "magnitude": acc_magnitude[-1]
        }
    
    def stop(self):
        """Stop processing"""
        self.running = False
        self.consumer.disconnect()

# Usage
processor = RealtimeProcessor(window_size=2.0, update_rate=10)
processor.start()  # Blocks until stopped
```

---

## Best Practices

### 1. Device Connection Management

```python
class MuseConnection:
    def __init__(self, address):
        self.address = address
        self.streamer = None
        self.consumer = None
        
    def __enter__(self):
        """Context manager entry"""
        # Start streaming
        self.streamer = MuseStreamer(self.address)
        self.streamer.start()
        
        # Wait for streams to be available
        time.sleep(2.0)
        
        # Connect consumer
        self.consumer = MuseDataConsumer()
        self.consumer.connect()
        
        return self.consumer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.consumer:
            self.consumer.disconnect()
        if self.streamer:
            self.streamer.stop()

# Usage with automatic cleanup
with MuseConnection("XX:XX:XX:XX:XX:XX") as muse:
    eeg_data, _ = muse.get_eeg_data(window=5.0)
    # Process data...
# Automatically disconnects on exit
```

### 2. Error Handling

```python
def safe_stream(address, preset="p1041", max_retries=3):
    """Stream with automatic retry on failure"""
    for attempt in range(max_retries):
        try:
            stream(
                address=address,
                preset=preset,
                duration=None,
                verbose=True
            )
            break  # Success
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("Max retries reached. Giving up.")
                raise
```

### 3. Preset Selection

```python
# Preset configurations
PRESETS = {
    "eeg_only": "p20",           # EEG4 + Motion + Battery
    "eeg_basic": "p1035",        # EEG4 + Optics4 + Motion + Battery
    "eeg_ppg": "p1045",          # EEG8 + Optics4 + Motion + Battery
    "full_research": "p1041",    # EEG8 + Optics16 + Motion + Battery (default)
}

def stream_with_preset(address, config="eeg_ppg"):
    """Stream with named preset configuration"""
    preset = PRESETS.get(config, "p1041")
    stream(address=address, preset=preset, verbose=True)
```

### 4. Data Quality Monitoring

```python
class QualityMonitor:
    def __init__(self, consumer):
        self.consumer = consumer
        
    def check_eeg_quality(self, window=2.0):
        """Check EEG signal quality"""
        eeg_data, _ = self.consumer.get_eeg_data(window)
        
        if eeg_data is None:
            return {"status": "no_data"}
        
        # Check for flat lines
        std_per_channel = np.std(eeg_data, axis=1)
        flat_channels = np.where(std_per_channel < 1.0)[0]
        
        # Check for saturation
        max_values = np.max(np.abs(eeg_data), axis=1)
        saturated = np.where(max_values > 900)[0]  # Near 1000 μV limit
        
        return {
            "status": "ok" if len(flat_channels) == 0 else "poor",
            "flat_channels": flat_channels.tolist(),
            "saturated_channels": saturated.tolist(),
            "std_values": std_per_channel.tolist()
        }
    
    def check_motion(self, window=2.0, threshold=0.5):
        """Detect excessive motion artifacts"""
        motion_data, _ = self.consumer.get_motion_data(window)
        
        if motion_data is None:
            return {"motion_detected": False}
        
        # Check accelerometer magnitude
        acc_magnitude = np.sqrt(np.sum(motion_data[:3, :] ** 2, axis=0))
        motion_detected = np.any(acc_magnitude > (1.0 + threshold))
        
        return {
            "motion_detected": motion_detected,
            "max_acceleration": float(np.max(acc_magnitude))
        }

# Usage
monitor = QualityMonitor(consumer)
eeg_quality = monitor.check_eeg_quality()
motion_check = monitor.check_motion()
```

### 5. Timestamp Synchronization

```python
def synchronize_streams(consumer, window=1.0):
    """Get synchronized data from all streams"""
    # Get all data
    eeg_data, eeg_times = consumer.get_eeg_data(window)
    motion_data, motion_times = consumer.get_motion_data(window)
    ppg_data, ppg_times = consumer.get_ppg_data(window)
    
    # Find common time range
    if all(t is not None for t in [eeg_times, motion_times, ppg_times]):
        start_time = max(eeg_times[0], motion_times[0], ppg_times[0])
        end_time = min(eeg_times[-1], motion_times[-1], ppg_times[-1])
        
        # Extract synchronized segments
        eeg_mask = (eeg_times >= start_time) & (eeg_times <= end_time)
        motion_mask = (motion_times >= start_time) & (motion_times <= end_time)
        ppg_mask = (ppg_times >= start_time) & (ppg_times <= end_time)
        
        return {
            "eeg": (eeg_data[:, eeg_mask], eeg_times[eeg_mask]),
            "motion": (motion_data[:, motion_mask], motion_times[motion_mask]),
            "ppg": (ppg_data[:, ppg_mask], ppg_times[ppg_mask]),
            "time_range": (start_time, end_time)
        }
    
    return None
```

---

## Complete Example: Real-Time Neurofeedback

```python
import numpy as np
from collections import deque

class NeurofeedbackSystem:
    def __init__(self, address):
        self.address = address
        self.streamer = None
        self.consumer = None
        self.running = False
        
        # Neurofeedback parameters
        self.alpha_history = deque(maxlen=100)
        self.beta_history = deque(maxlen=100)
        
    def start(self):
        """Initialize and start the system"""
        # Start streaming
        self.streamer = MuseStreamer(self.address)
        self.streamer.start()
        time.sleep(2)
        
        # Connect consumer
        self.consumer = MuseDataConsumer()
        self.consumer.connect()
        
        self.running = True
        self.run_feedback_loop()
    
    def compute_band_power(self, data, fs, band):
        """Compute power in frequency band"""
        from scipy import signal
        
        # Apply bandpass filter
        filtered = apply_butter_filter(data, band, fs, "band")
        
        # Compute power
        power = np.mean(filtered ** 2, axis=1)
        return np.mean(power)  # Average across channels
    
    def run_feedback_loop(self):
        """Main feedback loop"""
        while self.running:
            # Get 2 seconds of EEG data
            eeg_data, _ = self.consumer.get_eeg_data(window=2.0)
            
            if eeg_data is not None and eeg_data.shape[1] > 0:
                # Compute alpha (8-12 Hz) and beta (13-30 Hz) power
                alpha_power = self.compute_band_power(eeg_data, 256, (8, 12))
                beta_power = self.compute_band_power(eeg_data, 256, (13, 30))
                
                # Store history
                self.alpha_history.append(alpha_power)
                self.beta_history.append(beta_power)
                
                # Compute alpha/beta ratio
                if len(self.alpha_history) > 10:
                    alpha_avg = np.mean(self.alpha_history)
                    beta_avg = np.mean(self.beta_history)
                    ratio = alpha_avg / (beta_avg + 1e-10)
                    
                    # Generate feedback
                    self.provide_feedback(ratio)
            
            time.sleep(0.1)  # 10 Hz update rate
    
    def provide_feedback(self, alpha_beta_ratio):
        """Provide neurofeedback based on alpha/beta ratio"""
        # High alpha/beta ratio indicates relaxation
        if alpha_beta_ratio > 1.5:
            print(f"✓ Relaxed state (ratio: {alpha_beta_ratio:.2f})")
        elif alpha_beta_ratio < 0.8:
            print(f"⚠ Alert state (ratio: {alpha_beta_ratio:.2f})")
        else:
            print(f"○ Neutral state (ratio: {alpha_beta_ratio:.2f})")
    
    def stop(self):
        """Stop the system"""
        self.running = False
        if self.consumer:
            self.consumer.disconnect()
        if self.streamer:
            self.streamer.stop()

# Usage
system = NeurofeedbackSystem("XX:XX:XX:XX:XX:XX")
try:
    system.start()
except KeyboardInterrupt:
    system.stop()
```

---

## Troubleshooting

### Common Issues

1. **No devices found**
   - Ensure Muse is powered on (blue light visible)
   - Check Bluetooth is enabled
   - On Linux, may need sudo or proper permissions

2. **Connection timeouts**
   - Device may be paired with another application
   - Try power cycling the device
   - Increase timeout parameter

3. **Poor signal quality**
   - Ensure good electrode contact
   - Moisten electrodes slightly
   - Check for movement artifacts

4. **LSL streams not appearing**
   - Verify streaming is active in another terminal
   - Check firewall settings
   - Ensure mne-lsl is properly installed

---

## Additional Resources

- **GitHub**: [OpenMuse Repository](https://github.com/DominiqueMakowski/OpenMuse)
- **LSL Documentation**: [Lab Streaming Layer](https://labstreaminglayer.readthedocs.io/)
- **MNE-LSL**: [MNE-LSL Documentation](https://mne.tools/mne-lsl/)
- **Muse Device**: [Muse S Athena](https://choosemuse.com/products/muse-s-athena)

---

## License & Disclaimer

OpenMuse is NOT an official product of InteraXon Inc and is not affiliated with or endorsed by the company. It does not come with any warranty and should be considered experimental software developed for research purposes only.