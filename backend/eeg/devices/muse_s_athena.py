"""
Muse S Athena device adapter using OpenMuse.

Implements the EEGDeviceInterface contract using OpenMuse for BLE communication
and LSL streaming. OpenMuse handles the low-level BLE protocol while this class
provides the interface expected by the Synesthesia pipeline.

Installation:
    pip install https://github.com/DominiqueMakowski/OpenMuse/zipball/main
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.eeg.device_interface import EEGDeviceInterface

logger = get_logger(__name__)

# Channel definitions from OpenMuse
EEG_CHANNELS: Tuple[str, ...] = (
    "EEG_TP9",   # Left ear
    "EEG_AF7",   # Left forehead
    "EEG_AF8",   # Right forehead
    "EEG_TP10",  # Right ear
    "AUX_1",     # Auxiliary 1
    "AUX_2",     # Auxiliary 2
    "AUX_3",     # Auxiliary 3
    "AUX_4",     # Auxiliary 4
)

ACCGYRO_CHANNELS: Tuple[str, ...] = (
    "ACC_X", "ACC_Y", "ACC_Z",
    "GYRO_X", "GYRO_Y", "GYRO_Z",
)

OPTICS_CHANNELS: Tuple[str, ...] = (
    "OPTICS_LO_NIR", "OPTICS_RO_NIR", "OPTICS_LO_IR", "OPTICS_RO_IR",
    "OPTICS_LI_NIR", "OPTICS_RI_NIR", "OPTICS_LI_IR", "OPTICS_RI_IR",
    "OPTICS_LO_RED", "OPTICS_RO_RED", "OPTICS_LO_AMB", "OPTICS_RO_AMB",
    "OPTICS_LI_RED", "OPTICS_RI_RED", "OPTICS_LI_AMB", "OPTICS_RI_AMB",
)

# Preset configurations for OpenMuse
PRESETS = {
    "eeg_only": "p20",           # EEG4 + Motion + Battery
    "eeg_basic": "p1035",        # EEG4 + Optics4 + Motion + Battery
    "eeg_ppg": "p1045",          # EEG8 + Optics4 + Motion + Battery
    "full_research": "p1041",    # EEG8 + Optics16 + Motion + Battery (default)
}


class MuseStreamer:
    """
    Background thread manager for OpenMuse streaming.
    
    Handles the async event loop in a separate thread to allow
    synchronous control from the main application.
    """
    
    def __init__(self, address: str, preset: str = "p1041"):
        self.address = address
        self.preset = preset
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False
        
    def start(self) -> None:
        """Start streaming in a background thread."""
        if self._running:
            logger.warning("muse_streamer_already_running")
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._run_async, daemon=True)
        self._thread.start()
        logger.info("muse_streamer_started", address=self.address, preset=self.preset)
        
    def _run_async(self) -> None:
        """Run async streaming loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            from OpenMuse.stream import _stream_async
            
            self._task = self._loop.create_task(
                _stream_async(
                    address=self.address,
                    preset=self.preset,
                    duration=None,
                    raw_data_file=None,
                    verbose=True
                )
            )
            self._loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            logger.info("muse_streamer_cancelled")
        except ImportError as e:
            logger.error("openmuse_import_error", error=str(e))
            self._running = False
        except Exception as e:
            logger.error("muse_streamer_error", error=str(e))
            self._running = False
        finally:
            if self._loop and not self._loop.is_closed():
                self._loop.close()
            self._running = False
    
    def stop(self) -> None:
        """Stop streaming gracefully."""
        if not self._running:
            return
            
        if self._task and not self._task.done() and self._loop:
            self._loop.call_soon_threadsafe(self._task.cancel)
            
        if self._thread:
            self._thread.join(timeout=5.0)
            
        self._running = False
        logger.info("muse_streamer_stopped")
    
    @property
    def is_running(self) -> bool:
        return self._running


class MuseDataConsumer:
    """
    Consumes data from OpenMuse LSL streams.
    
    Connects to the LSL streams created by OpenMuse and provides
    methods to retrieve buffered data.
    """
    
    def __init__(self, buffer_size: float = 10.0):
        self.buffer_size = buffer_size
        self._streams: Dict[str, any] = {}
        self._connected = False
        
    def connect(self, timeout: float = 10.0) -> bool:
        """
        Connect to all available Muse LSL streams.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            True if at least EEG stream connected
        """
        try:
            from mne_lsl.stream import StreamLSL
        except ImportError as e:
            logger.error("mne_lsl_import_error", error=str(e))
            return False
        
        stream_names = ["Muse_EEG", "Muse_ACCGYRO", "Muse_OPTICS", "Muse_BATTERY"]
        
        for name in stream_names:
            try:
                stream = StreamLSL(bufsize=self.buffer_size, name=name)
                stream.connect(timeout=timeout)
                self._streams[name] = stream
                logger.info("lsl_stream_connected", stream_name=name)
            except Exception as e:
                logger.warning("lsl_stream_connect_failed", stream_name=name, error=str(e))
        
        self._connected = "Muse_EEG" in self._streams
        return self._connected
    
    def get_eeg_data(self, window: float = 1.0) -> Tuple[Optional[NDArray], Optional[NDArray]]:
        """
        Get recent EEG data.
        
        Args:
            window: Time window in seconds
            
        Returns:
            Tuple of (data, timestamps) where data shape is (n_channels, n_samples)
            or (None, None) if not available
        """
        if "Muse_EEG" not in self._streams:
            return None, None
        
        try:
            data, timestamps = self._streams["Muse_EEG"].get_data(winsize=window)
            return data, timestamps
        except Exception as e:
            logger.warning("eeg_data_read_error", error=str(e))
            return None, None
    
    def get_motion_data(self, window: float = 1.0) -> Tuple[Optional[NDArray], Optional[NDArray]]:
        """
        Get recent accelerometer/gyroscope data.
        
        Args:
            window: Time window in seconds
            
        Returns:
            Tuple of (data, timestamps) where data shape is (6, n_samples)
        """
        if "Muse_ACCGYRO" not in self._streams:
            return None, None
        
        try:
            data, timestamps = self._streams["Muse_ACCGYRO"].get_data(winsize=window)
            return data, timestamps
        except Exception as e:
            logger.warning("motion_data_read_error", error=str(e))
            return None, None
    
    def get_ppg_data(self, window: float = 1.0) -> Tuple[Optional[NDArray], Optional[NDArray]]:
        """
        Get recent optical (PPG/fNIRS) data.
        
        Args:
            window: Time window in seconds
            
        Returns:
            Tuple of (data, timestamps) where data shape is (16, n_samples)
        """
        if "Muse_OPTICS" not in self._streams:
            return None, None
        
        try:
            data, timestamps = self._streams["Muse_OPTICS"].get_data(winsize=window)
            return data, timestamps
        except Exception as e:
            logger.warning("ppg_data_read_error", error=str(e))
            return None, None
    
    def disconnect(self) -> None:
        """Disconnect all streams."""
        for name, stream in self._streams.items():
            try:
                stream.disconnect()
                logger.info("lsl_stream_disconnected", stream_name=name)
            except Exception as e:
                logger.warning("lsl_stream_disconnect_error", stream_name=name, error=str(e))
        self._streams.clear()
        self._connected = False
    
    @property
    def is_connected(self) -> bool:
        return self._connected


class MuseSAthenaDevice(EEGDeviceInterface):
    """
    Muse S Athena (Gen 3) multi-modal headset using OpenMuse.
    
    Streams EEG (4 on-head + 4 auxiliary), fNIRS/PPG (16 optical channels),
    and IMU (accelerometer + gyroscope). Uses BLE 5.3 transport via OpenMuse.
    
    Architecture:
        1. MuseStreamer runs OpenMuse in background thread, creating LSL streams
        2. MuseDataConsumer connects to LSL streams for data retrieval
        3. This class provides the EEGDeviceInterface for the Synesthesia pipeline
    """
    
    def __init__(
        self,
        address: Optional[str] = None,
        preset: str = "full_research",
        enable_fnirs: bool | None = None,
        enable_ppg: bool | None = None,
        enable_imu: bool | None = None,
        connection_timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        ble_name: Optional[str] = None,
        lsl_buffer_size: float = 10.0,
    ) -> None:
        """
        Initialize Muse S Athena device.
        
        Args:
            address: BLE MAC address (None for auto-discovery)
            preset: OpenMuse preset (eeg_only, eeg_basic, eeg_ppg, full_research)
            enable_fnirs: Enable fNIRS data (requires optics preset)
            enable_ppg: Enable PPG data (requires optics preset)
            enable_imu: Enable IMU data
            connection_timeout: BLE connection timeout in seconds
            max_retries: Maximum connection retry attempts
            ble_name: BLE device name filter for discovery
            lsl_buffer_size: LSL buffer size in seconds
        """
        self._address = address
        self._preset = PRESETS.get(preset, preset)  # Allow raw preset codes too
        self._connected = False
        self._streaming = False
        self._last_stream_start: Optional[float] = None
        
        # Configuration (fall back to settings defaults)
        self.enable_fnirs = settings.muse_enable_fnirs if enable_fnirs is None else enable_fnirs
        self.enable_ppg = settings.muse_enable_ppg if enable_ppg is None else enable_ppg
        self.enable_imu = settings.muse_enable_imu if enable_imu is None else enable_imu
        self.connection_timeout = connection_timeout or settings.muse_connection_timeout
        self.max_retries = max_retries or settings.muse_max_retries
        self.ble_name = ble_name or settings.muse_ble_name
        self._lsl_buffer_size = lsl_buffer_size
        
        # OpenMuse components
        self._streamer: Optional[MuseStreamer] = None
        self._consumer: Optional[MuseDataConsumer] = None
        
        # Internal sample buffer for read_samples compatibility
        self._sample_buffer: deque = deque(maxlen=2560)  # ~10 seconds at 256 Hz
        
        logger.info(
            "muse_athena_initialized",
            address=self._address,
            preset=self._preset,
            enable_fnirs=self.enable_fnirs,
            enable_ppg=self.enable_ppg,
            enable_imu=self.enable_imu,
            connection_timeout=self.connection_timeout,
            max_retries=self.max_retries,
            ble_name=self.ble_name
        )
    
    @staticmethod
    def find_devices(timeout: float = 10.0) -> List[Dict[str, str]]:
        """
        Scan for available Muse devices.
        
        Args:
            timeout: Scan timeout in seconds
            
        Returns:
            List of device dicts with 'name' and 'address' keys
        """
        try:
            from OpenMuse import find_devices
            devices = find_devices(timeout=timeout, verbose=True)
            logger.info("muse_devices_found", count=len(devices))
            return devices
        except ImportError as e:
            logger.error("openmuse_import_error", error=str(e))
            return []
        except Exception as e:
            logger.error("muse_device_scan_error", error=str(e))
            return []
    
    def connect(self) -> bool:
        """
        Connect to the Muse S Athena headset.
        
        If no address is specified, performs auto-discovery.
        
        Returns:
            True if connection successful
        """
        address = self._address
        
        # Auto-discover if no address provided
        if not address:
            logger.info("muse_auto_discovery_starting")
            devices = self.find_devices(timeout=self.connection_timeout)
            
            if not devices:
                logger.error("muse_no_devices_found")
                return False
            
            # Filter by name if specified
            if self.ble_name:
                devices = [d for d in devices if self.ble_name.lower() in d.get('name', '').lower()]
                if not devices:
                    logger.error("muse_device_name_not_found", ble_name=self.ble_name)
                    return False
            
            address = devices[0]['address']
            logger.info("muse_device_selected", address=address, name=devices[0].get('name'))
        
        self._address = address
        
        # Start OpenMuse streaming (creates LSL streams)
        for attempt in range(self.max_retries):
            try:
                self._streamer = MuseStreamer(address=address, preset=self._preset)
                self._streamer.start()
                
                # Wait for LSL streams to become available
                time.sleep(2.0)
                
                if self._streamer.is_running:
                    self._connected = True
                    logger.info("muse_athena_connected", address=address, preset=self._preset)
                    return True
                    
            except Exception as e:
                logger.warning(
                    "muse_connection_attempt_failed",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e)
                )
                if self._streamer:
                    self._streamer.stop()
                time.sleep(2.0)
        
        logger.error("muse_connection_failed", address=address)
        return False
    
    def disconnect(self) -> None:
        """Disconnect from the headset and stop all streams."""
        if self._streaming:
            self.stop_stream()
        
        if self._consumer:
            self._consumer.disconnect()
            self._consumer = None
        
        if self._streamer:
            self._streamer.stop()
            self._streamer = None
        
        self._connected = False
        logger.info("muse_athena_disconnected")
    
    def is_connected(self) -> bool:
        """Return current connection state."""
        return self._connected and (self._streamer is not None and self._streamer.is_running)
    
    def start_stream(self) -> None:
        """
        Start streaming data from the device.
        
        Connects to LSL streams created by OpenMuse.
        """
        if not self._connected:
            raise RuntimeError("Muse S Athena not connected")
        
        if self._streaming:
            logger.warning("muse_stream_already_active")
            return
        
        # Connect to LSL streams
        self._consumer = MuseDataConsumer(buffer_size=self._lsl_buffer_size)
        
        if not self._consumer.connect(timeout=self.connection_timeout):
            raise RuntimeError("Failed to connect to Muse LSL streams")
        
        self._streaming = True
        self._last_stream_start = time.time()
        
        logger.info(
            "muse_athena_stream_started",
            enable_fnirs=self.enable_fnirs,
            enable_ppg=self.enable_ppg,
            enable_imu=self.enable_imu
        )
    
    def stop_stream(self) -> None:
        """Stop streaming from the headset."""
        if not self._streaming:
            return
        
        duration = time.time() - self._last_stream_start if self._last_stream_start else 0
        
        if self._consumer:
            self._consumer.disconnect()
            self._consumer = None
        
        self._streaming = False
        self._sample_buffer.clear()
        
        logger.info("muse_athena_stream_stopped", duration_seconds=duration)
    
    def read_samples(self, n_samples: int = 1) -> Optional[NDArray[np.float64]]:
        """
        Read EEG samples from the device.
        
        Args:
            n_samples: Number of samples to read
            
        Returns:
            Array of shape (n_samples, n_channels) or None if no data available
        """
        if not self._streaming or not self._consumer:
            return None
        
        # Calculate window size needed
        window_seconds = max(0.1, n_samples / self.sampling_rate)
        
        # Get data from LSL stream
        data, timestamps = self._consumer.get_eeg_data(window=window_seconds)
        
        if data is None or data.shape[1] == 0:
            return None
        
        # Transpose from (n_channels, n_samples) to (n_samples, n_channels)
        data = data.T
        
        # Return requested number of samples (from end of buffer)
        if data.shape[0] >= n_samples:
            return data[-n_samples:, :]
        else:
            return data
    
    def get_eeg_data(self, window: float = 1.0) -> Tuple[Optional[NDArray], Optional[NDArray]]:
        """
        Get EEG data with timestamps.
        
        Args:
            window: Time window in seconds
            
        Returns:
            Tuple of (data, timestamps) where data shape is (n_channels, n_samples)
        """
        if not self._streaming or not self._consumer:
            return None, None
        return self._consumer.get_eeg_data(window=window)
    
    def get_motion_data(self, window: float = 1.0) -> Tuple[Optional[NDArray], Optional[NDArray]]:
        """
        Get accelerometer/gyroscope data with timestamps.
        
        Args:
            window: Time window in seconds
            
        Returns:
            Tuple of (data, timestamps) where data shape is (6, n_samples)
        """
        if not self._streaming or not self._consumer or not self.enable_imu:
            return None, None
        return self._consumer.get_motion_data(window=window)
    
    def get_ppg_data(self, window: float = 1.0) -> Tuple[Optional[NDArray], Optional[NDArray]]:
        """
        Get PPG/fNIRS optical data with timestamps.
        
        Args:
            window: Time window in seconds
            
        Returns:
            Tuple of (data, timestamps) where data shape is (16, n_samples)
        """
        if not self._streaming or not self._consumer or not (self.enable_ppg or self.enable_fnirs):
            return None, None
        return self._consumer.get_ppg_data(window=window)
    
    def get_info(self) -> Dict:
        """Return device metadata and current stream state."""
        return {
            "device_type": "muse_s_athena",
            "connected": self._connected,
            "streaming": self._streaming,
            "address": self._address,
            "preset": self._preset,
            "n_channels": self.n_channels,
            "sampling_rate": self.sampling_rate,
            "channels": self.channel_names,
            "modalities": {
                "eeg": {
                    "enabled": True,
                    "channels": list(EEG_CHANNELS),
                    "sampling_rate_hz": 256,
                    "resolution_bits": 14
                },
                "fnirs": {
                    "enabled": self.enable_fnirs,
                    "channels": list(OPTICS_CHANNELS),
                    "sampling_rate_hz": 64
                },
                "ppg": {
                    "enabled": self.enable_ppg,
                    "wavelengths": ["IR", "NIR", "RED"],
                    "sampling_rate_hz": 64,
                    "resolution_bits": 20
                },
                "imu": {
                    "enabled": self.enable_imu,
                    "channels": list(ACCGYRO_CHANNELS),
                    "sampling_rate_hz": 52
                }
            },
            "transport": {
                "ble_version": "5.3",
                "ble_name": self.ble_name
            },
            "lsl_streams": {
                "eeg": "Muse_EEG",
                "motion": "Muse_ACCGYRO",
                "optics": "Muse_OPTICS",
                "battery": "Muse_BATTERY"
            }
        }
    
    @property
    def sampling_rate(self) -> int:
        """EEG sampling rate (Hz)."""
        return 256
    
    @property
    def n_channels(self) -> int:
        """EEG channel count (4 on-head + 4 auxiliary)."""
        return 8
    
    @property
    def channel_names(self) -> list[str]:
        """EEG channel labels."""
        return list(EEG_CHANNELS)
