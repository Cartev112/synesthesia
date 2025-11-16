"""
HDF5 session recorder for storing full EEG data.
"""

from pathlib import Path
from typing import Dict

import h5py
import numpy as np
from numpy.typing import NDArray

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class SessionRecorder:
    """
    Records full session data to HDF5 format.
    
    Stores:
    - Raw EEG data
    - Timestamps
    - Brain state features
    - Metadata
    """
    
    def __init__(self, session_id: str, sampling_rate: int = 256, n_channels: int = 8) -> None:
        """
        Initialize session recorder.
        
        Args:
            session_id: Unique session identifier
            sampling_rate: EEG sampling rate in Hz
            n_channels: Number of EEG channels
        """
        self.session_id = session_id
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        
        # Create recordings directory if it doesn't exist
        recordings_dir = Path(settings.recording_dir)
        recordings_dir.mkdir(parents=True, exist_ok=True)
        
        # Create HDF5 file
        self.filename = recordings_dir / f"{session_id}.h5"
        self.file = h5py.File(str(self.filename), 'w')
        
        # Create datasets with chunking and compression
        self.eeg_data = self.file.create_dataset(
            'eeg_raw',
            shape=(0, n_channels),
            maxshape=(None, n_channels),
            chunks=(sampling_rate, n_channels),  # 1 second chunks
            dtype='float32',
            compression='gzip',
            compression_opts=4
        )
        
        self.timestamps = self.file.create_dataset(
            'timestamps',
            shape=(0,),
            maxshape=(None,),
            chunks=(sampling_rate,),
            dtype='float64',
            compression='gzip',
            compression_opts=4
        )
        
        self.brain_states = self.file.create_dataset(
            'brain_states',
            shape=(0, 8),  # 8 features: 5 band powers + focus + relax + asymmetry
            maxshape=(None, 8),
            chunks=(100, 8),
            dtype='float32',
            compression='gzip',
            compression_opts=4
        )
        
        self.brain_state_timestamps = self.file.create_dataset(
            'brain_state_timestamps',
            shape=(0,),
            maxshape=(None,),
            chunks=(100,),
            dtype='float64',
            compression='gzip',
            compression_opts=4
        )
        
        # Store metadata
        self.file.attrs['session_id'] = session_id
        self.file.attrs['sampling_rate'] = sampling_rate
        self.file.attrs['n_channels'] = n_channels
        self.file.attrs['channel_names'] = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2'][:n_channels]
        
        logger.info(
            "session_recorder_initialized",
            session_id=session_id,
            filename=str(self.filename)
        )
    
    def append_eeg(self, data: NDArray[np.float64], timestamp: float) -> None:
        """
        Append EEG samples to recording.
        
        Args:
            data: EEG data of shape (n_samples, n_channels)
            timestamp: Unix timestamp of first sample
        """
        n_new = data.shape[0]
        
        # Resize and append EEG data
        current_size = self.eeg_data.shape[0]
        self.eeg_data.resize(current_size + n_new, axis=0)
        self.eeg_data[current_size:] = data.astype(np.float32)
        
        # Generate timestamps for each sample
        sample_timestamps = timestamp + np.arange(n_new) / self.sampling_rate
        
        # Resize and append timestamps
        self.timestamps.resize(current_size + n_new, axis=0)
        self.timestamps[current_size:] = sample_timestamps
    
    def append_brain_state(self, state: Dict[str, float], timestamp: float) -> None:
        """
        Append brain state features to recording.
        
        Args:
            state: Dictionary of brain state features
            timestamp: Unix timestamp
        """
        # Extract features in consistent order
        features = np.array([
            state.get('delta_power', 0.0),
            state.get('theta_power', 0.0),
            state.get('alpha_power', 0.0),
            state.get('beta_power', 0.0),
            state.get('gamma_power', 0.0),
            state.get('focus_metric', 0.0),
            state.get('relax_metric', 0.0),
            state.get('hemispheric_asymmetry', 0.0)
        ], dtype=np.float32)
        
        # Resize and append
        current_size = self.brain_states.shape[0]
        self.brain_states.resize(current_size + 1, axis=0)
        self.brain_states[current_size] = features
        
        self.brain_state_timestamps.resize(current_size + 1, axis=0)
        self.brain_state_timestamps[current_size] = timestamp
    
    def set_metadata(self, key: str, value) -> None:
        """
        Set metadata attribute.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.file.attrs[key] = value
    
    def close(self) -> None:
        """Close the HDF5 file."""
        if self.file:
            self.file.close()
            logger.info(
                "session_recorder_closed",
                session_id=self.session_id,
                filename=str(self.filename),
                eeg_samples=self.eeg_data.shape[0],
                brain_state_samples=self.brain_states.shape[0]
            )
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()



