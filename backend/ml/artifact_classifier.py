"""
Artifact classifier for real-time EEG artifact detection.

Detects artifacts such as:
- Eye blinks
- Muscle activity
- Movement artifacts
"""

from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from pathlib import Path

from backend.core.logging import get_logger

logger = get_logger(__name__)


class ArtifactCNN(nn.Module):
    """
    1D CNN for artifact detection.
    
    Architecture:
    - Input: (batch, channels=8, time_samples)
    - Conv1D layers to capture temporal patterns
    - Global average pooling
    - Binary classification output
    """
    
    def __init__(self, n_channels: int = 8, n_samples: int = 128):
        """
        Initialize artifact classifier CNN.
        
        Args:
            n_channels: Number of EEG channels
            n_samples: Number of time samples per window
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, time_samples)
            
        Returns:
            Artifact probability (batch, 1)
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x


class ArtifactClassifier:
    """
    Real-time artifact detection classifier.
    
    Uses a lightweight CNN for binary classification:
    - 0: Clean signal
    - 1: Artifact present
    """
    
    def __init__(
        self,
        n_channels: int = 8,
        window_samples: int = 128,
        threshold: float = 0.5,
        model_path: Optional[Path] = None
    ):
        """
        Initialize artifact classifier.
        
        Args:
            n_channels: Number of EEG channels
            window_samples: Number of samples per window (typically 0.5s at 256Hz)
            threshold: Classification threshold (0-1)
            model_path: Path to pre-trained model (optional)
        """
        self.n_channels = n_channels
        self.window_samples = window_samples
        self.threshold = threshold
        
        # Initialize model
        self.model = ArtifactCNN(n_channels=n_channels, n_samples=window_samples)
        self.model.eval()  # Set to evaluation mode
        
        # Track whether model is trained
        self.is_trained = False
        
        # Load pre-trained weights if available
        if model_path and model_path.exists():
            self._load_model(model_path)
            self.is_trained = True
            logger.info("artifact_classifier_loaded", model_path=str(model_path))
        else:
            logger.warning(
                "artifact_classifier_not_trained",
                message="Using untrained model. Artifact detection disabled."
            )
        
        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(
            "artifact_classifier_initialized",
            n_channels=n_channels,
            window_samples=window_samples,
            threshold=threshold,
            device=str(self.device)
        )
    
    def detect_artifact(
        self,
        eeg_window: NDArray[np.float64]
    ) -> Tuple[bool, float]:
        """
        Detect if artifact is present in EEG window.
        
        Args:
            eeg_window: EEG data of shape (n_channels, n_samples)
            
        Returns:
            is_artifact: True if artifact detected
            confidence: Artifact probability (0-1)
        """
        # Skip artifact detection if model is not trained
        # Untrained model produces random outputs that incorrectly block data
        if not self.is_trained:
            return False, 0.0
        
        # Validate input shape
        if eeg_window.shape != (self.n_channels, self.window_samples):
            logger.warning(
                "invalid_window_shape",
                expected=(self.n_channels, self.window_samples),
                received=eeg_window.shape
            )
            # Resize or pad if needed
            eeg_window = self._resize_window(eeg_window)
        
        # Convert to tensor
        x = torch.from_numpy(eeg_window).float().unsqueeze(0)  # Add batch dimension
        x = x.to(self.device)
        
        # Inference
        with torch.no_grad():
            artifact_prob = self.model(x).item()
        
        is_artifact = artifact_prob > self.threshold
        
        if is_artifact:
            logger.debug(
                "artifact_detected",
                probability=artifact_prob,
                threshold=self.threshold
            )
        
        return is_artifact, artifact_prob
    
    def _resize_window(
        self,
        window: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Resize or pad window to expected shape.
        
        Args:
            window: Input window
            
        Returns:
            Resized window of shape (n_channels, window_samples)
        """
        target_shape = (self.n_channels, self.window_samples)
        new_window = np.zeros(target_shape)
        
        # Copy what we can
        n_ch = min(window.shape[0], self.n_channels)
        n_samples = min(window.shape[1], self.window_samples)
        
        new_window[:n_ch, :n_samples] = window[:n_ch, :n_samples]
        
        return new_window
    
    def _load_model(self, model_path: Path):
        """Load pre-trained model weights."""
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            logger.info("model_loaded_successfully", path=str(model_path))
        except Exception as e:
            logger.error("model_load_failed", path=str(model_path), error=str(e))
            raise
    
    def save_model(self, model_path: Path):
        """
        Save model weights.
        
        Args:
            model_path: Path to save model
        """
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        logger.info("model_saved", path=str(model_path))
    
    def train_model(
        self,
        train_data: NDArray[np.float64],
        train_labels: NDArray[np.int64],
        val_data: Optional[NDArray[np.float64]] = None,
        val_labels: Optional[NDArray[np.int64]] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Train the artifact classifier.
        
        Args:
            train_data: Training data of shape (n_samples, n_channels, n_timepoints)
            train_labels: Training labels (0=clean, 1=artifact)
            val_data: Validation data (optional)
            val_labels: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        logger.info(
            "training_started",
            n_samples=len(train_data),
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Set model to training mode
        self.model.train()
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Convert to tensors
        train_data_tensor = torch.from_numpy(train_data).float()
        train_labels_tensor = torch.from_numpy(train_labels).float().unsqueeze(1)
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data_tensor[i:i+batch_size].to(self.device)
                batch_labels = train_labels_tensor[i:i+batch_size].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            
            # Validation
            if val_data is not None and val_labels is not None:
                val_loss, val_acc = self._validate(
                    val_data, val_labels, criterion
                )
                logger.info(
                    "epoch_complete",
                    epoch=epoch+1,
                    train_loss=avg_loss,
                    val_loss=val_loss,
                    val_accuracy=val_acc
                )
            else:
                logger.info(
                    "epoch_complete",
                    epoch=epoch+1,
                    train_loss=avg_loss
                )
        
        # Set back to evaluation mode
        self.model.eval()
        logger.info("training_complete")
    
    def _validate(
        self,
        val_data: NDArray[np.float64],
        val_labels: NDArray[np.int64],
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Validate model on validation set.
        
        Returns:
            val_loss: Validation loss
            val_accuracy: Validation accuracy
        """
        self.model.eval()
        
        val_data_tensor = torch.from_numpy(val_data).float().to(self.device)
        val_labels_tensor = torch.from_numpy(val_labels).float().unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(val_data_tensor)
            loss = criterion(outputs, val_labels_tensor)
            
            # Calculate accuracy
            predictions = (outputs > self.threshold).float()
            accuracy = (predictions == val_labels_tensor).float().mean().item()
        
        self.model.train()
        return loss.item(), accuracy


