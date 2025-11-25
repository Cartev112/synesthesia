"""
Custom exceptions for Synesthesia BCI system.
"""


class SynesthesiaError(Exception):
    """Base exception for all Synesthesia errors."""
    
    def __init__(self, message: str, code: str = "SYNESTHESIA_ERROR") -> None:
        self.message = message
        self.code = code
        super().__init__(self.message)


class DeviceError(SynesthesiaError):
    """EEG device connection or reading errors."""
    
    def __init__(self, message: str) -> None:
        super().__init__(message, code="DEVICE_ERROR")


class CalibrationError(SynesthesiaError):
    """Calibration failed or insufficient data."""
    
    def __init__(self, message: str) -> None:
        super().__init__(message, code="CALIBRATION_ERROR")


class ProcessingError(SynesthesiaError):
    """Signal processing errors."""
    
    def __init__(self, message: str) -> None:
        super().__init__(message, code="PROCESSING_ERROR")


class SessionError(SynesthesiaError):
    """Session management errors."""
    
    def __init__(self, message: str) -> None:
        super().__init__(message, code="SESSION_ERROR")


class ModelError(SynesthesiaError):
    """Machine learning model errors."""
    
    def __init__(self, message: str) -> None:
        super().__init__(message, code="MODEL_ERROR")


class ValidationError(SynesthesiaError):
    """Data validation errors."""
    
    def __init__(self, message: str) -> None:
        super().__init__(message, code="VALIDATION_ERROR")





