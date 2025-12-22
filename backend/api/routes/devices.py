"""
Device management API routes.

Provides endpoints for EEG device discovery, connection, and streaming control.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from backend.core.logging import get_logger
from backend.eeg.devices.muse_s_athena import MuseSAthenaDevice

logger = get_logger(__name__)

router = APIRouter(prefix="/devices")

# Global device instance (singleton pattern for device management)
_active_device: Optional[MuseSAthenaDevice] = None


# Request/Response Models
class DeviceInfo(BaseModel):
    """Device information response."""
    name: str
    address: str


class ScanRequest(BaseModel):
    """Device scan request."""
    timeout: float = Field(default=10.0, ge=1.0, le=60.0, description="Scan timeout in seconds")


class ScanResponse(BaseModel):
    """Device scan response."""
    devices: List[DeviceInfo]
    count: int


class ConnectRequest(BaseModel):
    """Device connection request."""
    address: Optional[str] = Field(default=None, description="BLE MAC address (None for auto-discovery)")
    preset: str = Field(default="full_research", description="OpenMuse preset")
    ble_name: Optional[str] = Field(default=None, description="BLE device name filter")


class ConnectResponse(BaseModel):
    """Device connection response."""
    connected: bool
    address: Optional[str]
    preset: str
    message: str


class StreamRequest(BaseModel):
    """Stream control request."""
    action: str = Field(..., pattern="^(start|stop)$", description="Stream action: start or stop")


class StreamResponse(BaseModel):
    """Stream control response."""
    streaming: bool
    message: str


class DeviceStatusResponse(BaseModel):
    """Device status response."""
    connected: bool
    streaming: bool
    address: Optional[str]
    preset: Optional[str]
    device_info: Optional[dict]


class DataRequest(BaseModel):
    """Data retrieval request."""
    window: float = Field(default=1.0, ge=0.1, le=30.0, description="Time window in seconds")
    modality: str = Field(default="eeg", pattern="^(eeg|motion|ppg)$", description="Data modality")


class DataResponse(BaseModel):
    """Data retrieval response."""
    modality: str
    shape: List[int]
    samples: int
    data: Optional[List[List[float]]]
    timestamps: Optional[List[float]]


@router.get("/muse/scan", response_model=ScanResponse)
async def scan_devices(timeout: float = 10.0):
    """
    Scan for available Muse devices.
    
    Args:
        timeout: Scan timeout in seconds (1-60)
    
    Returns:
        List of discovered devices with name and address
    """
    if timeout < 1.0 or timeout > 60.0:
        raise HTTPException(status_code=400, detail="Timeout must be between 1 and 60 seconds")
    
    logger.info("muse_device_scan_requested", timeout=timeout)
    
    try:
        devices = MuseSAthenaDevice.find_devices(timeout=timeout)
        device_list = [
            DeviceInfo(name=d.get("name", "Unknown"), address=d.get("address", ""))
            for d in devices
        ]
        
        return ScanResponse(devices=device_list, count=len(device_list))
    except Exception as e:
        logger.error("muse_scan_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Device scan failed: {str(e)}")


@router.post("/muse/connect", response_model=ConnectResponse)
async def connect_device(request: ConnectRequest):
    """
    Connect to a Muse S Athena device.
    
    If no address is provided, auto-discovery will be used.
    """
    global _active_device
    
    # Disconnect existing device if connected
    if _active_device is not None:
        try:
            _active_device.disconnect()
        except Exception:
            pass
        _active_device = None
    
    logger.info(
        "muse_connect_requested",
        address=request.address,
        preset=request.preset,
        ble_name=request.ble_name
    )
    
    try:
        device = MuseSAthenaDevice(
            address=request.address,
            preset=request.preset,
            ble_name=request.ble_name
        )
        
        if device.connect():
            _active_device = device
            return ConnectResponse(
                connected=True,
                address=device._address,
                preset=device._preset,
                message="Successfully connected to Muse S Athena"
            )
        else:
            return ConnectResponse(
                connected=False,
                address=None,
                preset=request.preset,
                message="Failed to connect to device"
            )
    except Exception as e:
        logger.error("muse_connect_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Connection failed: {str(e)}")


@router.post("/muse/disconnect")
async def disconnect_device():
    """
    Disconnect from the current Muse device.
    """
    global _active_device
    
    if _active_device is None:
        return {"disconnected": True, "message": "No device was connected"}
    
    try:
        _active_device.disconnect()
        _active_device = None
        logger.info("muse_disconnected")
        return {"disconnected": True, "message": "Device disconnected successfully"}
    except Exception as e:
        logger.error("muse_disconnect_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Disconnect failed: {str(e)}")


@router.post("/muse/stream", response_model=StreamResponse)
async def control_stream(request: StreamRequest):
    """
    Start or stop data streaming from the connected device.
    """
    global _active_device
    
    if _active_device is None:
        raise HTTPException(status_code=400, detail="No device connected")
    
    try:
        if request.action == "start":
            if not _active_device.is_connected():
                raise HTTPException(status_code=400, detail="Device not connected")
            
            _active_device.start_stream()
            return StreamResponse(streaming=True, message="Streaming started")
        
        else:  # stop
            _active_device.stop_stream()
            return StreamResponse(streaming=False, message="Streaming stopped")
            
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("muse_stream_error", action=request.action, error=str(e))
        raise HTTPException(status_code=500, detail=f"Stream control failed: {str(e)}")


@router.get("/muse/status", response_model=DeviceStatusResponse)
async def get_device_status():
    """
    Get current device connection and streaming status.
    """
    global _active_device
    
    if _active_device is None:
        return DeviceStatusResponse(
            connected=False,
            streaming=False,
            address=None,
            preset=None,
            device_info=None
        )
    
    return DeviceStatusResponse(
        connected=_active_device.is_connected(),
        streaming=_active_device._streaming,
        address=_active_device._address,
        preset=_active_device._preset,
        device_info=_active_device.get_info()
    )


@router.get("/muse/data", response_model=DataResponse)
async def get_device_data(window: float = 1.0, modality: str = "eeg"):
    """
    Get recent data from the streaming device.
    
    Args:
        window: Time window in seconds (0.1-30)
        modality: Data type - eeg, motion, or ppg
    
    Returns:
        Data array and timestamps
    """
    global _active_device
    
    if _active_device is None:
        raise HTTPException(status_code=400, detail="No device connected")
    
    if not _active_device._streaming:
        raise HTTPException(status_code=400, detail="Device not streaming")
    
    if window < 0.1 or window > 30.0:
        raise HTTPException(status_code=400, detail="Window must be between 0.1 and 30 seconds")
    
    try:
        if modality == "eeg":
            data, timestamps = _active_device.get_eeg_data(window=window)
        elif modality == "motion":
            data, timestamps = _active_device.get_motion_data(window=window)
        elif modality == "ppg":
            data, timestamps = _active_device.get_ppg_data(window=window)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown modality: {modality}")
        
        if data is None:
            return DataResponse(
                modality=modality,
                shape=[0, 0],
                samples=0,
                data=None,
                timestamps=None
            )
        
        return DataResponse(
            modality=modality,
            shape=list(data.shape),
            samples=data.shape[1] if len(data.shape) > 1 else data.shape[0],
            data=data.tolist(),
            timestamps=timestamps.tolist() if timestamps is not None else None
        )
        
    except Exception as e:
        logger.error("muse_data_error", modality=modality, error=str(e))
        raise HTTPException(status_code=500, detail=f"Data retrieval failed: {str(e)}")


@router.get("/muse/info")
async def get_device_info():
    """
    Get detailed device information and capabilities.
    """
    global _active_device
    
    if _active_device is None:
        # Return static device capabilities even when not connected
        return {
            "connected": False,
            "device_type": "muse_s_athena",
            "library": "OpenMuse",
            "capabilities": {
                "eeg": {
                    "channels": 8,
                    "sampling_rate_hz": 256,
                    "channel_names": ["EEG_TP9", "EEG_AF7", "EEG_AF8", "EEG_TP10", "AUX_1", "AUX_2", "AUX_3", "AUX_4"]
                },
                "motion": {
                    "channels": 6,
                    "sampling_rate_hz": 52,
                    "channel_names": ["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"]
                },
                "optics": {
                    "channels": 16,
                    "sampling_rate_hz": 64
                }
            },
            "presets": {
                "eeg_only": "p20",
                "eeg_basic": "p1035",
                "eeg_ppg": "p1045",
                "full_research": "p1041"
            }
        }
    
    return _active_device.get_info()
