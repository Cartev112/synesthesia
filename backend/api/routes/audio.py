"""
Audio configuration API endpoints.
"""

from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# Helper function to get audio engine from session
def get_audio_engine_from_session(session_id: Optional[str] = None):
    """
    Get audio engine from active session.
    
    Backend audio is now disabled; API is kept for compatibility.
    """
    raise HTTPException(status_code=503, detail="Backend audio engine is disabled; use frontend audio instead")


# Request/Response models
class SynthesizerConfig(BaseModel):
    """Synthesizer configuration."""
    type: str = Field(..., description="Synthesizer type")
    
    class Config:
        schema_extra = {
            "example": {
                "type": "fm"
            }
        }


class EffectConfig(BaseModel):
    """Effect configuration."""
    type: str = Field(..., description="Effect type")
    parameters: Optional[Dict] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "type": "reverb",
                "parameters": {"room_size": 0.7, "wet_level": 0.4}
            }
        }


class TrackConfig(BaseModel):
    """Track configuration."""
    name: str
    synthesizer: str
    volume: float = Field(0.8, ge=0.0, le=1.0)
    mute: bool = False
    solo: bool = False
    effects: List[str] = Field(default_factory=list)


class VolumeUpdate(BaseModel):
    """Volume update."""
    volume: float = Field(..., ge=0.0, le=1.0)


# Available synthesizers and effects
AVAILABLE_SYNTHESIZERS = [
    {
        "id": "sine",
        "name": "Sine Wave",
        "description": "Pure sine wave - clean, simple tone",
        "best_for": ["melody", "harmony"]
    },
    {
        "id": "square",
        "name": "Square Wave",
        "description": "Bright, hollow tone",
        "best_for": ["bass", "melody"]
    },
    {
        "id": "sawtooth",
        "name": "Sawtooth Wave",
        "description": "Bright, buzzy tone",
        "best_for": ["bass", "texture"]
    },
    {
        "id": "triangle",
        "name": "Triangle Wave",
        "description": "Mellow, soft tone",
        "best_for": ["texture", "harmony"]
    },
    {
        "id": "fm",
        "name": "FM Synthesis",
        "description": "Complex, evolving tones",
        "best_for": ["melody", "texture"]
    },
    {
        "id": "subtractive",
        "name": "Subtractive Synthesis",
        "description": "Rich, analog-style tones",
        "best_for": ["bass", "melody"]
    }
]

AVAILABLE_EFFECTS = [
    {
        "id": "reverb",
        "name": "Reverb",
        "description": "Spatial ambience",
        "parameters": {
            "room_size": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "damping": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "wet_level": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.3}
        }
    },
    {
        "id": "delay",
        "name": "Delay",
        "description": "Echo effect",
        "parameters": {
            "delay_time": {"type": "float", "min": 0.0, "max": 2.0, "default": 0.25},
            "feedback": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.4},
            "wet_level": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.3}
        }
    },
    {
        "id": "filter",
        "name": "Filter",
        "description": "Frequency filter",
        "parameters": {
            "filter_type": {"type": "string", "options": ["lowpass", "highpass", "bandpass"], "default": "lowpass"},
            "cutoff_freq": {"type": "float", "min": 20.0, "max": 20000.0, "default": 1000.0},
            "resonance": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.7}
        }
    },
    {
        "id": "compressor",
        "name": "Compressor",
        "description": "Dynamic range compression",
        "parameters": {
            "threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "ratio": {"type": "float", "min": 1.0, "max": 20.0, "default": 4.0}
        }
    }
]


@router.get("/synthesizers")
async def get_available_synthesizers():
    """Get list of available synthesizers."""
    return {
        "synthesizers": AVAILABLE_SYNTHESIZERS
    }


@router.get("/effects")
async def get_available_effects():
    """Get list of available effects."""
    return {
        "effects": AVAILABLE_EFFECTS
    }


@router.get("/tracks")
async def get_tracks():
    """
    Get all audio tracks configuration.
    
    Note: This would typically get from a session-specific audio engine.
    For now, returns default configuration.
    """
    # TODO: Get from session-specific audio engine
    default_tracks = [
        {
            "name": "bass",
            "synthesizer": "subtractive",
            "volume": 0.7,
            "mute": False,
            "solo": False,
            "effects": ["compressor"]
        },
        {
            "name": "harmony",
            "synthesizer": "sine",
            "volume": 0.5,
            "mute": False,
            "solo": False,
            "effects": ["reverb"]
        },
        {
            "name": "melody",
            "synthesizer": "fm",
            "volume": 0.6,
            "mute": False,
            "solo": False,
            "effects": ["reverb", "delay"]
        },
        {
            "name": "texture",
            "synthesizer": "triangle",
            "volume": 0.4,
            "mute": False,
            "solo": False,
            "effects": ["reverb", "filter"]
        }
    ]
    
    return {
        "tracks": default_tracks
    }


@router.put("/tracks/{track_name}/synthesizer")
async def update_track_synthesizer(
    track_name: str,
    config: SynthesizerConfig
):
    """
    Update synthesizer for a track.
    
    Args:
        track_name: Track name (bass, harmony, melody, texture)
        config: Synthesizer configuration
    """
    try:
        engine = get_audio_engine_from_session()
        engine.set_track_synthesizer(track_name, config.type)
        
        logger.info(
            "track_synthesizer_updated",
            track=track_name,
            synthesizer=config.type
        )
        
        return {
            "success": True,
            "track": track_name,
            "synthesizer": config.type
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("synthesizer_update_failed", track=track_name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tracks/{track_name}/effects")
async def add_track_effect(
    track_name: str,
    config: EffectConfig
):
    """
    Add an effect to a track.
    
    Args:
        track_name: Track name
        config: Effect configuration
    """
    try:
        engine = get_audio_engine_from_session()
        params = config.parameters or {}
        engine.add_track_effect(track_name, config.type, **params)
        
        logger.info(
            "track_effect_added",
            track=track_name,
            effect=config.type,
            parameters=params
        )
        
        return {
            "success": True,
            "track": track_name,
            "effect": config.type
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("effect_add_failed", track=track_name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tracks/{track_name}/effects/{effect_index}")
async def remove_track_effect(
    track_name: str,
    effect_index: int
):
    """
    Remove an effect from a track.
    
    Args:
        track_name: Track name
        effect_index: Effect index in chain
    """
    try:
        engine = get_audio_engine_from_session()
        track = engine.mixer.get_track(track_name)
        if track:
            track.remove_effect(effect_index)
            logger.info(
                "track_effect_removed",
                track=track_name,
                effect_index=effect_index
            )
        else:
            raise HTTPException(status_code=404, detail=f"Track {track_name} not found")
        
        return {
            "success": True,
            "track": track_name,
            "effect_index": effect_index
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("effect_remove_failed", track=track_name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/tracks/{track_name}/volume")
async def update_track_volume(
    track_name: str,
    update: VolumeUpdate
):
    """
    Update track volume.
    
    Args:
        track_name: Track name
        update: Volume update
    """
    try:
        engine = get_audio_engine_from_session()
        engine.mixer.set_track_volume(track_name, update.volume)
        
        logger.info(
            "track_volume_updated",
            track=track_name,
            volume=update.volume
        )
        
        return {
            "success": True,
            "track": track_name,
            "volume": update.volume
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("volume_update_failed", track=track_name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/tracks/{track_name}/mute")
async def toggle_track_mute(
    track_name: str,
    mute: bool
):
    """
    Mute or unmute a track.
    
    Args:
        track_name: Track name
        mute: Mute state
    """
    try:
        engine = get_audio_engine_from_session()
        engine.mixer.set_track_mute(track_name, mute)
        
        logger.info(
            "track_mute_toggled",
            track=track_name,
            mute=mute
        )
        
        return {
            "success": True,
            "track": track_name,
            "mute": mute
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("mute_toggle_failed", track=track_name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/tracks/{track_name}/solo")
async def toggle_track_solo(
    track_name: str,
    solo: bool
):
    """
    Solo or unsolo a track.
    
    Args:
        track_name: Track name
        solo: Solo state
    """
    try:
        engine = get_audio_engine_from_session()
        engine.mixer.set_track_solo(track_name, solo)
        
        logger.info(
            "track_solo_toggled",
            track=track_name,
            solo=solo
        )
        
        return {
            "success": True,
            "track": track_name,
            "solo": solo
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("solo_toggle_failed", track=track_name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/master/volume")
async def update_master_volume(update: VolumeUpdate):
    """
    Update master volume.
    
    Args:
        update: Volume update
    """
    try:
        engine = get_audio_engine_from_session()
        engine.mixer.set_master_volume(update.volume)
        
        logger.info("master_volume_updated", volume=update.volume)
        
        return {
            "success": True,
            "volume": update.volume
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("master_volume_update_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug/mixer-state")
async def get_mixer_state():
    """Get current mixer state for debugging."""
    try:
        engine = get_audio_engine_from_session()
        tracks = {}
        for name, track in engine.mixer.tracks.items():
            tracks[name] = {
                "volume": track.volume,
                "mute": track.mute,
                "solo": track.solo,
                "synthesizer": track.synthesizer_type
            }
        
        return {
            "master_volume": engine.mixer.master_volume,
            "tracks": tracks,
            "track_count": len(tracks)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("mixer_state_debug_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
