"""
Visual configuration API endpoints.
"""

from typing import Dict, List
from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.visual.algorithms import VisualAlgorithmFactory
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# Request/Response models
class VisualPresetRequest(BaseModel):
    """Visual preset request."""
    preset_name: str = Field(..., description="Preset name")
    
    class Config:
        schema_extra = {
            "example": {
                "preset_name": "calm"
            }
        }


class AlgorithmRequest(BaseModel):
    """Algorithm selection request."""
    algorithm_type: str = Field(..., description="Algorithm type")
    
    class Config:
        schema_extra = {
            "example": {
                "algorithm_type": "lissajous"
            }
        }


class PointGenerationRequest(BaseModel):
    """Point generation request."""
    algorithm_type: str = Field(..., description="Algorithm type")
    parameters: Dict = Field(..., description="Visual parameters")
    num_points: int = Field(1000, description="Number of points")
    duration: float = Field(10.0, description="Duration in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "algorithm_type": "lissajous",
                "parameters": {
                    "frequency_ratio_x": 3.0,
                    "frequency_ratio_y": 2.0,
                    "phase_offset": 0.0
                },
                "num_points": 1000,
                "duration": 10.0
            }
        }


@router.get("/algorithms")
async def get_available_algorithms():
    """
    Get list of available visual algorithms.
    
    Returns:
        List of algorithm descriptions
    """
    algorithms = VisualAlgorithmFactory.get_available_algorithms()
    
    return {
        "algorithms": algorithms
    }


@router.get("/presets")
async def get_available_presets():
    """
    Get list of available visual presets.
    
    Returns:
        List of preset descriptions
    """
    presets = [
        {
            "id": "calm",
            "name": "Calm",
            "description": "Slow, simple patterns for relaxation",
            "hue": 200,  # Blue
            "complexity": "low"
        },
        {
            "id": "energetic",
            "name": "Energetic",
            "description": "Fast, complex patterns for focus",
            "hue": 30,  # Orange
            "complexity": "high"
        },
        {
            "id": "meditative",
            "name": "Meditative",
            "description": "Flowing patterns for meditation",
            "hue": 280,  # Purple
            "complexity": "medium"
        }
    ]
    
    visual_algorithms = [
        {
            "id": "lissajous",
            "name": "Lissajous Curves",
            "description": "Simple parametric curves",
            "complexity": "low"
        },
        {
            "id": "harmonograph",
            "name": "Harmonograph",
            "description": "Multiple damped pendulum simulation",
            "complexity": "medium"
        },
        {
            "id": "lorenz",
            "name": "Lorenz Attractor",
            "description": "Chaotic strange attractor system",
            "complexity": "high"
        },
        {
            "id": "reaction_diffusion",
            "name": "Reaction-Diffusion",
            "description": "Organic patterns from Gray-Scott model",
            "complexity": "high"
        },
        {
            "id": "hyperspace_portal",
            "name": "Hyperspace Portal",
            "description": "Layered radial waves with spiral warp",
            "complexity": "high"
        }
    ]
    
    return {
        "presets": presets,
        "visual_algorithms": visual_algorithms
    }


@router.post("/preset")
async def get_preset_parameters(request: VisualPresetRequest):
    """
    Get parameters for a visual preset.
    
    Args:
        request: Preset request
        
    Returns:
        Visual parameters
    """
    from backend.visual import VisualParameterGenerator
    
    generator = VisualParameterGenerator()
    params = generator.get_preset_params(request.preset_name)
    
    logger.info("preset_parameters_requested", preset=request.preset_name)
    
    return {
        "preset": request.preset_name,
        "parameters": params
    }


@router.post("/generate_points")
async def generate_visual_points(request: PointGenerationRequest):
    """
    Generate visual points for an algorithm.
    
    Args:
        request: Point generation request
        
    Returns:
        List of (x, y) points
    """
    # Create algorithm generator
    generator = VisualAlgorithmFactory.create(request.algorithm_type)
    
    # Generate points
    points = generator.generate_points(
        params=request.parameters,
        num_points=request.num_points,
        duration=request.duration
    )
    
    # Get formula
    formula = generator.generate_formula(request.parameters)
    
    logger.info(
        "visual_points_generated",
        algorithm=request.algorithm_type,
        num_points=len(points)
    )
    
    return {
        "algorithm": request.algorithm_type,
        "num_points": len(points),
        "points": points,
        "formula": formula
    }


@router.post("/formula")
async def get_algorithm_formula(request: AlgorithmRequest):
    """
    Get mathematical formula for an algorithm.
    
    Args:
        request: Algorithm request
        
    Returns:
        Formula strings
    """
    # Create algorithm generator
    generator = VisualAlgorithmFactory.create(request.algorithm_type)
    
    # Get formula with default parameters
    default_params = {
        'frequency_ratio_x': 3.0,
        'frequency_ratio_y': 2.0,
        'phase_offset': 0.0,
        'num_harmonics': 4,
        'damping_x': 0.02,
        'damping_y': 0.02,
        'num_epicycles': 5,
        'epicycle_decay': 0.7,
        'rotation_speed': 0.5
    }
    
    formula = generator.generate_formula(default_params)
    
    return {
        "algorithm": request.algorithm_type,
        "formula": formula
    }


@router.get("/parameter_ranges")
async def get_parameter_ranges():
    """
    Get valid ranges for visual parameters.
    
    Returns:
        Parameter ranges and descriptions
    """
    ranges = {
        "frequency_ratio_x": {
            "min": 1.0,
            "max": 10.0,
            "default": 3.0,
            "description": "X-axis frequency ratio"
        },
        "frequency_ratio_y": {
            "min": 1.0,
            "max": 10.0,
            "default": 2.0,
            "description": "Y-axis frequency ratio"
        },
        "phase_offset": {
            "min": -3.14159,
            "max": 3.14159,
            "default": 0.0,
            "description": "Phase offset in radians"
        },
        "amplitude_x": {
            "min": 0.1,
            "max": 1.0,
            "default": 1.0,
            "description": "X-axis amplitude"
        },
        "amplitude_y": {
            "min": 0.1,
            "max": 1.0,
            "default": 1.0,
            "description": "Y-axis amplitude"
        },
        "rotation_speed": {
            "min": -1.0,
            "max": 1.0,
            "default": 0.0,
            "description": "Rotation speed in rad/sec"
        },
        "num_harmonics": {
            "min": 1,
            "max": 10,
            "default": 4,
            "description": "Number of harmonics"
        },
        "hue_base": {
            "min": 0,
            "max": 360,
            "default": 180,
            "description": "Base hue in HSV (degrees)"
        },
        "saturation": {
            "min": 0.0,
            "max": 1.0,
            "default": 0.8,
            "description": "Color saturation"
        },
        "brightness": {
            "min": 0.0,
            "max": 1.0,
            "default": 0.9,
            "description": "Color brightness"
        },
        "color_cycle_speed": {
            "min": 0.0,
            "max": 1.0,
            "default": 0.2,
            "description": "Color cycling speed"
        },
        "point_density": {
            "min": 100,
            "max": 5000,
            "default": 1000,
            "description": "Number of points to render"
        },
        "trail_length": {
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
            "description": "Trail fade amount"
        },
        "distortion_amount": {
            "min": 0.0,
            "max": 0.5,
            "default": 0.1,
            "description": "Distortion effect amount"
        }
    }
    
    return {
        "parameters": ranges
    }
