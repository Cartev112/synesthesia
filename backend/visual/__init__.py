"""
Visual parameter generation for synesthesia BCI system.

Generates parameters for geometric art visualization based on brain states.
Backend generates parameters only - frontend renders the visuals.
"""

from backend.visual.parameter_generator import VisualParameterGenerator
from backend.visual.algorithms import (
    LissajousGenerator,
    HarmonographGenerator,
    LorenzAttractorGenerator,
    ReactionDiffusionGenerator
)

__all__ = [
    'VisualParameterGenerator',
    'LissajousGenerator',
    'HarmonographGenerator',
    'LorenzAttractorGenerator',
    'ReactionDiffusionGenerator'
]
