"""
Visual parameter generator.

Maps brain states to visual parameters for geometric art generation.
"""

from typing import Dict, Optional
import numpy as np
import time

from backend.core.logging import get_logger

logger = get_logger(__name__)

# Default baseline for all visual parameters. Only a handful of fields will be
# mapped dynamically from brain-state input; the rest remain constant.
# NOTE: These defaults MUST match frontend/src/features/visualizer/ParameterControls.tsx
DEFAULT_VISUAL_PARAMS: Dict[str, float] = {
    'frequency_ratio_x': 3.0,
    'frequency_ratio_y': 2.0,
    'phase_offset': 0.0,
    'amplitude_x': 0.8,
    'amplitude_y': 0.8,
    'rotation_speed': 0.0,
    'num_harmonics': 5,
    'hue_base': 180.0,
    'saturation': 0.7,
    'brightness': 0.8,
    'color_cycle_speed': 0.2,
    'recursion_depth': 2,
    'point_density': 1024,
    'trail_length': 0.9,
    'distortion_amount': 0.1,
    'speed_multiplier': 1.0,
    'pulse_frequency': 1.0,
    'pulse_amplitude': 0.0,
    'damping_x': 0.03,
    'damping_y': 0.03,
    'num_epicycles': 5,
    'epicycle_decay': 0.7,
    'portal_symmetry': 6,
    'portal_radial_frequency': 6.0,
    'portal_angular_frequency': 2.0,
    'portal_warp': 0.15,
    'portal_spiral': -1.5,
    'portal_layers': 4,
    'portal_radius': 0.48,
    'portal_ripple': 0.2,
    'portal_depth_skew': 0.35
}

DEFAULT_BRAIN_STATE = {
    'focus': 0.5,
    'relax': 0.5,
    'neutral': 0.5,
    'asymmetry': 0.0,
    'stability': 0.5
}

MAPPED_KEYS = ['trail_length', 'rotation_speed', 'speed_multiplier', 'portal_layers']


class VisualParameterGenerator:
    """
    Maps brain state to visual parameters for geometric art.
    
    Only the following parameters are brain-state aware:
    - trail_length
    - rotation_speed
    - speed_multiplier
    - portal_layers
    
    All other parameters remain fixed at DEFAULT_VISUAL_PARAMS.
    """
    
    def __init__(self):
        """Initialize visual parameter generator."""
        self.last_params = None
        self.smoothing_factor = 0.2  # Smooth transitions (higher = more responsive)
        
        logger.info("visual_parameter_generator_initialized")
    
    def generate_params(self, brain_state: Dict[str, float]) -> Dict:
        """
        Generate visual parameters from brain state.
        
        Args:
            brain_state: Dictionary with brain state metrics
                - focus: 0-1
                - relax: 0-1
                - neutral: 0-1
                - hemispheric_asymmetry: -1 to 1
                - stability: 0-1 (optional)
                
        Returns:
            Dictionary of visual parameters
        """
        # Extract brain state values
        focus = brain_state.get('focus', 0.5)
        relax = brain_state.get('relax', 0.5)
        neutral = brain_state.get('neutral', 0.5)
        asymmetry = brain_state.get('hemispheric_asymmetry', 0.0)
        stability = brain_state.get('stability', 0.5)
        
        overrides = {
            # Trail length: relax = long trails (0.95), focus = short trails (0.7)
            # Wider range for more visible effect
            'trail_length': np.clip(
                0.8 + relax * 0.25 - focus * 0.2,
                0.6,
                0.95
            ),
            # Rotation speed: asymmetry controls direction, but also add subtle
            # rotation based on state difference for visual interest
            'rotation_speed': np.clip(
                asymmetry * 0.4 + (relax - focus) * 0.15,
                -0.5,
                0.5
            ),
            # Speed: focus = faster (1.4), relax = slower (0.5)
            'speed_multiplier': np.clip(0.5 + focus * 0.9, 0.5, 1.4),
            # Portal layers: more layers with higher focus/stability (3-7)
            'portal_layers': int(np.clip(3 + stability * 2 + focus * 2, 3, 7)),
            # Color cycle speed: focus = fast (0.5), neutral = medium (0.2), relax = slow (0.05)
            # Weighted blend of all three states
            'color_cycle_speed': np.clip(
                (focus * 0.5 + neutral * 0.2 + relax * 0.05) / max(focus + neutral + relax, 0.01),
                0.05,
                0.5
            )
        }
        
        new_params = self._build_param_set(
            overrides=overrides,
            brain_state={
                'focus': focus,
                'relax': relax,
                'neutral': neutral,
                'asymmetry': asymmetry,
                'stability': stability
            }
        )
        
        # Apply smoothing if we have previous parameters
        if self.last_params is not None:
            new_params = self._smooth_params(self.last_params, new_params)
        
        self.last_params = new_params
        
        logger.debug(
            "visual_params_generated",
            focus=focus,
            relax=relax,
            rotation=new_params['rotation_speed'],
            portal_layers=new_params['portal_layers']
        )
        
        return new_params
    
    def _smooth_params(self, old_params: Dict, new_params: Dict) -> Dict:
        """
        Smooth transition between parameter sets.
        
        Args:
            old_params: Previous parameters
            new_params: New parameters
            
        Returns:
            Smoothed parameters
        """
        smoothed = new_params.copy()
        alpha = self.smoothing_factor
        
        # Smooth numeric parameters that are brain-state driven
        for key in MAPPED_KEYS:
            if key in old_params and key in new_params:
                old_val = old_params[key]
                new_val = new_params[key]
                smoothed[key] = old_val * (1 - alpha) + new_val * alpha
        
        # Special handling for hue (circular)
        if 'hue_base' in old_params and 'hue_base' in new_params:
            old_hue = old_params['hue_base']
            new_hue = new_params['hue_base']
            
            # Handle wrap-around
            diff = new_hue - old_hue
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            
            smoothed['hue_base'] = (old_hue + diff * alpha) % 360
        
        return smoothed
    
    def set_smoothing(self, factor: float):
        """
        Set smoothing factor for parameter transitions.
        
        Args:
            factor: Smoothing factor (0=no smoothing, 1=maximum smoothing)
        """
        self.smoothing_factor = np.clip(factor, 0.0, 1.0)
        logger.info("smoothing_factor_updated", factor=self.smoothing_factor)
    
    def reset(self):
        """Reset generator state."""
        self.last_params = None
        logger.info("visual_generator_reset")
    
    def get_preset_params(self, preset_name: str) -> Dict:
        """
        Get predefined parameter presets.
        
        Args:
            preset_name: Name of preset
            
        Returns:
            Parameter dictionary
        """
        presets = {
            'calm': {
                'frequency_ratio_x': 2.0,
                'frequency_ratio_y': 1.0,
                'phase_offset': 0.0,
                'amplitude_x': 1.0,
                'amplitude_y': 1.0,
                'rotation_speed': 0.1,
                'num_harmonics': 3,
                'hue_base': 200,  # Blue
                'saturation': 0.5,
                'brightness': 0.8,
                'color_cycle_speed': 0.1,
                'recursion_depth': 1,
                'point_density': 500,
                'trail_length': 0.6,
                'distortion_amount': 0.05,
                'speed_multiplier': 0.5,
                'pulse_frequency': 0.5,
                'pulse_amplitude': 0.1,
                'damping_x': 0.02,
                'damping_y': 0.02,
                'num_epicycles': 3,
                'epicycle_decay': 0.7,
                'portal_symmetry': 6,
                'portal_radial_frequency': 4.0,
                'portal_angular_frequency': 1.2,
                'portal_warp': 0.25,
                'portal_spiral': 0.0,
                'portal_layers': 3,
                'portal_radius': 0.5,
                'portal_ripple': 0.2,
                'portal_depth_skew': 0.3
            },
            'energetic': {
                'frequency_ratio_x': 5.0,
                'frequency_ratio_y': 3.0,
                'phase_offset': 1.57,
                'amplitude_x': 1.0,
                'amplitude_y': 1.0,
                'rotation_speed': 0.4,
                'num_harmonics': 8,
                'hue_base': 30,  # Orange
                'saturation': 0.8,
                'brightness': 1.0,
                'color_cycle_speed': 0.3,
                'recursion_depth': 3,
                'point_density': 1500,
                'trail_length': 0.2,
                'distortion_amount': 0.2,
                'speed_multiplier': 1.2,
                'pulse_frequency': 1.5,
                'pulse_amplitude': 0.25,
                'damping_x': 0.04,
                'damping_y': 0.04,
                'num_epicycles': 8,
                'epicycle_decay': 0.5,
                'portal_symmetry': 10,
                'portal_radial_frequency': 7.0,
                'portal_angular_frequency': 2.8,
                'portal_warp': 0.65,
                'portal_spiral': 0.5,
                'portal_layers': 6,
                'portal_radius': 0.42,
                'portal_ripple': 0.32,
                'portal_depth_skew': 0.55
            },
            'meditative': {
                'frequency_ratio_x': 3.0,
                'frequency_ratio_y': 2.0,
                'phase_offset': 0.0,
                'amplitude_x': 0.9,
                'amplitude_y': 0.9,
                'rotation_speed': 0.05,
                'num_harmonics': 5,
                'hue_base': 280,  # Purple
                'saturation': 0.6,
                'brightness': 0.7,
                'color_cycle_speed': 0.05,
                'recursion_depth': 2,
                'point_density': 800,
                'trail_length': 0.7,
                'distortion_amount': 0.05,
                'speed_multiplier': 0.3,
                'pulse_frequency': 0.3,
                'pulse_amplitude': 0.15,
                'damping_x': 0.01,
                'damping_y': 0.01,
                'num_epicycles': 5,
                'epicycle_decay': 0.8,
                'portal_symmetry': 7,
                'portal_radial_frequency': 5.0,
                'portal_angular_frequency': 1.6,
                'portal_warp': 0.35,
                'portal_spiral': 0.1,
                'portal_layers': 4,
                'portal_radius': 0.6,
                'portal_ripple': 0.25,
                'portal_depth_skew': 0.4
            }
        }
        
        if preset_name not in presets:
            logger.warning("unknown_preset", preset=preset_name)
            preset_name = 'calm'
        
        return self._build_param_set(
            overrides=presets[preset_name],
            brain_state=DEFAULT_BRAIN_STATE.copy()
        )

    def _build_param_set(self, overrides: Dict[str, float], brain_state: Dict[str, float]) -> Dict:
        """Merge overrides with defaults and append metadata."""
        params = DEFAULT_VISUAL_PARAMS.copy()
        params.update(overrides)
        params['timestamp'] = time.time()
        params['brain_state'] = brain_state
        return params
