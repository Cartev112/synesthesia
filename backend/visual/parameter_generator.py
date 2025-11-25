"""
Visual parameter generator.

Maps brain states to visual parameters for geometric art generation.
"""

from typing import Dict, Optional
import numpy as np
import time

from backend.core.logging import get_logger

logger = get_logger(__name__)


class VisualParameterGenerator:
    """
    Maps brain state to visual parameters for geometric art.
    
    Generates parameters for:
    - Lissajous curves
    - Harmonographs
    - Fourier epicycles
    
    Frontend uses these parameters to render visuals.
    """
    
    def __init__(self):
        """Initialize visual parameter generator."""
        self.last_params = None
        self.smoothing_factor = 0.1  # Smooth transitions
        
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
                - theta_power: 0-1 (optional)
                - alpha_power: 0-1 (optional)
                - beta_power: 0-1 (optional)
                - stability: 0-1 (optional)
                
        Returns:
            Dictionary of visual parameters
        """
        # Extract brain state values
        focus = brain_state.get('focus', 0.5)
        relax = brain_state.get('relax', 0.5)
        neutral = brain_state.get('neutral', 0.5)
        asymmetry = brain_state.get('hemispheric_asymmetry', 0.0)
        theta = brain_state.get('theta_power', 0.5)
        alpha = brain_state.get('alpha_power', 0.5)
        beta = brain_state.get('beta_power', 0.5)
        stability = brain_state.get('stability', 0.5)
        arousal = beta
        
        # Generate new parameters
        new_params = {
            'timestamp': time.time(),
            
            # Wave function parameters (Lissajous/Harmonograph)
            'frequency_ratio_x': 2.0 + focus * 4.0,  # 2-6
            'frequency_ratio_y': 1.0 + focus * 3.0,  # 1-4
            'phase_offset': asymmetry * np.pi,  # -π to π radians
            'amplitude_x': 0.8 + relax * 0.2,  # 0.8-1.0
            'amplitude_y': 0.8 + relax * 0.2,  # 0.8-1.0
            'rotation_speed': asymmetry * 0.5,  # -0.5 to 0.5 rad/sec
            'num_harmonics': int(3 + focus * 7),  # 3-10 harmonics
            
            # Color parameters (HSV)
            'hue_base': (theta * 360) % 360,  # 0-360 degrees
            'saturation': 0.5 + (focus * 0.3),  # 0.5-0.8
            'brightness': 0.7 + (relax * 0.3),  # 0.7-1.0
            'color_cycle_speed': 0.1 + (beta * 0.3),  # 0.1-0.4
            
            # Complexity parameters
            'recursion_depth': int(1 + focus * 3),  # 1-4 levels
            'point_density': int(500 + focus * 1500),  # 500-2000 points
            'trail_length': relax * 0.8,  # 0-0.8 (more trails when relaxed)
            'distortion_amount': (1 - relax) * 0.3,  # 0-0.3 (less when relaxed)
            
            # Animation parameters
            'speed_multiplier': 0.5 + (focus * 1.0),  # 0.5-1.5
            'pulse_frequency': 0.5 + (alpha * 1.5),  # 0.5-2.0 Hz
            'pulse_amplitude': 0.1 + (stability * 0.2),  # 0.1-0.3
            
            # Damping (for harmonograph)
            'damping_x': 0.01 + ((1 - stability) * 0.04),  # 0.01-0.05
            'damping_y': 0.01 + ((1 - stability) * 0.04),  # 0.01-0.05
            
            # Epicycle parameters
            'num_epicycles': int(3 + focus * 7),  # 3-10 circles
            'epicycle_decay': 0.5 + (relax * 0.3),  # 0.5-0.8

            # Hyperspace portal parameters (new algorithm)
            'portal_symmetry': int(5 + focus * 5 + arousal * 2),  # 5-12 spokes
            'portal_radial_frequency': 3.0 + focus * 6.0,  # 3-9
            'portal_angular_frequency': 1.0 + relax * 2.0,  # 1-3
            'portal_warp': np.clip(0.15 + arousal * 0.55 + (1 - relax) * 0.25, 0.1, 0.95),
            'portal_spiral': asymmetry * 0.9 + (focus - 0.5) * 0.6,  # left/right twist
            'portal_layers': int(3 + stability * 3 + focus * 1.5),  # 3-7 depth slices
            'portal_radius': np.clip(0.35 + relax * 0.35 - focus * 0.08, 0.28, 0.85),
            'portal_ripple': 0.12 + theta * 0.25 + alpha * 0.1,  # 0.12-0.47
            'portal_depth_skew': 0.25 + (1 - stability) * 0.4,  # 0.25-0.65
            
            # Brain state (for reference)
            'brain_state': {
                'focus': focus,
                'relax': relax,
                'neutral': neutral,
                'asymmetry': asymmetry,
                'stability': stability
            }
        }
        
        # Apply smoothing if we have previous parameters
        if self.last_params is not None:
            new_params = self._smooth_params(self.last_params, new_params)
        
        self.last_params = new_params
        
        logger.debug(
            "visual_params_generated",
            focus=focus,
            relax=relax,
            hue=new_params['hue_base'],
            complexity=new_params['num_harmonics']
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
        
        # Smooth numeric parameters
        numeric_keys = [
            'frequency_ratio_x', 'frequency_ratio_y', 'phase_offset',
            'amplitude_x', 'amplitude_y', 'rotation_speed',
            'hue_base', 'saturation', 'brightness', 'color_cycle_speed',
            'trail_length', 'distortion_amount', 'speed_multiplier',
            'pulse_frequency', 'pulse_amplitude', 'damping_x', 'damping_y',
            'epicycle_decay', 'portal_warp', 'portal_spiral', 'portal_radius',
            'portal_ripple', 'portal_depth_skew', 'portal_radial_frequency',
            'portal_angular_frequency'
        ]
        
        for key in numeric_keys:
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
        
        params = presets[preset_name].copy()
        params['timestamp'] = time.time()
        params['brain_state'] = {
            'focus': 0.5,
            'relax': 0.5,
            'neutral': 0.5,
            'asymmetry': 0.0,
            'stability': 0.5
        }
        
        return params
