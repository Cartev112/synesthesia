"""
Visual algorithm implementations.

Provides mathematical formulas for geometric art generation.
These generate point data that frontend can render.
"""

from typing import Dict, List, Tuple
import numpy as np

from backend.core.logging import get_logger

logger = get_logger(__name__)


class LissajousGenerator:
    """
    Lissajous curve generator.
    
    Generates parametric curves:
    x(t) = A_x * sin(ω_x * t + φ)
    y(t) = A_y * sin(ω_y * t)
    """
    
    def __init__(self):
        """Initialize Lissajous generator."""
        pass
    
    def generate_points(
        self,
        params: Dict,
        num_points: int = 1000,
        duration: float = 10.0
    ) -> List[Tuple[float, float]]:
        """
        Generate Lissajous curve points.
        
        Args:
            params: Visual parameters
            num_points: Number of points to generate
            duration: Time duration in seconds
            
        Returns:
            List of (x, y) coordinate tuples
        """
        # Extract parameters
        freq_x = params.get('frequency_ratio_x', 3.0)
        freq_y = params.get('frequency_ratio_y', 2.0)
        phase = params.get('phase_offset', 0.0)
        amp_x = params.get('amplitude_x', 1.0)
        amp_y = params.get('amplitude_y', 1.0)
        
        # Generate time points
        t = np.linspace(0, duration, num_points)
        
        # Calculate positions
        x = amp_x * np.sin(freq_x * t + phase)
        y = amp_y * np.sin(freq_y * t)
        
        # Convert to list of tuples
        points = [(float(x[i]), float(y[i])) for i in range(num_points)]
        
        return points
    
    def generate_formula(self, params: Dict) -> Dict[str, str]:
        """
        Generate formula strings for frontend.
        
        Args:
            params: Visual parameters
            
        Returns:
            Dictionary with formula strings
        """
        freq_x = params.get('frequency_ratio_x', 3.0)
        freq_y = params.get('frequency_ratio_y', 2.0)
        phase = params.get('phase_offset', 0.0)
        amp_x = params.get('amplitude_x', 1.0)
        amp_y = params.get('amplitude_y', 1.0)
        
        return {
            'x': f"{amp_x:.2f} * sin({freq_x:.2f} * t + {phase:.2f})",
            'y': f"{amp_y:.2f} * sin({freq_y:.2f} * t)",
            'type': 'lissajous'
        }


class HarmonographGenerator:
    """
    Harmonograph generator (multiple pendulum simulation).
    
    Generates curves with damping:
    x(t) = Σ A_i * sin(ω_i * t + φ_i) * exp(-d_i * t)
    y(t) = Σ B_i * sin(ν_i * t + ψ_i) * exp(-e_i * t)
    """
    
    def __init__(self):
        """Initialize harmonograph generator."""
        pass
    
    def generate_points(
        self,
        params: Dict,
        num_points: int = 2000,
        duration: float = 20.0
    ) -> List[Tuple[float, float]]:
        """
        Generate harmonograph points.
        
        Args:
            params: Visual parameters
            num_points: Number of points to generate
            duration: Time duration in seconds
            
        Returns:
            List of (x, y) coordinate tuples
        """
        # Extract parameters
        num_harmonics = params.get('num_harmonics', 4)
        damping_x = params.get('damping_x', 0.02)
        damping_y = params.get('damping_y', 0.02)
        phase = params.get('phase_offset', 0.0)
        
        # Generate time points
        t = np.linspace(0, duration, num_points)
        
        # Initialize positions
        x = np.zeros(num_points)
        y = np.zeros(num_points)
        
        # Add harmonics
        for i in range(num_harmonics):
            # Frequency ratios
            freq_x = 1.0 + i * 0.5
            freq_y = 1.5 + i * 0.3
            
            # Amplitudes (decay with harmonic number)
            amp_x = 1.0 / (i + 1)
            amp_y = 1.0 / (i + 1)
            
            # Phase offsets
            phase_x = phase + i * np.pi / 4
            phase_y = i * np.pi / 6
            
            # Add damped oscillations
            x += amp_x * np.sin(freq_x * t + phase_x) * np.exp(-damping_x * t)
            y += amp_y * np.sin(freq_y * t + phase_y) * np.exp(-damping_y * t)
        
        # Normalize
        if np.max(np.abs(x)) > 0:
            x /= np.max(np.abs(x))
        if np.max(np.abs(y)) > 0:
            y /= np.max(np.abs(y))
        
        # Convert to list of tuples
        points = [(float(x[i]), float(y[i])) for i in range(num_points)]
        
        return points
    
    def generate_formula(self, params: Dict) -> Dict[str, str]:
        """
        Generate formula strings for frontend.
        
        Args:
            params: Visual parameters
            
        Returns:
            Dictionary with formula strings
        """
        num_harmonics = params.get('num_harmonics', 4)
        damping_x = params.get('damping_x', 0.02)
        damping_y = params.get('damping_y', 0.02)
        
        return {
            'x': f"Σ(i=0 to {num_harmonics}) A_i * sin(ω_i * t + φ_i) * exp(-{damping_x:.3f} * t)",
            'y': f"Σ(i=0 to {num_harmonics}) B_i * sin(ν_i * t + ψ_i) * exp(-{damping_y:.3f} * t)",
            'type': 'harmonograph',
            'num_harmonics': num_harmonics
        }


class LorenzAttractorGenerator:
    """
    Lorenz strange attractor generator.
    
    Generates chaotic patterns from the Lorenz system:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz
    """
    
    def __init__(self):
        """Initialize Lorenz attractor generator."""
        pass
    
    def generate_points(
        self,
        params: Dict,
        num_points: int = 2000,
        duration: float = 50.0
    ) -> List[Tuple[float, float]]:
        """
        Generate Lorenz attractor points.
        
        Args:
            params: Visual parameters
            num_points: Number of points to generate
            duration: Time duration in seconds
            
        Returns:
            List of (x, y) coordinate tuples (projecting 3D to 2D)
        """
        # Lorenz parameters (modulated by brain state)
        sigma = 10.0 + params.get('frequency_ratio_x', 3.0) * 2.0  # 10-20
        rho = 20.0 + params.get('frequency_ratio_y', 2.0) * 5.0    # 20-40
        beta = 2.0 + params.get('phase_offset', 0.0)                # 2-5
        
        # Time step
        dt = duration / num_points
        
        # Initial conditions
        x, y, z = 1.0, 1.0, 1.0
        
        points = []
        
        for _ in range(num_points):
            # Lorenz equations
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            
            x += dx
            y += dy
            z += dz
            
            # Project to 2D (use x-z plane for interesting view)
            points.append((x, z))
        
        # Normalize to [-1, 1]
        if points:
            x_vals = [p[0] for p in points]
            z_vals = [p[1] for p in points]
            
            x_min, x_max = min(x_vals), max(x_vals)
            z_min, z_max = min(z_vals), max(z_vals)
            
            x_range = max(x_max - x_min, 1e-10)
            z_range = max(z_max - z_min, 1e-10)
            
            points = [
                (
                    2 * (x - x_min) / x_range - 1,
                    2 * (z - z_min) / z_range - 1
                )
                for x, z in points
            ]
        
        return points
    
    def generate_formula(self, params: Dict) -> Dict[str, str]:
        """
        Generate formula strings for frontend.
        
        Args:
            params: Visual parameters
            
        Returns:
            Dictionary with formula strings
        """
        sigma = 10.0 + params.get('frequency_ratio_x', 3.0) * 2.0
        rho = 20.0 + params.get('frequency_ratio_y', 2.0) * 5.0
        beta = 2.0 + params.get('phase_offset', 0.0)
        
        return {
            'dx/dt': f"{sigma:.1f}(y - x)",
            'dy/dt': f"x({rho:.1f} - z) - y",
            'dz/dt': f"xy - {beta:.1f}z",
            'type': 'lorenz_attractor',
            'projection': 'x-z plane'
        }


class ReactionDiffusionGenerator:
    """
    Reaction-Diffusion pattern generator.
    
    Generates organic, flowing patterns based on Gray-Scott model.
    """
    
    def __init__(self):
        """Initialize reaction-diffusion generator."""
        pass
    
    def generate_points(
        self,
        params: Dict,
        num_points: int = 1500,
        duration: float = 10.0
    ) -> List[Tuple[float, float]]:
        """
        Generate reaction-diffusion pattern points.
        
        Args:
            params: Visual parameters
            num_points: Number of points to generate
            duration: Time duration in seconds
            
        Returns:
            List of (x, y) coordinate tuples
        """
        # Grid size
        size = 64
        
        # Parameters modulated by brain state
        feed_rate = 0.02 + params.get('frequency_ratio_x', 3.0) * 0.01  # 0.02-0.08
        kill_rate = 0.04 + params.get('frequency_ratio_y', 2.0) * 0.01  # 0.04-0.08
        
        # Diffusion rates
        Du = 0.16
        Dv = 0.08
        
        # Initialize concentrations
        u = np.ones((size, size))
        v = np.zeros((size, size))
        
        # Add initial perturbation in center
        center = size // 2
        radius = size // 8
        y, x = np.ogrid[-center:size-center, -center:size-center]
        mask = x*x + y*y <= radius*radius
        v[mask] = 1.0
        
        # Run simulation steps
        steps = 100
        for _ in range(steps):
            # Laplacian (diffusion)
            laplacian_u = (
                np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
                np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u
            )
            laplacian_v = (
                np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) +
                np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 4 * v
            )
            
            # Reaction-diffusion equations
            uvv = u * v * v
            u += Du * laplacian_u - uvv + feed_rate * (1 - u)
            v += Dv * laplacian_v + uvv - (feed_rate + kill_rate) * v
        
        # Extract contour points from the pattern
        threshold = 0.5
        points = []
        
        for i in range(size):
            for j in range(size):
                if v[i, j] > threshold:
                    # Convert grid coordinates to normalized [-1, 1]
                    x = (j / size) * 2 - 1
                    y = (i / size) * 2 - 1
                    points.append((x, y))
        
        # If we have too many points, sample them
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            points = [points[i] for i in indices]
        
        # Add some motion by rotating points based on time
        rotation = params.get('rotation_speed', 0.0) * duration
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        
        rotated_points = []
        for x, y in points:
            x_rot = x * cos_r - y * sin_r
            y_rot = x * sin_r + y * cos_r
            rotated_points.append((x_rot, y_rot))
        
        return rotated_points
    
    def generate_formula(self, params: Dict) -> Dict[str, str]:
        """
        Generate formula strings for frontend.
        
        Args:
            params: Visual parameters
            
        Returns:
            Dictionary with formula strings
        """
        feed_rate = 0.02 + params.get('frequency_ratio_x', 3.0) * 0.01
        kill_rate = 0.04 + params.get('frequency_ratio_y', 2.0) * 0.01
        
        return {
            'du/dt': f"D_u∇²u - uv² + F(1-u)",
            'dv/dt': f"D_v∇²v + uv² - (F+k)v",
            'F': f"{feed_rate:.3f}",
            'k': f"{kill_rate:.3f}",
            'type': 'reaction_diffusion',
            'model': 'Gray-Scott'
        }


class HyperspacePortalGenerator:
    """
    Psychedelic radial vortex generator.
    
    Builds layered polar wavefronts with radial symmetry, spiral drift, and
    controllable warp for "hyperspace portal" visuals.
    """

    def __init__(self):
        """Initialize hyperspace portal generator."""
        pass

    def generate_points(
        self,
        params: Dict,
        num_points: int = 2200,
        duration: float = 12.0
    ) -> List[Tuple[float, float]]:
        """
        Generate portal points.
        
        Args:
            params: Visual parameters
            num_points: Number of points to generate
            duration: Time duration in seconds
            
        Returns:
            List of (x, y) coordinate tuples
        """
        symmetry = max(3, int(params.get('portal_symmetry', 8)))
        radial_freq = params.get('portal_radial_frequency', 6.0)
        angular_freq = params.get('portal_angular_frequency', 2.0)
        warp = params.get('portal_warp', 0.4)
        spiral = params.get('portal_spiral', 0.6)
        layers = max(2, int(params.get('portal_layers', 4)))
        base_radius = params.get('portal_radius', 0.5)
        ripple = params.get('portal_ripple', 0.25)
        depth_skew = params.get('portal_depth_skew', 0.4)
        rotation_speed = params.get('rotation_speed', 0.0)
        speed_mult = params.get('speed_multiplier', 1.0)
        phase = params.get('phase_offset', 0.0)
        stability = params.get('stability', 0.6)

        # Points per layer for consistent density
        points_per_layer = max(24, num_points // layers)
        total_points = []

        # Time-driven term for animated drift when rendered
        time_term = duration * 0.12 * speed_mult
        rotation = rotation_speed * duration

        for layer in range(layers):
            depth = layer / max(layers - 1, 1)
            # Polar angle sweep for this layer
            theta = np.linspace(0, np.pi * 2 * angular_freq, points_per_layer)
            # Spiral drift increases with depth
            swirl = theta + spiral * (time_term + depth * 1.8) + phase

            # Base radius grows with depth to create tunnel effect
            radius_scale = base_radius * (1 + depth * (0.7 + depth_skew))

            # Interference waves: radial ripples + symmetry warp
            radial_wave = np.sin(theta * radial_freq + time_term) * ripple
            symmetry_wave = np.sin(symmetry * theta + rotation) * warp * (1 - 0.45 * depth)
            breathing = 1 + 0.15 * np.cos(depth * 3.1 + time_term * 1.2)

            r = radius_scale * (1 + radial_wave + symmetry_wave) * breathing

            # Low stability introduces subtle turbulence
            if stability < 0.5:
                jitter = (0.5 - stability) * 0.08
                r += jitter * np.sin(theta * 2.7 + depth * 3.3)

            x = r * np.cos(swirl)
            y = r * np.sin(swirl)

            total_points.extend(list(zip(x, y)))

        # Normalize to [-1, 1]
        if total_points:
            pts = np.array(total_points)
            max_abs = np.max(np.abs(pts)) or 1.0
            pts = pts / max_abs
            total_points = [(float(x), float(y)) for x, y in pts]

        return total_points

    def generate_formula(self, params: Dict) -> Dict[str, str]:
        """
        Generate formula strings for frontend.
        
        Args:
            params: Visual parameters
            
        Returns:
            Dictionary with formula strings
        """
        symmetry = params.get('portal_symmetry', 8)
        radial_freq = params.get('portal_radial_frequency', 6.0)
        angular_freq = params.get('portal_angular_frequency', 2.0)
        warp = params.get('portal_warp', 0.4)
        spiral = params.get('portal_spiral', 0.6)

        return {
            'r(theta)': (
                f"R0(1 + {warp:.2f} sin({symmetry}theta) + "
                f"sin({radial_freq:.1f}theta))"
            ),
            'theta(t)': f"{angular_freq:.1f}theta + {spiral:.2f}t",
            'type': 'hyperspace_portal',
            'notes': 'Layered polar waves with symmetry warp and spiral drift'
        }


class VisualAlgorithmFactory:
    """Factory for creating visual algorithm generators."""
    
    @staticmethod
    def create(algorithm_type: str):
        """
        Create a visual algorithm generator.
        
        Args:
            algorithm_type: Type of algorithm ('lissajous', 'harmonograph', 'lorenz', 'reaction_diffusion', 'hyperspace_portal')
            
        Returns:
            Algorithm generator instance
        """
        algorithms = {
            'lissajous': LissajousGenerator,
            'harmonograph': HarmonographGenerator,
            'lorenz': LorenzAttractorGenerator,
            'attractor': LorenzAttractorGenerator,  # Alias
            'reaction_diffusion': ReactionDiffusionGenerator,
            'diffusion': ReactionDiffusionGenerator,  # Alias
            'hyperspace_portal': HyperspacePortalGenerator,
            'portal': HyperspacePortalGenerator,
            'vortex': HyperspacePortalGenerator
        }
        
        if algorithm_type not in algorithms:
            logger.warning(
                "unknown_algorithm_type",
                type=algorithm_type,
                available=list(algorithms.keys())
            )
            algorithm_type = 'lissajous'  # Default
        
        return algorithms[algorithm_type]()
    
    @staticmethod
    def get_available_algorithms() -> List[Dict]:
        """
        Get list of available algorithms.
        
        Returns:
            List of algorithm descriptions
        """
        return [
            {
                'id': 'lissajous',
                'name': 'Lissajous Curves',
                'description': 'Simple parametric curves with frequency ratios',
                'complexity': 'low',
                'best_for': 'calm, meditative states'
            },
            {
                'id': 'harmonograph',
                'name': 'Harmonograph',
                'description': 'Multiple damped pendulum simulation',
                'complexity': 'medium',
                'best_for': 'complex, evolving patterns'
            },
            {
                'id': 'lorenz',
                'name': 'Lorenz Attractor',
                'description': 'Chaotic strange attractor system',
                'complexity': 'high',
                'best_for': 'chaotic, unpredictable patterns'
            },
            {
                'id': 'reaction_diffusion',
                'name': 'Reaction-Diffusion',
                'description': 'Organic patterns from Gray-Scott model',
                'complexity': 'high',
                'best_for': 'organic, cellular patterns'
            },
            {
                'id': 'hyperspace_portal',
                'name': 'Hyperspace Portal',
                'description': 'Layered radial waves with spiral warp',
                'complexity': 'high',
                'best_for': 'psychedelic tunnel / DMT portal'
            }
        ]
