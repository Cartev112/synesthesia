"""
Cellular automaton-based music generator.

Generates musical patterns using 2D cellular automata rules
modulated by brain state.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from backend.music.scales import Scale, get_scale
from backend.core.logging import get_logger

logger = get_logger(__name__)


class MidiEvent:
    """Represents a MIDI note event."""
    
    def __init__(
        self,
        note: int,
        velocity: int,
        duration: float,
        time: float = 0.0
    ):
        """
        Initialize MIDI event.
        
        Args:
            note: MIDI note number (0-127)
            velocity: Note velocity (0-127)
            duration: Note duration in seconds
            time: Event time in seconds
        """
        self.note = note
        self.velocity = velocity
        self.duration = duration
        self.time = time
    
    def __repr__(self) -> str:
        return f"MidiEvent(note={self.note}, vel={self.velocity}, dur={self.duration})"


class MusicalCellularAutomaton:
    """
    2D cellular automaton for music generation.
    
    Grid cells represent potential musical events.
    Rules determine note generation based on brain state.
    """
    
    def __init__(
        self,
        width: int = 16,
        height: int = 8,
        scale_name: str = 'major',
        root_note: int = 60,
        base_tempo: float = 120.0
    ):
        """
        Initialize musical cellular automaton.
        
        Args:
            width: Grid width (time steps, e.g., 16th notes in a bar)
            height: Grid height (pitch range, scale degrees)
            scale_name: Musical scale to use
            root_note: Root note MIDI number
            base_tempo: Base tempo in BPM
        """
        self.width = width
        self.height = height
        self.scale = get_scale(scale_name, root_note)
        self.base_tempo = base_tempo
        
        # Initialize grid (False = dead, True = alive)
        self.grid = np.zeros((height, width), dtype=bool)
        
        # Current position in grid
        self.current_position = 0
        
        # Brain state parameters
        self.tempo = base_tempo
        self.density = 0.3
        self.pitch_center = root_note
        self.complexity = 0.5
        
        # CA rules parameters
        self.birth_min = 2
        self.birth_max = 3
        self.survive_min = 2
        self.survive_max = 4
        self.rhythm_mask = self._build_rhythm_mask()
        self.accent_pattern = self._build_accent_pattern()
        
        # Initialize with random pattern
        self._initialize_pattern()
        
        logger.info(
            "musical_ca_initialized",
            width=width,
            height=height,
            scale=scale_name,
            tempo=base_tempo
        )
    
    def _initialize_pattern(self):
        """Initialize grid with random pattern."""
        self.grid = np.random.random((self.height, self.width)) < self.density
        self.rhythm_mask = self._build_rhythm_mask()

    def _build_rhythm_mask(self) -> NDArray[np.float64]:
        """
        Create a rhythmic mask that gates which columns may produce notes.
        
        The mask emphasizes downbeats to keep the CA output feeling metered
        even when brain-state updates or websocket ticks are jittery.
        """
        # Strong beats on 1 and 3, lighter syncopation elsewhere
        base = np.array([
            1.0, 0.45, 0.8, 0.55,
            0.95, 0.6, 0.75, 0.5,
            1.0, 0.45, 0.8, 0.55,
            0.95, 0.6, 0.75, 0.5,
        ])
        
        density_scale = np.clip(self.density + 0.2, 0.2, 1.0)
        mask = np.clip(base * density_scale, 0.0, 1.0)
        return mask.astype(float)

    def _build_accent_pattern(self) -> NDArray[np.float64]:
        """Accent pattern used to scale velocities on strong beats."""
        return np.array([
            1.2, 0.9, 1.05, 0.95,
            1.15, 0.9, 1.05, 0.95,
            1.2, 0.9, 1.05, 0.95,
            1.1, 0.9, 1.05, 0.95,
        ])
    
    def update_from_brain_state(self, brain_state: Dict[str, float]):
        """
        Update CA parameters based on brain state.
        
        Args:
            brain_state: Dictionary with keys:
                - focus: 0-1, higher = more active/complex
                - relax: 0-1, higher = simpler/slower
                - neutral: 0-1, baseline state
        """
        focus = brain_state.get('focus', 0.5)
        relax = brain_state.get('relax', 0.5)
        
        # Tempo modulation: focus increases tempo, relax decreases
        self.tempo = self.base_tempo + (focus * 40) - (relax * 30)
        self.tempo = np.clip(self.tempo, 60, 180)
        
        # Density modulation: focus increases note density
        self.density = 0.2 + (focus * 0.4) - (relax * 0.15)
        self.density = np.clip(self.density, 0.1, 0.7)
        
        # Pitch center: focus raises pitch, relax lowers
        pitch_offset = int((focus - relax) * 12)  # Up to 1 octave
        self.pitch_center = self.scale.root + pitch_offset
        
        # Complexity (affects CA rules): focus increases complexity
        self.complexity = 0.3 + (focus * 0.5) - (relax * 0.2)
        self.complexity = np.clip(self.complexity, 0.2, 0.8)
        
        # Adjust CA rules based on complexity
        if self.complexity > 0.6:
            # More complex: easier birth, harder survival
            self.birth_min = 2
            self.birth_max = 4
            self.survive_min = 2
            self.survive_max = 5
        elif self.complexity < 0.4:
            # Simpler: harder birth, easier survival
            self.birth_min = 3
            self.birth_max = 3
            self.survive_min = 2
            self.survive_max = 3
        else:
            # Balanced
            self.birth_min = 2
            self.birth_max = 3
            self.survive_min = 2
            self.survive_max = 4
        
        # Adjust density toward target
        current_density = np.mean(self.grid)
        if current_density < self.density - 0.1:
            # Add some cells
            mask = np.random.random(self.grid.shape) < 0.1
            self.grid = np.logical_or(self.grid, mask)
        elif current_density > self.density + 0.1:
            # Remove some cells
            mask = np.random.random(self.grid.shape) < 0.1
            self.grid = np.logical_and(self.grid, ~mask)
        
        # Update rhythmic emphasis to reflect new density/complexity
        self.rhythm_mask = self._build_rhythm_mask()
        self.accent_pattern = self._build_accent_pattern()
        
        logger.debug(
            "brain_state_update",
            tempo=self.tempo,
            density=self.density,
            complexity=self.complexity
        )
    
    def step(self, time_offset: float = 0.0) -> List[MidiEvent]:
        """
        Advance CA one step and generate MIDI events.
        
        Args:
            time_offset: Base time offset (seconds) to apply to generated events
        
        Returns:
            List of MIDI events to play
        """
        # Apply CA rules to evolve grid
        self.grid = self._apply_rules(self.grid)
        
        # Extract notes from current column
        events = []
        active_cells = np.where(self.grid[:, self.current_position])[0]
        rhythm_gate = self.rhythm_mask[self.current_position]
        
        if rhythm_gate <= 0.05:
            active_cells = np.array([], dtype=int)
        elif rhythm_gate < 0.99 and active_cells.size > 0:
            keep = np.random.random(active_cells.shape) < rhythm_gate
            active_cells = active_cells[keep]
        
        accent = self.accent_pattern[self.current_position]
        
        for cell_y in active_cells:
            midi_note = self._cell_to_midi(cell_y)
            velocity = self._compute_velocity(cell_y)
            duration = self._compute_duration()
            
            adjusted_velocity = int(np.clip(velocity * accent, 1, 127))
            
            events.append(MidiEvent(
                note=midi_note,
                velocity=adjusted_velocity,
                duration=duration,
                time=time_offset
            ))
        
        # Advance position
        self.current_position = (self.current_position + 1) % self.width
        
        return events
    
    def _apply_rules(self, grid: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """
        Apply cellular automaton rules.
        
        Rules (inspired by Conway's Life, musicalized):
        - Birth: cell becomes alive if neighbors in [birth_min, birth_max]
        - Survival: cell stays alive if neighbors in [survive_min, survive_max]
        - Death: otherwise
        
        Args:
            grid: Current grid state
            
        Returns:
            New grid state
        """
        new_grid = grid.copy()
        
        for y in range(self.height):
            for x in range(self.width):
                neighbors = self._count_neighbors(grid, y, x)
                
                if grid[y, x]:  # Cell is alive
                    new_grid[y, x] = (self.survive_min <= neighbors <= self.survive_max)
                else:  # Cell is dead
                    new_grid[y, x] = (self.birth_min <= neighbors <= self.birth_max)
        
        return new_grid
    
    def _count_neighbors(
        self,
        grid: NDArray[np.bool_],
        y: int,
        x: int
    ) -> int:
        """
        Count alive neighbors (Moore neighborhood with wrapping).
        
        Args:
            grid: Grid state
            y: Cell y position
            x: Cell x position
            
        Returns:
            Number of alive neighbors
        """
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                
                ny = (y + dy) % self.height
                nx = (x + dx) % self.width
                
                if grid[ny, nx]:
                    count += 1
        
        return count
    
    def _cell_to_midi(self, cell_y: int) -> int:
        """
        Convert cell y-position to MIDI note.
        
        Args:
            cell_y: Cell y coordinate
            
        Returns:
            MIDI note number
        """
        # Map cell position to scale degree
        scale_degree = cell_y
        midi_note = self.scale.get_note(scale_degree)
        
        # Apply pitch center offset
        offset = self.pitch_center - self.scale.root
        midi_note += offset
        
        # Clamp to valid MIDI range
        return int(np.clip(midi_note, 0, 127))
    
    def _compute_velocity(self, cell_y: int) -> int:
        """
        Compute note velocity based on cell position.
        
        Higher cells = louder notes.
        
        Args:
            cell_y: Cell y coordinate
            
        Returns:
            MIDI velocity (0-127)
        """
        # Base velocity
        base_vel = 60
        
        # Height-based variation (higher = louder)
        height_factor = cell_y / self.height
        velocity = base_vel + int(height_factor * 40)
        
        # Add some randomness
        velocity += np.random.randint(-10, 10)
        
        return int(np.clip(velocity, 20, 127))
    
    def _compute_duration(self) -> float:
        """
        Compute note duration based on tempo.
        
        Returns:
            Duration in seconds
        """
        # Duration of one grid step (16th note)
        step_duration = 60.0 / self.tempo / 4.0  # Quarter note / 4
        
        # Favor shorter notes to keep motion while allowing occasional longer holds
        num_steps = np.random.choice([1, 2, 3], p=[0.55, 0.35, 0.10])
        
        return step_duration * num_steps
    
    def get_step_duration(self) -> float:
        """
        Get duration of one CA step in seconds.
        
        Returns:
            Step duration
        """
        return 60.0 / self.tempo / 4.0  # 16th note duration
    
    def set_scale(self, scale_name: str, root_note: Optional[int] = None):
        """
        Change the musical scale.
        
        Args:
            scale_name: New scale name
            root_note: Optional new root note
        """
        if root_note is None:
            root_note = self.scale.root
        
        self.scale = get_scale(scale_name, root_note)
        logger.info("scale_changed", scale=scale_name, root=root_note)
    
    def reset(self):
        """Reset the automaton to initial state."""
        self._initialize_pattern()
        self.current_position = 0
        logger.debug("ca_reset")
    
    def get_grid_state(self) -> NDArray[np.bool_]:
        """Get current grid state (for visualization)."""
        return self.grid.copy()
