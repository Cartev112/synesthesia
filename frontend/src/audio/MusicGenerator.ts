/**
 * Music Generator
 * Generates musical patterns based on brain state
 */

import { MidiEvent } from './AudioTrack';

export interface BrainState {
  focus: number;   // 0-1 (sum of probabilities)
  neutral: number; // 0-1 (sum of probabilities)
  relax: number;   // 0-1 (sum of probabilities)
}

// Musical scales
const SCALES = {
  major: [0, 2, 4, 5, 7, 9, 11],
  minor: [0, 2, 3, 5, 7, 8, 10],
  pentatonic: [0, 2, 4, 7, 9],
  blues: [0, 3, 5, 6, 7, 10],
  dorian: [0, 2, 3, 5, 7, 9, 10],
  phrygian: [0, 1, 3, 5, 7, 8, 10],
  lydian: [0, 2, 4, 6, 7, 9, 11],
  mixolydian: [0, 2, 4, 5, 7, 9, 10],
};

// Common chord progressions (scale degrees)
const PROGRESSIONS = [
  [0, 4, 0, 4],       // I-V-I-V (classic)
  [1, 4, 0, 5],       // I-IV-V-I (authentic cadence)
  [2, 5, 3, 4],       // I-vi-IV-V (50s progression)
  [0, 4, 5, 3],       // I-V-vi-IV (pop progression)
  [2, 3, 5, 4],       // I-IV-vi-V
  [5, 3, 4, 0],       // vi-IV-V-I (emotional)
  [0, 2, 3, 4],       // I-iii-IV-V
  [1, 4, 2, 5],       // I-V-iii-vi (deceptive)
];

export class MusicGenerator {
  private stepCount: number = 0;
  private currentScale: number[] = SCALES.major;
  private currentRoot: number = 60; // Middle C - fixed
  private currentChord: number[] = [0, 4, 7]; // Current triad (scale degrees)
  private currentChordRoot: number = 0; // Root note of current chord
  private nextChord: number[] = [0, 4, 7]; // Next triad for passing notes
  private nextChordRoot: number = 0; // Next chord root
  private textureArpIndex: number = 0; // Current position in arpeggio
  private progressionIndex: number = 0; // Position in current progression
  private currentProgression: number[] = PROGRESSIONS[0]; // Active progression
  private chordChangeCount: number = 0; // Track chord changes for progression

  /**
   * Generate events for all layers
   * @param brainState Current brain state
   * @param stepDuration Duration of one step in seconds (passed from engine)
   */
  generateStep(brainState: BrainState, stepDuration: number): Record<string, MidiEvent[]> {
    // Update chord every 4 beats (16 steps at 16th note resolution)
    if (this.stepCount % 16 === 0) {
      this.updateChord();
    }

    const events: Record<string, MidiEvent[]> = {
      bass: this.generateBass(stepDuration, brainState),
      harmony: this.generateHarmony(stepDuration),
      melody: this.generateMelody(stepDuration, brainState),
      texture: this.generateTexture(stepDuration, brainState),
    };

    this.stepCount++;
    return events;
  }

  /**
   * Update current chord (every 4 beats)
   */
  private updateChord(): void {
    // Move next chord to current
    this.currentChord = this.nextChord;
    this.currentChordRoot = this.nextChordRoot;
    this.chordChangeCount++;

    // Change progression every 8 chords (32 beats)
    if (this.chordChangeCount % 8 === 0) {
      this.currentProgression = PROGRESSIONS[Math.floor(Math.random() * PROGRESSIONS.length)];
      this.progressionIndex = 0;
    }

    // Get next chord from progression
    const nextScaleDegree = this.currentProgression[this.progressionIndex];
    this.progressionIndex = (this.progressionIndex + 1) % this.currentProgression.length;

    // Build chord with possible extensions and variations
    const chordNotes = this.buildChord(nextScaleDegree);
    this.nextChordRoot = this.currentScale[nextScaleDegree];
    this.nextChord = chordNotes;
  }

  /**
   * Build a chord with extensions and variations
   */
  private buildChord(scaleDegree: number): number[] {
    const root = this.currentScale[scaleDegree];
    const third = this.currentScale[(scaleDegree + 2) % this.currentScale.length];
    const fifth = this.currentScale[(scaleDegree + 4) % this.currentScale.length];

    // Randomly add extensions for variety
    const rand = Math.random();
    
    if (rand < 0.15) {
      // Add 7th (15% chance)
      const seventh = this.currentScale[(scaleDegree + 6) % this.currentScale.length];
      return [root, third, fifth, seventh];
    } else if (rand < 0.25) {
      // Add 9th/2nd (10% chance)
      const ninth = this.currentScale[(scaleDegree + 1) % this.currentScale.length];
      return [root, third, fifth, ninth];
    } else if (rand < 0.35) {
      // Sus4 chord (10% chance) - replace third with fourth
      const fourth = this.currentScale[(scaleDegree + 3) % this.currentScale.length];
      return [root, fourth, fifth];
    } else if (rand < 0.40) {
      // Power chord (5% chance) - just root and fifth
      return [root, fifth];
    } else {
      // Standard triad (60% chance)
      return [root, third, fifth];
    }
  }

  /**
   * Generate bass line - plays root of current chord with passing notes
   */
  private generateBass(stepDuration: number, brainState: BrainState): MidiEvent[] {
    const events: MidiEvent[] = [];

    // Play chord root on beat 1 of every 4 beats (when chord changes)
    if (this.stepCount % 16 === 0) {
      const note = this.currentRoot + this.currentChordRoot - 24; // Two octaves below
      // Shorten duration if passing note will play, with gap before next event
      const duration = brainState.focus > 0.5 ? 6.5 : 13; // Stop well before passing note or next chord
      events.push({
        note,
        velocity: 85 + Math.random() * 15,
        duration: stepDuration * duration,
        time: 0,
      });
    }
    // Add passing note on beat 3 (step 8) when focus is high
    else if (this.stepCount % 16 === 8 && brainState.focus > 0.5) {
      // Find a passing note between current and next chord root
      const currentNote = this.currentChordRoot;
      const nextNote = this.nextChordRoot;
      const distance = nextNote - currentNote;
      
      // Choose a scale tone between them
      let passingNote = currentNote;
      if (Math.abs(distance) > 2) {
        // Find a scale degree between current and next
        for (let i = 0; i < this.currentScale.length; i++) {
          const scaleTone = this.currentScale[i];
          if (distance > 0 && scaleTone > currentNote && scaleTone < nextNote) {
            passingNote = scaleTone;
            break;
          } else if (distance < 0 && scaleTone < currentNote && scaleTone > nextNote) {
            passingNote = scaleTone;
            break;
          }
        }
      }
      
      events.push({
        note: this.currentRoot + passingNote - 24,
        velocity: 70 + Math.random() * 15,
        duration: stepDuration * 2,
        time: 0,
      });
    }

    return events;
  }

  /**
   * Generate harmony - plays triad from current chord
   */
  private generateHarmony(stepDuration: number): MidiEvent[] {
    const events: MidiEvent[] = [];

    // Play full triad when chord changes
    if (this.stepCount % 16 === 0) {
      // Play all three notes of the triad
      for (const chordTone of this.currentChord) {
        events.push({
          note: this.currentRoot + chordTone,
          velocity: 45 + Math.random() * 15,
          duration: stepDuration * 13, // Hold for 3.25 beats with gap before next chord
          time: 0,
        });
      }
    }

    return events;
  }

  /**
   * Generate melody - mostly pentatonic scale with rhythm based on brain state
   */
  private generateMelody(stepDuration: number, brainState: BrainState): MidiEvent[] {
    const events: MidiEvent[] = [];

    // Probability and rhythm complexity based on brain state
    // Focus increases density, relax increases duration
    const noteProbability = 0.2 + brainState.focus * 0.4; // 0.2-0.6
    
    if (Math.random() < noteProbability) {
      // Use pentatonic subset of current scale
      const pentatonic = [0, 2, 4, 7, 9].map(degree => 
        this.currentScale[degree % this.currentScale.length]
      );
      
      const noteIndex = Math.floor(Math.random() * pentatonic.length);
      const octaveShift = 12 + (Math.floor(Math.random() * 2) * 12); // 1-2 octaves up
      const note = this.currentRoot + pentatonic[noteIndex] + octaveShift;

      // Duration based on brain state
      // High focus = shorter, more rhythmic notes
      // High relax = longer, sustained notes
      let duration: number;
      if (brainState.focus > 0.6) {
        // Fast rhythms when focused
        const durations = [1, 1, 2, 2, 3];
        duration = durations[Math.floor(Math.random() * durations.length)];
      } else if (brainState.relax > 0.6) {
        // Slow rhythms when relaxed
        const durations = [3, 4, 6, 8];
        duration = durations[Math.floor(Math.random() * durations.length)];
      } else {
        // Medium rhythms when neutral
        const durations = [2, 3, 4];
        duration = durations[Math.floor(Math.random() * durations.length)];
      }

      events.push({
        note,
        velocity: 65 + Math.random() * 35,
        duration: stepDuration * duration,
        time: 0,
      });
    }

    return events;
  }

  /**
   * Generate texture - arpeggiated chord tones with duration based on brain state
   */
  private generateTexture(stepDuration: number, brainState: BrainState): MidiEvent[] {
    const events: MidiEvent[] = [];

    // Arpeggio speed based on brain state (2x faster than before)
    // Focus = faster arpeggios, Relax = slower arpeggios
    let arpSpeed: number;
    if (brainState.focus > 0.6) {
      arpSpeed = 1; // Play every step (very fast)
    } else if (brainState.relax > 0.6) {
      arpSpeed = 4; // Play every 4 steps (medium)
    } else {
      arpSpeed = 2; // Play every 2 steps (fast)
    }

    // Play arpeggio note on the appropriate steps
    if (this.stepCount % arpSpeed === 0) {
      const chordTone = this.currentChord[this.textureArpIndex % this.currentChord.length];
      const octaveShift = 12 + (Math.floor(this.textureArpIndex / 3) % 2) * 12; // Alternate octaves
      const note = this.currentRoot + chordTone + octaveShift;

      // Duration based on brain state
      let duration: number;
      if (brainState.relax > 0.6) {
        duration = arpSpeed * 3; // Long overlapping notes when relaxed
      } else {
        duration = arpSpeed * 1.5; // Shorter notes otherwise
      }

      events.push({
        note,
        velocity: 30 + Math.random() * 15,
        duration: stepDuration * duration,
        time: 0,
      });

      this.textureArpIndex++;
    }

    return events;
  }

  /**
   * Reset generator state
   */
  reset(): void {
    this.stepCount = 0;
    this.currentChord = [0, 4, 7];
    this.currentChordRoot = 0;
    this.nextChord = [0, 4, 7];
    this.nextChordRoot = 0;
    this.textureArpIndex = 0;
    this.progressionIndex = 0;
    this.chordChangeCount = 0;
    this.currentProgression = PROGRESSIONS[Math.floor(Math.random() * PROGRESSIONS.length)];
  }
}
