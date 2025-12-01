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

// Musical scales - ambient-friendly modes
const SCALES = {
  major: [0, 2, 4, 5, 7, 9, 11],
  minor: [0, 2, 3, 5, 7, 8, 10],
  pentatonic: [0, 2, 4, 7, 9],
  dorian: [0, 2, 3, 5, 7, 9, 10],      // Minor with raised 6th - dreamy
  lydian: [0, 2, 4, 6, 7, 9, 11],      // Major with raised 4th - ethereal
  mixolydian: [0, 2, 4, 5, 7, 9, 10],  // Major with flat 7th - floating
  aeolian: [0, 2, 3, 5, 7, 8, 10],     // Natural minor - melancholic
};

// Ambient-friendly scales for melody
const AMBIENT_SCALES = {
  pentatonicMajor: [0, 2, 4, 7, 9],
  pentatonicMinor: [0, 3, 5, 7, 10],
  wholeTone: [0, 2, 4, 6, 8, 10],      // Dreamy, no resolution
  suspended: [0, 2, 5, 7, 9],          // Avoids 3rds - open sound
};

// Ambient chord progressions (scale degrees) - slow, floating, minimal movement
const PROGRESSIONS = [
  // Minimal movement - drone-like
  [0, 0, 4, 0],       // I-I-V-I (pedal tone)
  [0, 3, 0, 3],       // I-IV-I-IV (gentle oscillation)
  [0, 5, 0, 5],       // I-vi-I-vi (minor color)
  
  // Descending - melancholic ambient
  [0, 6, 5, 4],       // I-vii-vi-V (descending)
  [5, 4, 3, 0],       // vi-V-IV-I (aeolian descent)
  [0, 6, 3, 4],       // I-vii-IV-V (tension release)
  
  // Modal/suspended feel
  [0, 2, 0, 4],       // I-iii-I-V (modal)
  [0, 4, 3, 2],       // I-V-IV-iii (reverse pop)
  [3, 0, 4, 0],       // IV-I-V-I (plagal ambient)
  
  // Emotional/cinematic
  [5, 3, 4, 0],       // vi-IV-V-I (emotional)
  [0, 5, 3, 4],       // I-vi-IV-V (50s dreamy)
  [5, 0, 3, 4],       // vi-I-IV-V (hopeful)
  
  // Static/meditative
  [1, 4, 0, 5],       // I-I-I-V (very minimal)
  [0, 3, 3, 0],       // I-IV-IV-I (suspended feel)
  [5, 5, 0, 0],       // vi-vi-I-I (dark to light)
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
   * Build a chord with ambient extensions and voicings
   */
  private buildChord(scaleDegree: number): number[] {
    const root = this.currentScale[scaleDegree];
    const second = this.currentScale[(scaleDegree + 1) % this.currentScale.length];
    const third = this.currentScale[(scaleDegree + 2) % this.currentScale.length];
    const fourth = this.currentScale[(scaleDegree + 3) % this.currentScale.length];
    const fifth = this.currentScale[(scaleDegree + 4) % this.currentScale.length];
    const sixth = this.currentScale[(scaleDegree + 5) % this.currentScale.length];
    const seventh = this.currentScale[(scaleDegree + 6) % this.currentScale.length];

    // Ambient chord voicings - favor extensions, suspensions, and open voicings
    const rand = Math.random();
    
    if (rand < 0.20) {
      // Add9 chord (20% chance) - lush, shimmery
      return [root, third, fifth, second + 12];
    } else if (rand < 0.35) {
      // Maj7/min7 chord (15% chance) - smooth, jazzy
      return [root, third, fifth, seventh];
    } else if (rand < 0.50) {
      // Sus2 chord (15% chance) - open, ambiguous
      return [root, second, fifth];
    } else if (rand < 0.65) {
      // Sus4 chord (15% chance) - suspended, unresolved
      return [root, fourth, fifth];
    } else if (rand < 0.75) {
      // Add11 chord (10% chance) - wide, orchestral
      return [root, third, fifth, fourth + 12];
    } else if (rand < 0.85) {
      // 6th chord (10% chance) - warm, nostalgic
      return [root, third, fifth, sixth];
    } else if (rand < 0.92) {
      // Power chord with octave (7% chance) - open, powerful
      return [root, fifth, root + 12];
    } else {
      // Sparse - just root and fifth (8% chance) - minimal, drone-like
      return [root, fifth];
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
   * Generate harmony - wide voicings with staggered entry for ambient pad sound
   */
  private generateHarmony(stepDuration: number): MidiEvent[] {
    const events: MidiEvent[] = [];

    // Play chord when it changes, with staggered entry for lush sound
    if (this.stepCount % 16 === 0) {
      // Spread chord tones across octaves for wider voicing
      this.currentChord.forEach((chordTone, index) => {
        // Stagger entry slightly for each note (0, 0.5, 1 steps)
        const staggerTime = index * 0.3 * stepDuration;
        
        // Spread across octaves: bass note lower, upper notes higher
        let octaveAdjust = 0;
        if (index === 0) octaveAdjust = -12; // Root down an octave
        if (index >= 2) octaveAdjust = 12;   // Upper extensions up an octave
        
        events.push({
          note: this.currentRoot + chordTone + octaveAdjust,
          velocity: 35 + Math.random() * 15, // Softer for ambient
          duration: stepDuration * 14, // Long sustain
          time: staggerTime,
        });
      });
    }

    return events;
  }

  /**
   * Generate melody - sparse, floating notes with chord tone preference
   */
  private generateMelody(stepDuration: number, brainState: BrainState): MidiEvent[] {
    const events: MidiEvent[] = [];

    // Lower probability for ambient - more space between notes
    // Focus increases density slightly, relax makes it very sparse
    const noteProbability = 0.08 + brainState.focus * 0.15 - brainState.relax * 0.05; // 0.03-0.23
    
    if (Math.random() < noteProbability) {
      // Prefer chord tones for consonance, occasionally add color tones
      let note: number;
      const useChordTone = Math.random() < 0.7; // 70% chord tones
      
      if (useChordTone && this.currentChord.length > 0) {
        // Pick a chord tone
        const chordTone = this.currentChord[Math.floor(Math.random() * this.currentChord.length)];
        const octaveShift = 12 + (Math.floor(Math.random() * 2) * 12); // 1-2 octaves up
        note = this.currentRoot + chordTone + octaveShift;
      } else {
        // Use suspended/pentatonic scale for color
        const ambientScale = AMBIENT_SCALES.suspended;
        const scaleNote = ambientScale[Math.floor(Math.random() * ambientScale.length)];
        const octaveShift = 12 + (Math.floor(Math.random() * 2) * 12); // 2-3 octaves up for shimmer
        note = this.currentRoot + scaleNote + octaveShift;
      }

      // Longer durations for ambient - notes float and decay
      let duration: number;
      if (brainState.focus > 0.6) {
        // Still relatively long even when focused
        const durations = [4, 6, 8];
        duration = durations[Math.floor(Math.random() * durations.length)];
      } else if (brainState.relax > 0.6) {
        // Very long, sustained notes when relaxed
        const durations = [8, 12, 16];
        duration = durations[Math.floor(Math.random() * durations.length)];
      } else {
        // Medium-long for neutral
        const durations = [6, 8, 10];
        duration = durations[Math.floor(Math.random() * durations.length)];
      }

      events.push({
        note,
        velocity: 40 + Math.random() * 25, // Softer dynamics
        duration: stepDuration * duration,
        time: 0,
      });
    }

    return events;
  }

  /**
   * Generate texture - gentle, overlapping arpeggios with wide spacing
   */
  private generateTexture(stepDuration: number, brainState: BrainState): MidiEvent[] {
    const events: MidiEvent[] = [];

    // Slower arpeggios for ambient - more space between notes
    // Focus = medium arpeggios, Relax = very slow, sparse
    let arpSpeed: number;
    if (brainState.focus > 0.6) {
      arpSpeed = 3; // Play every 3 steps
    } else if (brainState.relax > 0.6) {
      arpSpeed = 6; // Play every 6 steps (very slow)
    } else {
      arpSpeed = 4; // Play every 4 steps
    }

    // Play arpeggio note on the appropriate steps
    if (this.stepCount % arpSpeed === 0) {
      const chordTone = this.currentChord[this.textureArpIndex % this.currentChord.length];
      
      // Wide octave spread for ambient shimmer
      const octaveOptions = [12, 24, 36]; // 1, 2, or 3 octaves up
      const octaveShift = octaveOptions[this.textureArpIndex % octaveOptions.length];
      const note = this.currentRoot + chordTone + octaveShift;

      // Long, overlapping durations for pad-like texture
      let duration: number;
      if (brainState.relax > 0.6) {
        duration = arpSpeed * 4; // Very long, heavily overlapping
      } else if (brainState.focus > 0.6) {
        duration = arpSpeed * 2; // Medium overlap
      } else {
        duration = arpSpeed * 3; // Good overlap for ambient
      }

      events.push({
        note,
        velocity: 20 + Math.random() * 15, // Very soft for texture
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
