/**
 * Web Audio API Synthesizers
 * Mirrors backend synthesizer functionality
 */

export interface ADSREnvelope {
  attack: number;   // seconds
  decay: number;    // seconds
  sustain: number;  // level 0-1
  release: number;  // seconds
}

export const DEFAULT_ENVELOPE: ADSREnvelope = {
  attack: 0.005,
  decay: 0.05,
  sustain: 0.8,
  release: 0.05,
};

export type WaveType = 'sine' | 'square' | 'sawtooth' | 'triangle';

export interface SynthesizerConfig {
  type: WaveType;
  envelope?: ADSREnvelope;
}

export class Synthesizer {
  private audioContext: AudioContext;
  private config: SynthesizerConfig;
  private envelope: ADSREnvelope;

  constructor(audioContext: AudioContext, config: SynthesizerConfig) {
    this.audioContext = audioContext;
    this.config = config;
    this.envelope = config.envelope || DEFAULT_ENVELOPE;
  }

  /**
   * Play a note with ADSR envelope
   */
  playNote(
    frequency: number,
    duration: number,
    velocity: number,
    destination: AudioNode,
    startTime?: number
  ): void {
    const now = startTime || this.audioContext.currentTime;
    
    // Create oscillator
    const osc = this.audioContext.createOscillator();
    osc.type = this.config.type;
    osc.frequency.value = frequency;

    // Create gain node for ADSR envelope
    const gainNode = this.audioContext.createGain();
    gainNode.gain.value = 0;

    // Connect: oscillator -> gain -> destination
    osc.connect(gainNode);
    gainNode.connect(destination);

    // Apply ADSR envelope
    const { attack, decay, sustain, release } = this.envelope;
    const sustainLevel = sustain * velocity;

    // Attack
    gainNode.gain.setValueAtTime(0, now);
    gainNode.gain.linearRampToValueAtTime(velocity, now + attack);

    // Decay to sustain
    gainNode.gain.linearRampToValueAtTime(sustainLevel, now + attack + decay);

    // Sustain (hold at sustain level)
    const sustainEnd = now + duration - release;
    gainNode.gain.setValueAtTime(sustainLevel, sustainEnd);

    // Release
    gainNode.gain.linearRampToValueAtTime(0, sustainEnd + release);

    // Start and stop oscillator
    osc.start(now);
    osc.stop(sustainEnd + release + 0.01); // Small buffer to ensure clean stop
  }

  /**
   * Update synthesizer type
   */
  setType(type: WaveType): void {
    this.config.type = type;
  }

  /**
   * Update envelope parameters
   */
  setEnvelope(envelope: Partial<ADSREnvelope>): void {
    this.envelope = { ...this.envelope, ...envelope };
  }
}

/**
 * Convert MIDI note number to frequency
 */
export function midiToFrequency(midiNote: number): number {
  return 440 * Math.pow(2, (midiNote - 69) / 12);
}
