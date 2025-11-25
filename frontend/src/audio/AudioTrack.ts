/**
 * Audio Track
 * Represents a single audio track with synthesizer and mixer controls
 */

import { Synthesizer, SynthesizerConfig, WaveType } from './Synthesizer';

export interface MidiEvent {
  note: number;      // MIDI note number
  velocity: number;  // 0-127
  duration: number;  // seconds
  time: number;      // seconds (relative to buffer start)
}

export interface TrackConfig {
  name: string;
  synthType: WaveType;
  volume: number;    // 0-1
  mute: boolean;
  solo: boolean;
}

export class AudioTrack {
  public name: string;
  public volume: number;
  public mute: boolean;
  public solo: boolean;
  
  private synthesizer: Synthesizer;
  private gainNode: GainNode;
  private audioContext: AudioContext;

  constructor(audioContext: AudioContext, config: TrackConfig) {
    this.audioContext = audioContext;
    this.name = config.name;
    this.volume = config.volume;
    this.mute = config.mute;
    this.solo = config.solo;

    // Create gain node for track volume
    this.gainNode = audioContext.createGain();
    this.gainNode.gain.value = this.volume;

    // Create synthesizer
    const synthConfig: SynthesizerConfig = {
      type: config.synthType,
    };
    this.synthesizer = new Synthesizer(audioContext, synthConfig);
  }

  /**
   * Get the track's gain node for routing
   */
  getGainNode(): GainNode {
    return this.gainNode;
  }

  /**
   * Play a MIDI event
   */
  playEvent(event: MidiEvent, startTime: number): void {
    if (this.mute) return;

    // Normalize velocity (0-127 -> 0-1)
    const normalizedVelocity = event.velocity / 127;

    // Play note through synthesizer -> track gain
    this.synthesizer.playNote(
      this.midiToFrequency(event.note),
      event.duration,
      normalizedVelocity,
      this.gainNode,
      startTime + event.time
    );
  }

  /**
   * Set track volume
   */
  setVolume(volume: number): void {
    this.volume = Math.max(0, Math.min(1, volume));
    this.gainNode.gain.setValueAtTime(this.volume, this.audioContext.currentTime);
  }

  /**
   * Set mute state
   */
  setMute(mute: boolean): void {
    this.mute = mute;
  }

  /**
   * Set solo state
   */
  setSolo(solo: boolean): void {
    this.solo = solo;
  }

  /**
   * Change synthesizer type
   */
  setSynthType(type: WaveType): void {
    this.synthesizer.setType(type);
  }

  /**
   * Convert MIDI note to frequency
   */
  private midiToFrequency(midiNote: number): number {
    return 440 * Math.pow(2, (midiNote - 69) / 12);
  }

  /**
   * Cleanup
   */
  dispose(): void {
    this.gainNode.disconnect();
  }
}
