/**
 * Frontend Audio Engine
 * Complete audio system running in the browser
 */

import { AudioTrack, TrackConfig } from './AudioTrack';
import { MusicGenerator, BrainState } from './MusicGenerator';
import { WaveType } from './Synthesizer';

export interface AudioEngineConfig {
  sampleRate?: number;
  stepInterval?: number; // milliseconds between music generation steps
}

const DEFAULT_TRACKS: TrackConfig[] = [
  { name: 'bass', synthType: 'sawtooth', volume: 0.7, mute: false, solo: false },
  { name: 'harmony', synthType: 'sine', volume: 0.5, mute: false, solo: false },
  { name: 'melody', synthType: 'triangle', volume: 0.6, mute: false, solo: false },
  { name: 'texture', synthType: 'sine', volume: 0.4, mute: false, solo: false },
];

export class FrontendAudioEngine {
  private audioContext: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private lowPassFilter: BiquadFilterNode | null = null;
  private reverbGain: GainNode | null = null;
  private dryGain: GainNode | null = null;
  private convolver: ConvolverNode | null = null;
  private reverbHighPass: BiquadFilterNode | null = null;
  private tracks: Map<string, AudioTrack> = new Map();
  private musicGenerator: MusicGenerator;
  private isPlaying: boolean = false;
  private generationInterval: number | null = null;
  private stepInterval: number;
  private nextStepTime: number = 0; // Scheduled audio time for next step
  private scheduleAheadTime: number = 0.1; // Schedule 100ms ahead
  private brainState: BrainState = {
    focus: 0.5,
    neutral: 0.5,
    relax: 0.5,
  };

  constructor(config: AudioEngineConfig = {}) {
    this.stepInterval = config.stepInterval || 125; // 125ms = 8 steps per second
    this.musicGenerator = new MusicGenerator();
  }

  /**
   * Initialize audio context and tracks
   */
  async initialize(): Promise<void> {
    if (this.audioContext) {
      console.warn('Audio engine already initialized');
      return;
    }

    // Create audio context
    this.audioContext = new AudioContext();
    
    // Create master effects chain:
    // Tracks → Master Gain → Low-Pass Filter → Dry/Wet Split → Reverb → Output
    
    // Master gain (pre-effects)
    this.masterGain = this.audioContext.createGain();
    this.masterGain.gain.value = 0.8;
    
    // Low-pass filter for warmth
    this.lowPassFilter = this.audioContext.createBiquadFilter();
    this.lowPassFilter.type = 'lowpass';
    this.lowPassFilter.frequency.value = 2000; // Will be modulated by brain state
    this.lowPassFilter.Q.value = 1.0;
    
    // Dry signal path
    this.dryGain = this.audioContext.createGain();
    this.dryGain.gain.value = 0.5; // 50% dry
    
    // Wet signal path (reverb)
    this.reverbGain = this.audioContext.createGain();
    this.reverbGain.gain.value = 0.5; // 50% wet (will be modulated)
    
    // High-pass filter on reverb for shimmer and clarity
    this.reverbHighPass = this.audioContext.createBiquadFilter();
    this.reverbHighPass.type = 'highpass';
    this.reverbHighPass.frequency.value = 200; // Cut low mud from reverb
    this.reverbHighPass.Q.value = 0.7;
    
    // Create massive reverb
    this.convolver = this.audioContext.createConvolver();
    await this.createReverbImpulse();
    
    // Connect the chain
    this.masterGain.connect(this.lowPassFilter);
    
    // Split to dry and wet
    this.lowPassFilter.connect(this.dryGain);
    this.lowPassFilter.connect(this.convolver);
    
    // Wet path through reverb with high-pass filter
    this.convolver.connect(this.reverbHighPass);
    this.reverbHighPass.connect(this.reverbGain);
    
    // Mix dry and wet to output
    this.dryGain.connect(this.audioContext.destination);
    this.reverbGain.connect(this.audioContext.destination);

    // Create default tracks
    for (const trackConfig of DEFAULT_TRACKS) {
      const track = new AudioTrack(this.audioContext, trackConfig);
      track.getGainNode().connect(this.masterGain);
      this.tracks.set(trackConfig.name, track);
    }

    console.log('✅ Frontend audio engine initialized with reverb');
  }

  /**
   * Create a massive reverb impulse response
   */
  private async createReverbImpulse(): Promise<void> {
    if (!this.audioContext || !this.convolver) return;

    const sampleRate = this.audioContext.sampleRate;
    const length = sampleRate * 6; // 6 second reverb tail (longer)
    const impulse = this.audioContext.createBuffer(2, length, sampleRate);
    const leftChannel = impulse.getChannelData(0);
    const rightChannel = impulse.getChannelData(1);

    // Create a lush, spacey reverb with exponential decay
    for (let i = 0; i < length; i++) {
      const decay = Math.exp(-i / (sampleRate * 2.0)); // 2.0s decay time (longer)
      
      // Add some randomness for diffusion
      const noise = (Math.random() * 2 - 1) * decay;
      
      // Add some early reflections
      const earlyReflection = i < sampleRate * 0.05 ? Math.sin(i * 0.01) * decay : 0;
      
      leftChannel[i] = noise + earlyReflection;
      rightChannel[i] = noise * 0.9 + earlyReflection * 1.1; // Slightly different for stereo width
    }

    this.convolver.buffer = impulse;
  }

  /**
   * Start music generation and playback
   */
  start(): void {
    if (!this.audioContext || !this.masterGain) {
      throw new Error('Audio engine not initialized');
    }

    if (this.isPlaying) {
      console.warn('Already playing');
      return;
    }

    // Resume audio context if suspended
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }

    this.isPlaying = true;
    this.musicGenerator.reset();
    
    // Initialize next step time to current audio time
    this.nextStepTime = this.audioContext.currentTime;

    // Start generation loop with more frequent checks (every 25ms)
    // This ensures we schedule steps ahead of time without drift
    this.generationInterval = window.setInterval(() => {
      this.scheduleSteps();
    }, 25);

    console.log('▶️  Audio engine started');
  }

  /**
   * Stop music generation and playback
   */
  stop(): void {
    if (this.generationInterval !== null) {
      clearInterval(this.generationInterval);
      this.generationInterval = null;
    }

    this.isPlaying = false;
    console.log('⏸️  Audio engine stopped');
  }

  /**
   * Schedule musical steps ahead of time to prevent drift
   */
  private scheduleSteps(): void {
    if (!this.audioContext) return;

    const currentTime = this.audioContext.currentTime;
    const stepDuration = this.stepInterval / 1000; // Convert ms to seconds

    // Schedule all steps that should play within the next scheduleAheadTime
    while (this.nextStepTime < currentTime + this.scheduleAheadTime) {
      this.generateAndPlayStep(this.nextStepTime);
      this.nextStepTime += stepDuration;
    }
  }

  /**
   * Generate and play one musical step at a specific time
   */
  private generateAndPlayStep(scheduleTime: number): void {
    if (!this.audioContext) return;

    const events = this.musicGenerator.generateStep(this.brainState);

    // Check for solo tracks
    const hasSolo = Array.from(this.tracks.values()).some(t => t.solo);

    // Play events on each track
    for (const [trackName, trackEvents] of Object.entries(events)) {
      const track = this.tracks.get(trackName);
      if (!track) continue;

      // Skip if muted or if another track is soloed
      if (track.mute || (hasSolo && !track.solo)) {
        continue;
      }

      // Play all events for this track at the scheduled time
      for (const event of trackEvents) {
        track.playEvent(event, scheduleTime);
      }
    }
  }

  /**
   * Update brain state from EEG data
   */
  updateBrainState(brainState: Partial<BrainState>): void {
    this.brainState = { ...this.brainState, ...brainState };
    this.updateMasterEffects();
  }

  /**
   * Update master effects based on brain state
   */
  private updateMasterEffects(): void {
    if (!this.audioContext || !this.lowPassFilter || !this.reverbGain) return;

    const now = this.audioContext.currentTime;
    
    // Low-pass filter cutoff based on brain state
    // Relax = lower cutoff (darker, warmer)
    // Focus = higher cutoff (brighter, clearer)
    const minCutoff = 800;   // Hz
    const maxCutoff = 4000;  // Hz
    const cutoff = minCutoff + (this.brainState.focus * (maxCutoff - minCutoff));
    
    // Smooth transition
    this.lowPassFilter.frequency.cancelScheduledValues(now);
    this.lowPassFilter.frequency.setTargetAtTime(cutoff, now, 0.1);
    
    // Reverb mix based on brain state
    // Relax = more reverb (spacey, ambient)
    // Focus = less reverb (dry, present)
    const minReverb = 0.3;  // 30% wet
    const maxReverb = 0.8;  // 80% wet
    const reverbMix = minReverb + (this.brainState.relax * (maxReverb - minReverb));
    
    // Adjust dry/wet balance
    this.reverbGain.gain.cancelScheduledValues(now);
    this.reverbGain.gain.setTargetAtTime(reverbMix, now, 0.1);
    
    if (this.dryGain) {
      this.dryGain.gain.cancelScheduledValues(now);
      this.dryGain.gain.setTargetAtTime(1 - reverbMix, now, 0.1);
    }
  }

  /**
   * Set track volume (0-1)
   */
  setTrackVolume(trackName: string, volume: number): void {
    const track = this.tracks.get(trackName);
    if (track) {
      track.setVolume(volume);
      console.log(`🎚️  ${trackName} volume: ${(volume * 100).toFixed(0)}%`);
    }
  }

  /**
   * Set track mute
   */
  setTrackMute(trackName: string, mute: boolean): void {
    const track = this.tracks.get(trackName);
    if (track) {
      track.setMute(mute);
      console.log(`🔇 ${trackName} ${mute ? 'muted' : 'unmuted'}`);
    }
  }

  /**
   * Set track solo
   */
  setTrackSolo(trackName: string, solo: boolean): void {
    const track = this.tracks.get(trackName);
    if (track) {
      track.setSolo(solo);
      console.log(`🎯 ${trackName} ${solo ? 'soloed' : 'unsoloed'}`);
    }
  }

  /**
   * Set master volume (0-1)
   */
  setMasterVolume(volume: number): void {
    if (this.masterGain && this.audioContext) {
      const clampedVolume = Math.max(0, Math.min(1, volume));
      this.masterGain.gain.setValueAtTime(clampedVolume, this.audioContext.currentTime);
      console.log(`🎚️  Master volume: ${(clampedVolume * 100).toFixed(0)}%`);
    }
  }

  /**
   * Change track synthesizer type
   */
  setTrackSynthType(trackName: string, synthType: WaveType): void {
    const track = this.tracks.get(trackName);
    if (track) {
      track.setSynthType(synthType);
      console.log(`🎹 ${trackName} synth: ${synthType}`);
    }
  }

  /**
   * Get current state
   */
  getState() {
    return {
      isPlaying: this.isPlaying,
      brainState: this.brainState,
      tracks: Array.from(this.tracks.entries()).map(([name, track]) => ({
        name,
        volume: track.volume,
        mute: track.mute,
        solo: track.solo,
      })),
      masterVolume: this.masterGain?.gain.value || 0.8,
    };
  }

  /**
   * Cleanup
   */
  dispose(): void {
    this.stop();
    
    for (const track of this.tracks.values()) {
      track.dispose();
    }
    
    this.tracks.clear();
    
    if (this.masterGain) {
      this.masterGain.disconnect();
    }
    
    if (this.audioContext) {
      this.audioContext.close();
    }

    console.log('🗑️  Audio engine disposed');
  }
}
