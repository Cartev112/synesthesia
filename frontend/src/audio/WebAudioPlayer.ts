/**
 * Web Audio API player for music events.
 * 
 * Receives MIDI-like events from backend and plays them using Web Audio API.
 */

interface MidiEvent {
  note: number;
  velocity: number;
  duration: number;
  time: number;
}

interface MusicEvents {
  [layer: string]: MidiEvent[];
}

export class WebAudioPlayer {
  private audioContext: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private isInitialized = false;
  private nextPlayTime = 0;
  private pendingBuffers: Array<{
    audioData: string;
    sampleRate: number;
    channels: number;
    dtype: string;
    shape: number[];
  }> = [];
  
  constructor() {
    // Audio context will be created on first user interaction
  }
  
  /**
   * Initialize audio context (must be called after user interaction)
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;
    
    try {
      this.audioContext = new AudioContext();
      this.masterGain = this.audioContext.createGain();
      this.masterGain.gain.value = 0.3; // Master volume
      this.masterGain.connect(this.audioContext.destination);
      
      this.isInitialized = true;
      console.log('dYZ� Web Audio API initialized');
      
      // Flush any buffers that arrived before user interaction
      this.flushPendingBuffers();
    } catch (error) {
      console.error('Failed to initialize Web Audio API:', error);
      throw error;
    }
  }
  
  /**
   * Play audio buffer from backend
   */
  playAudioBuffer(audioData: string, sampleRate: number, channels: number, _dtype: string, shape: number[]): void {
    if (!this.isInitialized || !this.audioContext || !this.masterGain) {
      // Try to initialize automatically (may still need a user gesture)
      this.initialize().catch(() => {
        console.warn('Audio not initialized - waiting for user gesture');
      });
      
      // Queue buffer so we can play once initialized
      this.enqueuePendingBuffer({ audioData, sampleRate, channels, dtype: _dtype, shape });
      return;
    }
    
    // Ensure the context is running (can suspend after tab inactivity)
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume().catch(() => {
        console.warn('Audio context resume blocked');
      });
    }
    
    console.log('dYZ� Playing audio buffer:', { sampleRate, channels, shape, bufferLength: shape[0] });
    
    try {
      // Decode base64 to binary
      const binaryString = atob(audioData);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      // Convert to Float32Array (assuming float32 from backend)
      const floatArray = new Float32Array(bytes.buffer);
      
      // Create audio buffer
      const bufferLength = shape[0]; // Number of samples
      const audioBuffer = this.audioContext.createBuffer(channels, bufferLength, sampleRate);
      
      // Fill audio buffer
      if (channels === 1) {
        audioBuffer.copyToChannel(floatArray, 0);
      } else {
        // Stereo - deinterleave
        const leftChannel = new Float32Array(bufferLength);
        const rightChannel = new Float32Array(bufferLength);
        for (let i = 0; i < bufferLength; i++) {
          leftChannel[i] = floatArray[i * 2];
          rightChannel[i] = floatArray[i * 2 + 1];
        }
        audioBuffer.copyToChannel(leftChannel, 0);
        audioBuffer.copyToChannel(rightChannel, 1);
      }
      
      // Schedule playback
      const source = this.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.masterGain);
      
      // Calculate when to play
      const now = this.audioContext.currentTime;
      const bufferDuration = bufferLength / sampleRate;
      
      // If we're behind schedule, catch up but don't create gaps
      // Allow small latency (50ms) to avoid glitches
      if (this.nextPlayTime < now - 0.05) {
        // We've fallen behind - schedule immediately but maintain continuity
        this.nextPlayTime = now;
        console.warn('⚠️ Audio scheduling behind, resetting to current time');
      } else if (this.nextPlayTime < now) {
        // Slightly behind but within tolerance - schedule ASAP
        this.nextPlayTime = now;
      }
      
      source.start(this.nextPlayTime);
      this.nextPlayTime += bufferDuration;
      
    } catch (error) {
      console.error('Failed to play audio buffer:', error);
    }
  }
  
  private enqueuePendingBuffer(buffer: { audioData: string; sampleRate: number; channels: number; dtype: string; shape: number[] }) {
    // Keep queue very small to minimize latency (max 2 buffers)
    if (this.pendingBuffers.length > 2) {
      this.pendingBuffers.shift();
    }
    this.pendingBuffers.push(buffer);
  }
  
  private flushPendingBuffers() {
    if (!this.isInitialized || !this.audioContext || !this.masterGain) return;
    
    const buffersToPlay = [...this.pendingBuffers];
    this.pendingBuffers = [];
    
    buffersToPlay.forEach(buf => {
      this.playAudioBuffer(buf.audioData, buf.sampleRate, buf.channels, buf.dtype, buf.shape);
    });
  }
  
  /**
   * Play music events (legacy - for MIDI-like events)
   */
  playEvents(events: MusicEvents): void {
    if (!this.isInitialized || !this.audioContext || !this.masterGain) {
      console.warn('Audio not initialized - call initialize() first');
      return;
    }
    
    const now = this.audioContext.currentTime;
    
    // Play each layer
    for (const [layer, layerEvents] of Object.entries(events)) {
      for (const event of layerEvents) {
        this.playNote(event, now, layer);
      }
    }
  }
  
  /**
   * Play a single note
   */
  private playNote(event: MidiEvent, startTime: number, layer: string): void {
    if (!this.audioContext || !this.masterGain) return;
    
    const frequency = this.midiToFrequency(event.note);
    const duration = event.duration;
    const velocity = event.velocity / 127; // Normalize to 0-1
    
    // Create oscillator
    const oscillator = this.audioContext.createOscillator();
    const gainNode = this.audioContext.createGain();
    
    // Set waveform based on layer
    oscillator.type = this.getWaveformForLayer(layer);
    oscillator.frequency.value = frequency;
    
    // ADSR envelope
    const attackTime = 0.01;
    const decayTime = 0.1;
    const sustainLevel = 0.7;
    const releaseTime = 0.1;
    
    const peakTime = startTime + attackTime;
    const sustainTime = startTime + attackTime + decayTime;
    const endTime = startTime + duration;
    const releaseEndTime = endTime + releaseTime;
    
    // Envelope
    gainNode.gain.setValueAtTime(0, startTime);
    gainNode.gain.linearRampToValueAtTime(velocity, peakTime);
    gainNode.gain.linearRampToValueAtTime(velocity * sustainLevel, sustainTime);
    gainNode.gain.setValueAtTime(velocity * sustainLevel, endTime);
    gainNode.gain.linearRampToValueAtTime(0, releaseEndTime);
    
    // Connect nodes
    oscillator.connect(gainNode);
    gainNode.connect(this.masterGain);
    
    // Start and stop
    oscillator.start(startTime);
    oscillator.stop(releaseEndTime);
  }
  
  /**
   * Convert MIDI note number to frequency
   */
  private midiToFrequency(note: number): number {
    return 440 * Math.pow(2, (note - 69) / 12);
  }
  
  /**
   * Get waveform type for layer
   */
  private getWaveformForLayer(layer: string): OscillatorType {
    switch (layer) {
      case 'bass':
        return 'sawtooth';
      case 'harmony':
        return 'sine';
      case 'melody':
        return 'triangle';
      case 'texture':
        return 'square';
      default:
        return 'sine';
    }
  }
  
  /**
   * Set master volume
   */
  setVolume(volume: number): void {
    if (this.masterGain) {
      this.masterGain.gain.value = Math.max(0, Math.min(1, volume));
    }
  }
  
  /**
   * Resume audio context (needed after page becomes inactive)
   */
  async resume(): Promise<void> {
    if (this.audioContext && this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
      console.log('dYZ� Audio context resumed');
    }
  }
  
  /**
   * Close audio context
   */
  async close(): Promise<void> {
    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
      this.masterGain = null;
      this.isInitialized = false;
      console.log('dYZ� Audio context closed');
    }
  }
}
