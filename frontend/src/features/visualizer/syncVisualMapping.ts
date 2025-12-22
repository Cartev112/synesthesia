/**
 * Sync-to-Visual Parameter Mapping
 * 
 * Maps synchronization state between users to visual parameters.
 * Key concept: When synced, visuals are harmonious and unified.
 * When desynced, visuals show "disagreement" - layers rotate at different speeds,
 * colors shift warmer, and there's more visual tension.
 */

export interface SyncVisualState {
  sync_score: number;       // 0-1 (1 = perfectly synced)
  dissonance_level: number; // 0-1 (0 = consonant, 1 = dissonant)
}

export interface SyncVisualOverrides {
  // Color shifts
  hue_shift: number;           // Added to base hue (-60 to +60)
  saturation_mult: number;     // Multiplier for saturation (0.7 - 1.3)
  brightness_mult: number;     // Multiplier for brightness (0.8 - 1.2)
  
  // Portal-specific: Layer rotation divergence
  // When synced = 0, all layers rotate together
  // When desynced > 0, layers rotate at increasingly different speeds
  layer_rotation_divergence: number;  // 0-1, how much layer speeds differ
  
  // Symmetry and warp
  symmetry_reduction: number;  // 0-1, reduce portal symmetry when desynced
  warp_increase: number;       // Additional warp when desynced
  ripple_intensity: number;    // Ripple multiplier
  
  // Motion
  jitter_amount: number;       // Random phase jitter (0-0.1)
  pulse_intensity: number;     // Pulse amplitude multiplier
  
  // Trails
  trail_fragmentation: number; // Reduce trail length when desynced (0-0.3)
}

// Smoothing state for gradual transitions
let smoothedSyncScore = 0.5;
let smoothedDissonance = 0.5;
const SMOOTHING_FACTOR = 0.08; // Lower = slower transitions

/**
 * Smooth a value towards a target using exponential smoothing
 */
function smoothValue(current: number, target: number, factor: number): number {
  return current + (target - current) * factor;
}

/**
 * Map sync state to visual parameter overrides
 * 
 * @param syncState Current sync state (or null for solo mode)
 * @returns Visual parameter overrides to apply
 */
export function mapSyncToVisualParams(syncState: SyncVisualState | null): SyncVisualOverrides {
  // Default neutral values for solo mode
  if (!syncState) {
    // Gradually return to neutral
    smoothedSyncScore = smoothValue(smoothedSyncScore, 0.5, SMOOTHING_FACTOR);
    smoothedDissonance = smoothValue(smoothedDissonance, 0.5, SMOOTHING_FACTOR);
    
    return {
      hue_shift: 0,
      saturation_mult: 1.0,
      brightness_mult: 1.0,
      layer_rotation_divergence: 0,
      symmetry_reduction: 0,
      warp_increase: 0,
      ripple_intensity: 1.0,
      jitter_amount: 0,
      pulse_intensity: 1.0,
      trail_fragmentation: 0,
    };
  }
  
  // Apply smoothing to prevent jarring visual changes
  smoothedSyncScore = smoothValue(smoothedSyncScore, syncState.sync_score, SMOOTHING_FACTOR);
  smoothedDissonance = smoothValue(smoothedDissonance, syncState.dissonance_level, SMOOTHING_FACTOR);
  
  const sync = smoothedSyncScore;
  const desync = smoothedDissonance;
  
  // === COLOR MAPPING ===
  // Synced: Cool colors (cyan/blue, hue ~180-220)
  // Desynced: Warm colors (orange/red, hue ~0-40)
  // Shift range: -60 (towards red) to +40 (towards blue)
  const hue_shift = (sync - 0.5) * 80; // -40 to +40
  
  // Synced: Smoother, more harmonious saturation
  // Desynced: Higher contrast, more aggressive
  const saturation_mult = 0.85 + desync * 0.3; // 0.85 - 1.15
  
  // Synced: Brighter, cleaner
  // Desynced: Slightly darker, moodier
  const brightness_mult = 1.1 - desync * 0.25; // 0.85 - 1.1
  
  // === LAYER ROTATION DIVERGENCE (Key visual indicator!) ===
  // When sync_score is high, all layers rotate at same speed (divergence = 0)
  // When sync_score is low, layers rotate at different speeds (divergence = 1)
  // This creates visual "disagreement" that's immediately perceivable
  const layer_rotation_divergence = Math.pow(1 - sync, 1.5); // 0 to 1, emphasize low sync
  
  // === SYMMETRY AND WARP ===
  // Synced: High symmetry, low warp - harmonious
  // Desynced: Break symmetry, increase warp - chaotic
  const symmetry_reduction = desync * 0.4; // Can reduce symmetry by up to 40%
  const warp_increase = desync * 0.15; // Add up to 0.15 extra warp
  const ripple_intensity = 1.0 + desync * 0.5; // 1.0 - 1.5
  
  // === MOTION ===
  // Synced: Smooth, predictable motion
  // Desynced: Add jitter and pulsing
  const jitter_amount = desync * 0.05; // 0 - 0.05 phase jitter
  const pulse_intensity = 1.0 + desync * 0.4; // 1.0 - 1.4
  
  // === TRAILS ===
  // Synced: Long, smooth trails
  // Desynced: Shorter, more fragmented
  const trail_fragmentation = desync * 0.25; // Reduce trail by up to 25%
  
  return {
    hue_shift,
    saturation_mult,
    brightness_mult,
    layer_rotation_divergence,
    symmetry_reduction,
    warp_increase,
    ripple_intensity,
    jitter_amount,
    pulse_intensity,
    trail_fragmentation,
  };
}

/**
 * Reset smoothing state (call when switching modes)
 */
export function resetSyncVisualSmoothing(): void {
  smoothedSyncScore = 0.5;
  smoothedDissonance = 0.5;
}

/**
 * Get a descriptive label for the current sync state
 */
export function getSyncVisualLabel(syncState: SyncVisualState | null): {
  label: string;
  color: string;
} {
  if (!syncState) {
    return { label: 'SOLO', color: '#888888' };
  }
  
  const score = syncState.sync_score;
  
  if (score >= 0.75) {
    return { label: 'HARMONIZED', color: '#00ff88' };
  } else if (score >= 0.5) {
    return { label: 'ALIGNED', color: '#00ffff' };
  } else if (score >= 0.3) {
    return { label: 'DRIFTING', color: '#ffaa00' };
  } else {
    return { label: 'DISSONANT', color: '#ff4444' };
  }
}

