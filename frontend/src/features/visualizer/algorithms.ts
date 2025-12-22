/**
 * Visual algorithm implementations for frontend rendering
 */

// Sync visual overrides passed from syncVisualMapping
export interface SyncVisualOverrides {
  hue_shift: number;
  saturation_mult: number;
  brightness_mult: number;
  layer_rotation_divergence: number;
  symmetry_reduction: number;
  warp_increase: number;
  ripple_intensity: number;
  jitter_amount: number;
  pulse_intensity: number;
  trail_fragmentation: number;
}

export interface VisualParams {
  frequency_ratio_x?: number;
  frequency_ratio_y?: number;
  phase_offset?: number;
  amplitude_x?: number;
  amplitude_y?: number;
  rotation_speed?: number;
  num_harmonics?: number;
  hue_base?: number;
  saturation?: number;
  brightness?: number;
  color_cycle_speed?: number;
  point_density?: number;
  trail_length?: number;
  speed_multiplier?: number;
  pulse_frequency?: number;
  pulse_amplitude?: number;
  damping_x?: number;
  damping_y?: number;
  num_epicycles?: number;
  epicycle_decay?: number;
  portal_symmetry?: number;
  portal_radial_frequency?: number;
  portal_angular_frequency?: number;
  portal_warp?: number;
  portal_spiral?: number;
  portal_layers?: number;
  portal_radius?: number;
  portal_ripple?: number;
  portal_depth_skew?: number;
  _accumulatedRotation?: number;
  _syncOverrides?: SyncVisualOverrides;
  [key: string]: any;
}

export interface ColoredLayer {
  path: Path2D;
  color: string;
  lineWidth?: number;
  shadowBlur?: number;
  alpha?: number;
  composite?: GlobalCompositeOperation;
}

export interface RenderResult {
  layers?: ColoredLayer[];
}

export type AlgorithmType = 'lissajous' | 'harmonograph' | 'lorenz' | 'reaction_diffusion' | 'hyperspace_portal';

export interface Point {
  x: number;
  y: number;
}

/**
 * Lissajous curve renderer
 * x(t) = A_x * sin(ω_x * t + φ)
 * y(t) = A_y * sin(ω_y * t)
 * 
 * SYNC EFFECTS: When desynced, adds phase wobble and amplitude jitter
 */
export function renderLissajous(
  ctx: CanvasRenderingContext2D,
  params: VisualParams,
  width: number,
  height: number,
  time: number
): RenderResult | void {
  // Get sync overrides
  const sync = params._syncOverrides ?? {
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
  
  const freqX = params.frequency_ratio_x ?? 3.0;
  const freqY = params.frequency_ratio_y ?? 2.0;
  const phaseOffset = params.phase_offset ?? 0;
  const ampX = params.amplitude_x ?? 0.8;
  const ampY = params.amplitude_y ?? 0.8;
  const pointDensity = params.point_density ?? 1024;
  
  const centerX = width / 2;
  const centerY = height / 2;
  const scale = Math.min(width, height) * 0.35;
  
  // Sync effects: phase wobble when desynced
  const divergence = sync.layer_rotation_divergence;
  const phaseWobble = divergence * 0.3 * Math.sin(time * 3);
  const ampWobble = 1 + divergence * 0.15 * Math.sin(time * 5);
  
  ctx.beginPath();
  
  for (let i = 0; i < pointDensity; i++) {
    const t = (i / pointDensity) * Math.PI * 4 + time * 0.5;
    
    // Add jitter to phase when desynced
    const jitter = sync.jitter_amount * Math.sin(t * 10) * 0.5;
    
    const x = ampX * ampWobble * Math.sin(freqX * t + phaseOffset + phaseWobble + jitter);
    const y = ampY * Math.sin(freqY * t + jitter * 0.7);
    
    const px = centerX + x * scale;
    const py = centerY + y * scale;
    
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
}

/**
 * Harmonograph renderer (damped pendulum)
 * x(t) = Σ A_i * sin(ω_i * t + φ_i) * exp(-d_i * t)
 * y(t) = Σ B_i * sin(ν_i * t + ψ_i) * exp(-e_i * t)
 * 
 * SYNC EFFECTS: When desynced, harmonics become misaligned and damping becomes uneven
 */
export function renderHarmonograph(
  ctx: CanvasRenderingContext2D,
  params: VisualParams,
  width: number,
  height: number,
  time: number
): RenderResult | void {
  // Get sync overrides
  const sync = params._syncOverrides ?? {
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
  
  const freqX = params.frequency_ratio_x ?? 3.0;
  const freqY = params.frequency_ratio_y ?? 2.0;
  const phaseOffset = params.phase_offset ?? 0;
  const ampX = params.amplitude_x ?? 0.8;
  const ampY = params.amplitude_y ?? 0.8;
  const numHarmonics = params.num_harmonics ?? 5;
  const pointDensity = params.point_density ?? 1024;
  const pulseFreq = params.pulse_frequency ?? 1.0;
  
  // Increase pulse when desynced
  const basePulseAmp = params.pulse_amplitude ?? 0.0;
  const pulseAmp = basePulseAmp * sync.pulse_intensity;
  
  const dampX = params.damping_x ?? 0.03;
  const dampY = params.damping_y ?? 0.03;
  
  const centerX = width / 2;
  const centerY = height / 2;
  const scale = Math.min(width, height) * 0.35;
  const rotation = params._accumulatedRotation ?? 0;
  
  // Sync effects
  const divergence = sync.layer_rotation_divergence;
  const jitter = sync.jitter_amount;
  
  ctx.beginPath();
  
  for (let i = 0; i < pointDensity; i++) {
    const angle = (i / pointDensity) * Math.PI * 2 * 3;
    
    let x = 0;
    let y = 0;
    
    for (let h = 0; h < numHarmonics; h++) {
      const harmonic = h + 1;
      
      // When desynced, each harmonic has slightly different damping (creates visual discord)
      const dampingMod = 1 + divergence * 0.5 * (h % 2 === 0 ? 1 : -1);
      const dampingFactorX = Math.exp(-dampX * dampingMod * angle * harmonic);
      const dampingFactorY = Math.exp(-dampY * dampingMod * angle * harmonic);
      
      // Add phase noise per harmonic when desynced
      const harmonicPhaseNoise = jitter * Math.sin(time * 7 + h * 3) * 0.3;
      
      x += (ampX / harmonic) * Math.sin(freqX * angle * harmonic + phaseOffset + harmonicPhaseNoise) * dampingFactorX;
      y += (ampY / harmonic) * Math.sin(freqY * angle * harmonic + harmonicPhaseNoise * 0.7) * dampingFactorY;
    }
    
    // Pulse effect (stronger when desynced)
    const pulse = 1 + pulseAmp * Math.sin(time * pulseFreq * Math.PI * 2);
    x *= pulse;
    y *= pulse;
    
    // Rotation with slight wobble when desynced
    const rotWobble = divergence * 0.05 * Math.sin(time * 4);
    const cosR = Math.cos(rotation + rotWobble);
    const sinR = Math.sin(rotation + rotWobble);
    const xRot = x * cosR - y * sinR;
    const yRot = x * sinR + y * cosR;
    
    const px = centerX + xRot * scale;
    const py = centerY + yRot * scale;
    
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
}

/**
 * Lorenz attractor renderer
 * Chaotic strange attractor system
 * 
 * SYNC EFFECTS: When desynced, the attractor becomes more chaotic with parameter perturbation
 */
export function renderLorenz(
  ctx: CanvasRenderingContext2D,
  params: VisualParams,
  width: number,
  height: number,
  time: number
): RenderResult | void {
  // Get sync overrides
  const sync = params._syncOverrides ?? {
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
  
  const freqX = params.frequency_ratio_x ?? 3.0;
  const freqY = params.frequency_ratio_y ?? 2.0;
  const phaseOffset = params.phase_offset ?? 0;
  const pointDensity = params.point_density ?? 2048;
  
  // Sync effect: perturb Lorenz parameters when desynced (more chaotic)
  const divergence = sync.layer_rotation_divergence;
  const jitter = sync.jitter_amount;
  const paramPerturbation = divergence * 2.0 * Math.sin(time * 0.5);
  
  // Lorenz parameters (with sync perturbation)
  const sigma = 10.0 + freqX * 2.0 + paramPerturbation;
  const rho = 20.0 + freqY * 5.0 + paramPerturbation * 0.5;
  const beta = 2.667 + phaseOffset;
  
  const centerX = width / 2;
  const centerY = height / 2;
  const scale = Math.min(width, height) * 0.012;
  
  // Initial conditions (vary with time for animation, add jitter when desynced)
  let x = 1.0 + Math.sin(time * 0.1) * 0.1 + jitter * Math.sin(time * 7);
  let y = 1.0 + Math.cos(time * 0.1) * 0.1 + jitter * Math.cos(time * 7);
  let z = 1.0;
  
  const dt = 0.01;
  
  // Rotation for animation (with wobble when desynced)
  const rotation = params._accumulatedRotation ?? 0;
  const rotWobble = divergence * 0.03 * Math.sin(time * 3);
  const cosR = Math.cos(rotation + rotWobble);
  const sinR = Math.sin(rotation + rotWobble);
  
  // Store points for proper centering
  const points: Point[] = [];
  
  // Skip initial transient
  for (let i = 0; i < 100; i++) {
    const dx = sigma * (y - x) * dt;
    const dy = (x * (rho - z) - y) * dt;
    const dz = (x * y - beta * z) * dt;
    x += dx;
    y += dy;
    z += dz;
  }
  
  // Generate points
  for (let i = 0; i < pointDensity; i++) {
    const dx = sigma * (y - x) * dt;
    const dy = (x * (rho - z) - y) * dt;
    const dz = (x * y - beta * z) * dt;
    
    x += dx;
    y += dy;
    z += dz;
    
    points.push({ x, y: z });
  }
  
  // Find bounds for proper centering
  const xVals = points.map(p => p.x);
  const yVals = points.map(p => p.y);
  const xMin = Math.min(...xVals);
  const xMax = Math.max(...xVals);
  const yMin = Math.min(...yVals);
  const yMax = Math.max(...yVals);
  const xMid = (xMin + xMax) / 2;
  const yMid = (yMin + yMax) / 2;
  
  ctx.beginPath();
  
  // Render centered and rotated
  points.forEach((point, i) => {
    // Center the attractor
    let px = (point.x - xMid) * scale;
    let py = (point.y - yMid) * scale;
    
    // Apply rotation
    const pxRot = px * cosR - py * sinR;
    const pyRot = px * sinR + py * cosR;
    
    const screenX = centerX + pxRot;
    const screenY = centerY + pyRot;
    
    if (i === 0) ctx.moveTo(screenX, screenY);
    else ctx.lineTo(screenX, screenY);
  });
}


export function renderReactionDiffusion(
  _ctx: CanvasRenderingContext2D,
  _params: VisualParams,
  _width: number,
  _height: number,
  _time: number
): RenderResult | void {
  /* Reaction diffusion renderer disabled */
  return;
}

/**
 * Hyperspace portal renderer
 * Layered polar waves with spiral warp and radial symmetry
 * 
 * SYNC VISUAL INDICATOR: Layer rotation divergence
 * - When synced (divergence = 0): All layers rotate at the same speed = visual harmony
 * - When desynced (divergence > 0): Layers rotate at different speeds = visual "disagreement"
 */
export function renderHyperspacePortal(
  _ctx: CanvasRenderingContext2D,
  params: VisualParams,
  width: number,
  height: number,
  time: number
): RenderResult {
  // Get sync overrides (or use neutral defaults)
  const sync = params._syncOverrides ?? {
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
  
  // Base parameters with sync modifications
  const baseSymmetry = params.portal_symmetry ?? 6;
  // Reduce symmetry when desynced (makes it look more chaotic)
  const symmetry = Math.max(3, Math.floor(baseSymmetry * (1 - sync.symmetry_reduction)));
  
  const radialFreq = params.portal_radial_frequency ?? 6.0;
  const angularFreq = params.portal_angular_frequency ?? 2.0;
  
  // Increase warp when desynced
  const baseWarp = params.portal_warp ?? 0.15;
  const warp = baseWarp + sync.warp_increase;
  
  const spiral = params.portal_spiral ?? -1.5;
  const layers = Math.max(2, Math.floor(params.portal_layers ?? 4));
  const baseRadius = params.portal_radius ?? 0.48;
  
  // Increase ripple intensity when desynced
  const baseRipple = params.portal_ripple ?? 0.2;
  const ripple = baseRipple * sync.ripple_intensity;
  
  const depthSkew = params.portal_depth_skew ?? 0.35;
  const pointDensity = params.point_density ?? 1800;
  
  // Increase pulse when desynced
  const basePulseAmp = params.pulse_amplitude ?? 0.0;
  const pulseAmp = basePulseAmp * sync.pulse_intensity;
  const pulseFreq = params.pulse_frequency ?? 1.0;

  const centerX = width / 2;
  const centerY = height / 2;
  const scale = Math.min(width, height) * 0.48;
  const tunnelPulse = 1 + pulseAmp * Math.sin(time * pulseFreq * Math.PI * 2);
  const rotation = params._accumulatedRotation ?? 0;
  const densityPerLayer = Math.max(64, Math.floor(pointDensity / layers));

  // Apply sync hue shift
  const baseHue = params.hue_base ?? 180;
  const hueBase = baseHue + sync.hue_shift;
  
  // Apply sync saturation/brightness modifications
  const baseSat = params.saturation ?? 0.7;
  const saturation = Math.min(1, Math.max(0.3, baseSat * sync.saturation_mult));
  
  const baseBright = params.brightness ?? 0.8;
  const brightness = Math.min(1, Math.max(0.4, baseBright * sync.brightness_mult));
  
  const colorCycleSpeed = params.color_cycle_speed ?? 0.2;

  const layersOut: ColoredLayer[] = [];
  
  // === KEY SYNC INDICATOR: Layer Rotation Divergence ===
  // When synced: all layers rotate at the same speed (rotation)
  // When desynced: each layer has its own rotation speed offset
  // This creates visual "disagreement" between layers
  const divergence = sync.layer_rotation_divergence;

  for (let layer = 0; layer < layers; layer++) {
    const depth = layer / Math.max(layers - 1, 1);
    const layerOffset = depth * 1.1;
    const depthRadius = baseRadius * (1 + depth * (0.7 + depthSkew));
    const turbulence = 1 + 0.1 * Math.cos(depth * 3 + time * 0.6);
    const hue = (hueBase + depth * 90 + time * colorCycleSpeed * 80) % 360;
    const path = new Path2D();
    
    // === LAYER ROTATION DIVERGENCE ===
    // Each layer gets a different rotation speed modifier based on divergence
    // Layer 0: base speed
    // Other layers: progressively different speeds based on divergence
    // Pattern: alternate faster/slower to create visible "disagreement"
    const layerSpeedMod = 1 + (layer % 2 === 0 ? 1 : -1) * divergence * (0.3 + depth * 0.4);
    const layerRotation = rotation * layerSpeedMod;
    
    // Add jitter when desynced (creates micro-tremors in the animation)
    const jitter = sync.jitter_amount * Math.sin(time * 20 + layer * 7) * Math.PI;

    for (let i = 0; i < densityPerLayer; i++) {
      const theta = (i / densityPerLayer) * Math.PI * 2 * angularFreq;
      const swirl = theta + spiral * (time * 0.45 + depth * 1.4) + layerRotation + layerOffset + jitter;

      const radialWave = Math.sin(theta * radialFreq + time * 0.8 + layerOffset) * ripple;
      const symmetryWave = Math.sin(theta * symmetry + layerRotation) * warp * (1 - depth * 0.35);
      const radius = depthRadius * (1 + radialWave + symmetryWave) * tunnelPulse * turbulence;

      const x = centerX + Math.cos(swirl) * radius * scale;
      const y = centerY + Math.sin(swirl) * radius * scale;

      if (i === 0) path.moveTo(x, y);
      else path.lineTo(x, y);
    }

    layersOut.push({
      path,
      color: hslToRgb(hue, saturation, brightness),
      lineWidth: 1.5 + (1 - depth) * 2,
      shadowBlur: 10 + depth * 25,
      alpha: 0.85 - depth * 0.35
    });
  }

  // Add radial flares for portal spokes
  const flareCount = Math.max(4, Math.floor(symmetry / 2));
  const flareRadius = scale * (baseRadius + 0.4);
  const flareHue = (hueBase + 200 + time * colorCycleSpeed * 120) % 360;

  for (let i = 0; i < flareCount; i++) {
    // Flares also affected by divergence - wobble more when desynced
    const flareJitter = divergence * 0.1 * Math.sin(time * 15 + i * 5);
    const angle = (i / flareCount) * Math.PI * 2 + rotation * 0.6 + flareJitter;
    const wobble = Math.sin(time * 0.7 + i) * (0.15 + divergence * 0.1);
    const innerR = scale * (baseRadius * 0.35);
    const outerR = flareRadius * (1 + wobble);

    const innerX = centerX + Math.cos(angle) * innerR;
    const innerY = centerY + Math.sin(angle) * innerR;
    const outerX = centerX + Math.cos(angle) * outerR;
    const outerY = centerY + Math.sin(angle) * outerR;

    const path = new Path2D();
    path.moveTo(innerX, innerY);
    path.lineTo(outerX, outerY);

    layersOut.push({
      path,
      color: hslToRgb(flareHue, Math.min(1, saturation + 0.15), 0.9),
      lineWidth: 2.5,
      shadowBlur: 25,
      alpha: 0.75
    });
  }

  return { layers: layersOut };
}

/**
 * HSL to RGB conversion
 */
export function hslToRgb(h: number, s: number, l: number): string {
  h = h / 360;
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  
  const hue2rgb = (t: number) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1/6) return p + (q - p) * 6 * t;
    if (t < 1/2) return q;
    if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
    return p;
  };
  
  const r = Math.round(hue2rgb(h + 1/3) * 255);
  const g = Math.round(hue2rgb(h) * 255);
  const b = Math.round(hue2rgb(h - 1/3) * 255);
  
  return `rgb(${r}, ${g}, ${b})`;
}
