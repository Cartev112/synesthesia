/**
 * Visual algorithm implementations for frontend rendering
 */

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
 */
export function renderLissajous(
  ctx: CanvasRenderingContext2D,
  params: VisualParams,
  width: number,
  height: number,
  time: number
): RenderResult | void {
  const freqX = params.frequency_ratio_x ?? 3.0;
  const freqY = params.frequency_ratio_y ?? 2.0;
  const phaseOffset = params.phase_offset ?? 0;
  const ampX = params.amplitude_x ?? 0.8;
  const ampY = params.amplitude_y ?? 0.8;
  const pointDensity = params.point_density ?? 1024;
  
  const centerX = width / 2;
  const centerY = height / 2;
  const scale = Math.min(width, height) * 0.35;
  
  ctx.beginPath();
  
  for (let i = 0; i < pointDensity; i++) {
    const t = (i / pointDensity) * Math.PI * 4 + time * 0.5;
    
    const x = ampX * Math.sin(freqX * t + phaseOffset);
    const y = ampY * Math.sin(freqY * t);
    
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
 */
export function renderHarmonograph(
  ctx: CanvasRenderingContext2D,
  params: VisualParams,
  width: number,
  height: number,
  time: number
): RenderResult | void {
  const freqX = params.frequency_ratio_x ?? 3.0;
  const freqY = params.frequency_ratio_y ?? 2.0;
  const phaseOffset = params.phase_offset ?? 0;
  const ampX = params.amplitude_x ?? 0.8;
  const ampY = params.amplitude_y ?? 0.8;
  const numHarmonics = params.num_harmonics ?? 5;
  const pointDensity = params.point_density ?? 1024;
  const pulseFreq = params.pulse_frequency ?? 1.0;
  const pulseAmp = params.pulse_amplitude ?? 0.0;
  const dampX = params.damping_x ?? 0.03;
  const dampY = params.damping_y ?? 0.03;
  
  const centerX = width / 2;
  const centerY = height / 2;
  const scale = Math.min(width, height) * 0.35;
  const rotation = params._accumulatedRotation ?? 0;
  
  ctx.beginPath();
  
  for (let i = 0; i < pointDensity; i++) {
    const angle = (i / pointDensity) * Math.PI * 2 * 3;
    
    let x = 0;
    let y = 0;
    
    for (let h = 0; h < numHarmonics; h++) {
      const harmonic = h + 1;
      const dampingFactorX = Math.exp(-dampX * angle * harmonic);
      const dampingFactorY = Math.exp(-dampY * angle * harmonic);
      
      x += (ampX / harmonic) * Math.sin(freqX * angle * harmonic + phaseOffset) * dampingFactorX;
      y += (ampY / harmonic) * Math.sin(freqY * angle * harmonic) * dampingFactorY;
    }
    
    // Pulse effect
    const pulse = 1 + pulseAmp * Math.sin(time * pulseFreq * Math.PI * 2);
    x *= pulse;
    y *= pulse;
    
    // Rotation
    const cosR = Math.cos(rotation);
    const sinR = Math.sin(rotation);
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
 */
export function renderLorenz(
  ctx: CanvasRenderingContext2D,
  params: VisualParams,
  width: number,
  height: number,
  time: number
): RenderResult | void {
  const freqX = params.frequency_ratio_x ?? 3.0;
  const freqY = params.frequency_ratio_y ?? 2.0;
  const phaseOffset = params.phase_offset ?? 0;
  const pointDensity = params.point_density ?? 2048;
  
  // Lorenz parameters
  const sigma = 10.0 + freqX * 2.0;
  const rho = 20.0 + freqY * 5.0;
  const beta = 2.667 + phaseOffset;
  
  const centerX = width / 2;
  const centerY = height / 2;
  const scale = Math.min(width, height) * 0.012;
  
  // Initial conditions (vary with time for animation)
  let x = 1.0 + Math.sin(time * 0.1) * 0.1;
  let y = 1.0 + Math.cos(time * 0.1) * 0.1;
  let z = 1.0;
  
  const dt = 0.01;
  
  // Rotation for animation
  const rotation = params._accumulatedRotation ?? 0;
  const cosR = Math.cos(rotation);
  const sinR = Math.sin(rotation);
  
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
 */
export function renderHyperspacePortal(
  _ctx: CanvasRenderingContext2D,
  params: VisualParams,
  width: number,
  height: number,
  time: number
): RenderResult {
  const symmetry = Math.max(3, Math.floor(params.portal_symmetry ?? 6));
  const radialFreq = params.portal_radial_frequency ?? 6.0;
  const angularFreq = params.portal_angular_frequency ?? 2.0;
  const warp = params.portal_warp ?? 0.15;
  const spiral = params.portal_spiral ?? -1.5;
  const layers = Math.max(2, Math.floor(params.portal_layers ?? 4));
  const baseRadius = params.portal_radius ?? 0.48;
  const ripple = params.portal_ripple ?? 0.2;
  const depthSkew = params.portal_depth_skew ?? 0.35;
  const pointDensity = params.point_density ?? 1800;
  const pulseAmp = params.pulse_amplitude ?? 0.0;
  const pulseFreq = params.pulse_frequency ?? 1.0;

  const centerX = width / 2;
  const centerY = height / 2;
  const scale = Math.min(width, height) * 0.48;
  const tunnelPulse = 1 + pulseAmp * Math.sin(time * pulseFreq * Math.PI * 2);
  const rotation = params._accumulatedRotation ?? 0;
  const densityPerLayer = Math.max(64, Math.floor(pointDensity / layers));

  const hueBase = params.hue_base ?? 180;
  const saturation = params.saturation ?? 0.7;
  const brightness = params.brightness ?? 0.8;
  const colorCycleSpeed = params.color_cycle_speed ?? 0.2;

  const layersOut: ColoredLayer[] = [];

  for (let layer = 0; layer < layers; layer++) {
    const depth = layer / Math.max(layers - 1, 1);
    const layerOffset = depth * 1.1;
    const depthRadius = baseRadius * (1 + depth * (0.7 + depthSkew));
    const turbulence = 1 + 0.1 * Math.cos(depth * 3 + time * 0.6);
    const hue = (hueBase + depth * 90 + time * colorCycleSpeed * 80) % 360;
    const path = new Path2D();

    for (let i = 0; i < densityPerLayer; i++) {
      const theta = (i / densityPerLayer) * Math.PI * 2 * angularFreq;
      const swirl = theta + spiral * (time * 0.45 + depth * 1.4) + rotation + layerOffset;

      const radialWave = Math.sin(theta * radialFreq + time * 0.8 + layerOffset) * ripple;
      const symmetryWave = Math.sin(theta * symmetry + rotation) * warp * (1 - depth * 0.35);
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
    const angle = (i / flareCount) * Math.PI * 2 + rotation * 0.6;
    const wobble = Math.sin(time * 0.7 + i) * 0.15;
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
