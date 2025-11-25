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
  const speedMult = params.speed_multiplier ?? 1.0;
  
  const centerX = width / 2;
  const centerY = height / 2;
  const scale = Math.min(width, height) * 0.35;
  
  ctx.beginPath();
  
  for (let i = 0; i < pointDensity; i++) {
    const t = (i / pointDensity) * Math.PI * 4 + time * speedMult * 0.5;
    
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
  const rotSpeed = params.rotation_speed ?? 0;
  const numHarmonics = params.num_harmonics ?? 5;
  const pointDensity = params.point_density ?? 1024;
  const speedMult = params.speed_multiplier ?? 1.0;
  const pulseFreq = params.pulse_frequency ?? 1.0;
  const pulseAmp = params.pulse_amplitude ?? 0.2;
  const dampX = params.damping_x ?? 0.03;
  const dampY = params.damping_y ?? 0.03;
  
  const centerX = width / 2;
  const centerY = height / 2;
  const scale = Math.min(width, height) * 0.35;
  const rotation = time * rotSpeed;
  
  ctx.beginPath();
  
  for (let i = 0; i < pointDensity; i++) {
    const angle = (i / pointDensity) * Math.PI * 2 * 3;
    
    let x = 0;
    let y = 0;
    
    for (let h = 0; h < numHarmonics; h++) {
      const harmonic = h + 1;
      const dampingFactorX = Math.exp(-dampX * angle * harmonic);
      const dampingFactorY = Math.exp(-dampY * angle * harmonic);
      
      x += (ampX / harmonic) * Math.sin(freqX * angle * harmonic + phaseOffset + time * speedMult * 0.5) * dampingFactorX;
      y += (ampY / harmonic) * Math.sin(freqY * angle * harmonic + time * speedMult * 0.3) * dampingFactorY;
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
  const speedMult = params.speed_multiplier ?? 1.0;
  const rotSpeed = params.rotation_speed ?? 0.1;
  
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
  
  const dt = 0.01 * speedMult;
  
  // Rotation for animation
  const rotation = time * rotSpeed;
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

/**
 * Reaction-Diffusion pattern renderer
 * Organic patterns from Gray-Scott model
 */
// Persistent state for reaction-diffusion simulation
let rdState: {
  u: number[][];
  v: number[][];
  lastUpdate: number;
  size: number;
} | null = null;

export function renderReactionDiffusion(
  ctx: CanvasRenderingContext2D,
  params: VisualParams,
  width: number,
  height: number,
  time: number
): RenderResult | void {
  const freqX = params.frequency_ratio_x ?? 3.0;
  const freqY = params.frequency_ratio_y ?? 2.0;
  const rotSpeed = params.rotation_speed ?? 0.05;
  
  const size = 64;
  
  // Time-varying parameters for evolving patterns
  // Tuned to maintain complex patterns (mitosis/soliton regime)
  const timeOffset = time * 0.04;
  const feedRate = 0.054 + Math.sin(timeOffset) * 0.008 + freqX * 0.003;
  const killRate = 0.063 + Math.cos(timeOffset * 0.8) * 0.006 + freqY * 0.003;
  
  const Du = 0.16;
  const Dv = 0.08;
  
  // Initialize or reset if needed
  if (!rdState || rdState.size !== size || time - rdState.lastUpdate > 2.0) {
    const u = Array.from({ length: size }, () => Array(size).fill(1.0));
    const v = Array.from({ length: size }, () => Array(size).fill(0.0));
    
    // Rich initial pattern with multiple seeds
    const center = size / 2;
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        // Dense random noise for organic start
        if (Math.random() < 0.08) {
          v[i][j] = Math.random();
        }
        
        const dx = i - center;
        const dy = j - center;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const angle = Math.atan2(dy, dx);
        
        // Multiple concentric rings
        if (Math.abs(dist - 8) < 1.5 || Math.abs(dist - 16) < 1.5 || Math.abs(dist - 24) < 1.5) {
          v[i][j] = 0.8 + Math.random() * 0.2;
        }
        
        // Radial spokes
        if (Math.abs(Math.sin(angle * 5)) > 0.9 && dist < 20) {
          v[i][j] = 0.7 + Math.random() * 0.3;
        }
      }
    }
    
    rdState = { u, v, lastUpdate: time, size };
  }
  
  // Continuous evolution - run a few steps each frame for smooth animation
  const stepsPerFrame = 3;
  
  for (let step = 0; step < stepsPerFrame; step++) {
    const uNew = rdState.u.map(row => [...row]);
    const vNew = rdState.v.map(row => [...row]);
    
    for (let i = 1; i < size - 1; i++) {
      for (let j = 1; j < size - 1; j++) {
        // 5-point Laplacian
        const lapU = rdState.u[i+1][j] + rdState.u[i-1][j] + rdState.u[i][j+1] + rdState.u[i][j-1] - 4 * rdState.u[i][j];
        const lapV = rdState.v[i+1][j] + rdState.v[i-1][j] + rdState.v[i][j+1] + rdState.v[i][j-1] - 4 * rdState.v[i][j];
        
        // Reaction-diffusion
        const uvv = rdState.u[i][j] * rdState.v[i][j] * rdState.v[i][j];
        uNew[i][j] = rdState.u[i][j] + Du * lapU - uvv + feedRate * (1 - rdState.u[i][j]);
        vNew[i][j] = rdState.v[i][j] + Dv * lapV + uvv - (feedRate + killRate) * rdState.v[i][j];
        
        // Clamp
        uNew[i][j] = Math.max(0, Math.min(1, uNew[i][j]));
        vNew[i][j] = Math.max(0, Math.min(1, vNew[i][j]));
      }
    }
    
    // Update state
    rdState.u = uNew;
    rdState.v = vNew;
  }
  
  // More frequent perturbations to maintain complexity
  if (Math.random() < 0.03) {
    const rx = Math.floor(Math.random() * (size - 4)) + 2;
    const ry = Math.floor(Math.random() * (size - 4)) + 2;
    const radius = 2;
    
    // Inject a small cluster instead of single point
    for (let di = -radius; di <= radius; di++) {
      for (let dj = -radius; dj <= radius; dj++) {
        if (di * di + dj * dj <= radius * radius) {
          const ni = rx + di;
          const nj = ry + dj;
          if (ni >= 0 && ni < size && nj >= 0 && nj < size) {
            rdState.v[ni][nj] = Math.random() * 0.4 + 0.4;
          }
        }
      }
    }
  }
  
  // Prevent total extinction by monitoring and reseeding if pattern dies
  let totalV = 0;
  let highConcentrationCells = 0;
  
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      totalV += rdState.v[i][j];
      if (rdState.v[i][j] > 0.4) highConcentrationCells++;
    }
  }
  
  const avgV = totalV / (size * size);
  
  // More aggressive extinction prevention
  // If pattern is dying (low average or few active cells), inject life
  if (avgV < 0.08 || highConcentrationCells < 20) {
    // Inject multiple clusters to revive the pattern
    for (let k = 0; k < 8; k++) {
      const rx = Math.floor(Math.random() * (size - 6)) + 3;
      const ry = Math.floor(Math.random() * (size - 6)) + 3;
      const clusterRadius = 2;
      
      for (let di = -clusterRadius; di <= clusterRadius; di++) {
        for (let dj = -clusterRadius; dj <= clusterRadius; dj++) {
          if (di * di + dj * dj <= clusterRadius * clusterRadius) {
            const ni = rx + di;
            const nj = ry + dj;
            if (ni >= 0 && ni < size && nj >= 0 && nj < size) {
              rdState.v[ni][nj] = 0.7 + Math.random() * 0.3;
            }
          }
        }
      }
    }
  }
  
  rdState.lastUpdate = time;
  const v = rdState.v;
  
  // Render (this happens every frame for smooth rotation)
  const centerX = width / 2;
  const centerY = height / 2;
  const scale = Math.min(width, height) * 0.48;
  const rotation = time * rotSpeed;
  const cosR = Math.cos(rotation);
  const sinR = Math.sin(rotation);
  const cellSize = (scale * 2) / size;
  
  // Batch rendering for better performance
  const baseColor = ctx.strokeStyle?.toString() || '#00f3ff';
  
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      const concentration = v[i][j];
      
      if (concentration > 0.25) {
        // Convert to screen coordinates
        let x = (j / size - 0.5) * 2;
        let y = (i / size - 0.5) * 2;
        
        // Rotate
        const xRot = x * cosR - y * sinR;
        const yRot = x * sinR + y * cosR;
        
        const px = centerX + xRot * scale;
        const py = centerY + yRot * scale;
        
        // Simplified rendering
        const alpha = Math.min(0.9, (concentration - 0.25) * 1.5);
        const radius = cellSize * (0.6 + concentration * 0.4);
        
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fillStyle = baseColor.replace(')', `, ${alpha})`).replace('rgb', 'rgba');
        ctx.fill();
      }
    }
  }
}

/**
 * Hyperspace portal renderer
 * Layered polar waves with spiral warp and radial symmetry
 */
export function renderHyperspacePortal(
  ctx: CanvasRenderingContext2D,
  params: VisualParams,
  width: number,
  height: number,
  time: number
): RenderResult {
  const symmetry = Math.max(3, Math.floor(params.portal_symmetry ?? 8));
  const radialFreq = params.portal_radial_frequency ?? 6.0;
  const angularFreq = params.portal_angular_frequency ?? 2.0;
  const warp = params.portal_warp ?? 0.4;
  const spiral = params.portal_spiral ?? 0.6;
  const layers = Math.max(2, Math.floor(params.portal_layers ?? 4));
  const baseRadius = params.portal_radius ?? 0.55;
  const ripple = params.portal_ripple ?? 0.25;
  const depthSkew = params.portal_depth_skew ?? 0.4;
  const pointDensity = params.point_density ?? 1800;
  const speed = params.speed_multiplier ?? 1.0;
  const pulseAmp = params.pulse_amplitude ?? 0.18;
  const pulseFreq = params.pulse_frequency ?? 1.1;
  const rotSpeed = params.rotation_speed ?? 0.2;

  const centerX = width / 2;
  const centerY = height / 2;
  const scale = Math.min(width, height) * 0.48;
  const tunnelPulse = 1 + pulseAmp * Math.sin(time * pulseFreq * Math.PI * 2);
  const rotation = time * rotSpeed;
  const densityPerLayer = Math.max(64, Math.floor(pointDensity / layers));

  const hueBase = params.hue_base ?? 200;
  const saturation = params.saturation ?? 0.75;
  const brightness = params.brightness ?? 0.7;
  const colorCycleSpeed = params.color_cycle_speed ?? 0.3;

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
      const swirl = theta + spiral * (time * 0.45 * speed + depth * 1.4) + rotation + layerOffset;

      const radialWave = Math.sin(theta * radialFreq + time * 0.8 * speed + layerOffset) * ripple;
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
