import { useEffect, useRef } from 'react';
import { Card } from '@/components/ui/card';
import { 
  renderLissajous, 
  renderHarmonograph, 
  renderLorenz,
  renderReactionDiffusion,
  renderHyperspacePortal,
  hslToRgb,
  type AlgorithmType,
  type ColoredLayer
} from './algorithms';

interface VisualParams {
  timestamp?: number;
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
  recursion_depth?: number;
  point_density?: number;
  trail_length?: number;
  distortion_amount?: number;
  speed_multiplier?: number;
  pulse_frequency?: number;
  pulse_amplitude?: number;
  damping_x?: number;
  damping_y?: number;
  num_epicycles?: number;
  epicycle_decay?: number;
  brain_state?: {
    focus?: number;
    relax?: number;
    neutral?: number;
  };
  [key: string]: any;
}

interface VisualCanvasProps {
  params?: VisualParams | null;
  algorithm?: AlgorithmType;
  isActive?: boolean; // Controls fade in/out
}

export function VisualCanvas({ params, algorithm = 'hyperspace_portal', isActive = false }: VisualCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const paramsRef = useRef(params);
  const algorithmRef = useRef(algorithm);
  const lastFrameTimeRef = useRef(Date.now());
  const accumulatedPhaseRef = useRef(0);
  const accumulatedRotationRef = useRef(0);
  const canvasOpacityRef = useRef(0); // Start at 0 (blank)
  const isActiveRef = useRef(isActive);

  // Keep refs in sync with props for animation loop
  useEffect(() => {
    paramsRef.current = params;
    algorithmRef.current = algorithm;
    isActiveRef.current = isActive;
  }, [params, algorithm, isActive]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      
      // Set canvas internal size to match display size * DPR for crisp rendering
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      
      // Scale context to account for DPR
      ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset transform
      ctx.scale(dpr, dpr);
    };

    window.addEventListener('resize', resize);
    resize();

    const draw = () => {
      // Use CSS dimensions (rect) for calculations - ctx is already scaled by DPR
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      
      // Ensure canvas buffer matches current size (handles dynamic resizing)
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
      }
      
      const p = paramsRef.current || {};
      const algo = algorithmRef.current;
      const active = isActiveRef.current;
      
      // Calculate delta time
      const now = Date.now();
      const dt = (now - lastFrameTimeRef.current) / 1000;
      lastFrameTimeRef.current = now;
      
      // Fade in/out based on active state (2s fade in, 1s fade out)
      const targetOpacity = active ? 1 : 0;
      const fadeSpeed = active ? 0.5 : 1.0; // 2s fade in, 1s fade out
      const opacityDiff = targetOpacity - canvasOpacityRef.current;
      canvasOpacityRef.current += opacityDiff * Math.min(dt * fadeSpeed, 1);
      
      // Extract parameters with defaults
      const hueBase = p.hue_base ?? 180;
      const saturation = p.saturation ?? 0.7;
      const brightness = p.brightness ?? 0.8;
      const colorCycleSpeed = p.color_cycle_speed ?? 0.2;
      const trailLength = p.trail_length ?? 0.9;
      const speedMult = p.speed_multiplier ?? 1.0;
      const rotSpeed = p.rotation_speed ?? 0;
      
      // Accumulate phase based on current speed (prevents jumps when speed changes)
      accumulatedPhaseRef.current += dt * speedMult;
      accumulatedRotationRef.current += dt * rotSpeed;
      
      // Use accumulated phase instead of raw time
      const t = accumulatedPhaseRef.current;
      const rotation = accumulatedRotationRef.current;
      
      // Clear canvas - use stronger clear when fading out for faster fade
      const clearAlpha = active ? (1 - trailLength) : Math.max(1 - trailLength, 0.1);
      ctx.fillStyle = `rgba(5, 5, 10, ${clearAlpha})`;
      ctx.fillRect(0, 0, width, height);
      
      // Skip rendering if fully faded out
      if (canvasOpacityRef.current < 0.01) {
        animationFrameId = requestAnimationFrame(draw);
        return;
      }

      // Render based on selected algorithm
      ctx.beginPath();
      let customLayers: ColoredLayer[] | undefined;
      
      // Pass both time and rotation to algorithms
      const renderParams = { ...p, _accumulatedRotation: rotation };
      
      switch (algo) {
        case 'lissajous':
          renderLissajous(ctx, renderParams, width, height, t);
          break;
        case 'lorenz':
          renderLorenz(ctx, renderParams, width, height, t);
          break;
        case 'reaction_diffusion':
          renderReactionDiffusion(ctx, renderParams, width, height, t);
          break;
        case 'hyperspace_portal':
          customLayers = renderHyperspacePortal(ctx, renderParams, width, height, t)?.layers;
          break;
        case 'harmonograph':
        default:
          renderHarmonograph(ctx, renderParams, width, height, t);
          break;
      }
      
      // Apply fade opacity to all rendering
      const fadeOpacity = canvasOpacityRef.current;
      
      if (customLayers && customLayers.length > 0) {
        ctx.save();
        customLayers.forEach(layer => {
          ctx.globalAlpha = (layer.alpha ?? 1) * fadeOpacity;
          ctx.lineWidth = layer.lineWidth ?? 2;
          ctx.strokeStyle = layer.color;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';
          ctx.shadowBlur = layer.shadowBlur ?? 0;
          ctx.shadowColor = layer.color;
          if (layer.composite) ctx.globalCompositeOperation = layer.composite;
          ctx.stroke(layer.path);
        });
        ctx.restore();
      } else {
        // Dynamic multi-pass stroke for richer color layering
        const hue = (hueBase + t * colorCycleSpeed * 60) % 360;
        const hues = [hue, (hue + 42) % 360, (hue + 300) % 360];
        const widths = [2.6, 1.6, 0.9];
        const blurs = [18, 10, 4];

        ctx.save();
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        hues.forEach((h, idx) => {
          const strokeColor = hslToRgb(h, saturation, brightness - idx * 0.08);
          ctx.strokeStyle = strokeColor;
          ctx.lineWidth = widths[idx];
          ctx.shadowBlur = blurs[idx];
          ctx.shadowColor = strokeColor;
          ctx.globalAlpha = (0.9 - idx * 0.2) * fadeOpacity;
          ctx.stroke();
        });

        ctx.restore();
      }

      animationFrameId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  // Calculate dominant state for display
  const brainState = params?.brain_state;
  const dominantState = brainState ? 
    Object.entries(brainState)
      .filter(([key]) => ['focus', 'relax', 'neutral'].includes(key))
      .reduce((a, b) => (b[1] as number) > (a[1] as number) ? b : a, ['neutral', 0])[0]
    : 'unknown';
  const dominantValue = brainState?.[dominantState as keyof typeof brainState] ?? 0;

  return (
    <Card className="h-full w-full overflow-hidden bg-black/80 border-syn-cyan/30 shadow-[0_0_30px_rgba(0,243,255,0.1)]">
      <canvas 
        ref={canvasRef} 
        className="w-full h-full block"
      />
      <div className="absolute top-4 right-4 text-xs font-mono text-syn-cyan/50 pointer-events-none flex flex-col items-end gap-1">
        <span>{algorithm.toUpperCase()}_ACTIVE</span>
        {params && (
          <>
            <span className="text-[10px] text-white/30">
              State: {dominantState.toUpperCase()} ({(dominantValue * 100).toFixed(0)}%)
            </span>
            <span className="text-[10px] text-white/30">
              {algorithm === 'harmonograph' && `Harmonics: ${params.num_harmonics ?? 5}`}
              {algorithm === 'lorenz' && `Chaotic Attractor`}
              {algorithm === 'reaction_diffusion' && `Organic Pattern`}
              {algorithm === 'hyperspace_portal' && `Portal Symmetry: ${params.portal_symmetry ?? 8}`}
              {algorithm === 'lissajous' && `Freq: ${(params.frequency_ratio_x ?? 3).toFixed(1)}:${(params.frequency_ratio_y ?? 2).toFixed(1)}`}
            </span>
          </>
        )}
      </div>
    </Card>
  );
}
