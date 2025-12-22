/**
 * Sync Meter
 * Visualizes the synchronization state between two users
 */

import { useEffect, useRef } from 'react';
import type { SyncState } from '@/hooks/useSyncSession';

interface SyncMeterProps {
  syncState: SyncState;
  syncHistory: SyncState[];
}

export function SyncMeter({ syncState, syncHistory }: SyncMeterProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Determine sync quality label
  const getSyncLabel = (score: number): { label: string; color: string } => {
    if (score >= 0.75) return { label: 'HARMONIZED', color: '#00ff88' };
    if (score >= 0.5) return { label: 'ALIGNED', color: '#00ffff' };
    if (score >= 0.3) return { label: 'DRIFTING', color: '#ffaa00' };
    return { label: 'DISSONANT', color: '#ff4444' };
  };

  const { label, color } = getSyncLabel(syncState.sync_score);

  // Draw sync history graph
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || syncHistory.length < 2) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear
    ctx.clearRect(0, 0, width, height);

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = (height / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw sync score line
    ctx.beginPath();
    ctx.strokeStyle = '#00ffff';
    ctx.lineWidth = 2;
    
    syncHistory.forEach((state, i) => {
      const x = (i / (syncHistory.length - 1)) * width;
      const y = height - (state.sync_score * height);
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Draw dissonance line
    ctx.beginPath();
    ctx.strokeStyle = '#ff4444';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    
    syncHistory.forEach((state, i) => {
      const x = (i / (syncHistory.length - 1)) * width;
      const y = height - (state.dissonance_level * height);
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw current position marker
    const lastState = syncHistory[syncHistory.length - 1];
    const lastX = width - 2;
    const lastY = height - (lastState.sync_score * height);
    
    ctx.beginPath();
    ctx.fillStyle = '#00ffff';
    ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
    ctx.fill();

  }, [syncHistory]);

  return (
    <div className="space-y-3">
      {/* Main Sync Display */}
      <div className="relative">
        {/* Large sync score arc */}
        <div className="flex items-center justify-center">
          <div className="relative w-32 h-16 overflow-hidden">
            {/* Background arc */}
            <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 50">
              <path
                d="M 5 50 A 45 45 0 0 1 95 50"
                fill="none"
                stroke="rgba(255,255,255,0.1)"
                strokeWidth="8"
                strokeLinecap="round"
              />
              {/* Filled arc based on sync score */}
              <path
                d="M 5 50 A 45 45 0 0 1 95 50"
                fill="none"
                stroke={color}
                strokeWidth="8"
                strokeLinecap="round"
                strokeDasharray={`${syncState.sync_score * 141.37} 141.37`}
                style={{ transition: 'stroke-dasharray 0.3s ease' }}
              />
            </svg>
            
            {/* Center value */}
            <div className="absolute inset-0 flex items-end justify-center pb-0">
              <div className="text-center">
                <div className="text-2xl font-display font-bold" style={{ color }}>
                  {Math.round(syncState.sync_score * 100)}%
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Status label */}
        <div className="text-center mt-1">
          <span 
            className="text-xs font-mono px-2 py-0.5 rounded"
            style={{ 
              color, 
              backgroundColor: `${color}22`,
              border: `1px solid ${color}44`
            }}
          >
            {label}
          </span>
        </div>
      </div>

      {/* Sync History Graph */}
      <div className="space-y-1">
        <div className="flex items-center justify-between text-[10px] text-muted-foreground">
          <span>SYNC HISTORY</span>
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              <span className="w-2 h-0.5 bg-syn-cyan" /> Sync
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-0.5 bg-red-500 opacity-50" style={{ borderTop: '1px dashed' }} /> Dissonance
            </span>
          </div>
        </div>
        <canvas 
          ref={canvasRef}
          width={240}
          height={60}
          className="w-full h-[60px] rounded bg-black/30 border border-white/5"
        />
      </div>

      {/* Detailed Metrics */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="p-2 rounded bg-black/20 border border-white/5">
          <div className="text-muted-foreground text-[10px]">ALPHA PLV</div>
          <div className="font-mono text-syn-cyan">{(syncState.alpha_plv * 100).toFixed(1)}%</div>
        </div>
        <div className="p-2 rounded bg-black/20 border border-white/5">
          <div className="text-muted-foreground text-[10px]">THETA PLV</div>
          <div className="font-mono text-syn-purple">{(syncState.theta_plv * 100).toFixed(1)}%</div>
        </div>
        <div className="p-2 rounded bg-black/20 border border-white/5">
          <div className="text-muted-foreground text-[10px]">CORRELATION</div>
          <div className="font-mono text-syn-green">{(syncState.bandpower_correlation * 100).toFixed(1)}%</div>
        </div>
        <div className="p-2 rounded bg-black/20 border border-white/5">
          <div className="text-muted-foreground text-[10px]">DATA QUALITY</div>
          <div className="font-mono" style={{ 
            color: syncState.quality > 0.7 ? '#00ff88' : syncState.quality > 0.3 ? '#ffaa00' : '#ff4444' 
          }}>
            {(syncState.quality * 100).toFixed(0)}%
          </div>
        </div>
      </div>
    </div>
  );
}

