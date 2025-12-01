import { useEffect, useState } from 'react';
import { VisualCanvas } from '@/features/visualizer/VisualCanvas';
import { VisualSettings } from '@/features/visualizer/VisualSettings';
import { ParameterControls } from '@/features/visualizer/ParameterControls';
import { AudioControls } from '@/features/audio/AudioControls';
import { EEGDisplay } from '@/features/eeg/EEGDisplay';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useAudioEngineContext } from '@/contexts/AudioEngineContext';
import { Button } from '@/components/ui/button';
import { Power, Wifi, WifiOff } from 'lucide-react';
import type { AlgorithmType } from '@/features/visualizer/algorithms';

function App() {
  const { 
    isConnected, 
    isSessionActive,
    brainState, 
    visualParams, 
    brainStateHistory,
    startSession,
    stopSession
  } = useWebSocket();

  const audioEngine = useAudioEngineContext();

  const [selectedAlgorithm, setSelectedAlgorithm] = useState<AlgorithmType>('hyperspace_portal');
  const [presetParams, setPresetParams] = useState<any>(null);
  const [manualParams, setManualParams] = useState<any>(null);
  
  // Merge params: WebSocket (base) <- Preset (overlay) <- Manual (overlay)
  // This allows brain-state mappings to continue while manual/preset adjustments override specific params
  const activeVisualParams = {
    ...visualParams,      // Base: brain-state mapped params from backend
    ...presetParams,      // Overlay: preset overrides (if any)
    ...manualParams,      // Overlay: manual overrides (if any)
  };

  // Auto-start/stop music based on session state
  useEffect(() => {
    if (isSessionActive && !audioEngine.isPlaying) {
      console.log('Session started - starting audio engine');
      audioEngine.start();
    } else if (!isSessionActive && audioEngine.isPlaying) {
      console.log('Session stopped - stopping audio engine');
      audioEngine.stop();
    }
  }, [isSessionActive, audioEngine]);

  // Update audio engine with brain state
  useEffect(() => {
    if (brainState && audioEngine.isPlaying) {
      // Map backend brain state to frontend format
      const mappedBrainState = {
        focus: brainState.focus || 0.5,
        neutral: brainState.neutral || 0.5,
        relax: brainState.relax || 0.5,
      };
      audioEngine.updateBrainState(mappedBrainState);
    }
  }, [brainState, audioEngine]);


  return (
    <div className="min-h-screen bg-background text-foreground font-sans p-4 md:p-8 overflow-hidden bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-syn-dark via-[#050510] to-black">
      <div className="h-[calc(100vh-4rem)] flex flex-col">
        <header className="mb-8 flex justify-between items-center flex-none relative">
          <div>
            <h1 className="text-4xl md:text-6xl font-bold font-display text-transparent bg-clip-text bg-gradient-to-r from-syn-cyan via-syn-purple to-syn-cyan animate-pulse tracking-tighter">
              SYNESTHESIA
            </h1>
            <p className="text-sm md:text-lg text-muted-foreground font-mono mt-1 border-l-2 border-syn-cyan pl-4">
              NEURAL_INTERFACE // {isConnected ? 'ONLINE (SIM)' : 'OFFLINE'}
            </p>
          </div>
          
          {/* Brain State - Absolutely centered in header */}
          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 inline-flex items-center gap-6 px-6 py-3 rounded-xl border border-white/10 bg-card/50 backdrop-blur-md shadow-lg">
            <div className="text-center">
              <div className="text-2xl font-display font-bold text-syn-green">
                {brainState ? (
                  brainState.focus > brainState.relax 
                    ? (brainState.focus > brainState.neutral ? 'FOCUS' : 'NEUTRAL')
                    : (brainState.relax > brainState.neutral ? 'RELAX' : 'NEUTRAL')
                ) : '--'}
              </div>
              <div className="text-[10px] text-muted-foreground mt-0.5">CURRENT STATE</div>
            </div>
          </div>

          <div className="text-right hidden md:flex flex-col items-end gap-2">
             <div className="flex items-center gap-2 text-xs font-mono">
               {isConnected ? (
                 <span className="flex items-center gap-1 text-syn-green"><Wifi className="w-3 h-3" /> CONNECTED</span>
               ) : (
                 <span className="flex items-center gap-1 text-destructive"><WifiOff className="w-3 h-3" /> DISCONNECTED</span>
               )}
             </div>
             <div className="text-xs font-mono text-muted-foreground">SYS.VER.0.1.0</div>
             
             <div className="flex gap-2 mt-2">
               <Button 
                size="sm" 
                variant={isSessionActive ? "destructive" : "neon"}
                onClick={() => {
                  console.log('Session button clicked', {
                    isSessionActive,
                    isConnected,
                    action: isSessionActive ? 'stop' : 'start'
                  });
                  if (isSessionActive) {
                    stopSession();
                  } else {
                    startSession();
                  }
                }}
                disabled={!isConnected}
                className="h-6 text-xs"
               >
                 <Power className="w-3 h-3 mr-1" />
                 {isSessionActive ? 'STOP SESSION' : 'START SESSION'}
               </Button>
             </div>
          </div>
        </header>
        
        <main className="grid grid-cols-1 md:grid-cols-12 gap-6 flex-1 min-h-0 items-stretch h-full">
          {/* Left Column: Controls */}
          <div className="md:col-span-3 flex flex-col gap-6 overflow-y-auto pr-2 h-full min-h-0">
            <div className="flex-none">
              <VisualSettings 
                onAlgorithmChange={(algo) => setSelectedAlgorithm(algo as AlgorithmType)}
                onPresetChange={(preset) => console.log('Preset selected:', preset)}
                onPresetParamsChange={(params) => {
                  console.log('Applying preset parameters:', params);
                  setPresetParams(params);
                  // Don't clear manual params - they merge together
                }}
              />
            </div>
            <div className="flex-none">
              <ParameterControls 
                currentParams={activeVisualParams}
                onParamsChange={(params) => {
                  console.log('Manual parameters:', params);
                  setManualParams(params);
                  // Don't clear preset - they merge together
                }}
              />
            </div>
          </div>

          {/* Center Column: Visualizer */}
          <div className="md:col-span-6 h-full min-h-[400px] relative">
            <div className="absolute inset-0">
              <VisualCanvas params={activeVisualParams} algorithm={selectedAlgorithm} isActive={isSessionActive} />
            </div>

            {(manualParams || presetParams) && (
              <div className="absolute top-4 left-4 flex gap-2">
                {presetParams && (
                  <div className="text-xs font-mono text-syn-purple bg-syn-purple/10 border border-syn-purple/30 px-3 py-1 rounded backdrop-blur">
                    PRESET
                  </div>
                )}
                {manualParams && (
                  <div className="text-xs font-mono text-syn-green bg-syn-green/10 border border-syn-green/30 px-3 py-1 rounded backdrop-blur">
                    MANUAL
                  </div>
                )}
              </div>
            )}
          </div>
          
          {/* Right Column: EEG + Audio */}
          <div className="md:col-span-3 flex flex-col gap-6 overflow-y-auto h-full min-h-0">
            <div className="flex-none">
              <EEGDisplay data={brainStateHistory} />
            </div>
            <div className="flex-none">
              <AudioControls />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
