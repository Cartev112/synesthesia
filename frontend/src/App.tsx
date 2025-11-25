import { useEffect, useRef, useState } from 'react';
import { StatusPanel } from '@/features/dashboard/StatusPanel';
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
    brainState, 
    visualParams, 
    brainStateHistory,
    startSession,
    stopSession
  } = useWebSocket();

  const audioEngine = useAudioEngineContext();

  const logsRef = useRef<HTMLDivElement>(null);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<AlgorithmType>('harmonograph');
  const [presetParams, setPresetParams] = useState<any>(null);
  const [manualParams, setManualParams] = useState<any>(null);
  
  // Priority: Manual params > Preset params > WebSocket params
  const activeVisualParams = manualParams || presetParams || visualParams;

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

  // Auto-scroll logs
  useEffect(() => {
    if (logsRef.current) {
      logsRef.current.scrollTop = logsRef.current.scrollHeight;
    }
  }, [brainState]);

  return (
    <div className="min-h-screen bg-background text-foreground font-sans p-4 md:p-8 overflow-hidden bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-syn-dark via-[#050510] to-black">
      <div className="max-w-[1600px] mx-auto h-[calc(100vh-4rem)] flex flex-col">
        <header className="mb-8 flex justify-between items-end flex-none">
          <div>
            <h1 className="text-4xl md:text-6xl font-bold font-display text-transparent bg-clip-text bg-gradient-to-r from-syn-cyan via-syn-purple to-syn-cyan animate-pulse tracking-tighter">
              SYNESTHESIA
            </h1>
            <p className="text-sm md:text-lg text-muted-foreground font-mono mt-1 border-l-2 border-syn-cyan pl-4">
              NEURAL_INTERFACE // {isConnected ? 'ONLINE' : 'OFFLINE'}
            </p>
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
                variant={isConnected ? "destructive" : "neon"}
                onClick={isConnected ? stopSession : startSession}
                className="h-6 text-xs"
               >
                 <Power className="w-3 h-3 mr-1" />
                 {isConnected ? 'STOP SESSION' : 'START SESSION'}
               </Button>
             </div>
          </div>
        </header>
        
        <main className="grid grid-cols-1 md:grid-cols-12 gap-6 flex-1 min-h-0">
          {/* Left Column: Controls & Status */}
          <div className="md:col-span-3 flex flex-col gap-6 overflow-y-auto pr-2">
            <div className="flex-none">
              <StatusPanel />
            </div>
            <div className="flex-none">
              <VisualSettings 
                onAlgorithmChange={(algo) => setSelectedAlgorithm(algo as AlgorithmType)}
                onPresetChange={(preset) => console.log('Preset selected:', preset)}
                onPresetParamsChange={(params) => {
                  console.log('Applying preset parameters:', params);
                  setPresetParams(params);
                  setManualParams(null); // Clear manual overrides
                  // Clear preset after 30 seconds to return to brain-responsive mode
                  setTimeout(() => setPresetParams(null), 30000);
                }}
              />
            </div>
            <div className="flex-none">
              <ParameterControls 
                currentParams={activeVisualParams}
                onParamsChange={(params) => {
                  console.log('Manual parameters:', params);
                  setManualParams(params);
                  setPresetParams(null); // Clear preset when manually adjusting
                }}
              />
            </div>
            <div className="flex-none">
              <AudioControls />
            </div>
          </div>

          {/* Center Column: Visualizer */}
          <div className="md:col-span-6 h-full min-h-[400px] relative">
            <div className="absolute inset-0">
              <VisualCanvas params={activeVisualParams} algorithm={selectedAlgorithm} />
            </div>
            {manualParams && (
              <div className="absolute top-4 left-4 text-xs font-mono text-syn-green bg-syn-green/10 border border-syn-green/30 px-3 py-1 rounded backdrop-blur">
                MANUAL CONTROL
              </div>
            )}
            {!manualParams && presetParams && (
              <div className="absolute top-4 left-4 text-xs font-mono text-syn-purple bg-syn-purple/10 border border-syn-purple/30 px-3 py-1 rounded backdrop-blur">
                PRESET ACTIVE
              </div>
            )}
          </div>
          
          {/* Right Column: EEG Data */}
          <div className="md:col-span-3 flex flex-col gap-6 overflow-y-auto">
            <div className="flex-none">
              <EEGDisplay data={brainStateHistory} />
            </div>
            <div className="h-[200px] rounded-xl border border-white/10 bg-card/30 backdrop-blur p-4 font-mono text-xs text-syn-cyan/70 overflow-hidden flex-none flex flex-col">
               <div className="mb-2 text-white/50 border-b border-white/10 pb-1 flex-none">SYSTEM LOGS</div>
               <div ref={logsRef} className="space-y-1 overflow-y-auto flex-1">
                 <p className="opacity-50">[SYSTEM] Initializing Neural Core...</p>
                 <p className="opacity-50">[SYSTEM] Audio Engine warm-up complete</p>
                 {isConnected ? (
                   <p className="text-syn-green">[NET] WebSocket Connected</p>
                 ) : (
                   <p className="text-destructive">[NET] WebSocket Disconnected</p>
                 )}
                 {brainState && (
                   <p className="text-syn-purple">
                     [EEG] Delta:{brainState.delta_power?.toFixed(2)} Alpha:{brainState.alpha_power?.toFixed(2)}
                   </p>
                 )}
               </div>
            </div>
            
            <div className="rounded-xl border border-white/10 bg-card/30 backdrop-blur p-4 flex-1">
                <div className="mb-2 text-white/50 border-b border-white/10 pb-1 text-xs font-mono">BRAIN STATE</div>
                <div className="grid grid-cols-2 gap-4 mt-4">
                    <div className="text-center">
                        <div className="text-2xl font-display font-bold text-syn-green">
                          {brainState ? (brainState.focus_metric > 0.6 ? 'FOCUS' : 'RELAX') : '--'}
                        </div>
                        <div className="text-[10px] text-muted-foreground">CURRENT STATE</div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-display font-bold text-syn-purple">
                          {brainState ? `${((brainState.focus_metric || 0) * 100).toFixed(0)}%` : '--'}
                        </div>
                        <div className="text-[10px] text-muted-foreground">INTENSITY</div>
                    </div>
                </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
