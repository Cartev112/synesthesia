import { useEffect, useState, useCallback } from 'react';
import { VisualCanvas } from '@/features/visualizer/VisualCanvas';
import { VisualSettings } from '@/features/visualizer/VisualSettings';
import { ParameterControls } from '@/features/visualizer/ParameterControls';
import { AudioControls } from '@/features/audio/AudioControls';
import { EEGDisplay } from '@/features/eeg/EEGDisplay';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useAudioEngineContext } from '@/contexts/AudioEngineContext';
import { Button } from '@/components/ui/button';
import { Power, Wifi, WifiOff, Menu, X, ChevronDown, ChevronUp } from 'lucide-react';
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
  
  // Mobile UI state
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [activePanel, setActivePanel] = useState<'visuals' | 'eeg' | 'audio' | null>(null);
  
  // Close mobile menu when clicking outside or starting session
  const closeMobileMenu = useCallback(() => {
    setMobileMenuOpen(false);
    setActivePanel(null);
  }, []);
  
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
    <div className="min-h-screen bg-background text-foreground font-sans p-2 sm:p-4 md:p-8 overflow-hidden bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-syn-dark via-[#050510] to-black">
      <div className="h-[calc(100vh-1rem)] sm:h-[calc(100vh-2rem)] md:h-[calc(100vh-4rem)] flex flex-col">
        <header className="mb-2 sm:mb-4 md:mb-8 flex justify-between items-center flex-none relative">
          <div className="flex items-center gap-2 sm:gap-4">
            {/* Mobile menu button */}
            <Button
              variant="ghost"
              size="sm"
              className="md:hidden h-10 w-10 p-0"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </Button>
            
            <div>
              <h1 className="text-2xl sm:text-4xl md:text-6xl font-bold font-display text-transparent bg-clip-text bg-gradient-to-r from-syn-cyan via-syn-purple to-syn-cyan animate-pulse tracking-tighter">
                SYNESTHESIA
              </h1>
              <p className="hidden sm:block text-sm md:text-lg text-muted-foreground font-mono mt-1 border-l-2 border-syn-cyan pl-4">
                NEURAL_INTERFACE // {isConnected ? 'ONLINE (SIM)' : 'OFFLINE'}
              </p>
            </div>
          </div>
          
          {/* Brain State - Absolutely centered in header (hidden on small mobile) */}
          <div className="hidden sm:inline-flex absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 items-center gap-3 sm:gap-6 px-3 sm:px-6 py-2 sm:py-3 rounded-xl border border-white/10 bg-card/50 backdrop-blur-md shadow-lg">
            <div className="text-center">
              <div className="text-lg sm:text-2xl font-display font-bold text-syn-green">
                {brainState ? (
                  brainState.focus > brainState.relax 
                    ? (brainState.focus > brainState.neutral ? 'FOCUS' : 'NEUTRAL')
                    : (brainState.relax > brainState.neutral ? 'RELAX' : 'NEUTRAL')
                ) : '--'}
              </div>
              <div className="text-[10px] text-muted-foreground mt-0.5">CURRENT STATE</div>
            </div>
          </div>

          {/* Desktop controls */}
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
          
          {/* Mobile session button + status */}
          <div className="flex md:hidden items-center gap-2">
            <div className="flex items-center gap-1 text-[10px] font-mono">
              {isConnected ? (
                <Wifi className="w-3 h-3 text-syn-green" />
              ) : (
                <WifiOff className="w-3 h-3 text-destructive" />
              )}
            </div>
            <Button 
              size="sm" 
              variant={isSessionActive ? "destructive" : "neon"}
              onClick={() => {
                if (isSessionActive) {
                  stopSession();
                } else {
                  startSession();
                  closeMobileMenu();
                }
              }}
              disabled={!isConnected}
              className="h-8 text-[10px] px-2"
            >
              <Power className="w-3 h-3 mr-1" />
              {isSessionActive ? 'STOP' : 'START'}
            </Button>
          </div>
        </header>
        
        {/* Mobile brain state indicator (shown below header on small screens) */}
        <div className="sm:hidden flex justify-center mb-2">
          <div className="inline-flex items-center gap-3 px-4 py-1.5 rounded-lg border border-white/10 bg-card/50 backdrop-blur-md">
            <div className="text-center">
              <div className="text-sm font-display font-bold text-syn-green">
                {brainState ? (
                  brainState.focus > brainState.relax 
                    ? (brainState.focus > brainState.neutral ? 'FOCUS' : 'NEUTRAL')
                    : (brainState.relax > brainState.neutral ? 'RELAX' : 'NEUTRAL')
                ) : '--'}
              </div>
              <div className="text-[8px] text-muted-foreground">STATE</div>
            </div>
          </div>
        </div>
        
        {/* Mobile slide-out panel */}
        {mobileMenuOpen && (
          <div className="md:hidden fixed inset-0 z-50 bg-black/80 backdrop-blur-sm" onClick={closeMobileMenu}>
            <div 
              className="absolute left-0 top-0 bottom-0 w-[85%] max-w-sm bg-syn-dark border-r border-white/10 overflow-y-auto p-4"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-display text-syn-cyan">CONTROLS</h2>
                <Button variant="ghost" size="sm" onClick={closeMobileMenu}>
                  <X className="w-4 h-4" />
                </Button>
              </div>
              
              {/* Collapsible Visual Settings */}
              <div className="mb-3">
                <button 
                  className="w-full flex justify-between items-center py-2 px-3 rounded-lg bg-card/50 border border-white/10"
                  onClick={() => setActivePanel(activePanel === 'visuals' ? null : 'visuals')}
                >
                  <span className="text-sm font-mono text-syn-purple">VISUAL SETTINGS</span>
                  {activePanel === 'visuals' ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
                {activePanel === 'visuals' && (
                  <div className="mt-2 space-y-3">
                    <VisualSettings 
                      onAlgorithmChange={(algo) => setSelectedAlgorithm(algo as AlgorithmType)}
                      onPresetChange={(preset) => console.log('Preset selected:', preset)}
                      onPresetParamsChange={(params) => {
                        setPresetParams(params);
                      }}
                    />
                    <ParameterControls 
                      currentParams={activeVisualParams}
                      onParamsChange={(params) => {
                        setManualParams(params);
                      }}
                    />
                  </div>
                )}
              </div>
              
              {/* Collapsible EEG Display */}
              <div className="mb-3">
                <button 
                  className="w-full flex justify-between items-center py-2 px-3 rounded-lg bg-card/50 border border-white/10"
                  onClick={() => setActivePanel(activePanel === 'eeg' ? null : 'eeg')}
                >
                  <span className="text-sm font-mono text-syn-green">EEG DISPLAY</span>
                  {activePanel === 'eeg' ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
                {activePanel === 'eeg' && (
                  <div className="mt-2">
                    <EEGDisplay data={brainStateHistory} />
                  </div>
                )}
              </div>
              
              {/* Collapsible Audio Controls */}
              <div className="mb-3">
                <button 
                  className="w-full flex justify-between items-center py-2 px-3 rounded-lg bg-card/50 border border-white/10"
                  onClick={() => setActivePanel(activePanel === 'audio' ? null : 'audio')}
                >
                  <span className="text-sm font-mono text-syn-cyan">AUDIO CONTROLS</span>
                  {activePanel === 'audio' ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
                {activePanel === 'audio' && (
                  <div className="mt-2">
                    <AudioControls />
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        <main className="grid grid-cols-1 md:grid-cols-12 gap-2 sm:gap-4 md:gap-6 flex-1 min-h-0 items-stretch h-full">
          {/* Left Column: Controls (hidden on mobile - shown in slide-out menu) */}
          <div className="hidden md:flex md:col-span-3 flex-col gap-6 overflow-y-auto pr-2 h-full min-h-0">
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

          {/* Center Column: Visualizer (full width on mobile) */}
          <div className="col-span-1 md:col-span-6 h-full min-h-[250px] sm:min-h-[350px] md:min-h-[400px] relative">
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
          
          {/* Right Column: EEG + Audio (hidden on mobile - shown in slide-out menu) */}
          <div className="hidden md:flex md:col-span-3 flex-col gap-6 overflow-y-auto h-full min-h-0">
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
