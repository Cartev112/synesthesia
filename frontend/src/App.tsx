import { useEffect, useState, useCallback } from 'react';
import { VisualCanvas } from '@/features/visualizer/VisualCanvas';
import { VisualSettings } from '@/features/visualizer/VisualSettings';
import { ParameterControls } from '@/features/visualizer/ParameterControls';
import { AudioControls } from '@/features/audio/AudioControls';
import { EEGDisplay } from '@/features/eeg/EEGDisplay';
import { CalibrationFlow } from '@/features/calibration/CalibrationFlow';
import { SyncSessionPanel } from '@/features/sync/SyncSessionPanel';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useAudioEngineContext } from '@/contexts/AudioEngineContext';
import { Button } from '@/components/ui/button';
import { DeviceConfigModal } from '@/components/DeviceConfigModal';
import { Power, Wifi, WifiOff, Menu, X, ChevronDown, ChevronUp, Settings, CheckCircle2, Users, User } from 'lucide-react';
import type { AlgorithmType } from '@/features/visualizer/algorithms';
import type { SyncState } from '@/hooks/useSyncSession';

type AppPhase = 'device_select' | 'calibration' | 'session';
type SessionMode = 'solo' | 'sync';

function App() {
  const { 
    isConnected, 
    isSessionActive,
    brainState, 
    visualParams, 
    brainStateHistory,
    deviceConfig,
    setDeviceConfig,
    startSession,
    stopSession,
    // Calibration state
    calibrationStatus,
    calibrationStage,
    calibrationProgress,
    calibrationResults,
    calibrationError,
    isCalibrated,
    // Calibration actions
    startCalibration,
    startCalibrationStage,
    stopCalibrationStage,
    trainCalibration,
    cancelCalibration
  } = useWebSocket();

  // App phase state
  const [appPhase, setAppPhase] = useState<AppPhase>('device_select');
  const [showDeviceModal, setShowDeviceModal] = useState(true);
  const [sessionMode, setSessionMode] = useState<SessionMode>('solo');

  const audioEngine = useAudioEngineContext();

  const [selectedAlgorithm, setSelectedAlgorithm] = useState<AlgorithmType>('hyperspace_portal');
  const [presetParams, setPresetParams] = useState<any>(null);
  const [manualParams, setManualParams] = useState<any>(null);
  
  // Mobile UI state
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [activePanel, setActivePanel] = useState<'visuals' | 'eeg' | 'audio' | 'sync' | null>(null);
  
  // Sync state for multi-user mode
  const [currentSyncState, setCurrentSyncState] = useState<SyncState | null>(null);

  // Handle device selection from modal
  const handleDeviceSelected = useCallback((deviceType: string, deviceAddress?: string, devicePreset?: string) => {
    console.log('Device selected:', { deviceType, deviceAddress, devicePreset });
    setDeviceConfig({
      deviceType,
      deviceAddress,
      devicePreset
    });
    setShowDeviceModal(false);
    // Move to calibration phase
    setAppPhase('calibration');
  }, [setDeviceConfig]);
  
  // Handle calibration complete - move to session phase
  const handleCalibrationComplete = useCallback(() => {
    setAppPhase('session');
  }, []);
  
  // Handle calibration cancel - go back to device selection
  const handleCalibrationCancel = useCallback(() => {
    cancelCalibration();
    setAppPhase('device_select');
    setShowDeviceModal(true);
  }, [cancelCalibration]);
  
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

  // Update audio engine with sync state (multi-user mode)
  useEffect(() => {
    if (currentSyncState && sessionMode === 'sync') {
      audioEngine.setSyncMode(true);
      audioEngine.updateSyncState({
        sync_score: currentSyncState.sync_score,
        dissonance_level: currentSyncState.dissonance_level,
      });
    } else {
      audioEngine.setSyncMode(false);
    }
  }, [currentSyncState, sessionMode, audioEngine]);
  
  // Handle sync state updates from the sync panel
  const handleSyncStateChange = useCallback((syncState: SyncState | null) => {
    setCurrentSyncState(syncState);
  }, []);

  // Render calibration phase
  if (appPhase === 'calibration') {
    return (
      <div className="min-h-screen bg-background text-foreground font-sans overflow-hidden bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-syn-dark via-[#050510] to-black">
        <div className="h-screen flex flex-col">
          {/* Minimal header during calibration */}
          <header className="p-4 md:p-8 flex justify-between items-center flex-none">
            <div>
              <h1 className="text-2xl md:text-4xl font-bold font-display text-transparent bg-clip-text bg-gradient-to-r from-syn-cyan via-syn-purple to-syn-cyan tracking-tighter">
                SYNESTHESIA
              </h1>
              <p className="text-sm text-muted-foreground font-mono mt-1 border-l-2 border-syn-cyan pl-4">
                CALIBRATION // {deviceConfig.deviceType === 'simulator' ? 'SIMULATOR' : 'MUSE S'}
              </p>
            </div>
            
            <div className="flex items-center gap-2 text-xs font-mono">
              {isConnected ? (
                <span className="flex items-center gap-1 text-syn-green"><Wifi className="w-3 h-3" /> CONNECTED</span>
              ) : (
                <span className="flex items-center gap-1 text-destructive"><WifiOff className="w-3 h-3" /> DISCONNECTED</span>
              )}
            </div>
          </header>
          
          {/* Calibration flow */}
          <main className="flex-1 min-h-0">
            <CalibrationFlow
              calibrationStatus={calibrationStatus}
              calibrationStage={calibrationStage}
              calibrationProgress={calibrationProgress}
              calibrationResults={calibrationResults}
              calibrationError={calibrationError}
              isCalibrated={isCalibrated}
              onStartCalibration={startCalibration}
              onStartStage={startCalibrationStage}
              onStopStage={stopCalibrationStage}
              onTrain={trainCalibration}
              onCancel={handleCalibrationCancel}
              onComplete={handleCalibrationComplete}
            />
          </main>
        </div>
      </div>
    );
  }

  // Render main session phase
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
                NEURAL_INTERFACE // {isConnected ? (deviceConfig.deviceType === 'simulator' ? 'ONLINE (SIM)' : 'ONLINE (MUSE)') : 'OFFLINE'}
                {isCalibrated && <span className="ml-2 text-syn-green">• CALIBRATED</span>}
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
               {isCalibrated && (
                 <span className="flex items-center gap-1 text-syn-green ml-2">
                   <CheckCircle2 className="w-3 h-3" /> CALIBRATED
                 </span>
               )}
             </div>
             <div className="text-xs font-mono text-muted-foreground">SYS.VER.0.1.0</div>
             
             {/* Mode toggle */}
             <div className="flex items-center gap-1 mt-2 p-1 rounded-lg bg-black/30 border border-white/10">
               <Button
                 size="sm"
                 variant={sessionMode === 'solo' ? 'neon' : 'ghost'}
                 onClick={() => setSessionMode('solo')}
                 className="h-6 text-xs px-2"
               >
                 <User className="w-3 h-3 mr-1" />
                 SOLO
               </Button>
               <Button
                 size="sm"
                 variant={sessionMode === 'sync' ? 'neon' : 'ghost'}
                 onClick={() => setSessionMode('sync')}
                 className="h-6 text-xs px-2"
               >
                 <Users className="w-3 h-3 mr-1" />
                 SYNC
               </Button>
             </div>
             
             <div className="flex gap-2 mt-2">
               <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  setAppPhase('device_select');
                  setShowDeviceModal(true);
                }}
                className="h-6 text-xs"
               >
                 <Settings className="w-3 h-3 mr-1" />
                 DEVICE
               </Button>
               {sessionMode === 'solo' && (
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
               )}
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
              
              {/* Collapsible Sync Session */}
              <div className="mb-3">
                <button 
                  className="w-full flex justify-between items-center py-2 px-3 rounded-lg bg-card/50 border border-white/10"
                  onClick={() => setActivePanel(activePanel === 'sync' ? null : 'sync')}
                >
                  <span className="text-sm font-mono text-syn-purple">MULTI-USER SYNC</span>
                  {activePanel === 'sync' ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
                {activePanel === 'sync' && (
                  <div className="mt-2">
                    <SyncSessionPanel onSyncStateChange={handleSyncStateChange} />
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
              <VisualCanvas 
                params={activeVisualParams} 
                algorithm={selectedAlgorithm} 
                isActive={isSessionActive}
                syncState={sessionMode === 'sync' && currentSyncState ? {
                  sync_score: currentSyncState.sync_score,
                  dissonance_level: currentSyncState.dissonance_level
                } : null}
              />
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
          
          {/* Right Column: EEG + Audio + Sync (hidden on mobile - shown in slide-out menu) */}
          <div className="hidden md:flex md:col-span-3 flex-col gap-6 overflow-y-auto h-full min-h-0">
            {/* Sync Panel - shown when in sync mode */}
            {sessionMode === 'sync' && (
              <div className="flex-none">
                <SyncSessionPanel onSyncStateChange={handleSyncStateChange} />
              </div>
            )}
            
            {/* Solo mode EEG display */}
            {sessionMode === 'solo' && (
              <div className="flex-none">
                <EEGDisplay data={brainStateHistory} />
              </div>
            )}
            
            <div className="flex-none">
              <AudioControls />
            </div>
          </div>
        </main>
      </div>

      {/* Device Configuration Modal */}
      <DeviceConfigModal
        isOpen={showDeviceModal}
        onClose={() => setShowDeviceModal(false)}
        onDeviceSelected={handleDeviceSelected}
      />
    </div>
  );
}

export default App;
