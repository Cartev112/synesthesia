import { useState, useEffect, useCallback, useRef } from 'react';
import { buildWebSocketUrl } from '@/utils/env';

interface BrainState {
  [key: string]: any;
}

interface MusicEvents {
  [key: string]: any;
}

interface VisualParams {
  [key: string]: any;
}

export interface DeviceConfig {
  deviceType: string;
  deviceAddress?: string;
  devicePreset?: string;
}

// Calibration types
export interface CalibrationProgress {
  active: boolean;
  stage?: string;
  elapsed?: number;
  remaining?: number;
  progress?: number;
  complete?: boolean;
  samples_collected?: number;
  sample_counts?: Record<string, number>;
}

export interface CalibrationStageInfo {
  stage: string;
  duration: number;
  instructions: string;
  state_label: string | null;
}

export interface CalibrationResults {
  validation_accuracy: number;
  sample_counts: Record<string, number>;
  training_time: number;
  feature_importance: Record<string, number>;
}

export type CalibrationStatus = 
  | 'idle'           // Not started
  | 'starting'       // Initializing calibration
  | 'ready'          // Calibration started, waiting for first stage
  | 'in_stage'       // Currently in a calibration stage
  | 'stage_complete' // Stage finished, ready for next
  | 'training'       // Training model
  | 'complete'       // Calibration complete
  | 'error';         // Error occurred

interface UseWebSocketReturn {
  // Connection state
  isConnected: boolean;
  isSessionActive: boolean;
  
  // Brain state data
  brainState: BrainState | null;
  musicEvents: MusicEvents | null;
  visualParams: VisualParams | null;
  brainStateHistory: BrainState[];
  
  // Device config
  deviceConfig: DeviceConfig;
  setDeviceConfig: (config: DeviceConfig) => void;
  
  // Session actions
  startSession: () => void;
  stopSession: () => void;
  
  // Calibration state
  calibrationStatus: CalibrationStatus;
  calibrationStage: CalibrationStageInfo | null;
  calibrationProgress: CalibrationProgress | null;
  calibrationResults: CalibrationResults | null;
  isCalibrated: boolean;
  calibrationError: string | null;
  
  // Calibration actions
  startCalibration: () => void;
  startCalibrationStage: (stage: 'baseline' | 'focus' | 'relax') => void;
  stopCalibrationStage: () => void;
  trainCalibration: () => void;
  cancelCalibration: () => void;
}

// Module-level singleton to prevent duplicate connections
let globalWs: WebSocket | null = null;
let globalSessionId: string | null = null;
let isGlobalConnecting = false;

export function useWebSocket(): UseWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [brainState, setBrainState] = useState<BrainState | null>(null);
  const [musicEvents, setMusicEvents] = useState<MusicEvents | null>(null);
  const [visualParams, setVisualParams] = useState<VisualParams | null>(null);
  const [brainStateHistory, setBrainStateHistory] = useState<BrainState[]>([]);
  const [deviceConfig, setDeviceConfig] = useState<DeviceConfig>({ deviceType: 'simulator' });
  
  // Calibration state
  const [calibrationStatus, setCalibrationStatus] = useState<CalibrationStatus>('idle');
  const [calibrationStage, setCalibrationStage] = useState<CalibrationStageInfo | null>(null);
  const [calibrationProgress, setCalibrationProgress] = useState<CalibrationProgress | null>(null);
  const [calibrationResults, setCalibrationResults] = useState<CalibrationResults | null>(null);
  const [isCalibrated, setIsCalibrated] = useState(false);
  const [calibrationError, setCalibrationError] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const messageCountRef = useRef(0);
  
  // Generate a random session ID for now (only once globally)
  if (!globalSessionId) {
    globalSessionId = `session-${Math.random().toString(36).substr(2, 9)}`;
  }
  const sessionId = globalSessionId;

  // Message handler - defined outside useEffect so it can be reattached
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message = JSON.parse(event.data);
      
      // Track message count for debugging
      messageCountRef.current++;
      
      // Log all message types for debugging (but not too verbose for brain_state)
      if (message.type !== 'brain_state' && message.type !== 'visual_params' && message.type !== 'music_events' && message.type !== 'calibration_progress') {
        console.log('WebSocket message received:', message.type, message);
      } else if (messageCountRef.current % 50 === 0) {
        console.log(`WebSocket: Received ${messageCountRef.current} messages`);
      }
      
      switch (message.type) {
        // Session messages
        case 'brain_state':
          setBrainState(message.data);
          setBrainStateHistory(prev => {
            const newHistory = [...prev, message.data];
            if (newHistory.length > 50) return newHistory.slice(newHistory.length - 50);
            return newHistory;
          });
          break;
        case 'music_events':
          setMusicEvents(message.data);
          break;
        case 'visual_params':
          setVisualParams(message.data);
          break;
        case 'session_started':
          console.log('✅ Session started message received!', message);
          setIsSessionActive(true);
          if (message.is_calibrated) {
            setIsCalibrated(true);
          }
          break;
        case 'session_stopped':
          console.log('⏹️ Session stopped message received!', message);
          setIsSessionActive(false);
          break;
        
        // Calibration messages
        case 'calibration_started':
          console.log('🎯 Calibration started:', message);
          setCalibrationStatus('ready');
          setCalibrationError(null);
          break;
        case 'calibration_stage_started':
          console.log('📍 Calibration stage started:', message);
          setCalibrationStage({
            stage: message.stage,
            duration: message.duration,
            instructions: message.instructions,
            state_label: message.state_label
          });
          setCalibrationStatus('in_stage');
          setCalibrationProgress({
            active: true,
            stage: message.stage,
            elapsed: 0,
            remaining: message.duration,
            progress: 0,
            complete: false
          });
          break;
        case 'calibration_stage_stopped':
          console.log('⏸️ Calibration stage stopped:', message);
          setCalibrationStatus('stage_complete');
          setCalibrationProgress(prev => ({
            ...prev,
            active: false,
            complete: true,
            sample_counts: message.sample_counts
          }));
          break;
        case 'calibration_progress':
          setCalibrationProgress(message.progress);
          break;
        case 'calibration_complete':
          console.log('✅ Calibration complete:', message);
          setCalibrationStatus('complete');
          setCalibrationResults({
            validation_accuracy: message.validation_accuracy,
            sample_counts: message.sample_counts,
            training_time: message.training_time,
            feature_importance: message.feature_importance
          });
          setIsCalibrated(true);
          break;
        case 'calibration_cancelled':
          console.log('❌ Calibration cancelled');
          setCalibrationStatus('idle');
          setCalibrationStage(null);
          setCalibrationProgress(null);
          break;
        
        // Error messages
        case 'error':
          console.error('Server error:', message);
          // Check if it's a calibration error
          if (message.code?.startsWith('CALIBRATION') || message.code === 'NO_CALIBRATION_SESSION') {
            setCalibrationError(message.message);
            setCalibrationStatus('error');
          }
          break;
        default:
          console.warn('Unknown message type:', message.type);
          break;
      }
    } catch (err) {
      console.error('Error parsing WebSocket message:', err);
    }
  }, []);

  useEffect(() => {
    // If global WebSocket already exists and is open/connecting, reuse it
    if (globalWs && globalWs.readyState !== WebSocket.CLOSED) {
      console.log('Reusing existing global WebSocket connection, reattaching handlers');
      wsRef.current = globalWs;
      
      // Always reattach handlers to ensure this component instance receives messages
      globalWs.onmessage = handleMessage;
      globalWs.onclose = () => {
        console.log('WebSocket Disconnected');
        setIsConnected(false);
        globalWs = null;
        isGlobalConnecting = false;
      };
      
      if (globalWs.readyState === WebSocket.OPEN) {
        setIsConnected(true);
      }
      return;
    }
    
    // Prevent duplicate connection attempts
    if (isGlobalConnecting) {
      console.log('WebSocket connection already in progress');
      return;
    }
    
    isGlobalConnecting = true;

    // Connect to WebSocket - use backend server directly
    const wsUrl = buildWebSocketUrl(`/ws/stream/${sessionId}`);
    
    console.log(`Connecting to WebSocket: ${wsUrl}`);
    const ws = new WebSocket(wsUrl);
    
    // Set both local ref and global singleton
    wsRef.current = ws;
    globalWs = ws;

    ws.onopen = () => {
      console.log('WebSocket Connected:', wsUrl);
      setIsConnected(true);
      isGlobalConnecting = false;
    };

    ws.onmessage = handleMessage;

    ws.onclose = () => {
      console.log('WebSocket Disconnected');
      setIsConnected(false);
      globalWs = null;
      isGlobalConnecting = false;
    };

    ws.onerror = (error) => {
      console.error('WebSocket Error:', error);
      console.error('  URL:', wsUrl);
      console.error('  ReadyState:', ws.readyState);
      isGlobalConnecting = false;
    };

    return () => {
      console.log('Cleaning up WebSocket connection');
      // Don't close the global connection on cleanup (StrictMode protection)
      // Only clear local ref
      if (wsRef.current === ws) {
        wsRef.current = null;
      }
    };
  }, [sessionId, handleMessage]);

  // Session actions
  const startSession = useCallback(() => {
    console.log('startSession called', {
      wsExists: !!wsRef.current,
      readyState: wsRef.current?.readyState,
      isOpen: wsRef.current?.readyState === WebSocket.OPEN
    });
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      // Re-attach message handler before starting session to ensure we receive data
      // This fixes race condition where handler might be stale after re-renders
      wsRef.current.onmessage = handleMessage;
      console.log('Sending start_session message (handler re-attached)', deviceConfig);
      wsRef.current.send(JSON.stringify({ 
        type: 'start_session',
        device_type: deviceConfig.deviceType,
        device_address: deviceConfig.deviceAddress,
        device_preset: deviceConfig.devicePreset
      }));
    } else {
      console.error('Cannot start session - WebSocket not connected. ReadyState:', wsRef.current?.readyState);
    }
  }, [handleMessage, deviceConfig]);

  const stopSession = useCallback(() => {
    console.log('stopSession called', {
      wsExists: !!wsRef.current,
      readyState: wsRef.current?.readyState,
      isOpen: wsRef.current?.readyState === WebSocket.OPEN
    });
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      // Re-attach message handler to ensure we receive stop confirmation
      wsRef.current.onmessage = handleMessage;
      console.log('Sending stop_session message');
      wsRef.current.send(JSON.stringify({ type: 'stop_session' }));
    } else {
      console.error('Cannot stop session - WebSocket not connected');
    }
  }, [handleMessage]);

  // Calibration actions
  const startCalibration = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      console.log('Starting calibration with device config:', deviceConfig);
      setCalibrationStatus('starting');
      setCalibrationError(null);
      wsRef.current.onmessage = handleMessage;
      wsRef.current.send(JSON.stringify({ 
        type: 'calibration_start',
        device_type: deviceConfig.deviceType,
        device_address: deviceConfig.deviceAddress,
        device_preset: deviceConfig.devicePreset
      }));
    } else {
      console.error('Cannot start calibration - WebSocket not connected');
    }
  }, [handleMessage, deviceConfig]);

  const startCalibrationStage = useCallback((stage: 'baseline' | 'focus' | 'relax') => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      console.log('Starting calibration stage:', stage);
      wsRef.current.onmessage = handleMessage;
      wsRef.current.send(JSON.stringify({ 
        type: 'calibration_start_stage',
        stage
      }));
    } else {
      console.error('Cannot start calibration stage - WebSocket not connected');
    }
  }, [handleMessage]);

  const stopCalibrationStage = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      console.log('Stopping calibration stage');
      wsRef.current.onmessage = handleMessage;
      wsRef.current.send(JSON.stringify({ 
        type: 'calibration_stop_stage'
      }));
    } else {
      console.error('Cannot stop calibration stage - WebSocket not connected');
    }
  }, [handleMessage]);

  const trainCalibration = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      console.log('Training calibration model');
      setCalibrationStatus('training');
      wsRef.current.onmessage = handleMessage;
      wsRef.current.send(JSON.stringify({ 
        type: 'calibration_train'
      }));
    } else {
      console.error('Cannot train calibration - WebSocket not connected');
    }
  }, [handleMessage]);

  const cancelCalibration = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      console.log('Cancelling calibration');
      wsRef.current.onmessage = handleMessage;
      wsRef.current.send(JSON.stringify({ 
        type: 'calibration_cancel'
      }));
    } else {
      console.error('Cannot cancel calibration - WebSocket not connected');
    }
  }, [handleMessage]);

  return {
    // Connection state
    isConnected,
    isSessionActive,
    
    // Brain state data
    brainState,
    musicEvents,
    visualParams,
    brainStateHistory,
    
    // Device config
    deviceConfig,
    setDeviceConfig,
    
    // Session actions
    startSession,
    stopSession,
    
    // Calibration state
    calibrationStatus,
    calibrationStage,
    calibrationProgress,
    calibrationResults,
    isCalibrated,
    calibrationError,
    
    // Calibration actions
    startCalibration,
    startCalibrationStage,
    stopCalibrationStage,
    trainCalibration,
    cancelCalibration
  };
}
