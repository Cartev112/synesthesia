/**
 * Hook for managing multi-user BCI sync sessions
 */

import { useState, useEffect, useCallback, useRef } from 'react';

export interface SyncState {
  sync_score: number;
  dissonance_level: number;
  alpha_plv: number;
  theta_plv: number;
  bandpower_correlation: number;
  asymmetry_correlation: number;
  baseline_delta: number;
  quality: number;
  timestamp: number;
}

export interface SyncSessionState {
  phase: 'idle' | 'connecting' | 'baseline' | 'active' | 'stopped';
  userAConnected: boolean;
  userBConnected: boolean;
  baselineComplete: boolean;
  baselineProgress: number;
  usersConnected: string[];
}

export interface UseSyncSessionReturn {
  // Connection state
  isConnected: boolean;
  sessionId: string | null;
  userId: string | null;
  
  // Session state
  sessionState: SyncSessionState;
  syncState: SyncState | null;
  
  // Sync history for visualization
  syncHistory: SyncState[];
  
  // Actions
  createSession: (deviceTypeA?: string, deviceTypeB?: string) => void;
  joinSession: (sessionId: string) => void;
  startSync: () => void;
  stopSync: () => void;
  disconnect: () => void;
  
  // For testing
  setSimulatorStates: (stateA: string, stateB: string) => void;
  
  // Errors
  error: string | null;
}

const WS_URL = 'ws://localhost:8000';
const SYNC_HISTORY_SIZE = 60; // 60 samples for ~15 seconds at 4Hz

export function useSyncSession(): UseSyncSessionReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const [sessionState, setSessionState] = useState<SyncSessionState>({
    phase: 'idle',
    userAConnected: false,
    userBConnected: false,
    baselineComplete: false,
    baselineProgress: 0,
    usersConnected: [],
  });
  
  const [syncState, setSyncState] = useState<SyncState | null>(null);
  const [syncHistory, setSyncHistory] = useState<SyncState[]>([]);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  // Connect to WebSocket
  const connect = useCallback((sessId: string, usrId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }
    
    const ws = new WebSocket(`${WS_URL}/ws/sync/${sessId}/${usrId}`);
    
    ws.onopen = () => {
      console.log('Sync WebSocket connected');
      setIsConnected(true);
      setError(null);
    };
    
    ws.onclose = () => {
      console.log('Sync WebSocket disconnected');
      setIsConnected(false);
    };
    
    ws.onerror = (e) => {
      console.error('Sync WebSocket error:', e);
      setError('WebSocket connection error');
    };
    
    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        handleMessage(message);
      } catch (e) {
        console.error('Failed to parse message:', e);
      }
    };
    
    wsRef.current = ws;
  }, []);

  // Handle incoming messages
  const handleMessage = useCallback((message: any) => {
    console.log('Sync message:', message.type, message);
    
    switch (message.type) {
      case 'sync_created':
        setSessionState(prev => ({
          ...prev,
          phase: 'connecting',
          usersConnected: message.users_connected || [],
        }));
        break;
        
      case 'sync_user_joined':
        setSessionState(prev => ({
          ...prev,
          usersConnected: message.users_connected || [],
          userAConnected: message.users_connected?.includes('user_a'),
          userBConnected: message.users_connected?.includes('user_b'),
        }));
        break;
        
      case 'sync_started':
        setSessionState(prev => ({
          ...prev,
          phase: 'baseline',
          baselineProgress: 0,
        }));
        break;
        
      case 'sync_phase_changed':
        setSessionState(prev => ({
          ...prev,
          phase: message.phase,
          baselineProgress: message.baseline_progress || 0,
          baselineComplete: message.baseline_complete || false,
        }));
        break;
        
      case 'sync_state':
        const newSyncState = message.data as SyncState;
        setSyncState(newSyncState);
        setSyncHistory(prev => {
          const updated = [...prev, newSyncState];
          return updated.slice(-SYNC_HISTORY_SIZE);
        });
        break;
        
      case 'sync_stopped':
        setSessionState(prev => ({
          ...prev,
          phase: 'stopped',
        }));
        break;
        
      case 'error':
        setError(message.message);
        break;
    }
  }, []);

  // Create a new sync session
  const createSession = useCallback((
    deviceTypeA: string = 'simulator',
    deviceTypeB: string = 'simulator'
  ) => {
    // Generate a random session ID
    const newSessionId = `sync_${Date.now().toString(36)}_${Math.random().toString(36).substr(2, 5)}`;
    const newUserId = 'user_a'; // Creator is always user_a
    
    setSessionId(newSessionId);
    setUserId(newUserId);
    
    // Connect and create
    connect(newSessionId, newUserId);
    
    // Wait for connection then send create message
    const checkAndSend = () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'sync_create',
          device_type_a: deviceTypeA,
          device_type_b: deviceTypeB,
          baseline_duration: 30,
        }));
      } else {
        setTimeout(checkAndSend, 100);
      }
    };
    setTimeout(checkAndSend, 100);
  }, [connect]);

  // Join an existing session
  const joinSession = useCallback((sessId: string) => {
    const newUserId = 'user_b'; // Joiner is always user_b
    
    setSessionId(sessId);
    setUserId(newUserId);
    
    // Connect and join
    connect(sessId, newUserId);
    
    // Wait for connection then send join message
    const checkAndSend = () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'sync_join',
        }));
      } else {
        setTimeout(checkAndSend, 100);
      }
    };
    setTimeout(checkAndSend, 100);
  }, [connect]);

  // Start the sync session
  const startSync = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'sync_start',
      }));
    }
  }, []);

  // Stop the sync session
  const stopSync = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'sync_stop',
      }));
    }
  }, []);

  // Disconnect from session
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
    setSessionId(null);
    setUserId(null);
    setSyncState(null);
    setSyncHistory([]);
    setSessionState({
      phase: 'idle',
      userAConnected: false,
      userBConnected: false,
      baselineComplete: false,
      baselineProgress: 0,
      usersConnected: [],
    });
  }, []);

  // Set simulator states (for testing)
  const setSimulatorStates = useCallback((stateA: string, stateB: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'set_simulator_state',
        state_a: stateA,
        state_b: stateB,
        intensity: 1.0,
      }));
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  return {
    isConnected,
    sessionId,
    userId,
    sessionState,
    syncState,
    syncHistory,
    createSession,
    joinSession,
    startSync,
    stopSync,
    disconnect,
    setSimulatorStates,
    error,
  };
}

