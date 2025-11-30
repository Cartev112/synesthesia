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

interface UseWebSocketReturn {
  isConnected: boolean;
  isSessionActive: boolean;
  brainState: BrainState | null;
  musicEvents: MusicEvents | null;
  visualParams: VisualParams | null;
  brainStateHistory: BrainState[];
  startSession: () => void;
  stopSession: () => void;
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
  const wsRef = useRef<WebSocket | null>(null);
  const messageCountRef = useRef(0);
  
  // Generate a random session ID for now (only once globally)
  if (!globalSessionId) {
    globalSessionId = `session-${Math.random().toString(36).substr(2, 9)}`;
  }
  const sessionId = globalSessionId;

  useEffect(() => {
    // Use global singleton to prevent duplicate connections
    if (isGlobalConnecting || (globalWs && globalWs.readyState !== WebSocket.CLOSED)) {
      console.log('Using existing global WebSocket connection');
      wsRef.current = globalWs;
      if (globalWs && globalWs.readyState === WebSocket.OPEN) {
        setIsConnected(true);
      }
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

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        // Track message count for debugging
        messageCountRef.current++;
        
        // Log all message types for debugging (but not too verbose for brain_state)
        if (message.type !== 'brain_state' && message.type !== 'visual_params' && message.type !== 'music_events') {
          console.log('WebSocket message received:', message.type, message);
        } else if (messageCountRef.current % 50 === 0) {
          console.log(`WebSocket: Received ${messageCountRef.current} messages`);
        }
        
        switch (message.type) {
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
            break;
          case 'session_stopped':
            console.log('⏹️ Session stopped message received!', message);
            setIsSessionActive(false);
            break;
          case 'error':
            console.error('Server error:', message);
            break;
          default:
            console.warn('Unknown message type:', message.type);
            break;
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket Disconnected');
      setIsConnected(false);
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
  }, [sessionId]);

  const startSession = useCallback(() => {
    console.log('startSession called', {
      wsExists: !!wsRef.current,
      readyState: wsRef.current?.readyState,
      isOpen: wsRef.current?.readyState === WebSocket.OPEN
    });
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      console.log('Sending start_session message');
      wsRef.current.send(JSON.stringify({ type: 'start_session' }));
    } else {
      console.error('Cannot start session - WebSocket not connected. ReadyState:', wsRef.current?.readyState);
    }
  }, []);

  const stopSession = useCallback(() => {
    console.log('stopSession called', {
      wsExists: !!wsRef.current,
      readyState: wsRef.current?.readyState,
      isOpen: wsRef.current?.readyState === WebSocket.OPEN
    });
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      console.log('Sending stop_session message');
      wsRef.current.send(JSON.stringify({ type: 'stop_session' }));
    } else {
      console.error('Cannot stop session - WebSocket not connected');
    }
  }, []);

  return {
    isConnected,
    isSessionActive,
    brainState,
    musicEvents,
    visualParams,
    brainStateHistory,
    startSession,
    stopSession
  };
}
