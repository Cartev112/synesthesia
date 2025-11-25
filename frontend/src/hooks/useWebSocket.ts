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
  brainState: BrainState | null;
  musicEvents: MusicEvents | null;
  visualParams: VisualParams | null;
  brainStateHistory: BrainState[];
  startSession: () => void;
  stopSession: () => void;
}

export function useWebSocket(): UseWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [brainState, setBrainState] = useState<BrainState | null>(null);
  const [musicEvents, setMusicEvents] = useState<MusicEvents | null>(null);
  const [visualParams, setVisualParams] = useState<VisualParams | null>(null);
  const [brainStateHistory, setBrainStateHistory] = useState<BrainState[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const messageCountRef = useRef(0);
  
  // Generate a random session ID for now
  const sessionId = useRef(`session-${Math.random().toString(36).substr(2, 9)}`).current;

  useEffect(() => {
    // Connect to WebSocket - use backend server directly
    const wsUrl = buildWebSocketUrl(`/ws/stream/${sessionId}`);
    
    console.log(`Connecting to WebSocket: ${wsUrl}`);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    
    // Track if this specific connection has started a session
    let sessionStarted = false;

    ws.onopen = () => {
      console.log('WebSocket Connected:', wsUrl);
      setIsConnected(true);
      
      // Auto-start session immediately after connection
      if (!sessionStarted) {
        ws.send(JSON.stringify({ type: 'start_session' }));
        sessionStarted = true;
      }
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        // Track message count for debugging
        messageCountRef.current++;
        if (messageCountRef.current % 50 === 0) {
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
          case 'session_stopped':
            // no-op; kept for potential UI indicators
            break;
          case 'error':
            console.error('Server error:', message);
            break;
          default:
            // Ignore unknown message types
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
    };

    return () => {
      ws.close();
    };
  }, [sessionId]);

  const startSession = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'start_session' }));
    } else {
      console.error('Cannot start session - WebSocket not connected. ReadyState:', wsRef.current?.readyState);
    }
  }, []);

  const stopSession = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'stop_session' }));
    }
  }, []);

  return {
    isConnected,
    brainState,
    musicEvents,
    visualParams,
    brainStateHistory,
    startSession,
    stopSession
  };
}
