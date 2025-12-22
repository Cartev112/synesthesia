/**
 * Sync Session Panel
 * UI for creating, joining, and managing multi-user sync sessions
 */

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { 
  Users, 
  Link2, 
  Copy, 
  Check, 
  Play, 
  Square,
  Wifi,
  WifiOff,
  Brain,
  Activity,
  Loader2
} from 'lucide-react';
import { useSyncSession, SyncState } from '@/hooks/useSyncSession';
import { SyncMeter } from './SyncMeter';

interface SyncSessionPanelProps {
  onSyncStateChange?: (syncState: SyncState | null) => void;
}

export function SyncSessionPanel({ onSyncStateChange }: SyncSessionPanelProps) {
  const {
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
  } = useSyncSession();

  const [joinCode, setJoinCode] = useState('');
  const [copied, setCopied] = useState(false);
  const [showTestControls, setShowTestControls] = useState(false);

  // Notify parent of sync state changes
  if (onSyncStateChange && syncState) {
    onSyncStateChange(syncState);
  }

  const copySessionId = () => {
    if (sessionId) {
      navigator.clipboard.writeText(sessionId);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleJoin = () => {
    if (joinCode.trim()) {
      joinSession(joinCode.trim());
    }
  };

  const bothUsersConnected = sessionState.usersConnected.length >= 2;

  // Not in a session - show create/join options
  if (!sessionId) {
    return (
      <div className="rounded-xl border border-white/10 bg-card/50 backdrop-blur-md p-4 space-y-4">
        <div className="flex items-center gap-2 mb-4">
          <Users className="w-5 h-5 text-syn-purple" />
          <h3 className="text-lg font-display text-syn-purple">MULTI-USER SYNC</h3>
        </div>

        <p className="text-sm text-muted-foreground">
          Connect two BCI users to synchronize brain states and generate music that 
          reflects your neural harmony.
        </p>

        {/* Create Session */}
        <div className="space-y-2">
          <Button 
            onClick={() => createSession()}
            className="w-full"
            variant="neon"
          >
            <Link2 className="w-4 h-4 mr-2" />
            Create Sync Session
          </Button>
          <p className="text-xs text-muted-foreground text-center">
            Create a new session and share the code with your partner
          </p>
        </div>

        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <span className="w-full border-t border-white/10" />
          </div>
          <div className="relative flex justify-center text-xs uppercase">
            <span className="bg-card px-2 text-muted-foreground">or</span>
          </div>
        </div>

        {/* Join Session */}
        <div className="space-y-2">
          <div className="flex gap-2">
            <Input
              placeholder="Enter session code..."
              value={joinCode}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setJoinCode(e.target.value)}
              className="font-mono text-sm"
            />
            <Button onClick={handleJoin} disabled={!joinCode.trim()}>
              Join
            </Button>
          </div>
          <p className="text-xs text-muted-foreground text-center">
            Enter a code shared by your partner to join their session
          </p>
        </div>

        {error && (
          <div className="text-sm text-destructive bg-destructive/10 border border-destructive/30 rounded p-2">
            {error}
          </div>
        )}
      </div>
    );
  }

  // In a session - show session info and controls
  return (
    <div className="rounded-xl border border-white/10 bg-card/50 backdrop-blur-md p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Users className="w-5 h-5 text-syn-purple" />
          <h3 className="text-lg font-display text-syn-purple">SYNC SESSION</h3>
        </div>
        <div className="flex items-center gap-2 text-xs">
          {isConnected ? (
            <span className="flex items-center gap-1 text-syn-green">
              <Wifi className="w-3 h-3" /> ONLINE
            </span>
          ) : (
            <span className="flex items-center gap-1 text-destructive">
              <WifiOff className="w-3 h-3" /> OFFLINE
            </span>
          )}
        </div>
      </div>

      {/* Session ID */}
      <div className="flex items-center gap-2 p-2 rounded-lg bg-black/30 border border-white/5">
        <span className="text-xs text-muted-foreground">SESSION:</span>
        <code className="text-xs font-mono text-syn-cyan flex-1 truncate">{sessionId}</code>
        <Button 
          variant="ghost" 
          size="sm" 
          className="h-6 w-6 p-0"
          onClick={copySessionId}
        >
          {copied ? <Check className="w-3 h-3 text-syn-green" /> : <Copy className="w-3 h-3" />}
        </Button>
      </div>

      {/* User Status */}
      <div className="grid grid-cols-2 gap-2">
        <div className={`p-3 rounded-lg border ${
          sessionState.usersConnected.includes('user_a') 
            ? 'border-syn-green/50 bg-syn-green/10' 
            : 'border-white/10 bg-black/20'
        }`}>
          <div className="flex items-center gap-2">
            <Brain className={`w-4 h-4 ${
              sessionState.usersConnected.includes('user_a') ? 'text-syn-green' : 'text-muted-foreground'
            }`} />
            <span className="text-xs font-mono">
              USER A {userId === 'user_a' && '(YOU)'}
            </span>
          </div>
          <div className="text-[10px] text-muted-foreground mt-1">
            {sessionState.usersConnected.includes('user_a') ? 'Connected' : 'Waiting...'}
          </div>
        </div>
        
        <div className={`p-3 rounded-lg border ${
          sessionState.usersConnected.includes('user_b')
            ? 'border-syn-green/50 bg-syn-green/10' 
            : 'border-white/10 bg-black/20'
        }`}>
          <div className="flex items-center gap-2">
            <Brain className={`w-4 h-4 ${
              sessionState.usersConnected.includes('user_b') ? 'text-syn-green' : 'text-muted-foreground'
            }`} />
            <span className="text-xs font-mono">
              USER B {userId === 'user_b' && '(YOU)'}
            </span>
          </div>
          <div className="text-[10px] text-muted-foreground mt-1">
            {sessionState.usersConnected.includes('user_b') ? 'Connected' : 'Waiting...'}
          </div>
        </div>
      </div>

      {/* Session Phase */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-muted-foreground">PHASE:</span>
          <span className={`font-mono uppercase ${
            sessionState.phase === 'active' ? 'text-syn-green' :
            sessionState.phase === 'baseline' ? 'text-syn-cyan' :
            'text-muted-foreground'
          }`}>
            {sessionState.phase}
          </span>
        </div>

        {/* Baseline Progress */}
        {sessionState.phase === 'baseline' && (
          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground">Calibrating baseline...</span>
              <span className="text-syn-cyan">{Math.round(sessionState.baselineProgress * 100)}%</span>
            </div>
            <div className="h-2 bg-black/30 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-syn-cyan to-syn-purple transition-all duration-300"
                style={{ width: `${sessionState.baselineProgress * 100}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Sync Meter */}
      {syncState && sessionState.phase === 'active' && (
        <SyncMeter syncState={syncState} syncHistory={syncHistory} />
      )}

      {/* Controls */}
      <div className="flex gap-2">
        {sessionState.phase === 'idle' || sessionState.phase === 'connecting' ? (
          <Button 
            onClick={startSync}
            disabled={!bothUsersConnected}
            className="flex-1"
            variant="neon"
          >
            {!bothUsersConnected ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Waiting for partner...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Start Sync
              </>
            )}
          </Button>
        ) : sessionState.phase === 'baseline' ? (
          <Button disabled className="flex-1" variant="outline">
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            Calibrating...
          </Button>
        ) : sessionState.phase === 'active' ? (
          <Button onClick={stopSync} className="flex-1" variant="destructive">
            <Square className="w-4 h-4 mr-2" />
            Stop Sync
          </Button>
        ) : (
          <Button onClick={startSync} className="flex-1" variant="neon">
            <Play className="w-4 h-4 mr-2" />
            Restart
          </Button>
        )}
        
        <Button onClick={disconnect} variant="ghost" size="icon">
          <WifiOff className="w-4 h-4" />
        </Button>
      </div>

      {/* Test Controls (for simulator) */}
      {sessionState.phase === 'active' && (
        <div className="pt-2 border-t border-white/10">
          <button
            onClick={() => setShowTestControls(!showTestControls)}
            className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1"
          >
            <Activity className="w-3 h-3" />
            {showTestControls ? 'Hide' : 'Show'} Test Controls
          </button>
          
          {showTestControls && (
            <div className="mt-2 grid grid-cols-2 gap-2">
              <Button 
                size="sm" 
                variant="outline"
                className="text-xs"
                onClick={() => setSimulatorStates('focus', 'focus')}
              >
                Both Focus
              </Button>
              <Button 
                size="sm" 
                variant="outline"
                className="text-xs"
                onClick={() => setSimulatorStates('relax', 'relax')}
              >
                Both Relax
              </Button>
              <Button 
                size="sm" 
                variant="outline"
                className="text-xs"
                onClick={() => setSimulatorStates('focus', 'relax')}
              >
                Focus vs Relax
              </Button>
              <Button 
                size="sm" 
                variant="outline"
                className="text-xs"
                onClick={() => setSimulatorStates('neutral', 'neutral')}
              >
                Both Neutral
              </Button>
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="text-sm text-destructive bg-destructive/10 border border-destructive/30 rounded p-2">
          {error}
        </div>
      )}
    </div>
  );
}

