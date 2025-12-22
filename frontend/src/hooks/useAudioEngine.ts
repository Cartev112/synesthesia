/**
 * React hook for frontend audio engine
 */

import { useEffect, useRef, useState } from 'react';
import { FrontendAudioEngine } from '../audio/FrontendAudioEngine';
import { BrainState, SyncState } from '../audio/MusicGenerator';

export function useAudioEngine() {
  const engineRef = useRef<FrontendAudioEngine | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  // Initialize engine on mount
  useEffect(() => {
    const engine = new FrontendAudioEngine();
    engineRef.current = engine;

    engine.initialize().then(() => {
      setIsInitialized(true);
      console.log('Audio engine ready');
    });

    return () => {
      engine.dispose();
    };
  }, []);

  const start = () => {
    if (engineRef.current && isInitialized) {
      engineRef.current.start();
      setIsPlaying(true);
    }
  };

  const stop = () => {
    if (engineRef.current) {
      engineRef.current.stop();
      setIsPlaying(false);
    }
  };

  const setTrackVolume = (trackName: string, volume: number) => {
    engineRef.current?.setTrackVolume(trackName, volume / 100); // Convert 0-100 to 0-1
  };

  const setTrackMute = (trackName: string, mute: boolean) => {
    engineRef.current?.setTrackMute(trackName, mute);
  };

  const setTrackSolo = (trackName: string, solo: boolean) => {
    engineRef.current?.setTrackSolo(trackName, solo);
  };

  const setMasterVolume = (volume: number) => {
    engineRef.current?.setMasterVolume(volume / 100); // Convert 0-100 to 0-1
  };

  const setTrackSynthType = (trackName: string, synthType: string) => {
    engineRef.current?.setTrackSynthType(trackName, synthType as any);
  };

  const updateBrainState = (brainState: Partial<BrainState>) => {
    engineRef.current?.updateBrainState(brainState);
  };

  const updateSyncState = (syncState: SyncState) => {
    engineRef.current?.updateSyncState(syncState);
  };

  const setSyncMode = (enabled: boolean) => {
    engineRef.current?.setSyncMode(enabled);
  };

  return {
    isInitialized,
    isPlaying,
    start,
    stop,
    setTrackVolume,
    setTrackMute,
    setTrackSolo,
    setMasterVolume,
    setTrackSynthType,
    updateBrainState,
    updateSyncState,
    setSyncMode,
  };
}
