/**
 * Audio Engine Context
 * Provides a single shared audio engine instance across the app
 */

import { createContext, useContext, ReactNode } from 'react';
import { useAudioEngine } from '@/hooks/useAudioEngine';

type AudioEngineContextType = ReturnType<typeof useAudioEngine>;

const AudioEngineContext = createContext<AudioEngineContextType | null>(null);

export function AudioEngineProvider({ children }: { children: ReactNode }) {
  const audioEngine = useAudioEngine();

  return (
    <AudioEngineContext.Provider value={audioEngine}>
      {children}
    </AudioEngineContext.Provider>
  );
}

export function useAudioEngineContext() {
  const context = useContext(AudioEngineContext);
  if (!context) {
    throw new Error('useAudioEngineContext must be used within AudioEngineProvider');
  }
  return context;
}
