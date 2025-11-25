import axios from 'axios';

const api = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface SystemStatus {
  status: string;
  components: {
    eeg_simulator: { status: string };
    signal_processing: { status: string };
    ml_models: { status: string };
    music_generation: { status: string };
    visual_generation: { status: string };
    real_time_pipeline: { status: string };
  };
}

export const systemApi = {
  getStatus: () => api.get<SystemStatus>('/system/status'),
  getCapabilities: () => api.get('/system/capabilities'),
};

export const sessionApi = {
  start: (userId: string) => api.post('/sessions', { user_id: userId }),
  stop: (sessionId: string) => api.post(`/sessions/${sessionId}/stop`),
  getCurrent: () => api.get('/sessions/current'),
};

// Audio API
export const audioApi = {
  // Get configuration
  getTracks: () => api.get('/audio/tracks'),
  getSynthesizers: () => api.get('/audio/synthesizers'),
  getEffects: () => api.get('/audio/effects'),
  
  // Track configuration
  setTrackSynthesizer: (trackName: string, synthType: string) =>
    api.put(`/audio/tracks/${trackName}/synthesizer`, { type: synthType }),
  
  addTrackEffect: (trackName: string, effectType: string, parameters?: Record<string, any>) =>
    api.post(`/audio/tracks/${trackName}/effects`, { type: effectType, parameters }),
  
  removeTrackEffect: (trackName: string, effectIndex: number) =>
    api.delete(`/audio/tracks/${trackName}/effects/${effectIndex}`),
  
  setTrackVolume: (trackName: string, volume: number) =>
    api.put(`/audio/tracks/${trackName}/volume`, { volume: volume / 100 }), // Convert 0-100 to 0-1
  
  setTrackMute: (trackName: string, mute: boolean) =>
    api.put(`/audio/tracks/${trackName}/mute`, null, { params: { mute } }),
  
  setTrackSolo: (trackName: string, solo: boolean) =>
    api.put(`/audio/tracks/${trackName}/solo`, null, { params: { solo } }),
  
  // Master controls
  setMasterVolume: (volume: number) =>
    api.put('/audio/master/volume', { volume: volume / 100 }), // Convert 0-100 to 0-1
};

export default api;
