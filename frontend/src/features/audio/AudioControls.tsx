import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ChevronDown, ChevronUp, Volume2, Music, Waves, Play, Pause } from 'lucide-react';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { useAudioEngineContext } from '@/contexts/AudioEngineContext';

interface TrackConfig {
  name: string;
  synth: string;
  volume: number;
  enabled: boolean;
  color: string;
}

const SYNTHESIZERS = [
  { id: 'sine', name: 'Sine', desc: 'Pure tone' },
  { id: 'square', name: 'Square', desc: 'Bright' },
  { id: 'sawtooth', name: 'Sawtooth', desc: 'Buzzy' },
  { id: 'triangle', name: 'Triangle', desc: 'Mellow' },
];

export function AudioControls() {
  const audioEngine = useAudioEngineContext();
  const [masterVolume, setMasterVolume] = useState(75);
  const [openTracks, setOpenTracks] = useState<Set<string>>(new Set(['melody']));
  
  // Update master volume
  const updateMasterVolume = (volume: number) => {
    audioEngine.setMasterVolume(volume);
  };
  
  // Update track volume
  const updateTrackVolume = (trackId: string, volume: number) => {
    audioEngine.setTrackVolume(trackId, volume);
  };
  
  const [tracks, setTracks] = useState<Record<string, TrackConfig>>({
    bass: {
      name: 'Bass',
      synth: 'sine',
      volume: 40,
      enabled: true,
      color: 'text-red-400'
    },
    harmony: {
      name: 'Harmony',
      synth: 'sawtooth',
      volume: 50,
      enabled: true,
      color: 'text-blue-400'
    },
    melody: {
      name: 'Melody',
      synth: 'square',
      volume: 60,
      enabled: true,
      color: 'text-syn-cyan'
    },
    texture: {
      name: 'Texture',
      synth: 'triangle',
      volume: 40,
      enabled: true,
      color: 'text-purple-400'
    },
  });

  const toggleTrack = (trackId: string) => {
    const newOpenTracks = new Set(openTracks);
    if (newOpenTracks.has(trackId)) {
      newOpenTracks.delete(trackId);
    } else {
      newOpenTracks.add(trackId);
    }
    setOpenTracks(newOpenTracks);
  };

  const updateTrackSynth = (trackId: string, synthId: string) => {
    audioEngine.setTrackSynthType(trackId, synthId);
    setTracks(prev => ({
      ...prev,
      [trackId]: { ...prev[trackId], synth: synthId }
    }));
  };

  const toggleTrackEnabled = (trackId: string) => {
    const newEnabled = !tracks[trackId].enabled;
    audioEngine.setTrackMute(trackId, !newEnabled);
    setTracks(prev => ({
      ...prev,
      [trackId]: { ...prev[trackId], enabled: newEnabled }
    }));
  };

  return (
    <Card className="bg-card/50 border-syn-cyan/30">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-mono flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Music className="w-4 h-4" />
            MUSIC ENGINE
          </div>
          <Button
            size="sm"
            variant={audioEngine.isPlaying ? "destructive" : "default"}
            onClick={() => audioEngine.isPlaying ? audioEngine.stop() : audioEngine.start()}
            disabled={!audioEngine.isInitialized}
            className="h-6 px-2"
          >
            {audioEngine.isPlaying ? (
              <><Pause className="w-3 h-3 mr-1" /> Stop</>
            ) : (
              <><Play className="w-3 h-3 mr-1" /> Start</>
            )}
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Master Volume */}
        <div className="space-y-2 pb-3 border-b border-border">
          <div className="flex justify-between text-[10px] text-muted-foreground">
            <span className="flex items-center gap-1">
              <Volume2 className="w-3 h-3" /> MASTER
            </span>
            <span className="text-syn-cyan">{masterVolume}%</span>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            value={masterVolume}
            onChange={(e) => {
              const newVolume = parseInt(e.target.value);
              setMasterVolume(newVolume);
            }}
            onMouseUp={(e) => {
              const newVolume = parseInt((e.target as HTMLInputElement).value);
              updateMasterVolume(newVolume);
            }}
            className="w-full h-1.5 bg-secondary rounded-lg appearance-none cursor-pointer accent-syn-cyan"
          />
        </div>

        {/* Tracks */}
        <div className="space-y-2 max-h-[500px] overflow-y-auto">
          {Object.entries(tracks).map(([trackId, track]) => (
            <Collapsible
              key={trackId}
              open={openTracks.has(trackId)}
              onOpenChange={() => toggleTrack(trackId)}
            >
              <div className="border border-border rounded-lg bg-background/30">
                <CollapsibleTrigger className="w-full px-3 py-2 flex items-center justify-between hover:bg-accent/50 transition-colors">
                  <div className="flex items-center gap-2">
                    <div
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleTrackEnabled(trackId);
                      }}
                      className="h-5 w-5 flex items-center justify-center cursor-pointer hover:bg-accent rounded"
                    >
                      <Waves className={`w-3 h-3 ${track.enabled ? track.color : 'text-muted-foreground'}`} />
                    </div>
                    <span className={`text-xs font-semibold ${track.color}`}>
                      {track.name.toUpperCase()}
                    </span>
                    <Badge variant="outline" className="text-[9px] px-1 py-0">
                      {SYNTHESIZERS.find(s => s.id === track.synth)?.name}
                    </Badge>
                  </div>
                  {openTracks.has(trackId) ? (
                    <ChevronUp className="w-3 h-3" />
                  ) : (
                    <ChevronDown className="w-3 h-3" />
                  )}
                </CollapsibleTrigger>

                <CollapsibleContent>
                  <div className="px-3 py-2 space-y-3 bg-background/50">
                    {/* Synthesizer Selection */}
                    <div>
                      <div className="text-[9px] text-muted-foreground mb-1">SYNTHESIZER</div>
                      <div className="grid grid-cols-2 gap-1">
                        {SYNTHESIZERS.map((synth) => (
                          <button
                            key={synth.id}
                            onClick={() => updateTrackSynth(trackId, synth.id)}
                            className={`text-[9px] px-2 py-1 rounded border transition-colors ${
                              track.synth === synth.id
                                ? 'bg-syn-cyan/10 border-syn-cyan text-syn-cyan'
                                : 'border-border hover:border-syn-cyan/50'
                            }`}
                          >
                            {synth.name}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Volume */}
                    <div>
                      <div className="flex justify-between text-[9px] text-muted-foreground mb-1">
                        <span>VOLUME</span>
                        <span className="text-syn-cyan">{track.volume}%</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={track.volume}
                        onChange={(e) => {
                          const newVolume = parseInt(e.target.value);
                          setTracks(prev => ({
                            ...prev,
                            [trackId]: { ...prev[trackId], volume: newVolume }
                          }));
                        }}
                        onMouseUp={(e) => {
                          const newVolume = parseInt((e.target as HTMLInputElement).value);
                          updateTrackVolume(trackId, newVolume);
                        }}
                        className="w-full h-1 bg-secondary rounded-lg appearance-none cursor-pointer accent-syn-cyan"
                      />
                    </div>
                  </div>
                </CollapsibleContent>
              </div>
            </Collapsible>
          ))}
        </div>

        {/* Info */}
        <div className="pt-2 border-t border-border">
          <div className="text-[9px] text-muted-foreground space-y-0.5">
            <p>• 4 musical layers (bass, harmony, melody, texture)</p>
            <p>• 4 synthesizers per track</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
