import { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Palette, Zap, Sparkles } from 'lucide-react';
import { buildApiUrl } from '@/utils/env';

interface Algorithm {
  id: string;
  name: string;
  description: string;
  complexity: string;
  best_for: string;
}

interface Preset {
  id: string;
  name: string;
  description: string;
  hue: number;
  complexity: string;
}

interface VisualSettingsProps {
  onAlgorithmChange?: (algorithm: string) => void;
  onPresetChange?: (preset: string) => void;
  onPresetParamsChange?: (params: any) => void;
}

const DISABLED_ALGORITHMS = new Set(['lissajous', 'reaction_diffusion', 'lorenz', 'harmonograph']);

export function VisualSettings({ onAlgorithmChange, onPresetChange, onPresetParamsChange }: VisualSettingsProps) {
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('hyperspace_portal');
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch available algorithms and presets
    const fetchOptions = async () => {
      try {
        const algorithmsUrl = buildApiUrl('/api/v1/visual/algorithms');
        const presetsUrl = buildApiUrl('/api/v1/visual/presets');
        const [algoRes, presetRes] = await Promise.all([
          fetch(algorithmsUrl),
          fetch(presetsUrl)
        ]);

        const algoData = await algoRes.json();
        const presetData = await presetRes.json();

        setAlgorithms(algoData.algorithms || []);
        setPresets(presetData.presets || []);
      } catch (error) {
        console.error('Failed to fetch visual options:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchOptions();
  }, []);

  const handleAlgorithmSelect = (algorithmId: string) => {
    setSelectedAlgorithm(algorithmId);
    onAlgorithmChange?.(algorithmId);
  };

  const handlePresetSelect = async (presetId: string) => {
    setSelectedPreset(presetId);
    onPresetChange?.(presetId);
    
    // Fetch and apply preset parameters
    try {
      const response = await fetch(buildApiUrl('/api/v1/visual/preset'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ preset_name: presetId }),
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Preset parameters loaded:', data.parameters);
        onPresetParamsChange?.(data.parameters);
      }
    } catch (error) {
      console.error('Failed to load preset parameters:', error);
    }
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'low': return 'text-syn-green';
      case 'medium': return 'text-syn-cyan';
      case 'high': return 'text-syn-purple';
      default: return 'text-muted-foreground';
    }
  };

  const getPresetIcon = (presetId: string) => {
    switch (presetId) {
      case 'calm': return <Sparkles className="w-4 h-4" />;
      case 'energetic': return <Zap className="w-4 h-4" />;
      case 'meditative': return <Palette className="w-4 h-4" />;
      default: return <Palette className="w-4 h-4" />;
    }
  };

  if (loading) {
    return (
      <Card className="bg-card/50 border-syn-cyan/30">
        <CardHeader>
          <CardTitle className="text-sm font-mono">VISUAL SETTINGS</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-xs text-muted-foreground">Loading...</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-card/50 border-syn-cyan/30">
      <CardHeader>
        <CardTitle className="text-sm font-mono flex items-center gap-2">
          <Palette className="w-4 h-4" />
          VISUAL SETTINGS
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Algorithm Selection */}
        <div>
          <div className="text-xs font-mono text-muted-foreground mb-2">ALGORITHM</div>
          <div className="space-y-2">
            {algorithms
              .filter((algo) => !DISABLED_ALGORITHMS.has(algo.id))
              .map((algo) => (
              <button
                key={algo.id}
                onClick={() => handleAlgorithmSelect(algo.id)}
                className={`w-full text-left p-3 rounded border transition-all ${
                  selectedAlgorithm === algo.id
                    ? 'bg-syn-cyan/10 border-syn-cyan'
                    : 'bg-background/50 border-border hover:border-syn-cyan/50'
                }`}
              >
                <div className="flex items-start justify-between mb-1">
                  <span className="text-xs font-semibold">{algo.name}</span>
                  <Badge 
                    variant="outline" 
                    className={`text-[10px] ${getComplexityColor(algo.complexity)}`}
                  >
                    {algo.complexity}
                  </Badge>
                </div>
                <div className="text-[10px] text-muted-foreground">{algo.description}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Preset Selection */}
        <div>
          <div className="text-xs font-mono text-muted-foreground mb-2">PRESETS</div>
          <div className="grid grid-cols-3 gap-2">
            {presets.map((preset) => (
              <Button
                key={preset.id}
                size="sm"
                variant={selectedPreset === preset.id ? 'neon' : 'outline'}
                onClick={() => handlePresetSelect(preset.id)}
                className="flex flex-col items-center gap-1 h-auto py-2"
              >
                {getPresetIcon(preset.id)}
                <span className="text-[10px]">{preset.name}</span>
              </Button>
            ))}
          </div>
          {selectedPreset && (
            <div className="mt-2 p-2 rounded bg-background/50 border border-border">
              <div className="text-[10px] text-muted-foreground">
                {presets.find(p => p.id === selectedPreset)?.description}
              </div>
            </div>
          )}
        </div>

        {/* Current Selection Info */}
        <div className="pt-2 border-t border-border">
          <div className="text-[10px] font-mono text-muted-foreground space-y-1">
            <div>Active: <span className="text-syn-cyan">{selectedAlgorithm}</span></div>
            {selectedPreset && (
              <div>Preset: <span className="text-syn-purple">{selectedPreset}</span></div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
