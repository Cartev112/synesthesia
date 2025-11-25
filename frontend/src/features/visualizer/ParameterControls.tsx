import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { ChevronDown, ChevronUp, RotateCcw, Settings2 } from 'lucide-react';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';

interface ParameterControlsProps {
  onParamsChange?: (params: any) => void;
  currentParams?: any;
}

interface ParamConfig {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

const PARAM_GROUPS = {
  wave: {
    title: 'Wave Parameters',
    params: [
      { key: 'frequency_ratio_x', label: 'Frequency X', min: 0.5, max: 10, step: 0.1, default: 3.0 },
      { key: 'frequency_ratio_y', label: 'Frequency Y', min: 0.5, max: 10, step: 0.1, default: 2.0 },
      { key: 'phase_offset', label: 'Phase Offset', min: 0, max: 6.28, step: 0.01, default: 0 },
      { key: 'amplitude_x', label: 'Amplitude X', min: 0.1, max: 2, step: 0.05, default: 0.8 },
      { key: 'amplitude_y', label: 'Amplitude Y', min: 0.1, max: 2, step: 0.05, default: 0.8 },
    ] as ParamConfig[]
  },
  portal: {
    title: 'Portal (Hyperspace)',
    params: [
      { key: 'portal_symmetry', label: 'Symmetry Arms', min: 3, max: 14, step: 1, default: 8 },
      { key: 'portal_radial_frequency', label: 'Radial Frequency', min: 1, max: 12, step: 0.1, default: 6.0 },
      { key: 'portal_angular_frequency', label: 'Angular Frequency', min: 0.5, max: 4, step: 0.05, default: 2.0 },
      { key: 'portal_warp', label: 'Warp Intensity', min: 0, max: 1, step: 0.05, default: 0.4 },
      { key: 'portal_spiral', label: 'Spiral Twist', min: -1.5, max: 1.5, step: 0.05, default: 0.6 },
      { key: 'portal_layers', label: 'Depth Layers', min: 2, max: 8, step: 1, default: 4 },
      { key: 'portal_radius', label: 'Base Radius', min: 0.2, max: 1, step: 0.02, default: 0.55 },
      { key: 'portal_ripple', label: 'Ripple Amplitude', min: 0, max: 0.6, step: 0.02, default: 0.25 },
      { key: 'portal_depth_skew', label: 'Depth Skew', min: 0, max: 1, step: 0.05, default: 0.4 },
    ] as ParamConfig[]
  },
  complexity: {
    title: 'Complexity',
    params: [
      { key: 'num_harmonics', label: 'Harmonics', min: 1, max: 12, step: 1, default: 5 },
      { key: 'num_epicycles', label: 'Epicycles', min: 1, max: 12, step: 1, default: 5 },
      { key: 'epicycle_decay', label: 'Epicycle Decay', min: 0.3, max: 0.95, step: 0.05, default: 0.7 },
      { key: 'point_density', label: 'Point Density', min: 128, max: 2048, step: 64, default: 1024 },
    ] as ParamConfig[]
  },
  color: {
    title: 'Color',
    params: [
      { key: 'hue_base', label: 'Hue', min: 0, max: 360, step: 1, default: 180 },
      { key: 'saturation', label: 'Saturation', min: 0, max: 1, step: 0.05, default: 0.7 },
      { key: 'brightness', label: 'Brightness', min: 0, max: 1, step: 0.05, default: 0.8 },
      { key: 'color_cycle_speed', label: 'Color Cycle Speed', min: 0, max: 1, step: 0.05, default: 0.2 },
    ] as ParamConfig[]
  },
  animation: {
    title: 'Animation',
    params: [
      { key: 'rotation_speed', label: 'Rotation Speed', min: 0, max: 1, step: 0.05, default: 0 },
      { key: 'speed_multiplier', label: 'Speed', min: 0.1, max: 3, step: 0.1, default: 1.0 },
      { key: 'pulse_frequency', label: 'Pulse Frequency', min: 0, max: 3, step: 0.1, default: 1.0 },
      { key: 'pulse_amplitude', label: 'Pulse Amplitude', min: 0, max: 0.5, step: 0.05, default: 0.2 },
      { key: 'trail_length', label: 'Trail Length', min: 0, max: 1, step: 0.05, default: 0.1 },
    ] as ParamConfig[]
  },
  damping: {
    title: 'Damping',
    params: [
      { key: 'damping_x', label: 'Damping X', min: 0, max: 0.1, step: 0.005, default: 0.03 },
      { key: 'damping_y', label: 'Damping Y', min: 0, max: 0.1, step: 0.005, default: 0.03 },
    ] as ParamConfig[]
  }
};

export function ParameterControls({ onParamsChange, currentParams }: ParameterControlsProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [openGroups, setOpenGroups] = useState<Set<string>>(new Set(['portal']));
  const [localParams, setLocalParams] = useState<Record<string, number>>({});

  const toggleGroup = (groupKey: string) => {
    const newOpenGroups = new Set(openGroups);
    if (newOpenGroups.has(groupKey)) {
      newOpenGroups.delete(groupKey);
    } else {
      newOpenGroups.add(groupKey);
    }
    setOpenGroups(newOpenGroups);
  };

  const handleParamChange = (key: string, value: number) => {
    const newParams = { ...localParams, [key]: value };
    setLocalParams(newParams);
    onParamsChange?.(newParams);
  };

  const resetToDefaults = () => {
    const defaults: Record<string, number> = {};
    Object.values(PARAM_GROUPS).forEach(group => {
      group.params.forEach(param => {
        defaults[param.key] = param.default;
      });
    });
    setLocalParams(defaults);
    onParamsChange?.(defaults);
  };

  const resetGroup = (groupKey: string) => {
    const group = PARAM_GROUPS[groupKey as keyof typeof PARAM_GROUPS];
    const newParams = { ...localParams };
    group.params.forEach(param => {
      newParams[param.key] = param.default;
    });
    setLocalParams(newParams);
    onParamsChange?.(newParams);
  };

  const getValue = (param: ParamConfig): number => {
    return localParams[param.key] ?? currentParams?.[param.key] ?? param.default;
  };

  return (
    <Card className="bg-card/50 border-syn-cyan/30">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-mono flex items-center gap-2">
            <Settings2 className="w-4 h-4" />
            PARAMETER CONTROLS
          </CardTitle>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="ghost"
              onClick={resetToDefaults}
              className="h-7 text-xs"
              title="Reset all to defaults"
            >
              <RotateCcw className="w-3 h-3" />
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setIsOpen(!isOpen)}
              className="h-7"
            >
              {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </Button>
          </div>
        </div>
      </CardHeader>
      
      {isOpen && (
        <CardContent className="space-y-3 max-h-[600px] overflow-y-auto">
          {Object.entries(PARAM_GROUPS).map(([groupKey, group]) => (
            <Collapsible
              key={groupKey}
              open={openGroups.has(groupKey)}
              onOpenChange={() => toggleGroup(groupKey)}
            >
              <div className="border border-border rounded-lg">
                <CollapsibleTrigger className="w-full px-3 py-2 flex items-center justify-between hover:bg-accent/50 transition-colors">
                  <span className="text-xs font-semibold text-syn-cyan">{group.title}</span>
                  <div className="flex items-center gap-2">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={(e) => {
                        e.stopPropagation();
                        resetGroup(groupKey);
                      }}
                      className="h-5 w-5 p-0"
                      title="Reset group"
                    >
                      <RotateCcw className="w-3 h-3" />
                    </Button>
                    {openGroups.has(groupKey) ? (
                      <ChevronUp className="w-3 h-3" />
                    ) : (
                      <ChevronDown className="w-3 h-3" />
                    )}
                  </div>
                </CollapsibleTrigger>
                
                <CollapsibleContent>
                  <div className="px-3 py-2 space-y-3 bg-background/30">
                    {group.params.map((param) => {
                      const value = getValue(param);
                      return (
                        <div key={param.key} className="space-y-1">
                          <div className="flex items-center justify-between">
                            <Label className="text-[10px] text-muted-foreground">
                              {param.label}
                            </Label>
                            <span className="text-[10px] font-mono text-syn-cyan">
                              {value.toFixed(param.step < 0.1 ? 3 : param.step < 1 ? 2 : 0)}
                            </span>
                          </div>
                          <Slider
                            value={[value]}
                            min={param.min}
                            max={param.max}
                            step={param.step}
                            onValueChange={([newValue]: number[]) => handleParamChange(param.key, newValue)}
                            className="cursor-pointer"
                          />
                        </div>
                      );
                    })}
                  </div>
                </CollapsibleContent>
              </div>
            </Collapsible>
          ))}
          
          <div className="pt-2 border-t border-border">
            <div className="text-[10px] text-muted-foreground space-y-1">
              <p>• Adjust parameters in real-time</p>
              <p>• Click group reset to restore defaults</p>
              <p>• Changes override brain state & presets</p>
            </div>
          </div>
        </CardContent>
      )}
    </Card>
  );
}
