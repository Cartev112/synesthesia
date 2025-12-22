import { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from '@/components/ui/card';
import { deviceApi, DeviceInfo } from '@/services/api';
import { Bluetooth, Loader2, Radio, Check, X, RefreshCw, Cpu } from 'lucide-react';

interface DeviceConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  onDeviceSelected: (deviceType: string, deviceAddress?: string, devicePreset?: string) => void;
}

type ScanState = 'idle' | 'scanning' | 'found' | 'error' | 'connecting';

export function DeviceConfigModal({ isOpen, onClose, onDeviceSelected }: DeviceConfigModalProps) {
  const [scanState, setScanState] = useState<ScanState>('idle');
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<DeviceInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedPreset, setSelectedPreset] = useState('full_research');

  const presets = [
    { id: 'eeg_only', name: 'EEG Only', description: 'EEG + Motion + Battery' },
    { id: 'eeg_basic', name: 'EEG Basic', description: 'EEG + Optics4 + Motion' },
    { id: 'eeg_ppg', name: 'EEG + PPG', description: 'EEG8 + Optics4 + Motion' },
    { id: 'full_research', name: 'Full Research', description: 'All sensors enabled' },
  ];

  const startScan = useCallback(async () => {
    setScanState('scanning');
    setError(null);
    setDevices([]);
    setSelectedDevice(null);

    try {
      const response = await deviceApi.scan(10);
      const foundDevices = response.data.devices;
      setDevices(foundDevices);
      setScanState(foundDevices.length > 0 ? 'found' : 'idle');
      
      if (foundDevices.length === 0) {
        setError('No Muse devices found. Make sure your device is powered on and in pairing mode.');
      }
    } catch (err: any) {
      console.error('Device scan error:', err);
      setScanState('error');
      setError(err.response?.data?.detail || 'Failed to scan for devices. Is the backend running?');
    }
  }, []);

  const handleConnect = useCallback(async () => {
    if (!selectedDevice) return;

    setScanState('connecting');
    setError(null);

    try {
      await deviceApi.connect(selectedDevice.address, selectedPreset);
      onDeviceSelected('muse_s_athena', selectedDevice.address, selectedPreset);
      onClose();
    } catch (err: any) {
      console.error('Device connect error:', err);
      setScanState('error');
      setError(err.response?.data?.detail || 'Failed to connect to device');
    }
  }, [selectedDevice, selectedPreset, onDeviceSelected, onClose]);

  const handleUseSimulator = useCallback(() => {
    onDeviceSelected('simulator');
    onClose();
  }, [onDeviceSelected, onClose]);

  // Auto-start scan when modal opens
  useEffect(() => {
    if (isOpen && scanState === 'idle') {
      startScan();
    }
  }, [isOpen, scanState, startScan]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/80 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal */}
      <Card className="relative z-10 w-full max-w-lg mx-4 border-syn-cyan/30 bg-syn-dark/95 shadow-[0_0_50px_rgba(0,243,255,0.15)]">
        <CardHeader className="border-b border-white/10">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-syn-cyan/10 border border-syn-cyan/30">
              <Bluetooth className="w-6 h-6 text-syn-cyan" />
            </div>
            <div>
              <CardTitle className="text-syn-cyan">Device Configuration</CardTitle>
              <CardDescription>Connect to your Muse S Athena or use the simulator</CardDescription>
            </div>
          </div>
        </CardHeader>

        <CardContent className="pt-6 space-y-6">
          {/* Scan Status */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-mono text-muted-foreground">DEVICE SCAN</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={startScan}
                disabled={scanState === 'scanning' || scanState === 'connecting'}
                className="h-7 text-xs"
              >
                <RefreshCw className={`w-3 h-3 mr-1 ${scanState === 'scanning' ? 'animate-spin' : ''}`} />
                Rescan
              </Button>
            </div>

            {/* Scanning indicator */}
            {scanState === 'scanning' && (
              <div className="flex items-center gap-3 p-4 rounded-lg border border-syn-purple/30 bg-syn-purple/5">
                <Loader2 className="w-5 h-5 text-syn-purple animate-spin" />
                <div>
                  <div className="text-sm font-medium text-syn-purple">Scanning for devices...</div>
                  <div className="text-xs text-muted-foreground">This may take up to 10 seconds</div>
                </div>
              </div>
            )}

            {/* Connecting indicator */}
            {scanState === 'connecting' && (
              <div className="flex items-center gap-3 p-4 rounded-lg border border-syn-green/30 bg-syn-green/5">
                <Loader2 className="w-5 h-5 text-syn-green animate-spin" />
                <div>
                  <div className="text-sm font-medium text-syn-green">Connecting to {selectedDevice?.name}...</div>
                  <div className="text-xs text-muted-foreground">Establishing BLE connection</div>
                </div>
              </div>
            )}

            {/* Error message */}
            {error && scanState !== 'scanning' && scanState !== 'connecting' && (
              <div className="flex items-start gap-3 p-4 rounded-lg border border-destructive/30 bg-destructive/5">
                <X className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
                <div className="text-sm text-destructive">{error}</div>
              </div>
            )}

            {/* Device list */}
            {devices.length > 0 && scanState !== 'connecting' && (
              <div className="space-y-2">
                {devices.map((device) => (
                  <button
                    key={device.address}
                    onClick={() => setSelectedDevice(device)}
                    className={`w-full flex items-center gap-3 p-3 rounded-lg border transition-all ${
                      selectedDevice?.address === device.address
                        ? 'border-syn-cyan bg-syn-cyan/10 shadow-[0_0_15px_rgba(0,243,255,0.2)]'
                        : 'border-white/10 bg-card/30 hover:border-white/20 hover:bg-card/50'
                    }`}
                  >
                    <Radio className={`w-5 h-5 ${
                      selectedDevice?.address === device.address ? 'text-syn-cyan' : 'text-muted-foreground'
                    }`} />
                    <div className="flex-1 text-left">
                      <div className="text-sm font-medium">{device.name}</div>
                      <div className="text-xs font-mono text-muted-foreground">{device.address}</div>
                    </div>
                    {selectedDevice?.address === device.address && (
                      <Check className="w-5 h-5 text-syn-cyan" />
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Preset Selection (shown when device selected) */}
          {selectedDevice && scanState !== 'connecting' && (
            <div className="space-y-3">
              <span className="text-sm font-mono text-muted-foreground">SENSOR PRESET</span>
              <div className="grid grid-cols-2 gap-2">
                {presets.map((preset) => (
                  <button
                    key={preset.id}
                    onClick={() => setSelectedPreset(preset.id)}
                    className={`p-3 rounded-lg border text-left transition-all ${
                      selectedPreset === preset.id
                        ? 'border-syn-purple bg-syn-purple/10'
                        : 'border-white/10 bg-card/30 hover:border-white/20'
                    }`}
                  >
                    <div className="text-sm font-medium">{preset.name}</div>
                    <div className="text-xs text-muted-foreground">{preset.description}</div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Simulator option */}
          <div className="pt-4 border-t border-white/10">
            <button
              onClick={handleUseSimulator}
              className="w-full flex items-center gap-3 p-3 rounded-lg border border-white/10 bg-card/30 hover:border-syn-green/30 hover:bg-syn-green/5 transition-all"
            >
              <Cpu className="w-5 h-5 text-syn-green" />
              <div className="flex-1 text-left">
                <div className="text-sm font-medium">Use Simulator</div>
                <div className="text-xs text-muted-foreground">Generate synthetic EEG data for testing</div>
              </div>
            </button>
          </div>
        </CardContent>

        <CardFooter className="border-t border-white/10 flex justify-between">
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="neon"
            onClick={handleConnect}
            disabled={!selectedDevice || scanState === 'connecting' || scanState === 'scanning'}
          >
            {scanState === 'connecting' ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Connecting...
              </>
            ) : (
              <>
                <Bluetooth className="w-4 h-4 mr-2" />
                Connect Device
              </>
            )}
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}
