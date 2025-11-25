import { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Activity, Cpu, Music, Eye, Zap } from 'lucide-react';
import { systemApi, type SystemStatus } from '@/services/api';

export function StatusPanel() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await systemApi.getStatus();
        setStatus(response.data);
      } catch (error) {
        console.error("Failed to fetch system status", error);
      } finally {
        setLoading(false);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    return status === 'available' || status === 'operational' ? 'success' : 'destructive';
  };

  if (loading) return <div className="animate-pulse p-4">Loading system status...</div>;

  return (
    <Card className="h-full bg-card/30 backdrop-blur-md border-white/10">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="text-syn-cyan" />
          System Status
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/30 border border-white/5">
          <div className="flex items-center gap-3">
            <Zap className="w-4 h-4 text-yellow-400" />
            <span>EEG Simulator</span>
          </div>
          <Badge variant={getStatusColor(status?.components.eeg_simulator.status || 'error')}>
            {status?.components.eeg_simulator.status || 'OFFLINE'}
          </Badge>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/30 border border-white/5">
          <div className="flex items-center gap-3">
            <Cpu className="w-4 h-4 text-blue-400" />
            <span>Signal Processing</span>
          </div>
          <Badge variant={getStatusColor(status?.components.signal_processing.status || 'error')}>
            {status?.components.signal_processing.status || 'OFFLINE'}
          </Badge>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/30 border border-white/5">
          <div className="flex items-center gap-3">
            <Music className="w-4 h-4 text-syn-purple" />
            <span>Music Engine</span>
          </div>
          <Badge variant={getStatusColor(status?.components.music_generation.status || 'error')}>
            {status?.components.music_generation.status || 'OFFLINE'}
          </Badge>
        </div>

        <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/30 border border-white/5">
          <div className="flex items-center gap-3">
            <Eye className="w-4 h-4 text-syn-green" />
            <span>Visual Engine</span>
          </div>
          <Badge variant={getStatusColor(status?.components.visual_generation.status || 'error')}>
            {status?.components.visual_generation.status || 'OFFLINE'}
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}
