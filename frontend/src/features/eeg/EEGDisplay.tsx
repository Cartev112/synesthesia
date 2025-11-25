import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { LineChart, Line, ResponsiveContainer, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { Activity } from 'lucide-react';

interface BrainState {
  delta_power?: number;
  theta_power?: number;
  alpha_power?: number;
  beta_power?: number;
  gamma_power?: number;
  [key: string]: any;
}

interface EEGDisplayProps {
  data?: BrainState[];
}

export function EEGDisplay({ data = [] }: EEGDisplayProps) {
  // Use provided data or fallback to mock data if empty (but we should try to pass real data)
  const displayData = data.length > 0 ? data : Array.from({ length: 50 }, (_, i) => ({
    time: i,
    delta_power: 50 + Math.sin(i * 0.1) * 10,
    theta_power: 40 + Math.cos(i * 0.15) * 8,
    alpha_power: 60 + Math.sin(i * 0.2) * 15,
    beta_power: 30 + Math.cos(i * 0.25) * 5,
    gamma_power: 20 + Math.sin(i * 0.3) * 3,
  }));

  return (
    <Card className="h-full bg-card/30 backdrop-blur-md border-white/10">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-syn-green text-sm uppercase tracking-widest">
          <Activity className="w-4 h-4" />
          EEG Telemetry (Band Power)
        </CardTitle>
      </CardHeader>
      <CardContent className="h-[200px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={displayData}>
            <YAxis hide domain={['auto', 'auto']} />
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
            <Tooltip 
                contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.1)' }}
                itemStyle={{ fontSize: '10px' }}
            />
            <Line 
              type="monotone" 
              dataKey="delta_power" 
              stroke="#ff0000" 
              strokeWidth={2} 
              dot={false} 
              isAnimationActive={false}
              name="Delta"
            />
            <Line 
              type="monotone" 
              dataKey="theta_power" 
              stroke="#ffa500" 
              strokeWidth={2} 
              dot={false} 
              isAnimationActive={false}
              name="Theta"
            />
            <Line 
              type="monotone" 
              dataKey="alpha_power" 
              stroke="#00f3ff" 
              strokeWidth={2} 
              dot={false} 
              isAnimationActive={false}
              name="Alpha"
            />
             <Line 
              type="monotone" 
              dataKey="beta_power" 
              stroke="#0aff0a" 
              strokeWidth={2} 
              dot={false} 
              isAnimationActive={false}
              name="Beta"
            />
            <Line 
              type="monotone" 
              dataKey="gamma_power" 
              stroke="#bc13fe" 
              strokeWidth={2} 
              dot={false} 
              isAnimationActive={false}
              name="Gamma"
            />
          </LineChart>
        </ResponsiveContainer>
        <div className="flex justify-between text-[10px] font-mono text-muted-foreground mt-2 px-2 flex-wrap gap-2">
          <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-red-500"></div> DELTA</div>
          <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-orange-500"></div> THETA</div>
          <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-syn-cyan"></div> ALPHA</div>
          <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-syn-green"></div> BETA</div>
          <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-syn-purple"></div> GAMMA</div>
        </div>
      </CardContent>
    </Card>
  );
}