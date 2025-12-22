import { useEffect, useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { 
  Brain, 
  Target, 
  Wind, 
  Play, 
  CheckCircle2, 
  XCircle, 
  Loader2,
  RotateCcw,
  ArrowRight,
  Sparkles
} from 'lucide-react';
import type { 
  CalibrationStatus, 
  CalibrationProgress, 
  CalibrationStageInfo, 
  CalibrationResults 
} from '@/hooks/useWebSocket';

interface CalibrationFlowProps {
  calibrationStatus: CalibrationStatus;
  calibrationStage: CalibrationStageInfo | null;
  calibrationProgress: CalibrationProgress | null;
  calibrationResults: CalibrationResults | null;
  calibrationError: string | null;
  isCalibrated: boolean;
  onStartCalibration: () => void;
  onStartStage: (stage: 'baseline' | 'focus' | 'relax') => void;
  onStopStage: () => void;
  onTrain: () => void;
  onCancel: () => void;
  onComplete: () => void;
}

const STAGES = [
  { 
    id: 'baseline' as const, 
    name: 'BASELINE', 
    icon: Brain, 
    color: 'text-syn-cyan',
    bgColor: 'bg-syn-cyan/10',
    borderColor: 'border-syn-cyan/30',
    description: 'Establish your neutral brain state'
  },
  { 
    id: 'focus' as const, 
    name: 'FOCUS', 
    icon: Target, 
    color: 'text-syn-purple',
    bgColor: 'bg-syn-purple/10',
    borderColor: 'border-syn-purple/30',
    description: 'Count backwards from 100 by 7s'
  },
  { 
    id: 'relax' as const, 
    name: 'RELAX', 
    icon: Wind, 
    color: 'text-syn-green',
    bgColor: 'bg-syn-green/10',
    borderColor: 'border-syn-green/30',
    description: 'Focus on your breathing'
  },
];

export function CalibrationFlow({
  calibrationStatus,
  calibrationStage,
  calibrationProgress,
  calibrationResults,
  calibrationError,
  isCalibrated,
  onStartCalibration,
  onStartStage,
  onStopStage,
  onTrain,
  onCancel,
  onComplete,
}: CalibrationFlowProps) {
  const [currentStageIndex, setCurrentStageIndex] = useState(-1);
  const [completedStages, setCompletedStages] = useState<Set<string>>(new Set());
  const [stageTimer, setStageTimer] = useState<number>(0);

  // Timer for current stage
  useEffect(() => {
    if (calibrationStatus !== 'in_stage' || !calibrationStage) {
      return;
    }

    const interval = setInterval(() => {
      setStageTimer(prev => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, [calibrationStatus, calibrationStage]);

  // Reset timer when stage changes
  useEffect(() => {
    setStageTimer(0);
  }, [calibrationStage?.stage]);

  // Track completed stages
  useEffect(() => {
    if (calibrationStatus === 'stage_complete' && calibrationStage) {
      setCompletedStages(prev => new Set([...prev, calibrationStage.stage]));
    }
  }, [calibrationStatus, calibrationStage]);

  // Calculate progress percentage
  const progressPercent = calibrationStage 
    ? Math.min((stageTimer / calibrationStage.duration) * 100, 100)
    : 0;

  // Check if all stages are complete
  const allStagesComplete = STAGES.every(s => completedStages.has(s.id));

  // Handle stage completion (auto-advance or wait for user)
  const handleStageEnd = useCallback(() => {
    onStopStage();
    const nextIndex = currentStageIndex + 1;
    if (nextIndex < STAGES.length) {
      setCurrentStageIndex(nextIndex);
    }
  }, [onStopStage, currentStageIndex]);

  // Auto-stop stage when timer reaches duration
  useEffect(() => {
    if (calibrationStage && stageTimer >= calibrationStage.duration) {
      handleStageEnd();
    }
  }, [stageTimer, calibrationStage, handleStageEnd]);

  // Idle state - show start button
  if (calibrationStatus === 'idle') {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-6 p-8">
        <div className="text-center space-y-2">
          <Brain className="w-16 h-16 text-syn-cyan mx-auto mb-4" />
          <h2 className="text-2xl font-display font-bold text-white">NEURAL CALIBRATION</h2>
          <p className="text-sm text-muted-foreground max-w-md">
            Calibrate the system to your unique brain patterns for personalized state detection.
            This takes about 3 minutes.
          </p>
        </div>
        
        <div className="flex gap-4">
          <Button 
            variant="neon" 
            size="lg"
            onClick={onStartCalibration}
            className="gap-2"
          >
            <Sparkles className="w-4 h-4" />
            BEGIN CALIBRATION
          </Button>
          
          {isCalibrated && (
            <Button 
              variant="outline" 
              size="lg"
              onClick={onComplete}
              className="gap-2"
            >
              SKIP (ALREADY CALIBRATED)
            </Button>
          )}
        </div>
      </div>
    );
  }

  // Starting state
  if (calibrationStatus === 'starting') {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <Loader2 className="w-12 h-12 text-syn-cyan animate-spin" />
        <p className="text-muted-foreground">Initializing calibration...</p>
      </div>
    );
  }

  // Error state
  if (calibrationStatus === 'error') {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-6 p-8">
        <XCircle className="w-16 h-16 text-destructive" />
        <div className="text-center space-y-2">
          <h2 className="text-xl font-display font-bold text-white">CALIBRATION ERROR</h2>
          <p className="text-sm text-destructive">{calibrationError}</p>
        </div>
        <Button variant="outline" onClick={onCancel} className="gap-2">
          <RotateCcw className="w-4 h-4" />
          TRY AGAIN
        </Button>
      </div>
    );
  }

  // Training state
  if (calibrationStatus === 'training') {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <Loader2 className="w-12 h-12 text-syn-purple animate-spin" />
        <p className="text-muted-foreground">Training your personalized model...</p>
      </div>
    );
  }

  // Complete state
  if (calibrationStatus === 'complete' && calibrationResults) {
    const accuracy = (calibrationResults.validation_accuracy * 100).toFixed(1);
    
    return (
      <div className="flex flex-col items-center justify-center h-full gap-6 p-8">
        <CheckCircle2 className="w-16 h-16 text-syn-green" />
        <div className="text-center space-y-2">
          <h2 className="text-2xl font-display font-bold text-white">CALIBRATION COMPLETE</h2>
          <p className="text-lg text-syn-green font-mono">{accuracy}% ACCURACY</p>
        </div>
        
        <div className="grid grid-cols-3 gap-4 text-center text-sm">
          {Object.entries(calibrationResults.sample_counts).map(([state, count]) => (
            <div key={state} className="p-3 rounded-lg bg-card/50 border border-white/10">
              <div className="text-muted-foreground uppercase text-xs">{state}</div>
              <div className="text-white font-mono">{count} samples</div>
            </div>
          ))}
        </div>
        
        <Button 
          variant="neon" 
          size="lg"
          onClick={onComplete}
          className="gap-2 mt-4"
        >
          <Play className="w-4 h-4" />
          START SESSION
        </Button>
      </div>
    );
  }

  // Ready state - show stage selection
  if (calibrationStatus === 'ready' || calibrationStatus === 'stage_complete') {
    return (
      <div className="flex flex-col h-full p-4 md:p-8">
        <div className="text-center mb-6">
          <h2 className="text-xl font-display font-bold text-white mb-2">CALIBRATION STAGES</h2>
          <p className="text-sm text-muted-foreground">
            Complete each stage to train your personalized model
          </p>
        </div>
        
        {/* Stage buttons */}
        <div className="flex flex-col gap-4 flex-1">
          {STAGES.map((stage, index) => {
            const Icon = stage.icon;
            const isComplete = completedStages.has(stage.id);
            const isNext = !isComplete && index === 
              STAGES.findIndex(s => !completedStages.has(s.id));
            
            return (
              <div
                key={stage.id}
                className={`
                  relative p-4 rounded-xl border transition-all
                  ${isComplete 
                    ? 'bg-syn-green/10 border-syn-green/30' 
                    : isNext 
                      ? `${stage.bgColor} ${stage.borderColor}` 
                      : 'bg-card/30 border-white/10 opacity-50'
                  }
                `}
              >
                <div className="flex items-center gap-4">
                  <div className={`
                    p-3 rounded-lg 
                    ${isComplete ? 'bg-syn-green/20' : stage.bgColor}
                  `}>
                    {isComplete ? (
                      <CheckCircle2 className="w-6 h-6 text-syn-green" />
                    ) : (
                      <Icon className={`w-6 h-6 ${stage.color}`} />
                    )}
                  </div>
                  
                  <div className="flex-1">
                    <h3 className={`font-display font-bold ${isComplete ? 'text-syn-green' : 'text-white'}`}>
                      {stage.name}
                    </h3>
                    <p className="text-sm text-muted-foreground">{stage.description}</p>
                  </div>
                  
                  {isComplete ? (
                    <span className="text-sm text-syn-green font-mono">COMPLETE</span>
                  ) : isNext ? (
                    <Button
                      variant="neon"
                      size="sm"
                      onClick={() => {
                        setCurrentStageIndex(index);
                        onStartStage(stage.id);
                      }}
                      className="gap-2"
                    >
                      START
                      <ArrowRight className="w-4 h-4" />
                    </Button>
                  ) : (
                    <span className="text-sm text-muted-foreground">PENDING</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
        
        {/* Action buttons */}
        <div className="flex justify-between mt-6">
          <Button variant="ghost" onClick={onCancel}>
            CANCEL
          </Button>
          
          {allStagesComplete && (
            <Button 
              variant="neon"
              onClick={onTrain}
              className="gap-2"
            >
              <Sparkles className="w-4 h-4" />
              TRAIN MODEL
            </Button>
          )}
        </div>
      </div>
    );
  }

  // In-stage state - show active stage UI
  if (calibrationStatus === 'in_stage' && calibrationStage) {
    const currentStage = STAGES.find(s => s.id === calibrationStage.stage);
    const Icon = currentStage?.icon || Brain;
    
    const remainingSeconds = Math.max(0, calibrationStage.duration - stageTimer);
    const minutes = Math.floor(remainingSeconds / 60);
    const seconds = remainingSeconds % 60;
    
    return (
      <div className="flex flex-col items-center justify-center h-full gap-6 p-8">
        {/* Stage icon with pulse animation */}
        <div className={`
          relative p-6 rounded-full 
          ${currentStage?.bgColor || 'bg-syn-cyan/10'}
          animate-pulse
        `}>
          <Icon className={`w-12 h-12 ${currentStage?.color || 'text-syn-cyan'}`} />
          <div className="absolute inset-0 rounded-full border-2 border-current opacity-30 animate-ping" 
               style={{ animationDuration: '2s' }} />
        </div>
        
        {/* Stage name and timer */}
        <div className="text-center space-y-2">
          <h2 className={`text-3xl font-display font-bold ${currentStage?.color || 'text-white'}`}>
            {currentStage?.name || 'CALIBRATING'}
          </h2>
          <p className="text-4xl font-mono text-white">
            {minutes}:{seconds.toString().padStart(2, '0')}
          </p>
        </div>
        
        {/* Instructions */}
        <div className="text-center max-w-md">
          <p className="text-lg text-muted-foreground">
            {calibrationStage.instructions}
          </p>
        </div>
        
        {/* Progress bar */}
        <div className="w-full max-w-md">
          <div className="h-2 bg-white/10 rounded-full overflow-hidden">
            <div 
              className={`h-full transition-all duration-1000 rounded-full ${
                currentStage?.color.replace('text-', 'bg-') || 'bg-syn-cyan'
              }`}
              style={{ width: `${progressPercent}%` }}
            />
          </div>
          <div className="flex justify-between mt-2 text-xs text-muted-foreground">
            <span>0:00</span>
            <span>{Math.floor(calibrationStage.duration / 60)}:{(calibrationStage.duration % 60).toString().padStart(2, '0')}</span>
          </div>
        </div>
        
        {/* Sample count */}
        {calibrationProgress?.samples_collected && (
          <div className="text-sm text-muted-foreground font-mono">
            {calibrationProgress.samples_collected} samples collected
          </div>
        )}
        
        {/* Skip button */}
        <Button 
          variant="ghost" 
          size="sm"
          onClick={handleStageEnd}
          className="mt-4"
        >
          SKIP STAGE
        </Button>
      </div>
    );
  }

  // Fallback
  return null;
}

