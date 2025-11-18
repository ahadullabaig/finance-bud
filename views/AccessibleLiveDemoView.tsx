/**
 * Accessible Live Demo View
 * 
 * Enhanced demo view with accessibility features and responsive design
 * Task 15 - Person D
 * Requirements: 11.4, 12.4
 */

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import { 
  Zap, AlertTriangle, TrendingDown, Briefcase, Activity, 
  Terminal, Volume2, VolumeX, Pause, Play 
} from 'lucide-react';
import { ErrorBoundary } from '../components/ErrorBoundary';
import { LoadingState } from '../components/LoadingState';
import { ErrorMessage } from '../components/ErrorMessage';
import { CMVLTriggerPanel } from './CMVLTriggerPanel';
import { ReasonGraphLive } from './ReasonGraphLive';
import { FinancialPlanComparison } from './FinancialPlanComparison';

type DemoState = 'idle' | 'triggered' | 'processing' | 'complete' | 'error';

export function AccessibleLiveDemoView() {
  const [demoState, setDemoState] = useState<DemoState>('idle');
  const [triggerType, setTriggerType] = useState<'market' | 'lifeevent' | null>(null);
  const [triggerDetails, setTriggerDetails] = useState<string>('');
  const [audioEnabled, setAudioEnabled] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Announce state changes for screen readers
  const announceStateChange = (message: string) => {
    const announcement = document.createElement('div');
    announcement.setAttribute('role', 'status');
    announcement.setAttribute('aria-live', 'polite');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    document.body.appendChild(announcement);
    setTimeout(() => document.body.removeChild(announcement), 1000);
  };

  useEffect(() => {
    const stateMessages: Record<DemoState, string> = {
      idle: 'Demo ready. Select a trigger to begin.',
      triggered: 'Trigger activated. System responding.',
      processing: 'Processing response. Please wait.',
      complete: 'Demo complete. Results available.',
      error: 'An error occurred. Please try again.'
    };
    
    announceStateChange(stateMessages[demoState]);
  }, [demoState]);

  const handleTrigger = async (type: 'market' | 'lifeevent', details: string) => {
    try {
      setError(null);
      setIsLoading(true);
      setTriggerType(type);
      setTriggerDetails(details);
      setDemoState('triggered');
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500));
      
      setDemoState('processing');
      setIsLoading(false);
      
      // Simulate processing
      await new Promise(resolve => setTimeout(resolve, 4000));
      
      if (!isPaused) {
        setDemoState('complete');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setDemoState('error');
      setIsLoading(false);
    }
  };

  const reset = () => {
    setDemoState('idle');
    setTriggerType(null);
    setTriggerDetails('');
    setError(null);
    setIsPaused(false);
    announceStateChange('Demo reset. Ready to start again.');
  };

  const togglePause = () => {
    setIsPaused(!isPaused);
    announceStateChange(isPaused ? 'Demo resumed' : 'Demo paused');
  };

  const toggleAudio = () => {
    setAudioEnabled(!audioEnabled);
    announceStateChange(audioEnabled ? 'Audio disabled' : 'Audio enabled');
  };

  return (
    <ErrorBoundary>
      <div className="space-y-8" role="main" aria-label="Live Demo">
        {/* Accessibility Controls */}
        <div className="flex justify-end gap-2" role="toolbar" aria-label="Demo controls">
          <Button
            onClick={toggleAudio}
            variant="outline"
            className="border-2 border-[var(--color-ink)]"
            aria-label={audioEnabled ? 'Disable audio announcements' : 'Enable audio announcements'}
            aria-pressed={audioEnabled}
          >
            {audioEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
            <span className="sr-only">
              {audioEnabled ? 'Audio enabled' : 'Audio disabled'}
            </span>
          </Button>
          
          {demoState === 'processing' && (
            <Button
              onClick={togglePause}
              variant="outline"
              className="border-2 border-[var(--color-ink)]"
              aria-label={isPaused ? 'Resume demo' : 'Pause demo'}
              aria-pressed={isPaused}
            >
              {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
              <span className="sr-only">
                {isPaused ? 'Paused' : 'Playing'}
              </span>
            </Button>
          )}
        </div>

        {/* Mission Control Panel */}
        <Card 
          className="border-2 border-[var(--color-ink)] bg-white p-8 relative overflow-hidden blueprint-corners stagger-1"
          role="region"
          aria-label="Mission Control Panel"
        >
          <div className="absolute top-0 right-0 w-64 h-64 opacity-5" aria-hidden="true">
            <svg viewBox="0 0 200 200" className="w-full h-full">
              <circle cx="100" cy="100" r="80" fill="none" stroke="var(--color-blueprint)" strokeWidth="0.5" />
              <circle cx="100" cy="100" r="60" fill="none" stroke="var(--color-blueprint)" strokeWidth="0.5" />
              <circle cx="100" cy="100" r="40" fill="none" stroke="var(--color-blueprint)" strokeWidth="0.5" />
              <line x1="100" y1="20" x2="100" y2="180" stroke="var(--color-blueprint)" strokeWidth="0.5" />
              <line x1="20" y1="100" x2="180" y2="100" stroke="var(--color-blueprint)" strokeWidth="0.5" />
            </svg>
          </div>

          <div className="relative">
            <div className="flex items-start justify-between mb-6">
              <div className="space-y-2">
                <div className="data-label">CONTINUOUS MONITORING & VERIFICATION LOOP</div>
                <h2 className="text-[var(--color-ink)]">Mission Control</h2>
                <p className="text-[var(--color-blueprint)]" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>
                  Interactive demonstration of real-time constraint monitoring and dynamic replanning
                </p>
              </div>
              
              <div 
                className="terminal-text px-4 py-2"
                role="status"
                aria-live="polite"
                aria-atomic="true"
              >
                <div style={{ fontFamily: 'var(--font-code)', fontSize: '0.75rem' }}>
                  STATE: <span className={
                    demoState === 'idle' ? 'text-[var(--color-cyan)]' :
                    demoState === 'triggered' ? 'text-[var(--color-amber)]' :
                    demoState === 'processing' ? 'text-[var(--color-amber)]' :
                    demoState === 'error' ? 'text-[var(--color-vermillion)]' :
                    'text-[var(--color-emerald)]'
                  }>
                    {demoState.toUpperCase()}
                  </span>
                </div>
              </div>
            </div>

            {/* System Status Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4" role="list" aria-label="System status">
              <div className="border-2 border-[var(--color-ink)] p-4 angular-card bg-white" role="listitem">
                <Activity className="w-5 h-5 text-[var(--color-emerald)] mb-2" aria-hidden="true" />
                <div className="data-label mb-1">SYSTEM STATUS</div>
                <div className="text-2xl" style={{ fontFamily: 'var(--font-display)' }}>
                  ACTIVE
                </div>
                <span className="sr-only">System status: Active</span>
              </div>
              
              <div className="border-2 border-[var(--color-ink)] p-4 angular-card bg-white" role="listitem">
                <Briefcase className="w-5 h-5 text-[var(--color-blueprint)] mb-2" aria-hidden="true" />
                <div className="data-label mb-1">ACTIVE AGENTS</div>
                <div className="text-2xl" style={{ fontFamily: 'var(--font-display)' }}>
                  5/5
                </div>
                <span className="sr-only">Active agents: 5 out of 5</span>
              </div>
              
              <div className="border-2 border-[var(--color-ink)] p-4 angular-card bg-white" role="listitem">
                <TrendingDown className="w-5 h-5 text-[var(--color-cyan)] mb-2" aria-hidden="true" />
                <div className="data-label mb-1">MONITORING</div>
                <div className="text-2xl" style={{ fontFamily: 'var(--font-display)' }}>
                  REAL-TIME
                </div>
                <span className="sr-only">Monitoring: Real-time</span>
              </div>
            </div>
          </div>
        </Card>

        {/* Loading State */}
        {isLoading && (
          <LoadingState 
            message="Initializing trigger response..." 
            variant="default"
          />
        )}

        {/* Error State */}
        {error && (
          <ErrorMessage
            title="Demo Error"
            message={error}
            severity="error"
            onRetry={reset}
            onDismiss={() => setError(null)}
          />
        )}

        {/* Trigger Panel */}
        {demoState === 'idle' && !isLoading && (
          <div className="stagger-2">
            <CMVLTriggerPanel 
              onTrigger={handleTrigger} 
              disabled={demoState !== 'idle'} 
            />
          </div>
        )}

        {/* Animation and Process Visualization */}
        <AnimatePresence mode="wait">
          {demoState === 'triggered' && (
            <motion.div
              initial={{ opacity: 0, clipPath: 'polygon(0 0, 100% 0, 100% 0, 0 0)' }}
              animate={{ opacity: 1, clipPath: 'polygon(0 0, 100% 0, 100% 100%, 0 100%)' }}
              exit={{ opacity: 0, clipPath: 'polygon(0 100%, 100% 100%, 100% 100%, 0 100%)' }}
              transition={{ duration: 0.5 }}
              role="alert"
              aria-live="assertive"
            >
              <Card className="border-2 border-[var(--color-vermillion)] bg-gradient-to-br from-[var(--color-vermillion)] to-red-600 p-6 text-white">
                <div className="flex items-center gap-3">
                  <AlertTriangle className="w-8 h-8" aria-hidden="true" />
                  <div>
                    <h3 className="text-2xl mb-1" style={{ fontFamily: 'var(--font-display)' }}>
                      ALERT: CMVL TRIGGERED
                    </h3>
                    <p style={{ fontFamily: 'var(--font-mono)' }}>
                      {triggerType === 'market' ? '⚠ MARKET_EVENT: ' : '⚠ LIFE_EVENT: '}
                      {triggerDetails}
                    </p>
                  </div>
                </div>
              </Card>
            </motion.div>
          )}

          {(demoState === 'processing' || demoState === 'complete') && !isPaused && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
              className="space-y-8"
              role="region"
              aria-label="Demo results"
            >
              <ReasonGraphLive 
                isProcessing={demoState === 'processing'} 
                triggerType={triggerType!}
              />
              
              {demoState === 'complete' && (
                <motion.div
                  initial={{ opacity: 0, y: 40 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3, duration: 0.6 }}
                >
                  <FinancialPlanComparison triggerType={triggerType!} />
                  <div className="mt-8 flex justify-center">
                    <Button 
                      onClick={reset} 
                      className="bg-[var(--color-ink)] text-[var(--color-cyan)] hover:bg-[var(--color-blueprint)] border-2 border-[var(--color-ink)] px-8 py-6 text-lg angular-card"
                      style={{ fontFamily: 'var(--font-mono)' }}
                      aria-label="Reset demo and start over"
                    >
                      <Terminal className="w-5 h-5 mr-2" aria-hidden="true" />
                      RESET_DEMO
                    </Button>
                  </div>
                </motion.div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Skip to results link for keyboard users */}
        {demoState === 'complete' && (
          <a 
            href="#demo-results" 
            className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:p-4 focus:bg-white focus:border-2 focus:border-[var(--color-ink)]"
          >
            Skip to demo results
          </a>
        )}
      </div>
    </ErrorBoundary>
  );
}
