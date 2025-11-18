import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import { Zap, AlertTriangle, TrendingDown, Briefcase, Activity, Terminal } from 'lucide-react';
import { CMVLTriggerPanel } from './CMVLTriggerPanel';
import ChatWidget from '../components/ChatWidget';
import { ReasonGraphLive } from './ReasonGraphLive';
import GraphRiskPanel from '../components/GraphRiskPanel';
import { FinancialPlanComparison } from './FinancialPlanComparison';

export function LiveDemoView() {
  const [demoState, setDemoState] = useState<'idle' | 'triggered' | 'processing' | 'complete'>('idle');
  const [triggerType, setTriggerType] = useState<'market' | 'lifeevent' | 'composite' | null>(null);
  const [triggerDetails, setTriggerDetails] = useState<string>('');
  const [triggersQueue, setTriggersQueue] = useState<Array<{ type: 'market' | 'lifeevent'; label: string; details: string; severity: string }>>([]);
  const gapTimerRef = useRef<number | null>(null);
  const GAP_WINDOW_MS = 3000; // collect overlapping triggers within 3s

  const handleTrigger = (trigger: { type: 'market' | 'lifeevent'; label: string; details: string; severity: string }) => {
    // add to queue
    setTriggersQueue((q) => [...q, trigger]);

    // If no existing gap timer, start one to collect overlapping triggers
    if (!gapTimerRef.current) {
      // Show triggered state immediately
      setDemoState('triggered');

      // Start timer to aggregate triggers after gap window
      gapTimerRef.current = window.setTimeout(() => {
        aggregateTriggers();
      }, GAP_WINDOW_MS) as unknown as number;
    } else {
      // extend the gap window slightly to allow additional overlapping triggers
      window.clearTimeout(gapTimerRef.current);
      gapTimerRef.current = window.setTimeout(() => {
        aggregateTriggers();
      }, GAP_WINDOW_MS) as unknown as number;
    }
  };

  const aggregateTriggers = () => {
    // read and clear queue
    setTriggersQueue((queue) => {
      const q = [...queue];
      // determine aggregated severity
      const severityRank: Record<string, number> = { low: 0, medium: 1, high: 2, critical: 3 };
      let aggSeverity = 'low';
      for (const t of q) {
        if (severityRank[t.severity] > severityRank[aggSeverity]) aggSeverity = t.severity;
      }

      // build composite details
      const labels = q.map((t) => t.label).join(' + ');
      const details = q.map((t) => `${t.label}: ${t.details}`).join('\n');

      // set composite trigger
      setTriggerType(q.length === 1 ? q[0].type : 'composite');
      setTriggerDetails(details);

      // clear timer
      if (gapTimerRef.current) {
        window.clearTimeout(gapTimerRef.current);
        gapTimerRef.current = null;
      }

      // move through processing -> complete flow
      setTimeout(() => setDemoState('processing'), 500);
      setTimeout(() => setDemoState('complete'), 4000);

      return [];
    });
  };

  const reset = () => {
    setDemoState('idle');
    setTriggerType(null);
    setTriggerDetails('');
  };

  return (
    <div className="space-y-8">
      {/* Mission Control Panel */}
      <Card className="border-2 border-[var(--color-ink)] bg-white p-8 relative overflow-hidden blueprint-corners stagger-1">
        <div className="absolute top-0 right-0 w-64 h-64 opacity-5">
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
            
            <div className="terminal-text px-4 py-2">
              <div style={{ fontFamily: 'var(--font-code)', fontSize: '0.75rem' }}>
                STATE: <span className={
                  demoState === 'idle' ? 'text-[var(--color-cyan)]' :
                  demoState === 'triggered' ? 'text-[var(--color-amber)]' :
                  demoState === 'processing' ? 'text-[var(--color-amber)]' :
                  'text-[var(--color-emerald)]'
                }>
                  {demoState.toUpperCase()}
                </span>
              </div>
            </div>
          </div>

          {/* System Status Grid */}
          <div className="grid grid-cols-3 gap-4">
            <div className="border-2 border-[var(--color-ink)] p-4 angular-card bg-white">
              <Activity className="w-5 h-5 text-[var(--color-emerald)] mb-2" />
              <div className="data-label mb-1">SYSTEM STATUS</div>
              <div className="text-2xl" style={{ fontFamily: 'var(--font-display)' }}>ACTIVE</div>
            </div>
            
            <div className="border-2 border-[var(--color-ink)] p-4 angular-card bg-white">
              <Briefcase className="w-5 h-5 text-[var(--color-blueprint)] mb-2" />
              <div className="data-label mb-1">ACTIVE AGENTS</div>
              <div className="text-2xl" style={{ fontFamily: 'var(--font-display)' }}>5/5</div>
            </div>
            
            <div className="border-2 border-[var(--color-ink)] p-4 angular-card bg-white">
              <TrendingDown className="w-5 h-5 text-[var(--color-cyan)] mb-2" />
              <div className="data-label mb-1">MONITORING</div>
              <div className="text-2xl" style={{ fontFamily: 'var(--font-display)' }}>REAL-TIME</div>
            </div>
          </div>
        </div>
      </Card>

      {/* Trigger Panel */}
      <div className="stagger-2">
        <CMVLTriggerPanel 
          onTrigger={handleTrigger} 
          disabled={demoState !== 'idle'} 
        />
      </div>

      {/* Chat Widget (Interactive) */}
      <div className="stagger-2">
        <ChatWidget />
        <div className="mt-6">
          <GraphRiskPanel defaultUserId="demo_user" />
        </div>
      </div>

      {/* Animation and Process Visualization */}
      <AnimatePresence mode="wait">
        {demoState === 'triggered' && (
          <motion.div
            initial={{ opacity: 0, clipPath: 'polygon(0 0, 100% 0, 100% 0, 0 0)' }}
            animate={{ opacity: 1, clipPath: 'polygon(0 0, 100% 0, 100% 100%, 0 100%)' }}
            exit={{ opacity: 0, clipPath: 'polygon(0 100%, 100% 100%, 100% 100%, 0 100%)' }}
            transition={{ duration: 0.5 }}
          >
            <Card className="border-2 border-[var(--color-vermillion)] bg-gradient-to-br from-[var(--color-vermillion)] to-red-600 p-6 text-white">
              <div className="flex items-center gap-3">
                <AlertTriangle className="w-8 h-8" />
                <div>
                  <h3 className="text-2xl mb-1" style={{ fontFamily: 'var(--font-display)' }}>
                    ALERT: CMVL TRIGGERED
                  </h3>
                  <p style={{ fontFamily: 'var(--font-mono)', whiteSpace: 'pre-wrap' }}>
                    {triggerType === 'market' ? '⚠ MARKET_EVENT: ' : triggerType === 'lifeevent' ? '⚠ LIFE_EVENT: ' : '⚠ COMPOSITE_EVENT: '}
                    {triggerDetails}
                  </p>
                </div>
              </div>
            </Card>
          </motion.div>
        )}

        {(demoState === 'processing' || demoState === 'complete') && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="space-y-8"
          >
            <ReasonGraphLive 
              isProcessing={demoState === 'processing'} 
              triggerType={(triggerType === 'composite' ? 'market' : triggerType)!}
            />
            
            {demoState === 'complete' && (
              <motion.div
                initial={{ opacity: 0, y: 40 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3, duration: 0.6 }}
              >
                <FinancialPlanComparison triggerType={(triggerType === 'composite' ? 'market' : triggerType)!} />
                <div className="mt-8 flex justify-center">
                  <Button 
                    onClick={reset} 
                    className="bg-[var(--color-ink)] text-[var(--color-cyan)] hover:bg-[var(--color-blueprint)] border-2 border-[var(--color-ink)] px-8 py-6 text-lg angular-card"
                    style={{ fontFamily: 'var(--font-mono)' }}
                  >
                    <Terminal className="w-5 h-5 mr-2" />
                    RESET_DEMO
                  </Button>
                </div>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
