import { Card } from '../components/ui/card';
import { GitBranch, Search, CheckCircle2, XCircle, AlertTriangle } from 'lucide-react';
import { motion } from 'motion/react';

export function ReasonGraphView() {
  const searchPaths = [
    {
      id: 'path-a',
      name: 'Path A: Conservative Rebalancing',
      status: 'rejected',
      reason: 'Insufficient growth potential for retirement goals',
      steps: [
        { label: 'Reduce stock allocation to 30%', verified: true },
        { label: 'Increase bond allocation to 60%', verified: true },
        { label: 'Project 10-year returns', verified: true },
        { label: 'Verify retirement goal feasibility', verified: false, constraint: 'Growth rate below required 5.2%' },
      ]
    },
    {
      id: 'path-b',
      name: 'Path B: Moderate Adjustment',
      status: 'approved',
      reason: 'Optimal balance of risk mitigation and growth',
      steps: [
        { label: 'Reduce stock allocation to 50%', verified: true },
        { label: 'Increase bond allocation to 40%', verified: true },
        { label: 'Increase emergency fund by 40%', verified: true },
        { label: 'Verify constraint satisfaction', verified: true },
        { label: 'Calculate risk-adjusted returns', verified: true },
        { label: 'Validate against user risk tolerance', verified: true },
      ]
    },
    {
      id: 'path-c',
      name: 'Path C: Minimal Change',
      status: 'rejected',
      reason: 'Inadequate risk mitigation for current volatility',
      steps: [
        { label: 'Reduce stock allocation to 65%', verified: true },
        { label: 'Increase bond allocation to 30%', verified: true },
        { label: 'Assess volatility exposure', verified: true },
        { label: 'Verify risk parameters', verified: false, constraint: 'VaR exceeds acceptable threshold' },
      ]
    },
  ];

  const heuristics = [
    { name: 'Information Gain', value: 0.87, description: 'Prioritizes paths with highest constraint satisfaction probability' },
    { name: 'State Similarity', value: 0.72, description: 'Evaluates proximity to historical successful plans' },
    { name: 'Constraint Complexity', value: 0.65, description: 'Accounts for number of hard constraints per path' },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <Card className="border-2 border-[var(--color-ink)] bg-white p-8 blueprint-corners stagger-1">
        <div className="data-label mb-2">THOUGHT_OF_SEARCH</div>
        <h2 className="mb-3" style={{ fontFamily: 'var(--font-display)' }}>
          ReasonGraph Visualization
        </h2>
        <p className="text-[var(--color-blueprint)]" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9375rem' }}>
          Advanced visualization of the Planning Agent's heuristic-driven search process, 
          showing explored paths, verification checkpoints, and constraint satisfaction.
        </p>
      </Card>

      {/* Heuristics Panel */}
      <Card className="border-2 border-[var(--color-ink)] bg-white p-6 blueprint-corners stagger-2">
        <div className="flex items-center gap-2 mb-6">
          <Search className="w-5 h-5" style={{ color: 'var(--color-amber)' }} />
          <div className="data-label">ACTIVE_HEURISTICS</div>
        </div>
        
        <div className="grid md:grid-cols-3 gap-6">
          {heuristics.map((heuristic, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="border-2 border-[var(--color-ink)] p-5 angular-card bg-white"
            >
              <div className="data-label mb-3" style={{ fontSize: '0.7rem' }}>
                {heuristic.name.toUpperCase().replace(' ', '_')}
              </div>
              
              <div className="flex items-end gap-2 mb-4">
                <span className="text-4xl" style={{ fontFamily: 'var(--font-display)', color: 'var(--color-cyan)' }}>
                  {heuristic.value}
                </span>
                <span className="text-xs mb-2" style={{ fontFamily: 'var(--font-code)', color: 'var(--color-blueprint)' }}>
                  weight
                </span>
              </div>
              
              <div className="h-2 bg-[var(--color-grid)] relative overflow-hidden">
                <motion.div 
                  className="absolute inset-y-0 left-0 bg-gradient-to-r from-[var(--color-cyan)] to-[var(--color-blueprint)]"
                  initial={{ width: 0 }}
                  animate={{ width: `${heuristic.value * 100}%` }}
                  transition={{ duration: 1, delay: idx * 0.1 }}
                />
              </div>
              
              <p className="text-xs mt-3" style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-blueprint)' }}>
                {heuristic.description}
              </p>
            </motion.div>
          ))}
        </div>
      </Card>

      {/* Search Paths */}
      <div className="space-y-6">
        {searchPaths.map((path, pathIdx) => {
          const isApproved = path.status === 'approved';
          const isRejected = path.status === 'rejected';

          return (
            <motion.div
              key={path.id}
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: pathIdx * 0.2 }}
              className={`stagger-${pathIdx + 3}`}
            >
              <Card className={`
                border-2 p-6 angular-card
                ${isApproved ? 'bg-gradient-to-br from-[var(--color-emerald)] to-green-600 text-white border-[var(--color-emerald)]' : ''}
                ${isRejected ? 'bg-white border-[var(--color-vermillion)]' : ''}
              `}>
                <div className="flex items-start justify-between mb-6">
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="text-xl" style={{ fontFamily: 'var(--font-display)' }}>
                        {path.name}
                      </h3>
                      <div 
                        className={`
                          px-3 py-1 border-2 text-xs
                          ${isApproved ? 'border-white text-white' : 'border-[var(--color-vermillion)] text-[var(--color-vermillion)]'}
                        `}
                        style={{ fontFamily: 'var(--font-code)' }}
                      >
                        {isApproved && <CheckCircle2 className="w-3 h-3 inline mr-1" />}
                        {isRejected && <XCircle className="w-3 h-3 inline mr-1" />}
                        {path.status.toUpperCase()}
                      </div>
                    </div>
                    <p 
                      className={`text-sm ${isApproved ? 'text-white/90' : 'text-[var(--color-blueprint)]'}`}
                      style={{ fontFamily: 'var(--font-mono)' }}
                    >
                      {path.reason}
                    </p>
                  </div>
                </div>

                <div className="space-y-3">
                  {path.steps.map((step, stepIdx) => (
                    <motion.div
                      key={stepIdx}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: pathIdx * 0.2 + stepIdx * 0.05 }}
                      className={`
                        flex items-start gap-3 p-4 border-2
                        ${step.verified 
                          ? isApproved 
                            ? 'bg-white/10 border-white/30' 
                            : 'bg-[var(--color-paper)] border-[var(--color-grid)]' 
                          : 'bg-[var(--color-vermillion)]/10 border-[var(--color-vermillion)]'
                        }
                      `}
                    >
                      <div className="mt-0.5">
                        {step.verified ? (
                          <CheckCircle2 className={`w-4 h-4 ${isApproved ? 'text-white' : 'text-[var(--color-emerald)]'}`} />
                        ) : (
                          <AlertTriangle className="w-4 h-4 text-[var(--color-vermillion)]" />
                        )}
                      </div>
                      <div className="flex-1">
                        <div 
                          className={`text-sm ${isApproved ? 'text-white' : 'text-[var(--color-ink)]'}`}
                          style={{ fontFamily: 'var(--font-mono)' }}
                        >
                          {step.label}
                        </div>
                        {step.constraint && (
                          <div className="text-xs mt-1" style={{ fontFamily: 'var(--font-code)', color: 'var(--color-vermillion)' }}>
                            ⚠ CONSTRAINT_VIOLATION: {step.constraint}
                          </div>
                        )}
                      </div>
                      <div 
                        className={`px-2 py-1 border text-xs ${isApproved ? 'border-white/50 text-white' : 'border-[var(--color-ink)]'}`}
                        style={{ fontFamily: 'var(--font-code)' }}
                      >
                        {step.verified ? 'VA_PASS' : 'VA_FAIL'}
                      </div>
                    </motion.div>
                  ))}
                </div>
              </Card>
            </motion.div>
          );
        })}
      </div>

      {/* Insights */}
      <Card className="border-2 border-[var(--color-ink)] bg-white p-8 blueprint-corners">
        <div className="data-label mb-4">SEARCH_INSIGHTS</div>
        <div className="grid md:grid-cols-4 gap-6">
          <div className="space-y-2">
            <div className="data-label text-xs" style={{ color: 'var(--color-blueprint)' }}>PATHS_EXPLORED</div>
            <div className="text-5xl" style={{ fontFamily: 'var(--font-display)', color: 'var(--color-ink)' }}>3</div>
            <div className="text-xs" style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-blueprint)' }}>
              Combinatorial complexity managed through heuristic pruning
            </div>
          </div>
          <div className="space-y-2">
            <div className="data-label text-xs" style={{ color: 'var(--color-blueprint)' }}>VERIFICATION_POINTS</div>
            <div className="text-5xl" style={{ fontFamily: 'var(--font-display)', color: 'var(--color-ink)' }}>14</div>
            <div className="text-xs" style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-blueprint)' }}>
              Every critical step validated by Verification Agent
            </div>
          </div>
          <div className="space-y-2">
            <div className="data-label text-xs" style={{ color: 'var(--color-blueprint)' }}>CONSTRAINTS_CAUGHT</div>
            <div className="text-5xl" style={{ fontFamily: 'var(--font-display)', color: 'var(--color-vermillion)' }}>2</div>
            <div className="text-xs" style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-blueprint)' }}>
              Prevented compounding errors before execution
            </div>
          </div>
          <div className="space-y-2">
            <div className="data-label text-xs" style={{ color: 'var(--color-blueprint)' }}>OPTIMAL_PATH</div>
            <div className="text-5xl" style={{ fontFamily: 'var(--font-display)', color: 'var(--color-emerald)' }}>B</div>
            <div className="text-xs" style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-blueprint)' }}>
              Approved by VA after rigorous constraint checking
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}
