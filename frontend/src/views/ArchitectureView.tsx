import { Card } from '../components/ui/card';
import { motion } from 'motion/react';
import { 
  Cpu, Database, GitBranch, Shield, CheckCircle2, 
  ArrowRight
} from 'lucide-react';

export function ArchitectureView() {
  const agents = [
    {
      id: 'oa',
      name: 'Orchestration Agent',
      code: 'OA',
      icon: Cpu,
      color: 'var(--color-blueprint)',
      description: 'Mission control for the entire VP-MAS. Manages workflow, coordinates agents, and monitors external triggers.',
      responsibilities: [
        'Receives high-level user goals',
        'Translates goals into executable sequences',
        'Monitors external triggers for CMVL',
        'Manages overall agent workflow'
      ]
    },
    {
      id: 'ira',
      name: 'Information Retrieval Agent',
      code: 'IRA',
      icon: Database,
      color: 'var(--color-cyan)',
      description: 'Specialist in accessing external data and tools. Integrates real-time market data APIs and RAG systems.',
      responsibilities: [
        'Retrieval-Augmented Generation (RAG)',
        'Real-time market data integration',
        'API orchestration and data fetching',
        'Monitors volatility indicators'
      ]
    },
    {
      id: 'pa',
      name: 'Planning Agent',
      code: 'PA',
      icon: GitBranch,
      color: 'var(--color-amber)',
      description: 'Core intelligence for complex sequence optimization using Guided Search Module (GSM) and Thought of Search (ToS).',
      responsibilities: [
        'Non-generative planning and sequencing',
        'Guided Search Module (GSM) execution',
        'Thought of Search (ToS) heuristics',
        'Path exploration and pruning'
      ]
    },
    {
      id: 'va',
      name: 'Verification Agent',
      code: 'VA',
      icon: Shield,
      color: 'var(--color-emerald)',
      description: 'Critical integrity layer. Enforces constraint satisfaction and acts as external feedback loop for self-correction.',
      responsibilities: [
        'Rigorous constraint checking',
        'Validates against financial rules',
        'External feedback for self-correction',
        'Approves or rejects plans'
      ]
    },
    {
      id: 'ea',
      name: 'Execution Agent',
      code: 'EA',
      icon: CheckCircle2,
      color: 'var(--color-vermillion)',
      description: 'Performs symbolic actions mandated by approved plans. Updates financial ledgers and commits validated steps.',
      responsibilities: [
        'Updates financial ledger',
        'Runs forecast models',
        'Commits validated plan steps',
        'Executes approved actions'
      ]
    }
  ];

  const dataFlow = [
    { from: 'OA', to: 'IRA', label: 'REQUEST_DATA' },
    { from: 'IRA', to: 'PA', label: 'CONTEXT_STREAM' },
    { from: 'PA', to: 'VA', label: 'PROPOSED_PLAN' },
    { from: 'VA', to: 'PA', label: 'REJECT_REPLAN' },
    { from: 'VA', to: 'EA', label: 'APPROVED_PLAN' },
    { from: 'EA', to: 'OA', label: 'EXEC_REPORT' },
  ];

  return (
    <div className="space-y-8">
      {/* Overview */}
      <Card className="border-2 border-[var(--color-ink)] bg-white p-8 blueprint-corners stagger-1">
        <div className="absolute top-4 left-4 w-12 h-12 border-2 border-[var(--color-blueprint)] opacity-20"></div>
        <div className="absolute bottom-4 right-4 w-12 h-12 border-2 border-[var(--color-blueprint)] opacity-20"></div>
        
        <div className="data-label mb-2">SYSTEM_ARCHITECTURE</div>
        <h2 className="mb-4" style={{ fontFamily: 'var(--font-display)' }}>
          VP-MAS Multi-Agent System
        </h2>
        <p className="text-[var(--color-blueprint)]" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9375rem' }}>
          A hierarchical Multi-Agent System with specialized roles, separated cognitive functions, 
          and rigorous data contracts ensuring verifiable execution.
        </p>
      </Card>

      {/* Agents Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {agents.map((agent, idx) => {
          const Icon = agent.icon;
          return (
            <motion.div
              key={agent.id}
              className={`stagger-${idx + 2}`}
            >
              <Card className="border-2 border-[var(--color-ink)] bg-white p-6 h-full angular-card hover:shadow-lg transition-shadow duration-300">
                <div className="space-y-4">
                  <div className="flex items-center gap-3 mb-4">
                    <div 
                      className="p-3 border-2"
                      style={{ 
                        borderColor: agent.color,
                        backgroundColor: 'white'
                      }}
                    >
                      <Icon className="w-6 h-6" style={{ color: agent.color }} />
                    </div>
                    <div>
                      <div className="data-label" style={{ color: agent.color }}>
                        {agent.code}
                      </div>
                      <h3 className="text-lg" style={{ fontFamily: 'var(--font-display)' }}>
                        {agent.name}
                      </h3>
                    </div>
                  </div>

                  <p className="text-sm text-[var(--color-blueprint)]" style={{ fontFamily: 'var(--font-mono)' }}>
                    {agent.description}
                  </p>

                  <div className="space-y-2">
                    <div className="data-label text-xs">KEY_RESPONSIBILITIES</div>
                    <ul className="space-y-1.5">
                      {agent.responsibilities.map((resp, i) => (
                        <li key={i} className="text-xs flex items-start gap-2" style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-blueprint)' }}>
                          <span style={{ color: agent.color }}>▸</span>
                          <span>{resp}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </Card>
            </motion.div>
          );
        })}
      </div>

      {/* Data Flow Visualization */}
      <Card className="border-2 border-[var(--color-ink)] bg-white p-8 blueprint-corners stagger-6">
        <div className="data-label mb-2">COMMUNICATION_PROTOCOL</div>
        <h3 className="mb-6" style={{ fontFamily: 'var(--font-display)' }}>
          Inter-Agent Data Flow
        </h3>

        <div className="space-y-4">
          {dataFlow.map((flow, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.15 }}
              className="flex items-center gap-4 p-4 bg-[var(--color-paper)] border-l-4 border-[var(--color-cyan)]"
            >
              <div 
                className="px-3 py-1 border-2 border-[var(--color-ink)] bg-white"
                style={{ fontFamily: 'var(--font-code)', fontSize: '0.75rem' }}
              >
                {flow.from}
              </div>
              <ArrowRight className="w-5 h-5" style={{ color: 'var(--color-blueprint)' }} />
              <div className="flex-1">
                <span className="text-sm" style={{ fontFamily: 'var(--font-code)', color: 'var(--color-blueprint)' }}>
                  {flow.label}
                </span>
              </div>
              <ArrowRight className="w-5 h-5" style={{ color: 'var(--color-blueprint)' }} />
              <div 
                className="px-3 py-1 border-2 border-[var(--color-ink)] bg-white"
                style={{ fontFamily: 'var(--font-code)', fontSize: '0.75rem' }}
              >
                {flow.to}
              </div>
            </motion.div>
          ))}
        </div>

        <div className="mt-8 p-6 bg-[var(--color-paper)] border-2 border-[var(--color-blueprint)]">
          <div className="text-sm" style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-ink)' }}>
            <strong>Structured Data Contracts:</strong> All communication uses rigorous Pydantic BaseModel definitions 
            in docstrings, ensuring accurate tool identification and parameter passing between LLM-powered agents.
          </div>
        </div>
      </Card>
    </div>
  );
}
