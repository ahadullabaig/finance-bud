import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { ThemeToggle } from './components/ThemeToggle';
import { ArchitectureView } from './views/ArchitectureView';
import { ReasonGraphView } from './views/ReasonGraphView';
import { DashboardView } from './views/DashboardView';
import { LiveDemoView } from './views/LiveDemoView';
import { Activity, GitBranch, LayoutDashboard, Play } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('demo');

  return (
    <div className="min-h-screen blueprint-grid">
      <div className="container mx-auto p-8 relative">
        {/* Header with Technical Design */}
        <div className="mb-12 relative stagger-1">
          <div className="absolute -left-4 top-0 bottom-0 w-1 bg-gradient-to-b from-[var(--color-cyan)] via-[var(--color-blueprint)] to-transparent"></div>

          <div className="space-y-3">
            <div className="flex items-baseline gap-4">
              <div className="data-label">SYSTEM_ID: VP-MAS-001</div>
              <div className="h-px flex-1 bg-[var(--color-grid)]"></div>
              <div className="data-label">STATUS: ACTIVE</div>
            </div>

            <div className="flex justify-end">
              <ThemeToggle />
            </div>

            <h1 className="text-[var(--color-ink)] leading-none">
              VP-MAS
            </h1>
            
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-[var(--color-emerald)] pulse-indicator"></div>
              <p className="text-lg text-[var(--color-blueprint)]" style={{ fontFamily: 'var(--font-mono)' }}>
                Verifiable Planning Multi-Agent System for Dynamic Financial Optimization
              </p>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-8">
          <TabsList className="bg-transparent border-2 border-[var(--color-ink)] p-1 gap-1 stagger-2">
            <TabsTrigger 
              value="demo" 
              className="data-[state=active]:bg-[var(--color-ink)] data-[state=active]:text-[var(--color-cyan)] font-mono border-0"
            >
              <Play className="w-4 h-4 mr-2" />
              LIVE_DEMO
            </TabsTrigger>
            <TabsTrigger 
              value="architecture"
              className="data-[state=active]:bg-[var(--color-ink)] data-[state=active]:text-[var(--color-cyan)] font-mono border-0"
            >
              <Activity className="w-4 h-4 mr-2" />
              ARCHITECTURE
            </TabsTrigger>
            <TabsTrigger 
              value="reasongraph"
              className="data-[state=active]:bg-[var(--color-ink)] data-[state=active]:text-[var(--color-cyan)] font-mono border-0"
            >
              <GitBranch className="w-4 h-4 mr-2" />
              REASONGRAPH
            </TabsTrigger>
            <TabsTrigger 
              value="dashboard"
              className="data-[state=active]:bg-[var(--color-ink)] data-[state=active]:text-[var(--color-cyan)] font-mono border-0"
            >
              <LayoutDashboard className="w-4 h-4 mr-2" />
              METRICS
            </TabsTrigger>
          </TabsList>

          <TabsContent value="demo" className="space-y-6">
            <LiveDemoView />
          </TabsContent>

          <TabsContent value="architecture" className="space-y-6">
            <ArchitectureView />
          </TabsContent>

          <TabsContent value="reasongraph" className="space-y-6">
            <ReasonGraphView />
          </TabsContent>

          <TabsContent value="dashboard" className="space-y-6">
            <DashboardView />
          </TabsContent>
        </Tabs>

        {/* Footer Technical Info */}
        <div className="mt-16 pt-8 border-t-2 border-[var(--color-grid)] flex justify-between items-center text-xs stagger-6" style={{ fontFamily: 'var(--font-code)' }}>
          <div className="text-[var(--color-blueprint)]">
            © 2025 VP-MAS | Multi-Agent Financial Planning System
          </div>
          <div className="flex gap-6 text-[var(--color-blueprint)]">
            <span>v1.0.0</span>
            <span>|</span>
            <span>AGENTS: 5/5 ACTIVE</span>
            <span>|</span>
            <span>UPTIME: 99.98%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
