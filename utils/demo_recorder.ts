/**
 * Demo Scenario Recording and Replay System
 * 
 * Records and replays demonstration scenarios for testing and presentation
 * Task 15 - Person D
 * Requirements: 11.4, 12.4
 */

export interface DemoAction {
  timestamp: number;
  type: 'trigger' | 'user_input' | 'system_response' | 'state_change';
  data: any;
  description: string;
}

export interface DemoScenario {
  id: string;
  name: string;
  description: string;
  duration: number;
  actions: DemoAction[];
  initialState: any;
  expectedOutcome: any;
  tags: string[];
  createdAt: string;
}

export class DemoRecorder {
  private recording: boolean = false;
  private currentScenario: DemoScenario | null = null;
  private startTime: number = 0;
  private actions: DemoAction[] = [];

  startRecording(scenarioName: string, description: string, initialState: any) {
    this.recording = true;
    this.startTime = Date.now();
    this.actions = [];
    
    this.currentScenario = {
      id: `demo_${Date.now()}`,
      name: scenarioName,
      description,
      duration: 0,
      actions: [],
      initialState,
      expectedOutcome: null,
      tags: [],
      createdAt: new Date().toISOString()
    };

    console.log(`[DemoRecorder] Started recording: ${scenarioName}`);
  }

  recordAction(type: DemoAction['type'], data: any, description: string) {
    if (!this.recording || !this.currentScenario) {
      return;
    }

    const action: DemoAction = {
      timestamp: Date.now() - this.startTime,
      type,
      data,
      description
    };

    this.actions.push(action);
    console.log(`[DemoRecorder] Recorded action:`, action);
  }

  stopRecording(expectedOutcome: any, tags: string[] = []): DemoScenario | null {
    if (!this.recording || !this.currentScenario) {
      return null;
    }

    this.recording = false;
    this.currentScenario.duration = Date.now() - this.startTime;
    this.currentScenario.actions = this.actions;
    this.currentScenario.expectedOutcome = expectedOutcome;
    this.currentScenario.tags = tags;

    const scenario = this.currentScenario;
    this.currentScenario = null;
    this.actions = [];

    console.log(`[DemoRecorder] Stopped recording. Duration: ${scenario.duration}ms`);
    return scenario;
  }

  isRecording(): boolean {
    return this.recording;
  }

  saveScenario(scenario: DemoScenario) {
    const scenarios = this.loadScenarios();
    scenarios.push(scenario);
    localStorage.setItem('demo_scenarios', JSON.stringify(scenarios));
    console.log(`[DemoRecorder] Saved scenario: ${scenario.name}`);
  }

  loadScenarios(): DemoScenario[] {
    const stored = localStorage.getItem('demo_scenarios');
    return stored ? JSON.parse(stored) : [];
  }

  getScenario(id: string): DemoScenario | null {
    const scenarios = this.loadScenarios();
    return scenarios.find(s => s.id === id) || null;
  }

  deleteScenario(id: string) {
    const scenarios = this.loadScenarios();
    const filtered = scenarios.filter(s => s.id !== id);
    localStorage.setItem('demo_scenarios', JSON.stringify(filtered));
    console.log(`[DemoRecorder] Deleted scenario: ${id}`);
  }
}

export class DemoPlayer {
  private playing: boolean = false;
  private currentScenario: DemoScenario | null = null;
  private currentActionIndex: number = 0;
  private startTime: number = 0;
  private playbackSpeed: number = 1.0;
  private onActionCallback: ((action: DemoAction) => void) | null = null;
  private onCompleteCallback: (() => void) | null = null;

  async playScenario(
    scenario: DemoScenario,
    onAction: (action: DemoAction) => void,
    onComplete: () => void,
    speed: number = 1.0
  ) {
    this.playing = true;
    this.currentScenario = scenario;
    this.currentActionIndex = 0;
    this.startTime = Date.now();
    this.playbackSpeed = speed;
    this.onActionCallback = onAction;
    this.onCompleteCallback = onComplete;

    console.log(`[DemoPlayer] Starting playback: ${scenario.name} at ${speed}x speed`);

    // Initialize with initial state
    if (onAction) {
      onAction({
        timestamp: 0,
        type: 'state_change',
        data: scenario.initialState,
        description: 'Initial state'
      });
    }

    // Play through actions
    for (let i = 0; i < scenario.actions.length; i++) {
      if (!this.playing) {
        console.log('[DemoPlayer] Playback stopped');
        break;
      }

      const action = scenario.actions[i];
      const nextAction = scenario.actions[i + 1];
      
      // Wait for the appropriate time
      if (nextAction) {
        const delay = (nextAction.timestamp - action.timestamp) / this.playbackSpeed;
        await this.sleep(delay);
      }

      if (this.playing && onAction) {
        onAction(action);
        this.currentActionIndex = i + 1;
      }
    }

    if (this.playing && onComplete) {
      onComplete();
    }

    this.playing = false;
    console.log('[DemoPlayer] Playback complete');
  }

  pause() {
    this.playing = false;
    console.log('[DemoPlayer] Playback paused');
  }

  resume() {
    if (this.currentScenario && this.currentActionIndex < this.currentScenario.actions.length) {
      this.playing = true;
      console.log('[DemoPlayer] Playback resumed');
      // Continue from current action
      this.continuePlayback();
    }
  }

  stop() {
    this.playing = false;
    this.currentScenario = null;
    this.currentActionIndex = 0;
    console.log('[DemoPlayer] Playback stopped');
  }

  setSpeed(speed: number) {
    this.playbackSpeed = speed;
    console.log(`[DemoPlayer] Playback speed set to ${speed}x`);
  }

  isPlaying(): boolean {
    return this.playing;
  }

  getProgress(): number {
    if (!this.currentScenario) return 0;
    return this.currentActionIndex / this.currentScenario.actions.length;
  }

  private async continuePlayback() {
    if (!this.currentScenario || !this.onActionCallback) return;

    for (let i = this.currentActionIndex; i < this.currentScenario.actions.length; i++) {
      if (!this.playing) break;

      const action = this.currentScenario.actions[i];
      const nextAction = this.currentScenario.actions[i + 1];
      
      if (nextAction) {
        const delay = (nextAction.timestamp - action.timestamp) / this.playbackSpeed;
        await this.sleep(delay);
      }

      if (this.playing && this.onActionCallback) {
        this.onActionCallback(action);
        this.currentActionIndex = i + 1;
      }
    }

    if (this.playing && this.onCompleteCallback) {
      this.onCompleteCallback();
    }

    this.playing = false;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Predefined demo scenarios
export const PredefinedScenarios: DemoScenario[] = [
  {
    id: 'market_crash_demo',
    name: 'Market Crash Response',
    description: 'Demonstrates CMVL response to sudden market crash',
    duration: 15000,
    actions: [
      {
        timestamp: 0,
        type: 'trigger',
        data: { type: 'market_crash', severity: 'critical', drop: -0.15 },
        description: 'Market crash detected: -15% drop'
      },
      {
        timestamp: 500,
        type: 'system_response',
        data: { cmvl_activated: true, priority: 'critical' },
        description: 'CMVL activated with critical priority'
      },
      {
        timestamp: 2000,
        type: 'system_response',
        data: { verification_started: true },
        description: 'Verification Agent analyzing current plan'
      },
      {
        timestamp: 4000,
        type: 'system_response',
        data: { replanning_initiated: true },
        description: 'Planning Agent generating alternative strategies'
      },
      {
        timestamp: 8000,
        type: 'system_response',
        data: { new_plan_generated: true, paths_explored: 5 },
        description: 'New plan generated with 5 alternative paths'
      },
      {
        timestamp: 10000,
        type: 'system_response',
        data: { verification_complete: true, status: 'approved' },
        description: 'New plan verified and approved'
      },
      {
        timestamp: 12000,
        type: 'state_change',
        data: { plan_updated: true, risk_reduced: true },
        description: 'Portfolio rebalanced to reduce risk'
      }
    ],
    initialState: {
      portfolio: { stocks: 0.7, bonds: 0.3 },
      market_volatility: 0.15
    },
    expectedOutcome: {
      portfolio: { stocks: 0.5, bonds: 0.5 },
      market_volatility: 0.45,
      plan_status: 'updated'
    },
    tags: ['market_event', 'cmvl', 'critical'],
    createdAt: '2025-01-01T00:00:00Z'
  },
  {
    id: 'job_loss_demo',
    name: 'Job Loss Adaptation',
    description: 'Demonstrates plan adaptation to job loss event',
    duration: 12000,
    actions: [
      {
        timestamp: 0,
        type: 'user_input',
        data: { event: 'job_loss', income_change: -1.0 },
        description: 'User reports job loss'
      },
      {
        timestamp: 500,
        type: 'trigger',
        data: { type: 'life_event', severity: 'high' },
        description: 'Life event trigger activated'
      },
      {
        timestamp: 2000,
        type: 'system_response',
        data: { constraint_update: true },
        description: 'Updating financial constraints'
      },
      {
        timestamp: 4000,
        type: 'system_response',
        data: { replanning_started: true },
        description: 'Generating emergency financial plan'
      },
      {
        timestamp: 7000,
        type: 'system_response',
        data: { new_plan_ready: true },
        description: 'Emergency plan generated'
      },
      {
        timestamp: 9000,
        type: 'state_change',
        data: { emergency_mode: true, spending_reduced: true },
        description: 'Emergency mode activated'
      }
    ],
    initialState: {
      monthly_income: 8000,
      emergency_fund: 30000,
      monthly_expenses: 6000
    },
    expectedOutcome: {
      monthly_income: 0,
      emergency_fund: 30000,
      monthly_expenses: 4000,
      plan_status: 'emergency_mode'
    },
    tags: ['life_event', 'emergency', 'high_priority'],
    createdAt: '2025-01-01T00:00:00Z'
  }
];
