import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { TrendingDown, UserX, Heart, DollarSign, AlertCircle } from 'lucide-react';

interface CMVLTriggerPanelProps {
  onTrigger: (trigger: { type: 'market' | 'lifeevent'; label: string; details: string; severity: string }) => void;
  disabled: boolean;
}

export function CMVLTriggerPanel({ onTrigger, disabled }: CMVLTriggerPanelProps) {
  const marketTriggers = [
    { 
      label: 'Market Crash (-15%)', 
      icon: TrendingDown, 
      details: 'S&P 500 drops 15% in 48 hours',
      severity: 'critical'
    },
    { 
      label: 'Volatility Spike', 
      icon: AlertCircle, 
      details: 'VIX index jumps above 40',
      severity: 'high'
    },
  ];

  const lifeTriggers = [
    { 
      label: 'Job Loss', 
      icon: UserX, 
      details: 'Unexpected termination, 6-month severance',
      severity: 'critical'
    },
    { 
      label: 'Medical Emergency', 
      icon: Heart, 
      details: 'Unplanned medical expense: $45,000',
      severity: 'critical'
    },
    { 
      label: 'Business Disruption', 
      icon: DollarSign, 
      details: 'Revenue down 60% due to supply chain',
      severity: 'high'
    },
  ];

  return (
    <div className="grid md:grid-cols-2 gap-6">
      {/* Market Triggers */}
      <Card className="border-2 border-[var(--color-ink)] bg-white p-6 blueprint-corners">
        <div className="space-y-4">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-1 h-8 bg-[var(--color-vermillion)]"></div>
            <div>
              <div className="data-label">CATEGORY_01</div>
              <h3 style={{ fontFamily: 'var(--font-display)' }}>Market Event Triggers</h3>
            </div>
          </div>
          
          <p className="text-sm text-[var(--color-blueprint)]" style={{ fontFamily: 'var(--font-mono)' }}>
            Simulate external market volatility events that activate the CMVL protocol
          </p>
          
          <div className="space-y-3 mt-6">
            {marketTriggers.map((trigger, idx) => (
              <Button
                key={idx}
                onClick={() => onTrigger({ type: 'market', label: trigger.label, details: trigger.details, severity: trigger.severity })}
                disabled={disabled}
                className={`
                  w-full justify-start border-2 border-[var(--color-ink)] 
                  bg-white hover:bg-[var(--color-vermillion)] hover:text-white
                  text-[var(--color-ink)] angular-card h-auto py-4
                  transition-all duration-300
                  ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
                `}
                style={{ fontFamily: 'var(--font-mono)' }}
              >
                <trigger.icon className="w-5 h-5 mr-3 flex-shrink-0" />
                <div className="flex flex-col items-start text-left">
                  <span className="font-semibold">{trigger.label}</span>
                  <span className="text-xs opacity-70">{trigger.details}</span>
                </div>
              </Button>
            ))}
          </div>
        </div>
      </Card>

      {/* Life Event Triggers */}
      <Card className="border-2 border-[var(--color-ink)] bg-white p-6 blueprint-corners">
        <div className="space-y-4">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-1 h-8 bg-[var(--color-amber)]"></div>
            <div>
              <div className="data-label">CATEGORY_02</div>
              <h3 style={{ fontFamily: 'var(--font-display)' }}>Life Event Triggers</h3>
            </div>
          </div>
          
          <p className="text-sm text-[var(--color-blueprint)]" style={{ fontFamily: 'var(--font-mono)' }}>
            Simulate personal life events requiring immediate plan recalibration
          </p>
          
          <div className="space-y-3 mt-6">
            {lifeTriggers.map((trigger, idx) => (
              <Button
                key={idx}
                onClick={() => onTrigger({ type: 'lifeevent', label: trigger.label, details: trigger.details, severity: trigger.severity })}
                disabled={disabled}
                className={`
                  w-full justify-start border-2 border-[var(--color-ink)] 
                  bg-white hover:bg-[var(--color-amber)] hover:text-[var(--color-ink)]
                  text-[var(--color-ink)] angular-card h-auto py-4
                  transition-all duration-300
                  ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
                `}
                style={{ fontFamily: 'var(--font-mono)' }}
              >
                <trigger.icon className="w-5 h-5 mr-3 flex-shrink-0" />
                <div className="flex flex-col items-start text-left">
                  <span className="font-semibold">{trigger.label}</span>
                  <span className="text-xs opacity-70">{trigger.details}</span>
                </div>
              </Button>
            ))}
          </div>
        </div>
      </Card>
    </div>
  );
}
