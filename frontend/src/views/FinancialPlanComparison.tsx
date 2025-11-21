import { Card } from '../components/ui/card';
import { ArrowRight, TrendingUp, TrendingDown } from 'lucide-react';
import { motion } from 'motion/react';

interface FinancialPlanComparisonProps {
  triggerType: 'market' | 'lifeevent';
}

export function FinancialPlanComparison({ triggerType }: FinancialPlanComparisonProps) {
  const beforePlan = {
    emergencyFund: 25000,
    stockAllocation: 70,
    bondAllocation: 25,
    cashAllocation: 5,
    monthlyInvestment: 3500,
    riskTolerance: 'Aggressive',
  };

  const afterPlan = triggerType === 'market' ? {
    emergencyFund: 35000,
    stockAllocation: 50,
    bondAllocation: 40,
    cashAllocation: 10,
    monthlyInvestment: 3000,
    riskTolerance: 'Moderate',
  } : {
    emergencyFund: 45000,
    stockAllocation: 40,
    bondAllocation: 45,
    cashAllocation: 15,
    monthlyInvestment: 2000,
    riskTolerance: 'Conservative',
  };

  const metrics = [
    { 
      label: 'EMERGENCY_FUND', 
      before: `$${beforePlan.emergencyFund.toLocaleString()}`, 
      after: `$${afterPlan.emergencyFund.toLocaleString()}`,
      change: afterPlan.emergencyFund - beforePlan.emergencyFund
    },
    { 
      label: 'STOCK_ALLOC', 
      before: `${beforePlan.stockAllocation}%`, 
      after: `${afterPlan.stockAllocation}%`,
      change: afterPlan.stockAllocation - beforePlan.stockAllocation
    },
    { 
      label: 'BOND_ALLOC', 
      before: `${beforePlan.bondAllocation}%`, 
      after: `${afterPlan.bondAllocation}%`,
      change: afterPlan.bondAllocation - beforePlan.bondAllocation
    },
    { 
      label: 'CASH_ALLOC', 
      before: `${beforePlan.cashAllocation}%`, 
      after: `${afterPlan.cashAllocation}%`,
      change: afterPlan.cashAllocation - beforePlan.cashAllocation
    },
    { 
      label: 'MONTHLY_INV', 
      before: `$${beforePlan.monthlyInvestment.toLocaleString()}`, 
      after: `$${afterPlan.monthlyInvestment.toLocaleString()}`,
      change: afterPlan.monthlyInvestment - beforePlan.monthlyInvestment
    },
  ];

  return (
    <Card className="border-2 border-[var(--color-emerald)] bg-gradient-to-br from-[var(--color-emerald)] to-green-600 p-8 text-white angular-card">
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="text-xs uppercase tracking-wider opacity-80 mb-2">VERIFICATION_COMPLETE</div>
          <h2 className="text-white" style={{ fontFamily: 'var(--font-display)' }}>
            Plan Successfully Adjusted
          </h2>
          <p className="text-sm opacity-90 mt-2" style={{ fontFamily: 'var(--font-mono)' }}>
            Optimized and verified by VP-MAS constraint satisfaction engine
          </p>
        </div>
        <div className="terminal-text border-white/30">
          <div style={{ fontFamily: 'var(--font-code)', fontSize: '0.75rem', color: 'white' }}>
            VA_STATUS: APPROVED
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-6">
        {/* Before */}
        <div className="space-y-3">
          <div className="data-label text-white/80 mb-3">ORIGINAL_PLAN</div>
          <div className="bg-white/10 backdrop-blur-sm p-5 border-2 border-white/30 angular-card">
            <table className="w-full" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>
              <tbody className="space-y-2">
                {metrics.map((metric, idx) => (
                  <tr key={idx} className="border-b border-white/20">
                    <td className="py-2 pr-4 text-white/70 text-xs">{metric.label}</td>
                    <td className="py-2 text-right text-white">{metric.before}</td>
                  </tr>
                ))}
                <tr>
                  <td className="pt-3 pr-4 text-white/70 text-xs">RISK_PROFILE</td>
                  <td className="pt-3 text-right text-white">{beforePlan.riskTolerance}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* After */}
        <div className="space-y-3">
          <div className="data-label text-white/80 mb-3">ADJUSTED_PLAN</div>
          <div className="bg-white/20 backdrop-blur-sm p-5 border-2 border-white angular-card">
            <table className="w-full" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>
              <tbody className="space-y-2">
                {metrics.map((metric, idx) => (
                  <motion.tr 
                    key={idx}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.1 }}
                    className="border-b border-white/30"
                  >
                    <td className="py-2 pr-4 text-white/80 text-xs">{metric.label}</td>
                    <td className="py-2 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <span className="text-white font-semibold">{metric.after}</span>
                        {metric.change > 0 ? (
                          <TrendingUp className="w-3 h-3" />
                        ) : metric.change < 0 ? (
                          <TrendingDown className="w-3 h-3" />
                        ) : null}
                      </div>
                    </td>
                  </motion.tr>
                ))}
                <tr>
                  <td className="pt-3 pr-4 text-white/80 text-xs">RISK_PROFILE</td>
                  <td className="pt-3 text-right text-white font-semibold">{afterPlan.riskTolerance}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-center gap-3 text-sm text-white/90" style={{ fontFamily: 'var(--font-code)' }}>
        <span>ORIGINAL</span>
        <ArrowRight className="w-5 h-5" />
        <span>CMVL_PROCESS</span>
        <ArrowRight className="w-5 h-5" />
        <span>VERIFIED_PLAN</span>
      </div>
    </Card>
  );
}
