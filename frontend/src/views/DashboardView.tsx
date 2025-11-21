import { Card } from '../components/ui/card';
import { 
  TrendingUp, TrendingDown, DollarSign, Activity, 
  AlertTriangle, CheckCircle2, Clock 
} from 'lucide-react';
import { LineChart, Line, AreaChart, Area, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { motion } from 'motion/react';

export function DashboardView() {
  const portfolioData = [
    { month: 'JAN', value: 125000, projected: 125000 },
    { month: 'FEB', value: 128000, projected: 127000 },
    { month: 'MAR', value: 132000, projected: 129000 },
    { month: 'APR', value: 135000, projected: 131000 },
    { month: 'MAY', value: 131000, projected: 133500 },
    { month: 'JUN', value: 128000, projected: 136000 },
    { month: 'JUL', value: 142000, projected: 138500 },
  ];

  const allocationData = [
    { name: 'STOCKS', value: 50, color: '#1B4965' },
    { name: 'BONDS', value: 40, color: '#06D6A0' },
    { name: 'CASH', value: 10, color: '#FFB703' },
  ];

  const riskMetrics = [
    { metric: 'VaR (95%)', value: '$8,500', status: 'normal', change: -5.2 },
    { metric: 'Sharpe Ratio', value: '1.42', status: 'good', change: 8.1 },
    { metric: 'Beta', value: '0.87', status: 'normal', change: -2.3 },
    { metric: 'Max Drawdown', value: '12.4%', status: 'warning', change: 3.1 },
  ];

  const marketIndicators = [
    { name: 'S&P 500', value: '4521.45', change: -2.3, status: 'down' },
    { name: 'VIX', value: '18.24', change: 12.5, status: 'up' },
    { name: '10Y Treasury', value: '4.12%', change: 0.8, status: 'up' },
  ];

  const cmvlHistory = [
    { date: '2024-01-15', trigger: 'Market Volatility', action: 'Rebalanced', status: 'SUCCESS' },
    { date: '2024-02-28', trigger: 'Job Loss Event', action: 'Emergency Fund Increased', status: 'SUCCESS' },
    { date: '2024-05-12', trigger: 'Market Recovery', action: 'Risk Adjusted', status: 'SUCCESS' },
  ];

  return (
    <div className="space-y-8">
      {/* KPI Cards */}
      <div className="grid md:grid-cols-4 gap-6">
        <motion.div className="stagger-1">
          <Card className="border-2 border-[var(--color-ink)] bg-white p-6 angular-card">
            <div className="flex items-start justify-between">
              <div>
                <div className="data-label mb-2">TOTAL_PORTFOLIO</div>
                <div className="text-4xl mb-2" style={{ fontFamily: 'var(--font-display)' }}>$142K</div>
                <div className="flex items-center gap-1 text-sm" style={{ fontFamily: 'var(--font-code)', color: 'var(--color-emerald)' }}>
                  <TrendingUp className="w-4 h-4" />
                  <span>+11.2%</span>
                </div>
              </div>
              <div className="p-3 border-2 border-[var(--color-blueprint)]">
                <DollarSign className="w-6 h-6" style={{ color: 'var(--color-blueprint)' }} />
              </div>
            </div>
          </Card>
        </motion.div>

        <motion.div className="stagger-2">
          <Card className="border-2 border-[var(--color-ink)] bg-white p-6 angular-card">
            <div className="flex items-start justify-between">
              <div>
                <div className="data-label mb-2">EMERGENCY_FUND</div>
                <div className="text-4xl mb-2" style={{ fontFamily: 'var(--font-display)' }}>$35K</div>
                <div className="flex items-center gap-1 text-sm" style={{ fontFamily: 'var(--font-code)', color: 'var(--color-emerald)' }}>
                  <CheckCircle2 className="w-4 h-4" />
                  <span>6 MONTHS</span>
                </div>
              </div>
              <div className="p-3 border-2 border-[var(--color-emerald)]">
                <Activity className="w-6 h-6" style={{ color: 'var(--color-emerald)' }} />
              </div>
            </div>
          </Card>
        </motion.div>

        <motion.div className="stagger-3">
          <Card className="border-2 border-[var(--color-ink)] bg-white p-6 angular-card">
            <div className="flex items-start justify-between">
              <div>
                <div className="data-label mb-2">RISK_SCORE</div>
                <div className="text-4xl mb-2" style={{ fontFamily: 'var(--font-display)' }}>MOD</div>
                <div className="flex items-center gap-1 text-sm" style={{ fontFamily: 'var(--font-code)', color: 'var(--color-cyan)' }}>
                  <Activity className="w-4 h-4" />
                  <span>OPTIMIZED</span>
                </div>
              </div>
              <div className="p-3 border-2 border-[var(--color-amber)]">
                <TrendingDown className="w-6 h-6" style={{ color: 'var(--color-amber)' }} />
              </div>
            </div>
          </Card>
        </motion.div>

        <motion.div className="stagger-4">
          <Card className="border-2 border-[var(--color-ink)] bg-white p-6 angular-card">
            <div className="flex items-start justify-between">
              <div>
                <div className="data-label mb-2">CMVL_TRIGGERS</div>
                <div className="text-4xl mb-2" style={{ fontFamily: 'var(--font-display)' }}>3</div>
                <div className="flex items-center gap-1 text-sm" style={{ fontFamily: 'var(--font-code)', color: 'var(--color-blueprint)' }}>
                  <Clock className="w-4 h-4" />
                  <span>30 DAYS</span>
                </div>
              </div>
              <div className="p-3 border-2 border-[var(--color-vermillion)]">
                <AlertTriangle className="w-6 h-6" style={{ color: 'var(--color-vermillion)' }} />
              </div>
            </div>
          </Card>
        </motion.div>
      </div>

      {/* Charts Row */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Portfolio Performance */}
        <Card className="border-2 border-[var(--color-ink)] bg-white p-6 blueprint-corners stagger-5">
          <div className="data-label mb-2">PERFORMANCE_ANALYSIS</div>
          <h3 className="mb-4" style={{ fontFamily: 'var(--font-display)' }}>
            Portfolio vs. Projection
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={portfolioData}>
              <defs>
                <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#1B4965" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#1B4965" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#CAE9FF" />
              <XAxis 
                dataKey="month" 
                stroke="#1B4965"
                style={{ fontFamily: 'var(--font-code)', fontSize: '0.75rem' }}
              />
              <YAxis 
                stroke="#1B4965"
                style={{ fontFamily: 'var(--font-code)', fontSize: '0.75rem' }}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#FAF9F6', 
                  border: '2px solid #0A1128',
                  fontFamily: 'var(--font-code)',
                  fontSize: '0.75rem'
                }}
              />
              <Area 
                type="monotone" 
                dataKey="value" 
                stroke="#00D9FF" 
                strokeWidth={3}
                fillOpacity={1} 
                fill="url(#colorValue)" 
                name="ACTUAL"
              />
              <Line 
                type="monotone" 
                dataKey="projected" 
                stroke="#E63946" 
                strokeWidth={2}
                strokeDasharray="5 5" 
                name="PROJECTED"
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </Card>

        {/* Asset Allocation */}
        <Card className="border-2 border-[var(--color-ink)] bg-white p-6 blueprint-corners stagger-5">
          <div className="data-label mb-2">ASSET_ALLOCATION</div>
          <h3 className="mb-4" style={{ fontFamily: 'var(--font-display)' }}>
            Current Distribution
          </h3>
          <div className="flex items-center justify-center">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={allocationData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name}: ${value}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                  style={{ fontFamily: 'var(--font-code)', fontSize: '0.75rem' }}
                >
                  {allocationData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} stroke="#0A1128" strokeWidth={2} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#FAF9F6', 
                    border: '2px solid #0A1128',
                    fontFamily: 'var(--font-code)',
                    fontSize: '0.75rem'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Risk Metrics */}
      <Card className="border-2 border-[var(--color-ink)] bg-white p-6 blueprint-corners stagger-6">
        <div className="data-label mb-4">RISK_METRICS</div>
        <div className="grid md:grid-cols-4 gap-6">
          {riskMetrics.map((metric, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: idx * 0.1 }}
              className={`
                p-5 border-2 angular-card
                ${metric.status === 'good' ? 'border-[var(--color-emerald)] bg-[var(--color-emerald)]/5' : ''}
                ${metric.status === 'warning' ? 'border-[var(--color-vermillion)] bg-[var(--color-vermillion)]/5' : ''}
                ${metric.status === 'normal' ? 'border-[var(--color-ink)] bg-white' : ''}
              `}
            >
              <div className="data-label mb-2 text-xs">{metric.metric.toUpperCase().replace(' ', '_')}</div>
              <div className="text-3xl mb-2" style={{ fontFamily: 'var(--font-display)' }}>
                {metric.value}
              </div>
              <div 
                className={`text-sm flex items-center gap-1`}
                style={{ 
                  fontFamily: 'var(--font-code)',
                  color: metric.change > 0 ? 'var(--color-vermillion)' : 'var(--color-emerald)'
                }}
              >
                {metric.change > 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                <span>{Math.abs(metric.change)}%</span>
              </div>
            </motion.div>
          ))}
        </div>
      </Card>

      {/* Market Indicators */}
      <Card className="border-2 border-[var(--color-ink)] bg-white p-6 blueprint-corners stagger-6">
        <div className="data-label mb-4">MARKET_INDICATORS</div>
        <div className="grid md:grid-cols-3 gap-6">
          {marketIndicators.map((indicator, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="border-2 border-[var(--color-ink)] p-5 angular-card bg-[var(--color-paper)]"
            >
              <div className="flex items-start justify-between">
                <div>
                  <div className="data-label mb-2 text-xs">{indicator.name.replace(' ', '_').toUpperCase()}</div>
                  <div className="text-3xl mb-2" style={{ fontFamily: 'var(--font-display)' }}>
                    {indicator.value}
                  </div>
                  <div 
                    className="text-sm flex items-center gap-1"
                    style={{ 
                      fontFamily: 'var(--font-code)',
                      color: indicator.status === 'up' ? 'var(--color-emerald)' : 'var(--color-vermillion)'
                    }}
                  >
                    {indicator.status === 'up' ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                    <span>{indicator.change}%</span>
                  </div>
                </div>
                <div 
                  className="px-2 py-1 border-2"
                  style={{ 
                    fontFamily: 'var(--font-code)',
                    fontSize: '0.65rem',
                    borderColor: indicator.status === 'up' ? 'var(--color-emerald)' : 'var(--color-vermillion)',
                    color: indicator.status === 'up' ? 'var(--color-emerald)' : 'var(--color-vermillion)'
                  }}
                >
                  {indicator.status.toUpperCase()}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </Card>

      {/* CMVL History */}
      <Card className="border-2 border-[var(--color-ink)] bg-white p-6 blueprint-corners stagger-6">
        <div className="data-label mb-4">CMVL_HISTORY</div>
        <div className="space-y-4">
          {cmvlHistory.map((event, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="flex items-center gap-4 p-5 border-2 border-[var(--color-ink)] bg-[var(--color-paper)] angular-card"
            >
              <CheckCircle2 className="w-5 h-5 flex-shrink-0" style={{ color: 'var(--color-emerald)' }} />
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-1">
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>
                    {event.trigger}
                  </span>
                  <div 
                    className="px-2 py-0.5 border border-[var(--color-ink)]"
                    style={{ fontFamily: 'var(--font-code)', fontSize: '0.65rem' }}
                  >
                    {event.action}
                  </div>
                </div>
                <div style={{ fontFamily: 'var(--font-code)', fontSize: '0.75rem', color: 'var(--color-blueprint)' }}>
                  {event.date}
                </div>
              </div>
              <div 
                className="px-3 py-1 border-2 border-[var(--color-emerald)]"
                style={{ 
                  fontFamily: 'var(--font-code)', 
                  fontSize: '0.75rem',
                  color: 'var(--color-emerald)'
                }}
              >
                {event.status}
              </div>
            </motion.div>
          ))}
        </div>
      </Card>
    </div>
  );
}
