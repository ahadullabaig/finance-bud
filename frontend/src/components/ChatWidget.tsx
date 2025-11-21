import React, { useEffect, useRef, useState } from 'react';

type Sender = 'user' | 'assistant' | 'system';

type Message = {
  id: string;
  sender: Sender;
  text: string;
  meta?: any;
};

export default function ChatWidget() {
  // Base URL for API requests.
  // Use Vite env variable if provided; otherwise use a relative path so the dev proxy (or same-origin) works.
  // In production, set VITE_API_URL to your deployed API base (e.g. https://api.myapp.com)
  const API_BASE = (typeof import.meta !== 'undefined' && (import.meta as any).env && (import.meta as any).env.VITE_API_URL)
    ? (import.meta as any).env.VITE_API_URL
    : '';

  const [messages, setMessages] = useState<Message[]>([
    { id: 'm1', sender: 'assistant', text: 'Hi — I can help parse goals, suggest actions, and explain plans. Try typing a goal (e.g. "I want to retire at 60 with $2M").' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // scroll to bottom on new message
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages, suggestions]);

  const pushMessage = (m: Message) => {
    setMessages((prev) => [...prev, m]);
  };

  const handleSend = async (text?: string) => {
    const content = (text ?? input).trim();
    if (!content) return;

    // push user message
    const userMsg: Message = { id: `u_${Date.now()}`, sender: 'user', text: content };
    pushMessage(userMsg);
    setInput('');
    setLoading(true);
    setSuggestions([]);

    try {
      const res = await fetch(`${API_BASE}/api/conversational/parse-goal`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_input: content, user_context: {} })
      });

      if (!res.ok) {
        // try to capture server error body for debugging
        let body = '';
        try { body = await res.text(); } catch (e) { /* ignore */ }
        throw new Error(`API error ${res.status}: ${body}`);
      }

      const parsed = await res.json();

      // Create assistant summary message
      const summaryParts: string[] = [];
      if (parsed.goal_type) summaryParts.push(`Goal: ${parsed.goal_type}`);
      if (parsed.target_amount) summaryParts.push(`Target: $${Number(parsed.target_amount).toLocaleString()}`);
      if (parsed.timeframe_years) summaryParts.push(`Timeframe: ${parsed.timeframe_years} years`);
      if (parsed.risk_tolerance) summaryParts.push(`Risk: ${parsed.risk_tolerance}`);

      const assistantText = summaryParts.length ? summaryParts.join(' • ') : 'I parsed your input but found no structured fields.';

      const assistantMsg: Message = { id: `a_${Date.now()}`, sender: 'assistant', text: assistantText, meta: parsed };
      pushMessage(assistantMsg);

      // Build suggestion chips from parsed result
      const chips: string[] = [];
      chips.push('Generate narrative');
      chips.push('Explain what-if (market crash)');
      chips.push('Suggest contribution plan');
      if (parsed.goal_type === 'retirement') chips.push('Estimate retirement savings');
      if (parsed.goal_type === 'debt_payoff') chips.push('Recommend payoff strategy');

      setSuggestions(chips);

    } catch (e: any) {
      // Fallback: do local mock suggestions and include error info in assistant text
      const errText = e?.message || String(e);
      const assistantMsg: Message = { id: `a_fallback_${Date.now()}`, sender: 'assistant', text: `Unable to reach conversational API — showing local suggestions. (${errText})` };
      pushMessage(assistantMsg);
      setSuggestions([
        'Generate narrative',
        'Explain what-if (job loss)',
        'Show risk analysis'
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestionClick = (s: string) => {
    // For now, suggestions are translated into user messages and trigger the API
    handleSend(s);
  };

  return (
    <div className="border-2 border-[var(--color-ink)] p-4 bg-white angular-card max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="data-label">CHAT ASSISTANT</div>
          <div className="text-sm text-[var(--color-blueprint)]" style={{ fontFamily: 'var(--font-mono)' }}>
            Interactive assistant (demo)
          </div>
        </div>
        <div className="text-xs text-[var(--color-blueprint)]">Real-time • Suggestions</div>
      </div>

      <div ref={listRef} className="h-64 overflow-y-auto p-2 mb-3 bg-[var(--color-grid)] rounded">
        {messages.map((m) => (
          <div key={m.id} className={`mb-2 flex ${m.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`${m.sender === 'user' ? 'bg-[var(--color-cyan)] text-white' : 'bg-white text-[var(--color-ink)]'} px-3 py-2 rounded shadow-sm max-w-[80%]`}>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9rem' }}>{m.text}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="mb-2">
        {suggestions.length > 0 && (
          <div className="flex gap-2 flex-wrap mb-2">
            {suggestions.map((s) => (
              <button key={s} onClick={() => handleSuggestionClick(s)} className="px-3 py-1 bg-[var(--color-ink)] text-[var(--color-cyan)] rounded font-mono text-sm hover:opacity-90">
                {s}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') handleSend(); }}
          placeholder="Type your goal or question..."
          className="flex-1 px-3 py-2 border-2 border-[var(--color-grid)] rounded bg-white"
        />
        <button onClick={() => handleSend()} disabled={loading} className="px-4 py-2 bg-[var(--color-ink)] text-[var(--color-cyan)] rounded">
          {loading ? '...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
