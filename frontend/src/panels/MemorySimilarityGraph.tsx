// frontend/src/panels/MemorySimilarityGraph.tsx
// Phase G (四.4): 记忆相似度图谱 — visualize the Hopfield associative
// memory state from the current snapshot.
//
// Renders:
//   1. A radial "memory capacity" gauge: number of stored memories vs
//      configured capacity (n_memories / capacity).
//   2. The current-cycle top-1 retrieval similarity as a horizontal bar
//      (0 → 1, with a 0.5 marker for the "associated" threshold).
//   3. A rolling time-series of top1_similarity over the last N
//      snapshots, so the user can see when the model's current state
//      matches a stored attractor (pattern completion) vs when it is
//      novel (low similarity → new memory being written).
//
// When use_hopfield is off, the panel renders an empty state explaining
// the upgrade is disabled.

import { useMemo } from 'react';
import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';
import { selectActiveSnapshot } from '../store/useModelStore';

const HISTORY_WINDOW = 60;

const wrapStyle: CSSProperties = {
  padding: '8px 12px',
  fontFamily: 'ui-monospace, Consolas, monospace',
  fontSize: '11px',
  color: '#9ca3af',
  height: '100%',
  overflow: 'auto',
};

const sectionTitleStyle: CSSProperties = {
  fontSize: '10px',
  color: '#aaa',
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
  margin: '8px 0 4px',
};

const metricRowStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '4px 0',
  borderBottom: '1px solid #1a1a1a',
};

const metricLabelStyle: CSSProperties = { color: '#aaa' };

const metricValueStyle: CSSProperties = {
  color: '#e0e0e0',
  fontFamily: 'inherit',
};

const barTrackStyle: CSSProperties = {
  flex: 1,
  height: '12px',
  background: '#1a1a22',
  borderRadius: '2px',
  position: 'relative',
  overflow: 'hidden',
  margin: '0 8px',
};

const sparklineStyle: CSSProperties = {
  display: 'block',
  width: '100%',
  height: '48px',
  background: '#0a0a0f',
  borderRadius: '2px',
};

function simColor(sim: number): string {
  // 0 → blue (#3b82f6), 0.5 → amber (#fbbf24), 1 → green (#4ade80)
  if (sim >= 0.75) return '#4ade80';
  if (sim >= 0.5) return '#fbbf24';
  return '#3b82f6';
}

export default function MemorySimilarityGraph() {
  const snapshot = useModelStore(selectActiveSnapshot);
  const history = useModelStore((s) => s.history);

  const mem = snapshot?.memory_retrieved;
  const hasMem =
    mem && (typeof mem.n_memories === 'number' || typeof mem.top1_similarity === 'number');

  // Rolling similarity time-series.
  const recent = useMemo(() => {
    return history
      .filter((s) => s.memory_retrieved && typeof s.memory_retrieved.top1_similarity === 'number')
      .slice(-HISTORY_WINDOW)
      .map((s) => ({
        step: s.step,
        sim: s.memory_retrieved!.top1_similarity as number,
      }));
  }, [history]);

  if (!hasMem && recent.length === 0) {
    return (
      <div style={wrapStyle}>
        <div style={{ color: '#666', padding: '12px 0' }}>
          <div style={{ fontWeight: 600, color: '#9ca3af', marginBottom: '4px' }}>
            Hopfield memory unavailable
          </div>
          <div>
            The associative memory (use_hopfield) appears to be disabled, or
            no memories have been stored yet. Enable it via the
            <code> ZDM_USE_HOPFIELD=1 </code>
            environment variable or by constructing the model with
            <code> use_hopfield=True</code>.
          </div>
        </div>
      </div>
    );
  }

  const nMem = typeof mem?.n_memories === 'number' ? mem!.n_memories! : 0;
  const top1Sim = typeof mem?.top1_similarity === 'number' ? mem!.top1_similarity! : 0;

  // Build the sparkline SVG path.
  const sparklinePath = useMemo(() => {
    if (recent.length < 2) return '';
    const w = 100;
    const h = 40;
    const max = Math.max(...recent.map((r) => r.sim), 1e-9);
    const min = Math.min(...recent.map((r) => r.sim), 0);
    const range = max - min || 1;
    return recent
      .map((r, i) => {
        const x = (i / (recent.length - 1)) * w;
        const y = h - ((r.sim - min) / range) * h;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(' ');
  }, [recent]);

  return (
    <div style={wrapStyle}>
      <div style={sectionTitleStyle}>Hopfield associative memory</div>

      <div style={metricRowStyle}>
        <span style={metricLabelStyle}>Stored memories</span>
        <span style={metricValueStyle}>{nMem}</span>
      </div>

      <div style={metricRowStyle}>
        <span style={metricLabelStyle}>Top-1 similarity</span>
        <span style={{ ...metricValueStyle, color: simColor(top1Sim) }}>
          {top1Sim.toFixed(4)}
        </span>
      </div>

      {/* Top-1 similarity horizontal bar with 0.5 marker */}
      <div style={{ display: 'flex', alignItems: 'center', marginTop: '6px' }}>
        <span style={{ width: '24px', color: '#aaa' }}>sim</span>
        <div style={barTrackStyle}>
          <div
            style={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: 0,
              width: `${Math.min(100, top1Sim * 100)}%`,
              background: simColor(top1Sim),
              borderRadius: '2px',
            }}
          />
          {/* 0.5 marker */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: '50%',
              width: '1px',
              background: '#444',
            }}
          />
        </div>
        <span style={{ width: '48px', textAlign: 'right', color: '#aaa' }}>
          {top1Sim.toFixed(3)}
        </span>
      </div>

      <div style={sectionTitleStyle}>Similarity over time (last {recent.length} cycles)</div>
      {recent.length >= 2 ? (
        <svg viewBox="0 0 100 40" preserveAspectRatio="none" style={sparklineStyle}>
          {/* 0.5 reference line */}
          <line x1="0" y1="20" x2="100" y2="20" stroke="#222" strokeWidth="0.5" strokeDasharray="2 2" />
          <path d={sparklinePath} fill="none" stroke="#3b82f6" strokeWidth="1" />
        </svg>
      ) : (
        <div style={{ color: '#666', padding: '8px 0' }}>
          Insufficient history to plot similarity trend.
        </div>
      )}

      <div style={{ marginTop: '8px', fontSize: '10px', color: '#666' }}>
        Modern Hopfield network stores experiences as attractors. High
        top-1 similarity indicates the current state matches a stored
        pattern (pattern completion); low similarity indicates novelty
        (the experience will be written as a new attractor).
      </div>
    </div>
  );
}
