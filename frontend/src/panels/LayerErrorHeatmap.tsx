// frontend/src/panels/LayerErrorHeatmap.tsx
// Phase G (四.4): 层级误差热图 — visualize the per-layer prediction-error
// norms from the Hierarchical Predictive Coding network (L0/L1/L2).
//
// Renders:
//   1. A vertical 3-row heatmap (L0/L1/L2 × recent time window) showing
//      ‖error‖ over the last N snapshots, so the user can see whether
//      errors decrease / propagate bottom-up as expected.
//   2. A current-cycle bar chart of the three layer error norms with a
//      "decreasing?" indicator (‖error_L0‖ > ‖error_L1‖ > ‖error_L2‖
//      being the healthy pattern).
//
// When use_pcn is off, the panel renders an empty state explaining the
// upgrade is disabled.

import { useMemo } from 'react';
import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';
import { selectActiveSnapshot } from '../store/useModelStore';

const LAYERS = ['L0', 'L1', 'L2'] as const;
const HISTORY_WINDOW = 60; // number of recent snapshots to render in the heatmap

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

const heatmapGridStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '24px repeat(60, 1fr)',
  gap: '1px',
  alignItems: 'center',
};

const heatmapCellStyle: CSSProperties = {
  height: '14px',
  minWidth: '0',
};

const rowLabelStyle: CSSProperties = {
  fontSize: '10px',
  color: '#aaa',
  textAlign: 'right',
  paddingRight: '4px',
};

const barRowStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
  marginBottom: '4px',
};

const barTrackStyle: CSSProperties = {
  flex: 1,
  height: '12px',
  background: '#1a1a22',
  borderRadius: '2px',
  position: 'relative',
  overflow: 'hidden',
};

const pillStyle = (healthy: boolean): CSSProperties => ({
  display: 'inline-block',
  padding: '1px 6px',
  borderRadius: '3px',
  fontSize: '10px',
  background: healthy ? '#1a3a1a' : '#3a2a1a',
  color: healthy ? '#4ade80' : '#fbbf24',
  marginLeft: '4px',
});

// Map a value in [0, max] to an HSL blue→red ramp.
function heatColor(value: number, max: number): string {
  if (max <= 0 || !Number.isFinite(value)) return '#1a1a22';
  const ratio = Math.min(1, Math.max(0, value / max));
  // Hue: 220 (blue) → 0 (red) as ratio goes 0 → 1
  const hue = 220 - 220 * ratio;
  const sat = 60 + 30 * ratio;
  const light = 25 + 25 * ratio;
  return `hsl(${hue}, ${sat}%, ${light}%)`;
}

export default function LayerErrorHeatmap() {
  const snapshot = useModelStore(selectActiveSnapshot);
  const history = useModelStore((s) => s.history);

  // Pull the last HISTORY_WINDOW snapshots that have layer_errors populated.
  const recent = useMemo(() => {
    return history
      .filter((s) => s.layer_errors && Object.keys(s.layer_errors).length > 0)
      .slice(-HISTORY_WINDOW);
  }, [history]);

  // Current-cycle error norms.
  const currentErrors: Record<string, number> = (() => {
    const le = snapshot?.layer_errors;
    if (!le) return {};
    const out: Record<string, number> = {};
    for (const L of LAYERS) {
      const v = le[L];
      if (typeof v === 'number' && Number.isFinite(v)) out[L] = v;
    }
    return out;
  })();

  const hasCurrent = Object.keys(currentErrors).length > 0;
  const hasHistory = recent.length > 0;

  // Max error across the heatmap window for color scaling.
  const globalMax = useMemo(() => {
    let m = 1e-9;
    for (const s of recent) {
      const le = s.layer_errors!;
      for (const L of LAYERS) {
        const v = le[L];
        if (typeof v === 'number' && Number.isFinite(v) && v > m) m = v;
      }
    }
    if (hasCurrent) {
      for (const L of LAYERS) {
        const v = currentErrors[L];
        if (typeof v === 'number' && v > m) m = v;
      }
    }
    return m;
  }, [recent, currentErrors, hasCurrent]);

  const localMax = hasCurrent
    ? Math.max(...LAYERS.map((L) => currentErrors[L] ?? 0), 1e-9)
    : 1e-9;

  // Healthy hierarchy: errors decrease bottom → top (L0 > L1 > L2).
  const decreasing = hasCurrent
    ? LAYERS.every((L, i) => {
        if (i === 0) return true;
        const prev = currentErrors[LAYERS[i - 1]];
        const cur = currentErrors[L];
        return typeof prev === 'number' && typeof cur === 'number' && prev >= cur;
      })
    : false;

  if (!hasCurrent && !hasHistory) {
    return (
      <div style={wrapStyle}>
        <div style={{ color: '#666', padding: '12px 0' }}>
          <div style={{ fontWeight: 600, color: '#9ca3af', marginBottom: '4px' }}>
            PCN layer errors unavailable
          </div>
          <div>
            The Hierarchical Predictive Coding network (use_pcn) appears to be
            disabled. Enable it via the <code>ZDM_USE_PCN=1</code> environment
            variable or by constructing the model with <code>use_pcn=True</code>.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={wrapStyle}>
      <div style={sectionTitleStyle}>
        Layer Error Heatmap · last {recent.length} cycles
        <span style={pillStyle(decreasing)}>
          {decreasing ? 'L0>L1>L2 ✓' : 'not decreasing'}
        </span>
      </div>

      {hasHistory && (
        <div style={heatmapGridStyle}>
          {/* Header row: empty cell over the label column */}
          <div />
          {recent.map((s) => (
            <div key={s.step} style={{ fontSize: '8px', color: '#444', textAlign: 'center' }}>
              {/* tick marks only every 10 cycles to avoid clutter */}
              {s.step % 10 === 0 ? s.step % 100 : ''}
            </div>
          ))}
          {LAYERS.map((L) => (
            <div key={L} style={{ display: 'contents' }}>
              <div style={rowLabelStyle}>{L}</div>
              {recent.map((s) => {
                const v = s.layer_errors![L];
                const isFiniteNum = typeof v === 'number' && Number.isFinite(v);
                return (
                  <div
                    key={`${L}-${s.step}`}
                    title={`step ${s.step} · ${L} = ${isFiniteNum ? v!.toFixed(4) : 'n/a'}`}
                    style={{
                      ...heatmapCellStyle,
                      background: isFiniteNum ? heatColor(v!, globalMax) : '#0a0a0f',
                    }}
                  />
                );
              })}
            </div>
          ))}
        </div>
      )}

      <div style={sectionTitleStyle}>Current cycle</div>
      <div>
        {LAYERS.map((L) => {
          const v = currentErrors[L];
          const has = typeof v === 'number';
          return (
            <div key={L} style={barRowStyle}>
              <span style={{ width: '24px', color: '#aaa' }}>{L}</span>
              <div style={barTrackStyle}>
                <div
                  style={{
                    position: 'absolute',
                    top: 0,
                    bottom: 0,
                    left: 0,
                    width: `${has ? Math.min(100, (v! / localMax) * 100) : 0}%`,
                    background: heatColor(has ? v! : 0, localMax),
                    borderRadius: '2px',
                  }}
                />
              </div>
              <span style={{ width: '64px', textAlign: 'right', color: '#aaa' }}>
                {has ? v!.toFixed(4) : '—'}
              </span>
            </div>
          );
        })}
      </div>

      <div style={{ marginTop: '8px', fontSize: '10px', color: '#666' }}>
        Bottom-up error propagation: L0 (perception) → L1 (short-term) → L2
        (abstract rules). A healthy hierarchy has ‖e_L0‖ ≥ ‖e_L1‖ ≥ ‖e_L2‖
        as errors are absorbed at each level.
      </div>
    </div>
  );
}
