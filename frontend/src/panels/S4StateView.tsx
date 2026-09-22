// frontend/src/panels/S4StateView.tsx
// Phase G (四.4): S4 状态可视化 — visualize the S4 (Diagonal State
// Space / HiPPO-initialized) hidden state from the current snapshot.
//
// The S4 layer replaces the fixed transition matrix in
// ActiveInferenceEngine for efficient long-range temporal prediction.
// This panel surfaces the first ≤8 dims of the hidden state vector so
// the user can watch how the HiPPO memory evolves across cycles.
//
// Renders:
//   1. A centered-diverging bar chart of the current s4_state vector
//      (signed values: blue for positive, orange for negative).
//   2. A rolling heatmap of the s4_state trajectory over the last N
//      snapshots (one row per dimension, columns = cycles).
//
// When use_s4 is off, the panel renders an empty state explaining the
// upgrade is disabled.

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

const barRowStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
  marginBottom: '4px',
};

const barTrackStyle: CSSProperties = {
  flex: 1,
  height: '10px',
  background: '#1a1a22',
  borderRadius: '2px',
  position: 'relative',
  overflow: 'hidden',
};

function barFillStyle(value: number, max: number): CSSProperties {
  const ratio = max > 0 ? Math.min(1, Math.abs(value) / max) : 0;
  const sign = value >= 0 ? 1 : -1;
  const widthPct = ratio * 50;
  return {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: sign >= 0 ? '50%' : `${50 - widthPct}%`,
    width: `${widthPct}%`,
    background: sign >= 0 ? '#3b82f6' : '#f97316',
    borderRadius: '2px',
  };
}

function heatColor(value: number, absMax: number): string {
  if (absMax <= 0 || !Number.isFinite(value)) return '#1a1a22';
  const ratio = Math.min(1, Math.abs(value) / absMax);
  if (value >= 0) {
    // blue ramp
    const light = 20 + 35 * ratio;
    return `hsl(220, 70%, ${light}%)`;
  }
  // orange ramp
  const light = 20 + 35 * ratio;
  return `hsl(25, 80%, ${light}%)`;
}

export default function S4StateView() {
  const snapshot = useModelStore(selectActiveSnapshot);
  const history = useModelStore((s) => s.history);

  const s4 = snapshot?.s4_state ?? [];

  // Pull last HISTORY_WINDOW snapshots with s4_state populated.
  const recent = useMemo(() => {
    return history
      .filter((s) => Array.isArray(s.s4_state) && s.s4_state!.length > 0)
      .slice(-HISTORY_WINDOW);
  }, [history]);

  const hasCurrent = s4.length > 0;
  const hasHistory = recent.length > 0;

  if (!hasCurrent && !hasHistory) {
    return (
      <div style={wrapStyle}>
        <div style={{ color: '#666', padding: '12px 0' }}>
          <div style={{ fontWeight: 600, color: '#9ca3af', marginBottom: '4px' }}>
            S4 hidden state unavailable
          </div>
          <div>
            The S4 state-space model (use_s4) appears to be disabled. Enable
            it via the <code>ZDM_USE_S4=1</code> environment variable or by
            constructing the model with <code>use_s4=True</code>.
          </div>
        </div>
      </div>
    );
  }

  const localMax = hasCurrent ? Math.max(...s4.map((v) => Math.abs(v)), 1e-9) : 1e-9;

  // Max absolute value across the heatmap window for color scaling.
  const globalMax = useMemo(() => {
    let m = 1e-9;
    for (const s of recent) {
      for (const v of s.s4_state!) {
        if (Number.isFinite(v) && Math.abs(v) > m) m = Math.abs(v);
      }
    }
    if (hasCurrent) {
      for (const v of s4) {
        if (Number.isFinite(v) && Math.abs(v) > m) m = Math.abs(v);
      }
    }
    return m;
  }, [recent, s4, hasCurrent]);

  // Determine the number of dimensions to render in the heatmap (max 8).
  const nDims = Math.max(
    hasCurrent ? s4.length : 0,
    hasHistory ? Math.max(...recent.map((s) => s.s4_state!.length)) : 0,
  );

  return (
    <div style={wrapStyle}>
      <div style={sectionTitleStyle}>S4 hidden state (current cycle)</div>
      {hasCurrent ? (
        <>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {s4.map((v, i) => (
              <div key={i} style={barRowStyle}>
                <span style={{ width: '24px', textAlign: 'right', color: '#aaa' }}>
                  {i}
                </span>
                <div style={barTrackStyle}>
                  <div style={barFillStyle(v, localMax)} />
                  <div
                    style={{
                      position: 'absolute',
                      left: '50%',
                      top: 0,
                      bottom: 0,
                      width: '1px',
                      background: '#333',
                    }}
                  />
                </div>
                <span style={{ width: '52px', color: '#aaa' }}>{v.toFixed(4)}</span>
              </div>
            ))}
          </div>
          <div style={{ marginTop: '4px', fontSize: '10px', color: '#666' }}>
            ‖·‖∞ = {localMax.toFixed(4)} · {s4.length} dims shown (capped at 8)
          </div>
        </>
      ) : (
        <div style={{ color: '#666', padding: '6px 0' }}>
          No S4 state in the current snapshot.
        </div>
      )}

      {hasHistory && (
        <>
          <div style={sectionTitleStyle}>
            Trajectory · last {recent.length} cycles · {nDims} dims
          </div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: `24px repeat(${recent.length}, 1fr)`,
              gap: '1px',
            }}
          >
            {/* Header row (cycle numbers, sparse) */}
            <div />
            {recent.map((s) => (
              <div
                key={`hdr-${s.step}`}
                style={{ fontSize: '7px', color: '#444', textAlign: 'center' }}
              >
                {s.step % 20 === 0 ? s.step % 100 : ''}
              </div>
            ))}
            {/* One row per dim */}
            {Array.from({ length: nDims }).map((_, dimIdx) => (
              <div key={`dim-${dimIdx}`} style={{ display: 'contents' }}>
                <div style={{ fontSize: '10px', color: '#aaa', textAlign: 'right', paddingRight: '4px' }}>
                  {dimIdx}
                </div>
                {recent.map((s) => {
                  const v = s.s4_state![dimIdx];
                  const isFiniteNum = typeof v === 'number' && Number.isFinite(v);
                  return (
                    <div
                      key={`cell-${dimIdx}-${s.step}`}
                      title={`step ${s.step} · dim ${dimIdx} = ${isFiniteNum ? v!.toFixed(4) : 'n/a'}`}
                      style={{
                        height: '12px',
                        minWidth: '0',
                        background: isFiniteNum ? heatColor(v!, globalMax) : '#0a0a0f',
                      }}
                    />
                  );
                })}
              </div>
            ))}
          </div>
        </>
      )}

      <div style={{ marginTop: '8px', fontSize: '10px', color: '#666' }}>
        S4 (Diagonal State Space) replaces the fixed transition matrix with
        a HiPPO-initialized recurrent state for long-range temporal
        prediction. The hidden state encodes the model's compressed memory
        of recent observations; stable oscillation indicates healthy
        sequence modelling, while divergence signals numerical instability.
      </div>
    </div>
  );
}
