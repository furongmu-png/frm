// frontend/src/panels/BeliefStateRaw.tsx
// Collapsible raw view of the current snapshot's belief_state vector.
//
// Renders a horizontal bar histogram (one bar per dimension) so the
// raw latent is visually scannable. In replay mode, follows the
// replayed snapshot.

import { useState } from 'react';
import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';

const wrapStyle: CSSProperties = {
  borderBottom: '1px solid #1a1a1a',
  background: '#0d0d12',
};

const headerStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '4px 12px',
  cursor: 'pointer',
  userSelect: 'none',
  fontSize: '11px',
  color: '#aaa',
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
  background: 'transparent',
  border: 'none',
  width: '100%',
  textAlign: 'left',
};

const bodyStyle: CSSProperties = {
  padding: '6px 12px 8px',
  display: 'flex',
  flexDirection: 'column',
  gap: '4px',
};

const barRowStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
  fontSize: '10px',
  color: '#aaa',
};

const barTrackStyle: CSSProperties = {
  flex: 1,
  height: '8px',
  background: '#1a1a22',
  borderRadius: '2px',
  position: 'relative',
  overflow: 'hidden',
};

function barFillStyle(value: number, max: number): CSSProperties {
  // Map value in [-max, max] to a width 0..100%, centred.
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

const textStyle: CSSProperties = {
  fontFamily: 'ui-monospace, Consolas, monospace',
  fontSize: '11px',
  color: '#9ca3af',
  wordBreak: 'break-all',
};

export default function BeliefStateRaw() {
  const snapshot = useModelStore((s) => s.snapshot);
  const playbackIndex = useModelStore((s) => s.playbackIndex);
  const [open, setOpen] = useState(false);

  const belief = snapshot?.belief_state ?? [];
  const max = belief.length > 0 ? Math.max(...belief.map((v) => Math.abs(v)), 1e-9) : 1;
  const inReplay = playbackIndex >= 0;

  return (
    <div style={wrapStyle}>
      <button
        type="button"
        style={headerStyle}
        aria-expanded={open}
        aria-controls="belief-state-raw-body"
        onClick={() => setOpen((o) => !o)}
      >
        <span>
          {open ? '▼' : '▶'} Belief State (raw){' '}
          {inReplay && (
            <span style={{ color: '#eab308', marginLeft: '4px' }}>· replay</span>
          )}
        </span>
        <span style={{ color: '#9ca3af' }}>
          dim={belief.length} · ‖·‖∞={max.toFixed(4)}
        </span>
      </button>
      {open && (
        <div id="belief-state-raw-body" style={bodyStyle}>
          {belief.length === 0 ? (
            <div style={textStyle}>No belief_state available.</div>
          ) : (
            <>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
                  gap: '3px 16px',
                }}
              >
                {belief.slice(0, 64).map((v, i) => (
                  <div key={i} style={barRowStyle}>
                    <span style={{ width: '24px', textAlign: 'right' }}>{i}</span>
                    <div style={barTrackStyle}>
                      <div style={barFillStyle(v, max)} />
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
                    <span style={{ width: '52px', color: '#aaa' }}>{v.toFixed(3)}</span>
                  </div>
                ))}
              </div>
              {belief.length > 64 && (
                <div style={textStyle}>
                  ... {belief.length - 64} more dimensions truncated
                </div>
              )}
              <details style={{ marginTop: '4px' }}>
                <summary style={{ cursor: 'pointer', color: '#9ca3af', fontSize: '10px' }}>
                  raw JSON array
                </summary>
                <pre style={textStyle}>
                  [{belief.map((v) => v.toFixed(4)).join(', ')}]
                </pre>
              </details>
            </>
          )}
        </div>
      )}
    </div>
  );
}
