// frontend/src/panels/ControlPanel.tsx
// Playback controls and system stats.
//
// Buttons:
//   - Play / Pause  : toggles the live stream (resume / pause).
//   - Step          : single-steps when paused (releases one snapshot).
// Speed slider (discrete): 0.5x, 1x, 2x, 5x.
// Timeline slider: range 0..history.length-1. Dragging enters replay
//   mode (playbackIndex >= 0); all other panels render the snapshot at
//   that index. Dragging to the far right returns to real-time mode.

import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';

const SPEED_OPTIONS = [0.5, 1, 2, 5];

const containerStyle: CSSProperties = {
  height: '100%',
  display: 'flex',
  flexDirection: 'row',
  flexWrap: 'wrap',
  alignItems: 'center',
  gap: '16px',
  padding: '8px 12px',
};

const btnStyle: CSSProperties = {
  padding: '6px 14px',
  fontSize: '12px',
  background: '#1a1a22',
  border: '1px solid #333',
  borderRadius: '4px',
  color: '#e0e0e0',
  cursor: 'pointer',
};

const groupStyle: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '4px',
  minWidth: '120px',
};

const labelStyle: CSSProperties = {
  fontSize: '11px',
  color: '#aaa',
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
};

const speedBtnRow: CSSProperties = {
  display: 'flex',
  gap: '4px',
};

const statsStyle: CSSProperties = {
  display: 'flex',
  gap: '16px',
  padding: '6px 10px',
  background: '#0d0d12',
  borderRadius: '4px',
  border: '1px solid #1a1a1a',
  fontSize: '12px',
  marginLeft: 'auto',
};

const timelineWrap: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  flex: '1 1 320px',
  minWidth: '240px',
  gap: '2px',
};

const badgeStyle = (live: boolean): CSSProperties => ({
  fontSize: '10px',
  padding: '1px 6px',
  borderRadius: '3px',
  background: live ? '#1a3a1a' : '#3a2a1a',
  color: live ? '#4ade80' : '#eab308',
  marginLeft: '6px',
});

export default function ControlPanel() {
  const paused = useModelStore((s) => s.paused);
  const speed = useModelStore((s) => s.speed);
  const currentStep = useModelStore((s) => s.currentStep);
  const beta = useModelStore((s) => s.beta);
  const connected = useModelStore((s) => s.connected);
  const knowledgeGraph = useModelStore((s) => s.knowledgeGraph);
  const history = useModelStore((s) => s.history);
  const playbackIndex = useModelStore((s) => s.playbackIndex);
  const sendCommand = useModelStore((s) => s.sendCommand);
  const setPlaybackIndex = useModelStore((s) => s.setPlaybackIndex);

  const inReplay = playbackIndex >= 0;
  const maxIdx = Math.max(0, history.length - 1);
  const sliderValue = inReplay ? playbackIndex : maxIdx;

  return (
    <div style={containerStyle}>
      {/* Playback buttons */}
      <div style={{ display: 'flex', gap: '8px' }}>
        <button
          type="button"
          style={btnStyle}
          aria-label={paused ? 'Play stream' : 'Pause stream'}
          onClick={() => sendCommand({ type: paused ? 'resume' : 'pause' })}
        >
          {paused ? '▶ Play' : '⏸ Pause'}
        </button>
        <button
          type="button"
          style={btnStyle}
          aria-label="Step forward one snapshot"
          onClick={() => sendCommand({ type: 'step' })}
          disabled={!paused}
          title={paused ? 'Advance one step' : 'Pause first to single-step'}
        >
          ⏭ Step
        </button>
      </div>

      {/* Speed: discrete options (toggle group) */}
      <div style={groupStyle} role="group" aria-label="Playback speed">
        <div style={{ ...labelStyle, display: 'flex', justifyContent: 'space-between' }}>
          <span>Speed</span>
          <span style={{ color: '#e0e0e0' }} aria-hidden="true">
            {speed}x
          </span>
        </div>
        <div style={speedBtnRow}>
          {SPEED_OPTIONS.map((opt) => {
            const active = speed === opt;
            return (
              <button
                key={opt}
                type="button"
                style={{
                  ...btnStyle,
                  padding: '4px 8px',
                  background: active ? '#2a3a4a' : '#1a1a22',
                  borderColor: active ? '#3b82f6' : '#333',
                }}
                aria-label={`Set playback speed to ${opt}x`}
                aria-pressed={active}
                onClick={() => sendCommand({ type: 'set_speed', speed: opt })}
              >
                {opt}x
              </button>
            );
          })}
        </div>
      </div>

      {/* Timeline slider */}
      <div style={timelineWrap}>
        <div style={{ ...labelStyle, display: 'flex', justifyContent: 'space-between' }}>
          <span>
            <label htmlFor="timeline-slider" style={{ cursor: 'pointer' }}>
              Timeline
            </label>
            <span
              role="status"
              aria-live="polite"
              aria-label={inReplay ? 'Replay mode' : 'Live mode'}
              style={badgeStyle(!inReplay)}
            >
              {inReplay ? 'REPLAY' : 'LIVE'}
            </span>
          </span>
          <span style={{ color: '#e0e0e0' }} aria-hidden="true">
            {inReplay ? `${playbackIndex} / ${maxIdx}` : `latest (${maxIdx})`}
          </span>
        </div>
        <input
          id="timeline-slider"
          type="range"
          min={0}
          max={maxIdx}
          step={1}
          value={sliderValue}
          aria-valuemin={0}
          aria-valuemax={maxIdx}
          aria-valuenow={sliderValue}
          aria-valuetext={
            inReplay ? `Replaying step ${playbackIndex} of ${maxIdx}` : `Live, latest step ${maxIdx}`
          }
          onChange={(e) => setPlaybackIndex(parseInt(e.target.value, 10))}
          style={{ width: '100%' }}
        />
      </div>

      {/* Live stats */}
      <div style={statsStyle} aria-label="System stats" role="group">
        <Stat label="Step" value={String(currentStep)} />
        <Stat label="β" value={beta.toFixed(3)} />
        <Stat label="KG Nodes" value={String(knowledgeGraph.nodes.length)} />
        <Stat label="Status" value={connected ? 'Connected' : 'Disconnected'} />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
      <span style={{ color: '#9ca3af', fontSize: '10px' }}>{label}</span>
      <span style={{ color: '#e0e0e0' }}>{value}</span>
    </div>
  );
}
