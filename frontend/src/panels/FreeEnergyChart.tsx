// frontend/src/panels/FreeEnergyChart.tsx
// Time-series chart of free energy and prediction error.
//
// Shows the last 500 points from the store's `history` array, with
// automatic X-axis scrolling (latest on the right). When in replay
// mode (playbackIndex >= 0), a ReferenceLine marks the replayed step.

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';

const MAX_POINTS = 500;

const emptyStyle: CSSProperties = {
  height: '100%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: '#9ca3af',
  fontSize: '13px',
};

export default function FreeEnergyChart() {
  const history = useModelStore((s) => s.history);
  const playbackIndex = useModelStore((s) => s.playbackIndex);

  if (history.length === 0) {
    return <div style={emptyStyle}>Waiting for data...</div>;
  }

  // Show only the last MAX_POINTS; slice preserves the original order.
  const start = Math.max(0, history.length - MAX_POINTS);
  const slice = history.slice(start);
  // Plot only the fields the chart needs (keeps Recharts lean).
  const data = slice.map((s) => ({
    step: s.step,
    free_energy: s.free_energy,
    prediction_error: s.prediction_error,
  }));

  // X-axis domain: auto-scroll to follow the latest step.
  const minStep = data[0]?.step ?? 0;
  const maxStep = data[data.length - 1]?.step ?? 0;

  // Reference line for the replayed step (if in replay mode and
  // the replayed step falls within the visible window).
  const replayStep =
    playbackIndex >= 0 && history[playbackIndex]
      ? history[playbackIndex].step
      : null;
  const showReplayRef =
    replayStep !== null && replayStep >= minStep && replayStep <= maxStep;

  return (
    <div
      style={{ height: '100%', width: '100%' }}
      role="figure"
      aria-label={`Free energy and prediction error time series chart, showing ${data.length} steps from step ${minStep} to step ${maxStep}${showReplayRef && replayStep !== null ? `, replay at step ${replayStep}` : ''}`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid stroke="#1a1a1a" strokeDasharray="3 3" />
          <XAxis
            dataKey="step"
            stroke="#9ca3af"
            tick={{ fontSize: 11, fill: '#9ca3af' }}
            domain={[minStep, maxStep]}
            type="number"
            allowDecimals={false}
          />
          <YAxis stroke="#9ca3af" tick={{ fontSize: 11, fill: '#9ca3af' }} />
          <Tooltip
            contentStyle={{
              background: '#111',
              border: '1px solid #333',
              borderRadius: '4px',
              fontSize: '12px',
            }}
            labelStyle={{ color: '#aaa' }}
            labelFormatter={(label) => `step ${label}`}
          />
          <Legend wrapperStyle={{ fontSize: '11px' }} />
          <Line
            type="monotone"
            dataKey="free_energy"
            stroke="#3b82f6"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="prediction_error"
            stroke="#f97316"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          {showReplayRef && replayStep !== null && (
            <ReferenceLine
              x={replayStep}
              stroke="#eab308"
              strokeDasharray="4 2"
              label={{ value: 'replay', fontSize: 10, fill: '#eab308', position: 'top' }}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
