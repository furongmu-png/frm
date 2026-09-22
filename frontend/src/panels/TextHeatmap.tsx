// frontend/src/panels/TextHeatmap.tsx
// Text-stream heatmap: prediction error vs. character position over time.
//
// X axis: text_pos buckets (100 chars each, up to 100k chars).
// Y axis: snapshot history index (most recent 200 rows, newest at bottom).
// Color: HSL interpolation — blue (low error) → red (high error),
//        with alpha 0.7.
// Interactions:
//   - Click a cell → setPlaybackIndex(historyIndex) jumps all panels
//     to that snapshot.
//   - Hover → tooltip showing step, error, text position.

import { useEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';

const MAX_ROWS = 200;          // most recent 200 text snapshots
const BUCKET_CHARS = 100;      // chars per X bucket
const MAX_BUCKETS = 1000;      // 1000 * 100 = 100k chars max range

const wrapStyle: CSSProperties = {
  height: '100%',
  width: '100%',
  display: 'flex',
  flexDirection: 'column',
};

const canvasStyle: CSSProperties = {
  flex: 1,
  display: 'block',
  cursor: 'crosshair',
  minHeight: 0,
};

const tooltipStyle: CSSProperties = {
  position: 'fixed',
  pointerEvents: 'none',
  background: '#111',
  border: '1px solid #333',
  borderRadius: '4px',
  padding: '4px 8px',
  fontSize: '11px',
  color: '#e0e0e0',
  zIndex: 1000,
  whiteSpace: 'nowrap',
};

const emptyStyle: CSSProperties = {
  height: '100%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: '#9ca3af',
  fontSize: '13px',
};

/** Map a normalised error value [0,1] to an HSL color string. */
function errorToColor(t: number, alpha = 0.7): string {
  // t in [0,1]: 0=blue (240°), 1=red (0°). Hue decreases.
  const hue = (1 - t) * 240;
  return `hsla(${hue.toFixed(0)}, 80%, 55%, ${alpha})`;
}

interface HoverInfo {
  x: number;
  y: number;
  step: number;
  error: number;
  textPos: number;
  histIdx: number;
}

export default function TextHeatmap() {
  const history = useModelStore((s) => s.history);
  const setPlaybackIndex = useModelStore((s) => s.setPlaybackIndex);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hover, setHover] = useState<HoverInfo | null>(null);
  // Track history length to trigger redraw only when it changes.
  const [histLen, setHistLen] = useState(0);

  // Filter to text snapshots (text_pos !== -1) and take last MAX_ROWS.
  const rows = useMemo(() => {
    const textSnaps = history.filter((s) => (s.text_pos ?? -1) !== -1);
    const slice = textSnaps.slice(-MAX_ROWS);
    // Record the original history index for each row so clicks can
    // call setPlaybackIndex with the correct index.
    return slice.map((snap, i) => {
      // Find the index in the full history array.
      const fullIdx = history.indexOf(snap);
      return {
        snap,
        histIdx: fullIdx,
        localIdx: i,
      };
    });
  }, [history]);

  // Redraw the canvas whenever history grows.
  useEffect(() => {
    setHistLen(history.length);
  }, [history.length]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 600;
    const cssH = canvas.clientHeight || 200;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Background.
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, cssW, cssH);

    if (rows.length === 0) return;

    // Compute error range for normalisation.
    let maxErr = 0.001;
    for (const r of rows) {
      if (r.snap.prediction_error > maxErr) maxErr = r.snap.prediction_error;
    }

    const rowH = Math.max(1, cssH / MAX_ROWS);
    const colW = cssW / MAX_BUCKETS;

    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      const tp = r.snap.text_pos ?? 0;
      const bucket = Math.min(MAX_BUCKETS - 1, Math.floor(tp / BUCKET_CHARS));
      const t = Math.min(1, r.snap.prediction_error / maxErr);
      const y = cssH - (i + 1) * rowH; // newest at bottom
      ctx.fillStyle = errorToColor(t, 0.7);
      ctx.fillRect(bucket * colW, y, Math.max(1, colW), rowH);
    }

    // Axis labels.
    ctx.fillStyle = '#9ca3af';
    ctx.font = '10px monospace';
    ctx.fillText('0', 2, cssH - 2);
    ctx.fillText(`${(MAX_BUCKETS * BUCKET_CHARS / 1000).toFixed(0)}k chars`, cssW - 50, cssH - 2);
    ctx.save();
    ctx.translate(8, 12);
    ctx.fillText('newest', 0, 0);
    ctx.restore();
  }, [rows, histLen]);

  const handleMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || rows.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const y = e.clientY - rect.top;
    const cssH = canvas.clientHeight;
    const rowH = cssH / MAX_ROWS;
    const rowFromBottom = Math.floor((cssH - y) / rowH);
    const idx = rows.length - 1 - rowFromBottom;
    if (idx < 0 || idx >= rows.length) {
      setHover(null);
      return;
    }
    const r = rows[idx];
    setHover({
      x: e.clientX,
      y: e.clientY,
      step: r.snap.step,
      error: r.snap.prediction_error,
      textPos: r.snap.text_pos ?? -1,
      histIdx: r.histIdx,
    });
  };

  const handleClick = () => {
    if (hover && hover.histIdx >= 0) {
      setPlaybackIndex(hover.histIdx);
    }
  };

  if (rows.length === 0) {
    return (
      <div style={emptyStyle}>
        No text snapshots yet (run with --modality=text)
      </div>
    );
  }

  return (
    <div style={wrapStyle}>
      <canvas
        ref={canvasRef}
        style={canvasStyle}
        role="img"
        aria-label={`Text stream heatmap of prediction error over character position. ${rows.length} rows shown, color ranges from blue (low error) to red (high error). Click a cell to replay that snapshot.`}
        onMouseMove={handleMove}
        onMouseLeave={() => setHover(null)}
        onClick={handleClick}
      >
        Text stream heatmap of prediction error over character position. This
        visualization requires a canvas-capable browser. Color ranges from blue
        (low error) to red (high error); click a cell to replay that snapshot.
      </canvas>
      {hover && (
        <div style={{ ...tooltipStyle, left: hover.x + 12, top: hover.y + 12 }}>
          step {hover.step} · err {hover.error.toFixed(4)}
          <br />
          pos {hover.textPos} · idx {hover.histIdx}
          <br />
          <span style={{ color: '#aaa' }}>click to replay</span>
        </div>
      )}
    </div>
  );
}
