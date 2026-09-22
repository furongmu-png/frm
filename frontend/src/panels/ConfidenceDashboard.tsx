// frontend/src/panels/ConfidenceDashboard.tsx
// 置信度仪表盘 —— 以半圆仪表显示元认知置信度、模式与平均不确定性。
// 元认知 (meta_cognition) 未启用时显示占位提示。

import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';
import type { Snapshot } from '../types';

type CogMode = 'explore' | 'exploit' | 'safe' | 'balanced';

interface MetaCognition {
  confidence?: number;
  mode?: CogMode;
  mean_uncertainty?: number;
  [k: string]: unknown;
}

interface CognitiveUpgrades {
  meta_cognition?: MetaCognition;
  [k: string]: unknown;
}

interface SnapshotMetadata {
  cognitive_upgrades?: CognitiveUpgrades;
}

function readMetadata(snap: Snapshot | null): SnapshotMetadata | undefined {
  if (!snap) return undefined;
  // 单层窄转换: Snapshot.metadata 已在 types.ts 中声明，无需 `as unknown as`。
  return snap.metadata as SnapshotMetadata | undefined;
}

const wrapStyle: CSSProperties = {
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  padding: '8px 12px',
  gap: '10px',
};

const titleStyle: CSSProperties = {
  fontSize: '12px',
  color: '#aaa',
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
  borderBottom: '1px solid #1a1a1a',
  paddingBottom: '4px',
};

const emptyStyle: CSSProperties = {
  flex: 1,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: '#9ca3af',
  fontSize: '13px',
};

const gaugeWrapStyle: CSSProperties = {
  position: 'relative',
  width: '220px',
  height: '110px',
  margin: '0 auto',
};

// 模式 → 颜色映射 (explore=蓝, exploit=绿, safe=红, balanced=灰)。
const MODE_COLORS: Record<CogMode, string> = {
  explore: '#3b82f6',
  exploit: '#4ade80',
  safe: '#ef4444',
  balanced: '#9ca3af',
};

function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
  const a = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
}

// 绘制从 startDeg 到 endDeg 的圆弧路径 (顺时针)。
function arcPath(cx: number, cy: number, r: number, startDeg: number, endDeg: number): string {
  const start = polarToCartesian(cx, cy, r, endDeg);
  const end = polarToCartesian(cx, cy, r, startDeg);
  const largeArc = endDeg - startDeg <= 180 ? 0 : 1;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y}`;
}

export default function ConfidenceDashboard() {
  const snapshot = useModelStore((s) => s.snapshot);
  const meta = readMetadata(snapshot)?.cognitive_upgrades?.meta_cognition;

  if (!snapshot || !meta) {
    return (
      <div style={wrapStyle}>
        <div style={titleStyle}>置信度仪表盘</div>
        <div style={emptyStyle}>
          {snapshot ? '元认知未启用' : '等待数据...'}
        </div>
      </div>
    );
  }

  // 本面板仅订阅 snapshot，不再订阅 history，以避免回放期间不必要的重渲染。

  // typeof === 'number' 不能排除 NaN；渲染数字时须同时校验 Number.isFinite，
  // 否则后端返回 NaN 会得到 "NaN%" 的无效显示。
  const confidence =
    typeof meta.confidence === 'number' && Number.isFinite(meta.confidence)
      ? Math.max(0, Math.min(100, meta.confidence))
      : 0;
  const mode: CogMode = meta.mode ?? 'balanced';
  const modeColor = MODE_COLORS[mode] ?? '#9ca3af';
  const meanUnc =
    typeof meta.mean_uncertainty === 'number' && Number.isFinite(meta.mean_uncertainty)
      ? Math.max(0, Math.min(1, meta.mean_uncertainty))
      : 0;

  // 半圆仪表: 0% 在左 (180°)，100% 在右 (0°)。置信度越高，填充弧越长。
  const angle = 180 - (confidence / 100) * 180;
  const cx = 110;
  const cy = 100;
  const r = 90;

  return (
    <div style={wrapStyle}>
      <div style={titleStyle}>置信度仪表盘</div>

      <div style={gaugeWrapStyle}>
        <svg
          viewBox="0 0 220 110"
          role="img"
          aria-label={`置信度仪表: ${confidence.toFixed(1)}%, 模式 ${mode}`}
          style={{ width: '100%', height: '100%', overflow: 'visible' }}
        >
          {/* 背景弧 (整段半圆) */}
          <path
            d={arcPath(cx, cy, r, 0, 180)}
            fill="none"
            stroke="#1a1a22"
            strokeWidth="10"
            strokeLinecap="round"
          />
          {/* 置信度填充弧 */}
          <path
            d={arcPath(cx, cy, r, angle, 180)}
            fill="none"
            stroke={modeColor}
            strokeWidth="10"
            strokeLinecap="round"
          />
        </svg>
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'flex-end',
            paddingBottom: '4px',
          }}
        >
          <div style={{ fontSize: '24px', color: '#e0e0e0', fontWeight: 600 }}>
            {confidence.toFixed(1)}%
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'center' }}>
        <span
          style={{
            fontSize: '12px',
            padding: '2px 10px',
            borderRadius: '3px',
            background: `${modeColor}22`,
            color: modeColor,
            border: `1px solid ${modeColor}66`,
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
          }}
        >
          {mode}
        </span>
      </div>

      <div>
        <div
          style={{
            fontSize: '11px',
            color: '#aaa',
            display: 'flex',
            justifyContent: 'space-between',
          }}
        >
          <span>平均不确定性</span>
          <span style={{ color: '#e0e0e0' }}>{meanUnc.toFixed(3)}</span>
        </div>
        <div
          style={{
            height: '8px',
            background: '#1a1a22',
            borderRadius: '2px',
            overflow: 'hidden',
            marginTop: '4px',
          }}
        >
          <div
            style={{
              height: '100%',
              width: `${(meanUnc * 100).toFixed(1)}%`,
              background: '#f97316',
            }}
          />
        </div>
      </div>
    </div>
  );
}
