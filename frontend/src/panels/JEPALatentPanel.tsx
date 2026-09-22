// frontend/src/panels/JEPALatentPanel.tsx
// JEPA 隐空间面板 —— 展示在线预测 vs 目标表示的对齐过程。
// 数据来源: snapshot.metadata.jpa。
// 在线预测与目标表示的 3D 投影叠加显示，配对齐度与 jepa_error 曲线。

import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';
import type { Snapshot } from '../types';

interface JEPAMetadata {
  enabled?: boolean;
  lambda_jepa?: number;
  jepa_error?: number;
  alignment?: number;
  combined_free_energy?: number;
  error_trend?: number[];
  online_3d?: number[];
  target_3d?: number[];
  error?: string;
}

interface MetadataWithJepa {
  jepa?: JEPAMetadata;
}

function readJepa(snap: Snapshot | null): JEPAMetadata | undefined {
  if (!snap?.metadata) return undefined;
  const md = snap.metadata as MetadataWithJepa;
  return md.jepa;
}

const wrapStyle: CSSProperties = {
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  padding: '8px 12px',
  gap: '8px',
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

const summaryStyle: CSSProperties = {
  display: 'flex',
  gap: '16px',
  fontSize: '11px',
  color: '#aaa',
  flexWrap: 'wrap',
};

const plotStyle: CSSProperties = {
  flex: 1,
  minHeight: '180px',
  background: '#0a0a0f',
  border: '1px solid #1a1a22',
  borderRadius: '4px',
  position: 'relative',
};

export default function JEPALatentPanel() {
  const snapshot = useModelStore((s) => s.snapshot);
  const jepa = readJepa(snapshot);

  if (!snapshot || !jepa) {
    return (
      <div style={wrapStyle}>
        <div style={titleStyle}>JEPA · 隐空间预测</div>
        <div style={emptyStyle}>
          {snapshot ? 'JEPA 模块未启用' : '等待数据...'}
        </div>
      </div>
    );
  }

  const online3d = jepa.online_3d ?? [0, 0, 0];
  const target3d = jepa.target_3d ?? [0, 0, 0];
  const alignment = jepa.alignment ?? 0;
  const jepaError = jepa.jepa_error ?? 0;
  const trend = jepa.error_trend ?? [];

  // 将 3D 投影到 2D 画布坐标（取前两维，居中缩放）
  const scale = 60;
  const cx = 100;
  const cy = 80;
  const onlineX = cx + online3d[0] * scale;
  const onlineY = cy - online3d[1] * scale;
  const targetX = cx + target3d[0] * scale;
  const targetY = cy - target3d[1] * scale;

  // 对齐度颜色：高对齐=绿，低对齐=红
  const alignColor = Math.abs(alignment) > 0.7 ? '#4ade80' : Math.abs(alignment) > 0.4 ? '#fbbf24' : '#ef4444';

  return (
    <div style={wrapStyle}>
      <div style={titleStyle}>JEPA · 隐空间预测</div>

      <div style={summaryStyle}>
        <span>
          λ_jepa: <span style={{ color: '#e0e0e0' }}>{jepa.lambda_jepa?.toFixed(2) ?? '—'}</span>
        </span>
        <span>
          jepa_error: <span style={{ color: '#fbbf24' }}>{jepaError.toFixed(4)}</span>
        </span>
        <span>
          对齐度: <span style={{ color: alignColor }}>{alignment.toFixed(4)}</span>
        </span>
        {jepa.combined_free_energy != null && (
          <span>
            合并自由能: <span style={{ color: '#60a5fa' }}>{jepa.combined_free_energy.toFixed(4)}</span>
          </span>
        )}
      </div>

      <div style={plotStyle}>
        <svg width="100%" height="100%" viewBox="0 0 200 160">
          {/* 坐标轴 */}
          <line x1={cx} y1={0} x2={cx} y2={160} stroke="#1a1a22" strokeWidth="0.5" />
          <line x1={0} y1={cy} x2={200} y2={cy} stroke="#1a1a22" strokeWidth="0.5" />

          {/* 连接线：在线→目标 */}
          <line
            x1={onlineX}
            y1={onlineY}
            x2={targetX}
            y2={targetY}
            stroke={alignColor}
            strokeWidth="1"
            strokeDasharray="2,2"
            opacity="0.6"
          />

          {/* 目标表示（EMA，蓝色圆） */}
          <circle cx={targetX} cy={targetY} r="4" fill="#60a5fa" />
          <text x={targetX + 6} y={targetY - 4} fontSize="8" fill="#60a5fa">target</text>

          {/* 在线预测（橙色方块） */}
          <rect x={onlineX - 4} y={onlineY - 4} width="8" height="8" fill="#fbbf24" />
          <text x={onlineX + 6} y={onlineY + 4} fontSize="8" fill="#fbbf24">online</text>
        </svg>
      </div>

      {/* 误差趋势迷你图 */}
      {trend.length > 1 && (
        <div style={{ height: '50px', background: '#0a0a0f', border: '1px solid #1a1a22', borderRadius: '4px', padding: '4px' }}>
          <div style={{ fontSize: '9px', color: '#6b7280', marginBottom: '2px' }}>jepa_error 趋势</div>
          <svg width="100%" height="38">
            <polyline
              fill="none"
              stroke="#fbbf24"
              strokeWidth="1"
              points={trend
                .map((v, i) => `${(i / (trend.length - 1)) * 100},${38 - (v / Math.max(...trend, 1)) * 34}`)
                .join(' ')}
            />
          </svg>
        </div>
      )}

      {jepa.error && (
        <div style={{ color: '#ef4444', fontSize: '10px' }}>{jepa.error}</div>
      )}
    </div>
  );
}
