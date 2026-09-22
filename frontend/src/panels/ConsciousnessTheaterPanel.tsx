// frontend/src/panels/ConsciousnessTheaterPanel.tsx
// GWT 意识剧场面板 —— 全局工作空间竞争 + 整合信息 Φ。
// 数据来源: snapshot.metadata.gwt，由 HierarchicalZeroDataModel
// 的 _run_gwt_cycle 每步写入，形如：
//   {
//     winner: string | null,
//     probabilities: { [module: string]: number },
//     broadcast_confidence: number,
//     global_timestamp: number,
//     phi: number,
//     is_conscious: boolean,
//     phi_history: number[],
//     selector: { recent_winners, winner_distribution, inhibited_modules, temperature, ... },
//     broadcaster: { last_winner, last_confidence, registered_modules, last_broadcast_3d, ... },
//     error?: string,
//   }
//
// 三块可视化:
//   1. 注意请求强度条形图 — 各模块竞争概率，胜者高亮。
//   2. 广播内容雷达图 — 胜者向量前 6 维 + 置信度扇区。
//   3. Φ 值时间曲线 — 标注意识阈值与高低意识事件。

import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';
import type { Snapshot } from '../types';

// ------------------------------------------------------------------ //
// 类型（窄转换：Snapshot.metadata 已声明带索引签名）
// ------------------------------------------------------------------ //

interface GWTMetadata {
  winner?: string | null;
  probabilities?: Record<string, number>;
  broadcast_confidence?: number;
  global_timestamp?: number;
  phi?: number;
  is_conscious?: boolean;
  phi_history?: number[];
  selector?: {
    temperature?: number;
    step?: number;
    inhibited_modules?: string[];
    recent_winners?: string[];
    winner_distribution?: Record<string, number>;
  };
  broadcaster?: {
    alignment_lr?: number;
    n_receivers?: number;
    registered_modules?: string[];
    last_winner?: string | null;
    last_confidence?: number;
    last_broadcast_3d?: number[];
  };
  error?: string;
}

function readGwt(snap: Snapshot | null): GWTMetadata | undefined {
  if (!snap?.metadata) return undefined;
  const md = snap.metadata as { gwt?: GWTMetadata };
  return md.gwt;
}

// ------------------------------------------------------------------ //
// 样式
// ------------------------------------------------------------------ //

const wrapStyle: CSSProperties = {
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  padding: '8px 12px',
  gap: '8px',
  overflow: 'auto',
};

const titleStyle: CSSProperties = {
  fontSize: '12px',
  color: '#aaa',
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
  borderBottom: '1px solid #1a1a1a',
  paddingBottom: '4px',
  flexShrink: 0,
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
  flexShrink: 0,
};

const cardStyle: CSSProperties = {
  background: '#0a0a0f',
  border: '1px solid #1a1a22',
  borderRadius: '4px',
  padding: '6px 8px',
};

const cardTitleStyle: CSSProperties = {
  fontSize: '9px',
  color: '#6b7280',
  marginBottom: '4px',
  textTransform: 'uppercase',
  letterSpacing: '0.4px',
};

// 模块颜色调色板（循环）
const MODULE_COLORS = [
  '#60a5fa', '#fbbf24', '#4ade80', '#f87171',
  '#a78bfa', '#34d399', '#fb923c', '#f472b6',
];

function moduleColor(_name: string, idx: number): string {
  return MODULE_COLORS[idx % MODULE_COLORS.length];
}

// ------------------------------------------------------------------ //
// 子组件 1：注意请求强度条形图
// ------------------------------------------------------------------ //

function AttentionBars({
  probs,
  winner,
}: {
  probs: Record<string, number>;
  winner: string | null | undefined;
}) {
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) {
    return <div style={{ fontSize: '10px', color: '#6b7280' }}>无竞争候选</div>;
  }
  const maxProb = Math.max(...entries.map(([, p]) => p), 1e-8);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
      {entries.map(([name, p], i) => {
        const isWinner = name === winner;
        const barWidth = (p / maxProb) * 100;
        const pct = (p * 100).toFixed(1);
        return (
          <div
            key={name}
            style={{
              display: 'grid',
              gridTemplateColumns: '90px 1fr 36px',
              gap: '6px',
              alignItems: 'center',
              fontSize: '10px',
            }}
          >
            <span
              style={{
                color: isWinner ? '#fbbf24' : '#9ca3af',
                fontWeight: isWinner ? 600 : 400,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
              title={name}
            >
              {isWinner ? '★ ' : ''}
              {name}
            </span>
            <div
              style={{
                background: '#1a1a22',
                height: '10px',
                borderRadius: '2px',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  width: `${barWidth}%`,
                  height: '100%',
                  background: isWinner ? '#fbbf24' : moduleColor(name, i),
                  opacity: isWinner ? 1 : 0.7,
                }}
              />
            </div>
            <span style={{ color: '#d1d5db', fontFamily: 'monospace', textAlign: 'right' }}>
              {pct}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ------------------------------------------------------------------ //
// 子组件 2：广播内容雷达图
// ------------------------------------------------------------------ //

function BroadcastRadar({
  vector,
  confidence,
  winner,
}: {
  vector: number[];
  confidence: number;
  winner: string | null | undefined;
}) {
  // 取前 6 维做雷达图，不足补 0
  const dims = 6;
  const vals = Array.from({ length: dims }, (_, i) => vector[i] ?? 0);
  const maxAbs = Math.max(...vals.map(Math.abs), 1e-8);
  const cx = 70;
  const cy = 70;
  const r = 50;
  // 雷达顶点
  const points = vals.map((v, i) => {
    const angle = (i / dims) * 2 * Math.PI - Math.PI / 2;
    const rad = (Math.abs(v) / maxAbs) * r * (confidence > 0 ? confidence : 0.2);
    return [cx + rad * Math.cos(angle), cy + rad * Math.sin(angle)];
  });
  const polygon = points.map((p) => p.join(',')).join(' ');

  return (
    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
      <svg width="140" height="140" viewBox="0 0 140 140">
        {/* 同心圆 */}
        {[0.25, 0.5, 0.75, 1].map((f) => (
          <circle
            key={f}
            cx={cx}
            cy={cy}
            r={r * f}
            fill="none"
            stroke="#1a1a22"
            strokeWidth="0.5"
          />
        ))}
        {/* 轴线 */}
        {vals.map((_, i) => {
          const angle = (i / dims) * 2 * Math.PI - Math.PI / 2;
          return (
            <line
              key={i}
              x1={cx}
              y1={cy}
              x2={cx + r * Math.cos(angle)}
              y2={cy + r * Math.sin(angle)}
              stroke="#1a1a22"
              strokeWidth="0.5"
            />
          );
        })}
        {/* 广播多边形 */}
        <polygon
          points={polygon}
          fill="#60a5fa"
          fillOpacity={0.2 + confidence * 0.4}
          stroke="#60a5fa"
          strokeWidth="1"
        />
        {/* 顶点 */}
        {points.map((p, i) => (
          <circle key={i} cx={p[0]} cy={p[1]} r="1.5" fill="#fbbf24" />
        ))}
      </svg>
      <div style={{ fontSize: '10px', color: '#9ca3af', flex: 1 }}>
        <div style={{ color: '#d1d5db', marginBottom: '2px' }}>
          胜者: <span style={{ color: '#fbbf24' }}>{winner ?? '—'}</span>
        </div>
        <div style={{ marginBottom: '2px' }}>
          置信度: <span style={{ color: '#4ade80' }}>{(confidence * 100).toFixed(1)}%</span>
        </div>
        <div style={{ fontFamily: 'monospace', fontSize: '9px' }}>
          v=[{vals.map((v) => v.toFixed(2)).join(', ')}]
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ //
// 子组件 3：Φ 值时间曲线
// ------------------------------------------------------------------ //

function PhiCurve({
  history,
  isConscious,
  sleepThreshold,
}: {
  history: number[];
  phi: number;
  isConscious: boolean;
  sleepThreshold: number;
}) {
  const w = 280;
  const h = 60;
  const pad = 6;
  if (history.length < 2) {
    return <div style={{ fontSize: '10px', color: '#6b7280' }}>等待 Φ 历史…</div>;
  }
  const maxPhi = Math.max(...history, sleepThreshold * 2, 0.1);
  const pts = history.map((v, i) => {
    const x = pad + (i / (history.length - 1)) * (w - 2 * pad);
    const y = h - pad - (v / maxPhi) * (h - 2 * pad);
    return `${x},${y}`;
  });
  // 阈值线
  const threshY = h - pad - (sleepThreshold / maxPhi) * (h - 2 * pad);
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      {/* 阈值线 */}
      <line
        x1={pad}
        y1={threshY}
        x2={w - pad}
        y2={threshY}
        stroke="#ef4444"
        strokeWidth="0.5"
        strokeDasharray="3,2"
        opacity="0.6"
      />
      <text x={pad} y={threshY - 2} fontSize="7" fill="#ef4444">
        sleep={sleepThreshold.toFixed(2)}
      </text>
      {/* Φ 曲线 */}
      <polyline
        fill="none"
        stroke={isConscious ? '#4ade80' : '#6b7280'}
        strokeWidth="1.2"
        points={pts.join(' ')}
      />
      {/* 当前点 */}
      {pts.length > 0 && (
        <circle
          cx={pts[pts.length - 1].split(',')[0]}
          cy={pts[pts.length - 1].split(',')[1]}
          r="2"
          fill={isConscious ? '#4ade80' : '#ef4444'}
        />
      )}
    </svg>
  );
}

// ------------------------------------------------------------------ //
// 主组件
// ------------------------------------------------------------------ //

export default function ConsciousnessTheaterPanel() {
  const snapshot = useModelStore((s) => s.snapshot);
  const gwt = readGwt(snapshot);

  if (!snapshot || !gwt) {
    return (
      <div style={wrapStyle}>
        <div style={titleStyle}>GWT · 意识剧场</div>
        <div style={emptyStyle}>
          {snapshot ? 'GWT 模块未启用' : '等待数据...'}
        </div>
      </div>
    );
  }

  if (gwt.error) {
    return (
      <div style={wrapStyle}>
        <div style={titleStyle}>GWT · 意识剧场</div>
        <div style={{ ...emptyStyle, color: '#ef4444' }}>{gwt.error}</div>
      </div>
    );
  }

  const probs = gwt.probabilities ?? {};
  const winner = gwt.winner;
  const phi = gwt.phi ?? 0;
  const isConscious = gwt.is_conscious ?? false;
  const phiHistory = gwt.phi_history ?? [];
  const broadcastConfidence = gwt.broadcast_confidence ?? 0;
  const selector = gwt.selector;
  const broadcaster = gwt.broadcaster;
  const broadcastVec = broadcaster?.last_broadcast_3d ?? [0, 0, 0];
  // 雷达图需要更多维度，从 broadcast_3d 取 3 维 + 用 confidence 填充后 3 维
  const radarVec = [
    ...broadcastVec,
    broadcastConfidence,
    broadcastConfidence * 0.8,
    broadcastConfidence * 0.6,
  ];
  // sleep_threshold 不在 metadata 中暴露，用经验值 0.1（与后端默认一致）
  const sleepThreshold = 0.1;

  const recentWinners = selector?.recent_winners ?? [];
  const winnerDist = selector?.winner_distribution ?? {};
  const inhibited = selector?.inhibited_modules ?? [];

  return (
    <div style={wrapStyle}>
      <div style={titleStyle}>GWT · 意识剧场</div>

      {/* 汇总 */}
      <div style={summaryStyle}>
        <span>
          Φ: <span style={{ color: isConscious ? '#4ade80' : '#ef4444' }}>{phi.toFixed(4)}</span>
        </span>
        <span>
          意识: <span style={{ color: isConscious ? '#4ade80' : '#9ca3af' }}>
            {isConscious ? '清醒' : '睡眠'}
          </span>
        </span>
        <span>
          胜者: <span style={{ color: '#fbbf24' }}>{winner ?? '—'}</span>
        </span>
        <span>
          时间步: <span style={{ color: '#d1d5db' }}>{gwt.global_timestamp ?? 0}</span>
        </span>
        {selector?.temperature != null && (
          <span>
            T: <span style={{ color: '#d1d5db' }}>{selector.temperature.toFixed(2)}</span>
          </span>
        )}
      </div>

      {/* 1. 注意竞争条形图 */}
      <div style={cardStyle}>
        <div style={cardTitleStyle}>注意请求强度 · 竞争概率</div>
        <AttentionBars probs={probs} winner={winner} />
      </div>

      {/* 2. 广播内容雷达图 */}
      <div style={cardStyle}>
        <div style={cardTitleStyle}>广播内容 · 胜者向量</div>
        <BroadcastRadar
          vector={radarVec}
          confidence={broadcastConfidence}
          winner={winner}
        />
      </div>

      {/* 3. Φ 值时间曲线 */}
      <div style={cardStyle}>
        <div style={cardTitleStyle}>整合信息 Φ · 时间曲线</div>
        <PhiCurve
          history={phiHistory}
          phi={phi}
          isConscious={isConscious}
          sleepThreshold={sleepThreshold}
        />
      </div>

      {/* 4. 意识流：胜者历史 */}
      {recentWinners.length > 0 && (
        <div style={cardStyle}>
          <div style={cardTitleStyle}>意识流 · 近期胜者</div>
          <div
            style={{
              display: 'flex',
              gap: '4px',
              flexWrap: 'wrap',
              fontSize: '9px',
              fontFamily: 'monospace',
            }}
          >
            {recentWinners.map((w, i) => (
              <span
                key={`${i}-${w}`}
                style={{
                  padding: '1px 4px',
                  background: '#1a1a22',
                  borderRadius: '2px',
                  color: '#9ca3af',
                }}
              >
                {w}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 5. 胜者分布统计 + 抑制模块 */}
      <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
        <div style={{ ...cardStyle, flex: 1 }}>
          <div style={cardTitleStyle}>胜者分布</div>
          <div style={{ fontSize: '9px', color: '#9ca3af' }}>
            {Object.keys(winnerDist).length === 0
              ? '—'
              : Object.entries(winnerDist)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 5)
                  .map(([n, f]) => `${n}: ${(f * 100).toFixed(0)}%`)
                  .join('  ')}
          </div>
        </div>
        <div style={{ ...cardStyle, flex: 1 }}>
          <div style={cardTitleStyle}>抑制模块</div>
          <div style={{ fontSize: '9px', color: '#9ca3af' }}>
            {inhibited.length === 0 ? '—' : inhibited.join(', ')}
          </div>
        </div>
      </div>

      {/* 6. 广播器统计 */}
      {broadcaster && (
        <div style={cardStyle}>
          <div style={cardTitleStyle}>广播器</div>
          <div style={{ fontSize: '10px', color: '#9ca3af', display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            <span>对齐率: <span style={{ color: '#d1d5db' }}>{broadcaster.alignment_lr?.toFixed(2) ?? '—'}</span></span>
            <span>接收者: <span style={{ color: '#d1d5db' }}>{broadcaster.n_receivers ?? 0}</span></span>
            <span>置信度: <span style={{ color: '#4ade80' }}>{((broadcaster.last_confidence ?? 0) * 100).toFixed(1)}%</span></span>
          </div>
        </div>
      )}
    </div>
  );
}
