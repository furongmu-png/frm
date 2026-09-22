// frontend/src/panels/SelfModelPanel.tsx
// 统一自我模型 (IWSM) 面板 —— 自我图式 + 自我 Φ + 自传体记忆 + 反事实自我。
// 数据来源: snapshot.metadata.iwsm，由 HierarchicalZeroDataModel
// 的 _run_iwsm_cycle 每步写入，形如：
//   {
//     self_schema: {
//       attn_error, action_error, affect_error,
//       attn_correct, action_correct, self_consistency, step
//     },
//     self_schema_stats: {
//       step, attention_accuracy, action_accuracy, affect_error, self_consistency
//     },
//     phi_self: number,
//     phi_self_history: number[],   // 最近 20 步
//     is_sleeping: boolean,
//     autobiographical: { n_episodes: number },
//     counterfactual: {
//       narratives_generated: number,
//       last_regret: number,
//       last_narrative: string
//     },
//     error?: string,
//   }
//
// 五块可视化:
//   1. 自我 Φ 曲线 + 清醒/睡眠状态。
//   2. 自我图式准确率仪表（注意/动作/情感）。
//   3. 自我一致性综合得分（self_consistency）。
//   4. 反事实叙述展示（最近生成的"如果…则…"叙述 + regret 指标）。
//   5. 自传体记忆容量 + 步数计数。

import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';
import type { Snapshot } from '../types';

// ------------------------------------------------------------------ //
// 类型
// ------------------------------------------------------------------ //

interface SelfSchemaDiag {
  attn_error?: number;
  action_error?: number;
  affect_error?: number;
  attn_correct?: boolean;
  action_correct?: boolean;
  self_consistency?: number;
  step?: number;
}

interface SelfSchemaStats {
  step?: number;
  attention_accuracy?: number;
  action_accuracy?: number;
  affect_error?: number;
  self_consistency?: number;
}

interface AutobiographicalSummary {
  n_episodes?: number;
}

interface CounterfactualSummary {
  narratives_generated?: number;
  last_regret?: number;
  last_narrative?: string;
}

interface IWSMMetadata {
  self_schema?: SelfSchemaDiag;
  self_schema_stats?: SelfSchemaStats;
  phi_self?: number;
  phi_self_history?: number[];
  is_sleeping?: boolean;
  autobiographical?: AutobiographicalSummary;
  counterfactual?: CounterfactualSummary;
  error?: string;
}

function readIwsm(snap: Snapshot | null): IWSMMetadata | undefined {
  if (!snap?.metadata) return undefined;
  const md = snap.metadata as { iwsm?: IWSMMetadata };
  return md.iwsm;
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

const summaryStyle: CSSProperties = {
  display: 'flex',
  gap: '14px',
  fontSize: '11px',
  color: '#aaa',
  flexWrap: 'wrap',
  flexShrink: 0,
};

const metricGridStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(3, 1fr)',
  gap: '6px',
};

const metricBoxStyle: CSSProperties = {
  background: '#0d0d12',
  border: '1px solid #1a1a22',
  borderRadius: '3px',
  padding: '4px 6px',
  textAlign: 'center',
};

const metricLabelStyle: CSSProperties = {
  fontSize: '9px',
  color: '#6b7280',
  marginTop: '2px',
};

const narrativeStyle: CSSProperties = {
  fontSize: '10px',
  color: '#d1d5db',
  background: '#0d0d12',
  padding: '6px 8px',
  borderRadius: '3px',
  borderLeft: '3px solid #a78bfa',
  fontStyle: 'italic',
  lineHeight: 1.5,
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
};

// ------------------------------------------------------------------ //
// 子组件 1：自我 Φ 曲线
// ------------------------------------------------------------------ //

function PhiSelfCurve({
  history,
  isSleeping,
  sleepThreshold,
}: {
  history: number[];
  isSleeping: boolean;
  sleepThreshold: number;
}) {
  const w = 280;
  const h = 70;
  const pad = 8;
  if (history.length < 2) {
    return <div style={{ fontSize: '10px', color: '#6b7280' }}>等待 Φ_self 历史…</div>;
  }
  const maxPhi = Math.max(...history, sleepThreshold * 2, 0.1);
  const pts = history.map((v, i) => {
    const x = pad + (i / (history.length - 1)) * (w - 2 * pad);
    const y = h - pad - (Math.max(0, v) / maxPhi) * (h - 2 * pad);
    return `${x},${y}`;
  });
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
      {/* Φ_self 曲线 */}
      <polyline
        fill="none"
        stroke={isSleeping ? '#6b7280' : '#4ade80'}
        strokeWidth="1.5"
        points={pts.join(' ')}
      />
      {/* 当前点 */}
      {pts.length > 0 && (
        <circle
          cx={parseFloat(pts[pts.length - 1].split(',')[0])}
          cy={parseFloat(pts[pts.length - 1].split(',')[1])}
          r="2.5"
          fill={isSleeping ? '#9ca3af' : '#4ade80'}
        />
      )}
    </svg>
  );
}

// ------------------------------------------------------------------ //
// 子组件 2：自我图式准确率仪表
// ------------------------------------------------------------------ //

function AccuracyGauge({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  const pct = Math.min(100, Math.max(0, value * 100));
  return (
    <div style={metricBoxStyle}>
      <div style={{ position: 'relative', height: '32px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <svg width="60" height="32" viewBox="0 0 60 32">
          {/* 背景半圆 */}
          <path d="M 5 30 A 25 25 0 0 1 55 30" fill="none" stroke="#1a1a22" strokeWidth="4" />
          {/* 前景半圆（按百分比填充） */}
          <path
            d="M 5 30 A 25 25 0 0 1 55 30"
            fill="none"
            stroke={color}
            strokeWidth="4"
            strokeDasharray={`${(pct / 100) * 78.5} 78.5`}
            strokeLinecap="round"
          />
        </svg>
        <span
          style={{
            position: 'absolute',
            fontSize: '11px',
            fontWeight: 600,
            color,
            fontFamily: 'monospace',
          }}
        >
          {pct.toFixed(0)}%
        </span>
      </div>
      <div style={metricLabelStyle}>{label}</div>
    </div>
  );
}

// ------------------------------------------------------------------ //
// 主组件
// ------------------------------------------------------------------ //

export default function SelfModelPanel() {
  const snapshot = useModelStore((s) => s.snapshot);
  const iwsm = readIwsm(snapshot);

  if (!snapshot || !iwsm) {
    return (
      <div style={wrapStyle}>
        <div style={titleStyle}>IWSM · 统一自我模型</div>
        <div style={emptyStyle}>
          {snapshot ? 'IWSM 模块未启用' : '等待数据...'}
        </div>
      </div>
    );
  }

  if (iwsm.error) {
    return (
      <div style={wrapStyle}>
        <div style={titleStyle}>IWSM · 统一自我模型</div>
        <div style={{ ...emptyStyle, color: '#ef4444' }}>{iwsm.error}</div>
      </div>
    );
  }

  const phiSelf = iwsm.phi_self ?? 0;
  const isSleeping = iwsm.is_sleeping ?? false;
  const phiHistory = iwsm.phi_self_history ?? [];
  const sleepThreshold = 0.1;

  const stats = iwsm.self_schema_stats ?? {};
  const attnAcc = stats.attention_accuracy ?? 0;
  const actionAcc = stats.action_accuracy ?? 0;
  const affectErr = stats.affect_error ?? 0;
  const selfConsistency = stats.self_consistency ?? 0;

  const diag = iwsm.self_schema ?? {};
  const ab = iwsm.autobiographical ?? {};
  const cf = iwsm.counterfactual ?? {};
  const lastNarrative = cf.last_narrative ?? '';
  const lastRegret = cf.last_regret ?? 0;
  const narrativesGenerated = cf.narratives_generated ?? 0;
  const nEpisodes = ab.n_episodes ?? 0;

  return (
    <div style={wrapStyle}>
      <div style={titleStyle}>IWSM · 统一自我模型</div>

      {/* 汇总 */}
      <div style={summaryStyle}>
        <span>
          Φ_self:{' '}
          <span style={{ color: isSleeping ? '#9ca3af' : '#4ade80' }}>
            {phiSelf.toFixed(4)}
          </span>
        </span>
        <span>
          状态:{' '}
          <span style={{ color: isSleeping ? '#9ca3af' : '#4ade80' }}>
            {isSleeping ? '睡眠' : '清醒'}
          </span>
        </span>
        <span>
          自我一致性:{' '}
          <span style={{ color: selfConsistency > 0.5 ? '#4ade80' : '#fbbf24' }}>
            {(selfConsistency * 100).toFixed(1)}%
          </span>
        </span>
        {stats.step != null && (
          <span>
            步数: <span style={{ color: '#d1d5db' }}>{stats.step}</span>
          </span>
        )}
      </div>

      {/* 1. 自我 Φ 曲线 */}
      <div style={cardStyle}>
        <div style={cardTitleStyle}>自我 Φ · 整合信息时间曲线</div>
        <PhiSelfCurve
          history={phiHistory}
          isSleeping={isSleeping}
          sleepThreshold={sleepThreshold}
        />
        <div style={{ fontSize: '9px', color: '#6b7280', marginTop: '4px' }}>
          Φ_self 仅考虑与自我图式相关模块（GWT、元认知、本体感觉）。高值=强自我表征。
        </div>
      </div>

      {/* 2. 自我图式准确率仪表 */}
      <div style={cardStyle}>
        <div style={cardTitleStyle}>自我图式 · 预测准确率</div>
        <div style={metricGridStyle}>
          <AccuracyGauge label="注意预测" value={attnAcc} color="#60a5fa" />
          <AccuracyGauge label="动作预测" value={actionAcc} color="#fbbf24" />
          <AccuracyGauge label="1 - 情感误差" value={Math.max(0, 1 - affectErr)} color="#f472b6" />
        </div>
      </div>

      {/* 3. 单步诊断 + 自我一致性 */}
      <div style={cardStyle}>
        <div style={cardTitleStyle}>单步诊断 · 实际 vs 预测</div>
        <div
          style={{
            fontSize: '10px',
            color: '#9ca3af',
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '4px 12px',
          }}
        >
          <span>
            注意误差:{' '}
            <span style={{ color: '#60a5fa', fontFamily: 'monospace' }}>
              {(diag.attn_error ?? 0).toFixed(4)}
            </span>
            <span style={{ marginLeft: '4px', color: diag.attn_correct ? '#4ade80' : '#f87171' }}>
              {diag.attn_correct ? '✓' : '✗'}
            </span>
          </span>
          <span>
            动作误差:{' '}
            <span style={{ color: '#fbbf24', fontFamily: 'monospace' }}>
              {(diag.action_error ?? 0).toFixed(4)}
            </span>
            <span style={{ marginLeft: '4px', color: diag.action_correct ? '#4ade80' : '#f87171' }}>
              {diag.action_correct ? '✓' : '✗'}
            </span>
          </span>
          <span>
            情感误差:{' '}
            <span style={{ color: '#f472b6', fontFamily: 'monospace' }}>
              {(diag.affect_error ?? 0).toFixed(4)}
            </span>
          </span>
          <span>
            自我一致性:{' '}
            <span
              style={{
                color: selfConsistency > 0.5 ? '#4ade80' : '#fbbf24',
                fontFamily: 'monospace',
              }}
            >
              {selfConsistency.toFixed(4)}
            </span>
          </span>
        </div>
      </div>

      {/* 4. 反事实自我叙述 */}
      <div style={cardStyle}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '4px',
          }}
        >
          <span style={cardTitleStyle}>反事实自我 · "如果当时..."</span>
          <span
            style={{
              fontSize: '9px',
              padding: '1px 4px',
              background: lastRegret > 0 ? '#fbbf2422' : '#4ade8022',
              color: lastRegret > 0 ? '#fbbf24' : '#4ade80',
              borderRadius: '2px',
              border: `1px solid ${lastRegret > 0 ? '#fbbf2444' : '#4ade8044'}`,
            }}
          >
            regret: {lastRegret.toFixed(3)}
          </span>
        </div>
        {lastNarrative && lastNarrative.trim().length > 0 ? (
          <div style={narrativeStyle}>"{lastNarrative}"</div>
        ) : (
          <div style={{ fontSize: '10px', color: '#6b7280' }}>
            暂无反事实叙述（每 20 步生成一次）
          </div>
        )}
        <div
          style={{
            marginTop: '4px',
            fontSize: '9px',
            color: '#6b7280',
            display: 'flex',
            gap: '10px',
          }}
        >
          <span>累计叙述: <span style={{ color: '#a78bfa', fontFamily: 'monospace' }}>{narrativesGenerated}</span></span>
        </div>
      </div>

      {/* 5. 自传体记忆 */}
      <div style={cardStyle}>
        <div style={cardTitleStyle}>自传体记忆 · Hopfield 自我片段</div>
        <div
          style={{
            display: 'flex',
            gap: '12px',
            alignItems: 'center',
            fontSize: '10px',
            color: '#9ca3af',
          }}
        >
          <span style={{ fontFamily: 'monospace', fontSize: '18px', color: '#34d399', fontWeight: 600 }}>
            {nEpisodes}
          </span>
          <span>个自我片段已存储</span>
        </div>
        <div style={{ fontSize: '9px', color: '#6b7280', marginTop: '4px' }}>
          每个片段含 (状态快照, 自我预测, 实际结果, 自由能, 时间戳)，离线巩固时被重新激活。
        </div>
      </div>
    </div>
  );
}
