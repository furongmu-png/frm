// frontend/src/panels/DiscoveryConsolePanel.tsx
// 科学发现控制台面板 —— 假设→实验→分析→论文 全闭环可视化。
// 数据来源: snapshot.metadata.discovery，由 HierarchicalZeroDataModel
// 的 _run_discovery_cycle 每步写入（ScienceLoop.snapshot()），形如：
//   {
//     enabled: boolean,
//     trigger_interval: number,
//     step: number,
//     cycle_count: number,
//     total_papers: number,
//     pending_approvals: number,
//     require_approval: boolean,
//     hypothesis_generator: { max_hypotheses, min_confidence, generated_count },
//     experiment_designer: { n_samples, designed_count, sandbox_attached },
//     result_analyzer: { accept_threshold, reject_threshold, total_analyses, accepted, rejected, inconclusive },
//     paper_writer: { output_dir, paper_count, recent_titles[], recent_decisions[] },
//     recent_cycles: [{ cycle_id, n_hypotheses, n_experiments, n_accepted, n_rejected, papers_written, findings[], error? }],
//     last_cycle?: { ... },  // 仅在本步触发循环时存在
//     error?: string,
//   }
//
// 四块可视化:
//   1. 汇总仪表盘 — 已生成假设/已设计实验/接受/拒绝/论文数 + 间隔进度条。
//   2. 实验队列 — 近期循环列表，含假设数/接受/拒绝/发现摘要。
//   3. 子模块状态 — 假设生成器/实验设计器/分析器/论文器指标。
//   4. 发现日志 — 近期论文标题与决策徽标。

import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';
import type { Snapshot } from '../types';

// ------------------------------------------------------------------ //
// 类型
// ------------------------------------------------------------------ //

interface RecentCycle {
  cycle_id: number;
  n_hypotheses: number;
  n_experiments: number;
  n_accepted: number;
  n_rejected: number;
  papers_written: number;
  findings: string[];
  error?: string | null;
}

interface DiscoveryMetadata {
  enabled?: boolean;
  trigger_interval?: number;
  step?: number;
  cycle_count?: number;
  total_papers?: number;
  pending_approvals?: number;
  require_approval?: boolean;
  hypothesis_generator?: {
    max_hypotheses?: number;
    min_confidence?: number;
    generated_count?: number;
  };
  experiment_designer?: {
    n_samples?: number;
    designed_count?: number;
    sandbox_attached?: boolean;
  };
  result_analyzer?: {
    accept_threshold?: number;
    reject_threshold?: number;
    total_analyses?: number;
    accepted?: number;
    rejected?: number;
    inconclusive?: number;
  };
  paper_writer?: {
    output_dir?: string;
    paper_count?: number;
    recent_titles?: string[];
    recent_decisions?: string[];
  };
  recent_cycles?: RecentCycle[];
  last_cycle?: RecentCycle;
  error?: string;
}

function readDiscovery(snap: Snapshot | null): DiscoveryMetadata | undefined {
  if (!snap?.metadata) return undefined;
  const md = snap.metadata as { discovery?: DiscoveryMetadata };
  return md.discovery;
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

const metricValueStyle: CSSProperties = {
  fontSize: '16px',
  fontWeight: 600,
  fontFamily: 'monospace',
};

const metricLabelStyle: CSSProperties = {
  fontSize: '9px',
  color: '#6b7280',
  marginTop: '2px',
};

function decisionColor(decision: string): string {
  switch (decision) {
    case 'accept':
      return '#4ade80';
    case 'reject':
      return '#f87171';
    case 'inconclusive':
      return '#fbbf24';
    default:
      return '#9ca3af';
  }
}

function decisionLabel(decision: string): string {
  switch (decision) {
    case 'accept':
      return '接受';
    case 'reject':
      return '拒绝';
    case 'inconclusive':
      return '不确定';
    default:
      return decision;
  }
}

// ------------------------------------------------------------------ //
// 子组件：触发间隔进度条
// ------------------------------------------------------------------ //

function TriggerProgress({ step, interval }: { step: number; interval: number }) {
  if (interval <= 0) return null;
  const inCycle = step % interval;
  const progress = (inCycle / interval) * 100;
  const remaining = interval - inCycle;
  return (
    <div style={cardStyle}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
        <span style={{ fontSize: '9px', color: '#6b7280' }}>下次触发</span>
        <span style={{ fontSize: '9px', color: '#9ca3af', fontFamily: 'monospace' }}>
          剩 {remaining} 步
        </span>
      </div>
      <div style={{ background: '#1a1a22', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
        <div
          style={{
            width: `${progress}%`,
            height: '100%',
            background: 'linear-gradient(90deg, #60a5fa, #a78bfa)',
            transition: 'width 0.3s',
          }}
        />
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ //
// 主组件
// ------------------------------------------------------------------ //

export default function DiscoveryConsolePanel() {
  const snapshot = useModelStore((s) => s.snapshot);
  const disc = readDiscovery(snapshot);

  if (!snapshot || !disc) {
    return (
      <div style={wrapStyle}>
        <div style={titleStyle}>科学发现 · 控制台</div>
        <div style={emptyStyle}>
          {snapshot ? '科学发现引擎未启用' : '等待数据...'}
        </div>
      </div>
    );
  }

  if (disc.error) {
    return (
      <div style={wrapStyle}>
        <div style={titleStyle}>科学发现 · 控制台</div>
        <div style={{ ...emptyStyle, color: '#ef4444' }}>{disc.error}</div>
      </div>
    );
  }

  if (disc.enabled === false) {
    return (
      <div style={wrapStyle}>
        <div style={titleStyle}>科学发现 · 控制台</div>
        <div style={emptyStyle}>科学发现引擎已禁用</div>
      </div>
    );
  }

  const hg = disc.hypothesis_generator;
  const ed = disc.experiment_designer;
  const ra = disc.result_analyzer;
  const pw = disc.paper_writer;
  const recentCycles = disc.recent_cycles ?? [];
  const lastCycle = disc.last_cycle;
  const totalAnalyses = ra?.total_analyses ?? 0;
  const accepted = ra?.accepted ?? 0;
  const rejected = ra?.rejected ?? 0;
  const inconclusive = ra?.inconclusive ?? Math.max(0, totalAnalyses - accepted - rejected);

  return (
    <div style={wrapStyle}>
      <div style={titleStyle}>科学发现 · 控制台</div>

      {/* 汇总指标 */}
      <div style={metricGridStyle}>
        <div style={metricBoxStyle}>
          <div style={{ ...metricValueStyle, color: '#60a5fa' }}>{disc.cycle_count ?? 0}</div>
          <div style={metricLabelStyle}>循环数</div>
        </div>
        <div style={metricBoxStyle}>
          <div style={{ ...metricValueStyle, color: '#fbbf24' }}>{hg?.generated_count ?? 0}</div>
          <div style={metricLabelStyle}>假设</div>
        </div>
        <div style={metricBoxStyle}>
          <div style={{ ...metricValueStyle, color: '#a78bfa' }}>{ed?.designed_count ?? 0}</div>
          <div style={metricLabelStyle}>实验</div>
        </div>
        <div style={metricBoxStyle}>
          <div style={{ ...metricValueStyle, color: '#4ade80' }}>{accepted}</div>
          <div style={metricLabelStyle}>接受</div>
        </div>
        <div style={metricBoxStyle}>
          <div style={{ ...metricValueStyle, color: '#f87171' }}>{rejected}</div>
          <div style={metricLabelStyle}>拒绝</div>
        </div>
        <div style={metricBoxStyle}>
          <div style={{ ...metricValueStyle, color: '#34d399' }}>{disc.total_papers ?? 0}</div>
          <div style={metricLabelStyle}>论文</div>
        </div>
      </div>

      {/* 触发进度 */}
      <TriggerProgress step={disc.step ?? 0} interval={disc.trigger_interval ?? 5000} />

      {/* 最近循环（last_cycle 高亮） */}
      {lastCycle && (
        <div style={{ ...cardStyle, borderColor: '#a78bfa' }}>
          <div style={cardTitleStyle}>本次循环 · #{lastCycle.cycle_id}</div>
          <div style={{ fontSize: '10px', color: '#9ca3af', display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            <span>假设: <span style={{ color: '#fbbf24' }}>{lastCycle.n_hypotheses}</span></span>
            <span>实验: <span style={{ color: '#a78bfa' }}>{lastCycle.n_experiments}</span></span>
            <span>接受: <span style={{ color: '#4ade80' }}>{lastCycle.n_accepted}</span></span>
            <span>拒绝: <span style={{ color: '#f87171' }}>{lastCycle.n_rejected}</span></span>
            <span>论文: <span style={{ color: '#34d399' }}>{lastCycle.papers_written}</span></span>
          </div>
          {lastCycle.findings.length > 0 && (
            <div style={{ marginTop: '4px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
              {lastCycle.findings.map((f, i) => (
                <div
                  key={i}
                  style={{
                    fontSize: '9px',
                    color: '#d1d5db',
                    background: '#0d0d12',
                    padding: '2px 4px',
                    borderRadius: '2px',
                    borderLeft: `2px solid ${f.startsWith('ACCEPT') ? '#4ade80' : '#f87171'}`,
                  }}
                >
                  {f}
                </div>
              ))}
            </div>
          )}
          {lastCycle.error && (
            <div style={{ fontSize: '9px', color: '#ef4444', marginTop: '4px' }}>
              {lastCycle.error}
            </div>
          )}
        </div>
      )}

      {/* 实验队列：历史循环 */}
      <div style={cardStyle}>
        <div style={cardTitleStyle}>实验队列 · 近期循环</div>
        {recentCycles.length === 0 ? (
          <div style={{ fontSize: '10px', color: '#6b7280' }}>暂无循环记录</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
            {recentCycles
              .slice()
              .reverse()
              .map((c) => (
                <div
                  key={c.cycle_id}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '40px 1fr auto',
                    gap: '6px',
                    alignItems: 'center',
                    fontSize: '10px',
                    padding: '3px 4px',
                    background: '#0d0d12',
                    border: '1px solid #1a1a22',
                    borderRadius: '3px',
                  }}
                >
                  <span style={{ color: '#9ca3af', fontFamily: 'monospace' }}>#{c.cycle_id}</span>
                  <span style={{ color: '#d1d5db', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {c.findings[0] ?? `H=${c.n_hypotheses} E=${c.n_experiments}`}
                  </span>
                  <span style={{ display: 'flex', gap: '3px', fontFamily: 'monospace' }}>
                    <span style={{ color: '#4ade80' }}>✓{c.n_accepted}</span>
                    <span style={{ color: '#f87171' }}>✗{c.n_rejected}</span>
                  </span>
                </div>
              ))}
          </div>
        )}
      </div>

      {/* 子模块状态 */}
      <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
        <div style={{ ...cardStyle, flex: 1 }}>
          <div style={cardTitleStyle}>假设生成器</div>
          <div style={{ fontSize: '9px', color: '#9ca3af', display: 'flex', flexDirection: 'column', gap: '1px' }}>
            <span>最大: <span style={{ color: '#d1d5db' }}>{hg?.max_hypotheses ?? '—'}</span></span>
            <span>已生成: <span style={{ color: '#fbbf24' }}>{hg?.generated_count ?? 0}</span></span>
          </div>
        </div>
        <div style={{ ...cardStyle, flex: 1 }}>
          <div style={cardTitleStyle}>实验设计器</div>
          <div style={{ fontSize: '9px', color: '#9ca3af', display: 'flex', flexDirection: 'column', gap: '1px' }}>
            <span>已设计: <span style={{ color: '#a78bfa' }}>{ed?.designed_count ?? 0}</span></span>
            <span>沙盒: <span style={{ color: ed?.sandbox_attached ? '#4ade80' : '#6b7280' }}>
              {ed?.sandbox_attached ? '已挂载' : '未挂载'}
            </span></span>
          </div>
        </div>
        <div style={{ ...cardStyle, flex: 1 }}>
          <div style={cardTitleStyle}>结果分析器</div>
          <div style={{ fontSize: '9px', color: '#9ca3af', display: 'flex', flexDirection: 'column', gap: '1px' }}>
            <span>分析: <span style={{ color: '#d1d5db' }}>{totalAnalyses}</span></span>
            <span>不确定: <span style={{ color: '#fbbf24' }}>{inconclusive}</span></span>
          </div>
        </div>
      </div>

      {/* 发现日志：论文列表 */}
      <div style={cardStyle}>
        <div style={cardTitleStyle}>发现日志 · 论文</div>
        {pw && pw.paper_count && pw.paper_count > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
            {(pw.recent_titles ?? []).map((t, i) => {
              const decision = pw.recent_decisions?.[i] ?? 'unknown';
              const color = decisionColor(decision);
              const label = decisionLabel(decision);
              return (
                <div
                  key={i}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'auto 1fr',
                    gap: '6px',
                    alignItems: 'center',
                    fontSize: '10px',
                    padding: '3px 4px',
                    background: '#0d0d12',
                    borderRadius: '3px',
                  }}
                >
                  <span
                    style={{
                      color,
                      fontSize: '9px',
                      padding: '1px 4px',
                      background: `${color}22`,
                      borderRadius: '2px',
                      border: `1px solid ${color}44`,
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {label}
                  </span>
                  <span
                    style={{ color: '#d1d5db', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                    title={t}
                  >
                    {t}
                  </span>
                </div>
              );
            })}
          </div>
        ) : (
          <div style={{ fontSize: '10px', color: '#6b7280' }}>暂无论文</div>
        )}
      </div>

      {/* 审批队列提示 */}
      {disc.require_approval && (disc.pending_approvals ?? 0) > 0 && (
        <div
          style={{
            ...cardStyle,
            borderColor: '#fbbf24',
            background: '#1a1a0a',
          }}
        >
          <div style={{ fontSize: '10px', color: '#fbbf24' }}>
            ⚠ 待审批实验: {disc.pending_approvals}（需人工确认）
          </div>
        </div>
      )}
    </div>
  );
}
