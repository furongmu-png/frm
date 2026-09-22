// frontend/src/panels/LogicPanel.tsx
// 逻辑约束 —— 展示逻辑规则违反情况。
// 数据来源: metadata.cognitive_upgrades.logic_violations。
// 单条违反的 penalty > 0.5 时标红显示；逻辑约束层未启用时显示占位提示。

import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';
import type { Snapshot } from '../types';

interface LogicViolation {
  rule?: string;
  inferred?: unknown;
  actual?: unknown;
  penalty?: number;
  [k: string]: unknown;
}

interface LogicViolationsUpgrade {
  n_violations?: number;
  total_penalty?: number;
  violations?: LogicViolation[];
  [k: string]: unknown;
}

interface CognitiveUpgrades {
  logic_violations?: LogicViolationsUpgrade;
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

// 安全格式化任意 unknown 值用于渲染: 数字做有限性校验后再 toFixed，
// 对象序列化为 JSON，避免 String(value) 把对象渲染成 "[object Object]"。
export function fmtVal(v: unknown): string {
  if (typeof v === 'number') return Number.isFinite(v) ? v.toFixed(4) : '—';
  if (typeof v === 'string') return v;
  if (v == null) return '—';
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
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
};

const listWrapStyle: CSSProperties = {
  flex: 1,
  overflowY: 'auto',
  minHeight: 0,
  display: 'flex',
  flexDirection: 'column',
  gap: '4px',
};

const noRecordsStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: '#9ca3af',
  fontSize: '11px',
  padding: '12px',
};

// penalty > 0.5 时使用红色边框与红色背景，否则使用常规暗色。
function violationStyle(penalty: number): CSSProperties {
  const severe = penalty > 0.5;
  return {
    background: severe ? '#2a0d0d' : '#0d0d12',
    border: `1px solid ${severe ? '#ef4444' : '#1a1a22'}`,
    borderRadius: '4px',
    padding: '5px 8px',
    fontSize: '11px',
    color: '#aaa',
    display: 'grid',
    gridTemplateColumns: '1fr 1fr 1fr 56px',
    gap: '6px',
    alignItems: 'center',
  };
}

export default function LogicPanel() {
  const snapshot = useModelStore((s) => s.snapshot);
  const lv = readMetadata(snapshot)?.cognitive_upgrades?.logic_violations;

  if (!snapshot || !lv) {
    return (
      <div style={wrapStyle}>
        <div style={titleStyle}>逻辑约束</div>
        <div style={emptyStyle}>
          {snapshot ? '逻辑约束层未启用' : '等待数据...'}
        </div>
      </div>
    );
  }

  // 本面板仅订阅 snapshot，不再订阅 history，以避免回放期间不必要的重渲染。

  const nViol = typeof lv.n_violations === 'number' ? lv.n_violations : 0;
  const totalPenalty = typeof lv.total_penalty === 'number' ? lv.total_penalty : 0;
  const violations = Array.isArray(lv.violations) ? lv.violations : [];

  return (
    <div style={wrapStyle}>
      <div style={titleStyle}>逻辑约束</div>

      <div style={summaryStyle}>
        <span>
          违反次数: <span style={{ color: '#e0e0e0' }}>{nViol}</span>
        </span>
        <span>
          总惩罚: <span style={{ color: '#ef4444' }}>{totalPenalty.toFixed(4)}</span>
        </span>
      </div>

      <div style={listWrapStyle}>
        {violations.length === 0 ? (
          <div style={noRecordsStyle}>暂无违反记录</div>
        ) : (
          violations.map((v, i) => {
            const penalty = typeof v.penalty === 'number' ? v.penalty : 0;
            return (
              // 违反记录无稳定唯一 id (rule 可能缺失或重复)，此处使用数组索引可接受。
              <div key={i} style={violationStyle(penalty)}>
                <span style={{ color: '#e0e0e0' }}>{v.rule ?? '-'}</span>
                <span>
                  <span style={{ color: '#9ca3af' }}>推断: </span>
                  {fmtVal(v.inferred)}
                </span>
                <span>
                  <span style={{ color: '#9ca3af' }}>实际: </span>
                  {fmtVal(v.actual)}
                </span>
                <span
                  style={{
                    color: penalty > 0.5 ? '#ef4444' : '#aaa',
                    textAlign: 'right',
                  }}
                >
                  {penalty.toFixed(3)}
                </span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
