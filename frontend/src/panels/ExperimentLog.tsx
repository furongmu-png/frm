// frontend/src/panels/ExperimentLog.tsx
// 实验日志 —— 展示实验规划器状态、参数不确定性与近期实验表。
// 数据来源: metadata.cognitive_upgrades.experiment。
// 实验规划器未启用时显示占位提示。

import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';
import type { Snapshot } from '../types';

interface ExperimentRecord {
  name?: string;
  intervention?: string;
  predicted_gain?: number;
  [k: string]: unknown;
}

interface ExperimentUpgrade {
  name?: string;
  intervention?: string;
  predicted_gain?: number;
  param_uncertainty?: number;
  history?: ExperimentRecord[];
  [k: string]: unknown;
}

interface CognitiveUpgrades {
  experiment?: ExperimentUpgrade;
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

// 安全格式化数值: 仅在传入有限数字时调用 toFixed，否则返回占位符，
// 避免后端返回 undefined/NaN/非数字时 .toFixed 抛出 TypeError 或渲染 "NaN"。
export function fmtErr(v: unknown): string {
  return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(4) : '—';
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

const cardStyle: CSSProperties = {
  background: '#0d0d12',
  border: '1px solid #1a1a22',
  borderRadius: '4px',
  padding: '6px 8px',
  fontSize: '11px',
  color: '#aaa',
  display: 'flex',
  flexDirection: 'column',
  gap: '3px',
};

const tableWrapStyle: CSSProperties = {
  flex: 1,
  overflowY: 'auto',
  minHeight: 0,
};

const tableStyle: CSSProperties = {
  width: '100%',
  borderCollapse: 'collapse',
  fontSize: '11px',
  color: '#aaa',
};

const thStyle: CSSProperties = {
  padding: '3px 4px',
  fontWeight: 400,
  textAlign: 'left',
  color: '#9ca3af',
  position: 'sticky',
  top: 0,
  background: '#0d0d12',
};

export default function ExperimentLog() {
  const snapshot = useModelStore((s) => s.snapshot);
  const exp = readMetadata(snapshot)?.cognitive_upgrades?.experiment;

  if (!snapshot || !exp) {
    return (
      <div style={wrapStyle}>
        <div style={titleStyle}>实验日志</div>
        <div style={emptyStyle}>
          {snapshot ? '实验规划器未启用' : '等待数据...'}
        </div>
      </div>
    );
  }

  // 本面板仅订阅 snapshot，不再订阅 history，以避免回放期间不必要的重渲染。

  const name = exp.name ?? '-';
  const intervention = exp.intervention ?? '-';
  const paramUnc =
    typeof exp.param_uncertainty === 'number' && Number.isFinite(exp.param_uncertainty)
      ? Math.max(0, Math.min(1, exp.param_uncertainty))
      : 0;
  const records = Array.isArray(exp.history) ? exp.history : [];

  return (
    <div style={wrapStyle}>
      <div style={titleStyle}>实验日志</div>

      <div style={cardStyle}>
        <div>
          <span style={{ color: '#9ca3af' }}>当前实验: </span>
          <span style={{ color: '#e0e0e0' }}>{name}</span>
        </div>
        <div>
          <span style={{ color: '#9ca3af' }}>干预: </span>
          <span style={{ color: '#e0e0e0' }}>{intervention}</span>
        </div>
        <div>
          <span style={{ color: '#9ca3af' }}>预期增益: </span>
          <span style={{ color: '#4ade80' }}>{fmtErr(exp.predicted_gain)}</span>
        </div>
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
          <span>参数不确定性</span>
          <span style={{ color: '#e0e0e0' }}>{paramUnc.toFixed(3)}</span>
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
              width: `${(paramUnc * 100).toFixed(1)}%`,
              background: '#f97316',
            }}
          />
        </div>
      </div>

      <div style={tableWrapStyle}>
        <table style={tableStyle}>
          <thead>
            <tr>
              <th scope="col" style={thStyle}>实验</th>
              <th scope="col" style={thStyle}>干预</th>
              <th scope="col" style={{ ...thStyle, textAlign: 'right' }}>预期增益</th>
            </tr>
          </thead>
          <tbody>
            {records.length === 0 ? (
              <tr>
                <td colSpan={3} style={{ padding: '6px 4px', color: '#9ca3af', textAlign: 'center' }}>
                  暂无历史实验
                </td>
              </tr>
            ) : (
              records.slice(-50).map((r, i) => (
                // 历史实验记录无稳定唯一 id (name 可能重复)，此处使用数组索引可接受。
                <tr key={i} style={{ borderTop: '1px solid #14141a' }}>
                  <td style={{ padding: '3px 4px', color: '#e0e0e0' }}>{r.name ?? '-'}</td>
                  <td style={{ padding: '3px 4px' }}>{r.intervention ?? '-'}</td>
                  <td style={{ padding: '3px 4px', textAlign: 'right', color: '#4ade80' }}>
                    {fmtErr(r.predicted_gain)}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
