// frontend/src/panels/ReasoningChainView.tsx
// 推理链视图 —— 展示神经符号推理的传递性、类比和缺省推理结果。
// 数据来源: metadata.cognitive_upgrades.reasoning_chain。
// 推理图未启用时显示占位提示。

import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';
import type { Snapshot } from '../types';

interface VirtualObservation {
  source?: string;
  target?: string;
  relation?: string;
  confidence?: number;
  type?: string;
  [k: string]: unknown;
}

interface ReasoningChain {
  pending_virtual_observations?: number;
  n_facts?: number;
  n_rules?: number;
  recent_inferences?: VirtualObservation[];
  [k: string]: unknown;
}

interface CognitiveUpgrades {
  reasoning_chain?: ReasoningChain;
  [k: string]: unknown;
}

interface SnapshotMetadata {
  cognitive_upgrades?: CognitiveUpgrades;
}

function readMetadata(snap: Snapshot | null): SnapshotMetadata | undefined {
  if (!snap) return undefined;
  return snap.metadata as SnapshotMetadata | undefined;
}

function fmtConf(v: unknown): string {
  return typeof v === 'number' && Number.isFinite(v)
    ? `${(v * 100).toFixed(1)}%`
    : '—';
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

const inferenceStyle: CSSProperties = {
  background: '#0d0d12',
  border: '1px solid #1a1a22',
  borderRadius: '4px',
  padding: '5px 8px',
  fontSize: '11px',
  color: '#aaa',
  display: 'flex',
  flexDirection: 'column',
  gap: '2px',
};

export default function ReasoningChainView() {
  const snapshot = useModelStore((s) => s.snapshot);
  const rc = readMetadata(snapshot)?.cognitive_upgrades?.reasoning_chain;

  if (!snapshot || !rc) {
    return (
      <div style={wrapStyle}>
        <div style={titleStyle}>推理链视图</div>
        <div style={emptyStyle}>
          {snapshot ? '推理图未启用' : '等待数据...'}
        </div>
      </div>
    );
  }

  const nFacts = typeof rc.n_facts === 'number' ? rc.n_facts : 0;
  const nRules = typeof rc.n_rules === 'number' ? rc.n_rules : 0;
  const nPending =
    typeof rc.pending_virtual_observations === 'number'
      ? rc.pending_virtual_observations
      : 0;
  const inferences = Array.isArray(rc.recent_inferences)
    ? rc.recent_inferences
    : [];

  return (
    <div style={wrapStyle}>
      <div style={titleStyle}>推理链视图</div>

      <div style={summaryStyle}>
        <span>
          事实: <span style={{ color: '#e0e0e0' }}>{nFacts}</span>
        </span>
        <span>
          规则: <span style={{ color: '#e0e0e0' }}>{nRules}</span>
        </span>
        <span>
          待处理虚拟观测: <span style={{ color: '#3b82f6' }}>{nPending}</span>
        </span>
      </div>

      <div style={listWrapStyle}>
        {inferences.length === 0 ? (
          <div style={noRecordsStyle}>暂无推理结果</div>
        ) : (
          inferences.map((inf, i) => (
            <div key={i} style={inferenceStyle}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#e0e0e0' }}>
                  {inf.source ?? '?'} → {inf.target ?? '?'}
                </span>
                <span style={{ color: '#4ade80' }}>
                  {fmtConf(inf.confidence)}
                </span>
              </div>
              <span>
                <span style={{ color: '#9ca3af' }}>关系: </span>
                {inf.relation ?? inf.type ?? '—'}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
