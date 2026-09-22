// frontend/src/panels/CommunicationLog.tsx
// 通信日志 —— 展示多智能体通信事件记录。
// 数据来源: metadata.cognitive_upgrades 中的 communication / multiagent 字段。
// 当前快照尚未包含多智能体数据时显示占位提示；新条目到达时自动滚动到底部。

import { type UIEvent, useCallback, useEffect, useRef } from 'react';
import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';
import type { Snapshot } from '../types';

interface CommEntry {
  step?: number;
  timestamp?: number;
  sender?: string;
  symbol?: string;
  event?: string;
  [k: string]: unknown;
}

interface CognitiveUpgrades {
  communication?: CommEntry[] | { entries?: CommEntry[]; [k: string]: unknown };
  multiagent?: CommEntry[] | { entries?: CommEntry[]; [k: string]: unknown };
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

// 从 communication/multiagent 字段中提取日志条目列表 (兼容数组或 { entries: [] } 两种形态)。
function extractEntries(meta: SnapshotMetadata | undefined): CommEntry[] {
  if (!meta?.cognitive_upgrades) return [];
  const cu = meta.cognitive_upgrades;
  const raw = cu.communication ?? cu.multiagent;
  if (!raw) return [];
  if (Array.isArray(raw)) return raw as CommEntry[];
  const entries = (raw as { entries?: CommEntry[] }).entries;
  return Array.isArray(entries) ? entries : [];
}

const wrapStyle: CSSProperties = {
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  padding: '8px 12px',
  gap: '6px',
};

const titleStyle: CSSProperties = {
  fontSize: '12px',
  color: '#aaa',
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
  borderBottom: '1px solid #1a1a1a',
  paddingBottom: '4px',
};

const logStyle: CSSProperties = {
  flex: 1,
  overflowY: 'auto',
  fontFamily: 'ui-monospace, Consolas, monospace',
  fontSize: '11px',
  color: '#aaa',
  display: 'flex',
  flexDirection: 'column',
  gap: '2px',
  minHeight: 0,
};

const emptyStyle: CSSProperties = {
  flex: 1,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: '#9ca3af',
  fontSize: '13px',
  textAlign: 'center',
  padding: '0 12px',
};

const rowStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '56px 76px 64px 1fr',
  gap: '6px',
  padding: '2px 4px',
  borderBottom: '1px solid #14141a',
};

const headerRowStyle: CSSProperties = {
  ...rowStyle,
  color: '#9ca3af',
  borderBottom: '1px solid #1a1a1a',
  position: 'sticky',
  top: 0,
  background: '#0d0d12',
};

export default function CommunicationLog() {
  const snapshot = useModelStore((s) => s.snapshot);
  const history = useModelStore((s) => s.history);
  const logRef = useRef<HTMLDivElement>(null);
  // 跟踪用户是否“钉”在底部附近；仅当钉住时才自动滚动，避免用户向上翻阅旧条目时被拉回底部。
  const stickToBottomRef = useRef(true);

  // 注意: entries 仅取自当前 snapshot (而非历史累积)。这是设计使然——通信记录随每条
  // 新快照整体刷新，回放某一步时展示的即是对应步的通信快照。
  const entries = extractEntries(readMetadata(snapshot));

  const onScroll = useCallback((e: UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 50;
  }, []);

  // 新条目到达时仅在用户已位于底部附近时自动滚动，否则保持当前阅读位置。
  useEffect(() => {
    if (stickToBottomRef.current && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [history.length]);

  return (
    <div style={wrapStyle}>
      <div style={titleStyle}>通信日志</div>
      {entries.length === 0 ? (
        <div style={emptyStyle}>
          {snapshot ? '启用 MultiAgentWorld 后将显示通信记录' : '等待数据...'}
        </div>
      ) : (
        <div ref={logRef} onScroll={onScroll} style={logStyle}>
          <div style={headerRowStyle}>
            <span>step</span>
            <span>sender</span>
            <span>symbol</span>
            <span>event</span>
          </div>
          {entries.map((e, i) => (
            // 条目无稳定唯一 id (step/symbol 均可能缺失或重复)，此处使用数组索引可接受。
            <div key={i} style={rowStyle}>
              <span style={{ color: '#9ca3af' }}>{e.step ?? e.timestamp ?? '-'}</span>
              <span style={{ color: '#3b82f6' }}>{e.sender ?? '-'}</span>
              <span style={{ color: '#eab308' }}>{e.symbol ?? '-'}</span>
              <span>{e.event ?? ''}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
