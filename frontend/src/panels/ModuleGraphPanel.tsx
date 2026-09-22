// frontend/src/panels/ModuleGraphPanel.tsx
// 模块误差图 —— 展示 6 个核心模块的当前误差，并以 HSL 颜色映射高低。
// 当架构升级 (architecture) 启用时，额外在下方显示 dormant/split 计数。

import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';
import type { Snapshot } from '../types';

// 6 个核心模块的规范名称 (用于在 module_errors 中查找误差)。
const CORE_MODULES = ['perception', 'memory', 'action', 'value', 'language', 'meta'];

// 认知升级元数据的局部类型扩展 (types.ts 中的 Snapshot.metadata 使用宽松索引签名，
// 此处仅声明本面板用到的具体子形状)。
interface ArchitectureUpgrade {
  dormant?: number;
  split?: number;
  [k: string]: unknown;
}

interface CognitiveUpgrades {
  architecture?: ArchitectureUpgrade;
  [k: string]: unknown;
}

interface SnapshotMetadata {
  cognitive_upgrades?: CognitiveUpgrades;
  module_errors?: Record<string, number>;
}

function readMetadata(snap: Snapshot | null): SnapshotMetadata | undefined {
  if (!snap) return undefined;
  // 单层窄转换: Snapshot.metadata 已在 types.ts 中声明，仅需收敛到本面板的
  // 具体子形状 (不再使用 `as unknown as` 双重转换)。
  return snap.metadata as SnapshotMetadata | undefined;
}

// 安全格式化误差值: 仅在传入有限数字时调用 toFixed，否则返回占位符，
// 避免后端返回 undefined/NaN/非数字时 .toFixed 抛出 TypeError 导致面板崩溃。
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

const gridStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(3, 1fr)',
  gridTemplateRows: 'repeat(2, 1fr)',
  gap: '8px',
  flex: 1,
  minHeight: 0,
};

const emptyStyle: CSSProperties = {
  flex: 1,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: '#9ca3af',
  fontSize: '13px',
};

const footerStyle: CSSProperties = {
  fontSize: '11px',
  color: '#aaa',
  display: 'flex',
  gap: '12px',
  flexWrap: 'wrap',
};

// 误差 0 → 绿色 (h=120)，误差 >=1 → 红色 (h=0)，使用 HSL 插值。
function errorColor(error: number): string {
  const clamped = Math.max(0, Math.min(1, error));
  const hue = 120 * (1 - clamped);
  return `hsl(${hue.toFixed(0)}, 70%, 45%)`;
}

function nodeStyle(color: string): CSSProperties {
  return {
    background: '#0d0d12',
    border: `1px solid ${color}`,
    borderRadius: '4px',
    padding: '6px 8px',
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
    minWidth: 0,
  };
}

const nodeNameStyle: CSSProperties = {
  fontSize: '11px',
  color: '#e0e0e0',
  textTransform: 'capitalize',
  whiteSpace: 'nowrap',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
};

const barTrackStyle: CSSProperties = {
  height: '6px',
  background: '#1a1a22',
  borderRadius: '2px',
  overflow: 'hidden',
};

export default function ModuleGraphPanel() {
  const snapshot = useModelStore((s) => s.snapshot);

  const metadata = readMetadata(snapshot);
  const moduleErrors = metadata?.module_errors;
  const architecture = metadata?.cognitive_upgrades?.architecture;

  if (!snapshot) {
    return (
      <div style={wrapStyle}>
        <div style={titleStyle}>模块误差图</div>
        <div style={emptyStyle}>等待数据...</div>
      </div>
    );
  }

  // 本面板仅订阅 snapshot (随 addSnapshot / setPlaybackIndex 更新)，
  // 不再订阅 history，以避免回放期间每条新快照触发不必要的重渲染。

  return (
    <div style={wrapStyle}>
      <div style={titleStyle}>模块误差图</div>
      <div style={gridStyle}>
        {CORE_MODULES.map((name) => {
          const err = moduleErrors ? (moduleErrors[name] ?? 0) : 0;
          const color = errorColor(err);
          // 条宽与误差成正比，上限 100%。
          const widthPct = Math.max(2, Math.min(100, err * 100));
          return (
            <div key={name} style={nodeStyle(color)}>
              <div style={nodeNameStyle}>{name}</div>
              <div style={barTrackStyle}>
                <div style={{ height: '100%', width: `${widthPct}%`, background: color }} />
              </div>
              <div style={{ fontSize: '10px', color: '#aaa' }}>
                error = <span style={{ color: '#e0e0e0' }}>{fmtErr(moduleErrors?.[name])}</span>
              </div>
            </div>
          );
        })}
      </div>
      {architecture && (
        <div style={footerStyle}>
          <span>
            架构升级:
            dormant = <span style={{ color: '#eab308' }}>{String(architecture.dormant ?? 0)}</span>,
            split = <span style={{ color: '#3b82f6' }}>{String(architecture.split ?? 0)}</span>
          </span>
        </div>
      )}
    </div>
  );
}
