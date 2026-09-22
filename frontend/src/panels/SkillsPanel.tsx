// frontend/src/panels/SkillsPanel.tsx
// 技能面板 —— 列出全部 20 项技能的启用状态、就绪度与关键指标。
//
// 数据来源: snapshot.metadata.skills，由 HierarchicalZeroDataModel
// 的 _run_skills_cycle 每步写入，形如：
//   { [skillName]: { enabled: bool, data: {...}, error?: string } }
//
// 面板按 5 个维度（感知/认知/交互/专业/元）分组展示，每项技能显示
// 状态徽标与从 data 提取的一两个关键指标。技能层未启用或无数据时
// 显示占位提示。

import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';
import type { Snapshot } from '../types';

// ------------------------------------------------------------------ //
// 类型（窄转换：Snapshot.metadata 已在 types.ts 声明为带索引签名）
// ------------------------------------------------------------------ //

interface SkillPayload {
  enabled?: boolean;
  data?: Record<string, unknown>;
  error?: string;
}

interface SkillsMetadata {
  skills?: Record<string, SkillPayload>;
}

function readSkills(snap: Snapshot | null): Record<string, SkillPayload> | undefined {
  if (!snap?.metadata) return undefined;
  const md = snap.metadata as SkillsMetadata;
  return md.skills;
}

// ------------------------------------------------------------------ //
// 维度与技能清单（与后端 src/skills/ 的 20 项一一对应）
// ------------------------------------------------------------------ //

type Dimension = 'perception' | 'cognition' | 'interaction' | 'expertise' | 'meta';

interface SkillDef {
  name: string;
  label: string;
  /** 从 data 中提取一个关键指标展示（返回 null 表示不展示）。 */
  metric?: (data: Record<string, unknown>) => string | null;
}

interface DimensionGroup {
  id: Dimension;
  label: string;
  skills: SkillDef[];
}

// 友好的数字格式化：非有限值显示为 —。
function fmtNum(v: unknown): string {
  if (typeof v === 'number') return Number.isFinite(v) ? v.toFixed(3) : '—';
  if (typeof v === 'string') return v;
  if (v == null) return '—';
  return String(v);
}

const DIMENSIONS: DimensionGroup[] = [
  {
    id: 'perception',
    label: '感知扩展',
    skills: [
      {
        name: 'depth_estimator',
        label: '深度估计',
        metric: (d) => {
          const mean = d.depth_mean;
          return typeof mean === 'number' ? `mean=${fmtNum(mean)}` : null;
        },
      },
      {
        name: 'tactile_sensor',
        label: '触觉感知',
        metric: (d) => {
          const p = d.mean_pressure;
          return typeof p === 'number' ? `pressure=${fmtNum(p)}` : null;
        },
      },
      {
        name: 'audio_scene',
        label: '音频场景',
        metric: (d) => (d.scene ? `scene=${String(d.scene)}` : null),
      },
      {
        name: 'olfaction',
        label: '嗅觉感知',
        metric: (d) => {
          const c = d.concentration;
          return typeof c === 'number' ? `conc=${fmtNum(c)}` : null;
        },
      },
    ],
  },
  {
    id: 'cognition',
    label: '认知深化',
    skills: [
      {
        name: 'theory_of_mind',
        label: '心理理论',
        metric: (d) => (d.predicted_action != null ? `pred=${fmtNum(d.predicted_action)}` : null),
      },
      {
        name: 'narrative',
        label: '叙事理解',
        metric: (d) => (d.continuation ? `…${String(d.continuation).slice(0, 24)}` : null),
      },
      {
        name: 'affect',
        label: '幽默情感',
        metric: (d) => (d.sentiment ? `sent=${String(d.sentiment)}` : null),
      },
      {
        name: 'analogy',
        label: '类比推理',
        metric: (d) => (d.analogy ? String(d.analogy).slice(0, 40) : null),
      },
    ],
  },
  {
    id: 'interaction',
    label: '交互协作',
    skills: [
      {
        name: 'dialogue',
        label: '自然语言对话',
        metric: (d) => (d.intent ? `intent=${String(d.intent)}` : null),
      },
      {
        name: 'learning_from_demo',
        label: '示教学习',
        metric: (d) => {
          const n = d.n_demos;
          return typeof n === 'number' ? `demos=${n}` : null;
        },
      },
      {
        name: 'machine_teaching',
        label: '主动教学',
        metric: (d) => {
          const m = d.mastery;
          return typeof m === 'number' ? `mastery=${fmtNum(m)}` : null;
        },
      },
      {
        name: 'multimodal_translation',
        label: '多模态翻译',
        metric: (d) => {
          const e = d.alignment_error;
          return typeof e === 'number' ? `align_err=${fmtNum(e)}` : null;
        },
      },
    ],
  },
  {
    id: 'expertise',
    label: '专业技能',
    skills: [
      {
        name: 'program_synthesis',
        label: '程序合成',
        metric: (d) => {
          const conv = d.converged;
          const err = d.error;
          return `conv=${conv ? 'yes' : 'no'}${typeof err === 'number' ? `, err=${fmtNum(err)}` : ''}`;
        },
      },
      {
        name: 'theorem_proving',
        label: '定理证明',
        metric: (d) => `proved=${d.proved ? 'yes' : 'no'}`,
      },
      {
        name: 'game_playing',
        label: '游戏策略',
        metric: (d) => {
          const n = d.games_played;
          return typeof n === 'number' ? `games=${n}` : null;
        },
      },
      {
        name: 'anomaly_detection',
        label: '异常检测',
        metric: (d) => (d.alert ? '⚠ ALERT' : (d.anomalies != null ? `n=${fmtNum(d.anomalies)}` : null)),
      },
    ],
  },
  {
    id: 'meta',
    label: '元技能',
    skills: [
      {
        name: 'meta_learning',
        label: '学习如何学习',
        metric: (d) => {
          const bl = d.best_loss;
          return typeof bl === 'number' ? `best_loss=${fmtNum(bl)}` : null;
        },
      },
      {
        name: 'curriculum',
        label: '课程学习',
        metric: (d) => (d.current_task ? `task=${String(d.current_task)}` : null),
      },
      {
        name: 'forgetting',
        label: '遗忘管理',
        metric: (d) => {
          const n = d.n_memories;
          return typeof n === 'number' ? `mem=${n}` : null;
        },
      },
      {
        name: 'energy_aware',
        label: '能源感知',
        metric: (d) => {
          const stats = d.stats as Record<string, unknown> | undefined;
          const mode = stats?.mode;
          return mode ? `mode=${String(mode)}` : null;
        },
      },
    ],
  },
];

// 全部技能名集合，用于核对后端是否完整下发。
const ALL_SKILL_NAMES = new Set(DIMENSIONS.flatMap((g) => g.skills.map((s) => s.name)));

// ------------------------------------------------------------------ //
// 样式
// ------------------------------------------------------------------ //

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

const listWrapStyle: CSSProperties = {
  flex: 1,
  overflowY: 'auto',
  minHeight: 0,
  display: 'flex',
  flexDirection: 'column',
  gap: '8px',
};

const groupLabelStyle: CSSProperties = {
  fontSize: '10px',
  color: '#6b7280',
  textTransform: 'uppercase',
  letterSpacing: '0.6px',
  marginTop: '4px',
};

const skillRowStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '10px 1fr auto',
  gap: '8px',
  alignItems: 'center',
  padding: '4px 6px',
  background: '#0d0d12',
  border: '1px solid #1a1a22',
  borderRadius: '4px',
  fontSize: '11px',
  color: '#d1d5db',
};

const metricStyle: CSSProperties = {
  color: '#9ca3af',
  fontSize: '10px',
  fontFamily: 'monospace',
  maxWidth: '160px',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
};

// 状态徽标颜色：启用就绪=绿，启用未就绪=灰，禁用=暗，出错=红。
function dotStyle(state: SkillState): CSSProperties {
  const color =
    state === 'error' ? '#ef4444' : state === 'ready' ? '#4ade80' : state === 'idle' ? '#6b7280' : '#374151';
  return {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    background: color,
    justifySelf: 'center',
  };
}

type SkillState = 'ready' | 'idle' | 'disabled' | 'error';

function classify(p: SkillPayload | undefined): SkillState {
  if (!p) return 'idle';
  if (p.enabled === false) return 'disabled';
  if (p.error) return 'error';
  if (p.data && Object.keys(p.data).length > 0) return 'ready';
  return 'idle';
}

// ------------------------------------------------------------------ //
// 主组件
// ------------------------------------------------------------------ //

export default function SkillsPanel() {
  const snapshot = useModelStore((s) => s.snapshot);
  const skills = readSkills(snapshot);

  if (!snapshot || !skills) {
    return (
      <div style={wrapStyle}>
        <div style={titleStyle}>技能面板</div>
        <div style={emptyStyle}>
          {snapshot ? '技能层未启用' : '等待数据...'}
        </div>
      </div>
    );
  }

  // 汇总统计。
  let enabledCount = 0;
  let readyCount = 0;
  let errorCount = 0;
  for (const name of ALL_SKILL_NAMES) {
    const p = skills[name];
    const st = classify(p);
    if (st === 'ready') readyCount++;
    if (st === 'error') errorCount++;
    // 只对后端实际下发的技能计数"已启用"：未声明的技能（p 为
    // undefined）既不计入已启用也不计入禁用，避免把后端未启用的
    // 技能误判为启用。总数仍恒为前端清单的 20。
    if (p && p.enabled !== false) enabledCount++;
  }
  const total = ALL_SKILL_NAMES.size;

  return (
    <div style={wrapStyle}>
      <div style={titleStyle}>技能面板 · Skills</div>

      <div style={summaryStyle}>
        <span>
          总数: <span style={{ color: '#e0e0e0' }}>{total}</span>
        </span>
        <span>
          已启用: <span style={{ color: '#4ade80' }}>{enabledCount}</span>
        </span>
        <span>
          就绪: <span style={{ color: '#60a5fa' }}>{readyCount}</span>
        </span>
        <span>
          错误: <span style={{ color: errorCount ? '#ef4444' : '#9ca3af' }}>{errorCount}</span>
        </span>
      </div>

      <div style={listWrapStyle}>
        {DIMENSIONS.map((group) => (
          <div key={group.id}>
            <div style={groupLabelStyle}>{group.label}</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '3px', marginTop: '3px' }}>
              {group.skills.map((def) => {
                const p = skills[def.name];
                const state = classify(p);
                // 关键指标仅在就绪态展示；出错/禁用/待数据时优先展示
                // 状态标签，使异常更醒目（错误还会以红色圆点 + tooltip 暴露）。
                const metric = state === 'ready' && p?.data && def.metric ? def.metric(p.data) : null;
                return (
                  <div key={def.name} style={skillRowStyle} title={p?.error ?? def.name}>
                    <span style={dotStyle(state)} />
                    <span>{def.label}</span>
                    <span style={metricStyle}>{metric ?? stateLabel(state)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function stateLabel(state: SkillState): string {
  switch (state) {
    case 'ready':
      return '就绪';
    case 'idle':
      return '待数据';
    case 'disabled':
      return '已禁用';
    case 'error':
      return '出错';
  }
}
