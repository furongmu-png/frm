// frontend/src/panels/EmbodiedBodyPanel.tsx
// 具身主动感知面板 —— 第一人称身体状态 + 感知-运动偶联。
// 数据来源: snapshot.metadata.embodiment，由 HierarchicalZeroDataModel
// 的 _run_embodiment_cycle 每步写入，形如：
//   {
//     action: number,
//     action_name: string,           // "physics" | "move_gaze" | "touch" | "apply_force"
//     gaze_yaw: number,              // 弧度
//     gaze_pitch: number,
//     touched_object: number,        // -1 表示无
//     proprioception_error: number,
//     body_schema_confidence: number,  // 0-1
//     visual_prediction_error: number,
//     tactile_prediction_error: number,
//     sensorimotor: { step, avg_visual_error, avg_tactile_error, s4_state_norm,
//                      s4_spectral_radius, w_vis_norm, w_tac_norm, recent_error },
//     body_schema: { step, avg_prediction_error, schema_confidence,
//                    schema_spectral_radius, prev_state[], recent_error },
//     body_snapshot?: { ... },       // 当前 EmbodiedBody.get_snapshot() 输出（如可用）
//     error?: string,
//   }
//
// 五块可视化:
//   1. 动作 + 视线方向罗盘 — 当前主动感知动作与 gaze 朝向。
//   2. 触觉网格 — 4×4 压力读数热力图。
//   3. 本体感觉误差 + 身体图式置信度 — 时间趋势小图。
//   4. 感知-运动预测误差 — 视觉/触觉误差对比柱状。
//   5. S4 与身体图式谱半径 — 长程记忆稳定性指标。

import type { CSSProperties } from 'react';
import { useModelStore } from '../store/useModelStore';
import type { Snapshot } from '../types';

// ------------------------------------------------------------------ //
// 类型
// ------------------------------------------------------------------ //

interface SensorimotorSnapshot {
  step?: number;
  avg_visual_error?: number;
  avg_tactile_error?: number;
  s4_state_norm?: number;
  s4_spectral_radius?: number;
  w_vis_norm?: number;
  w_tac_norm?: number;
  recent_error?: number;
}

interface BodySchemaSnapshot {
  step?: number;
  avg_prediction_error?: number;
  schema_confidence?: number;
  schema_spectral_radius?: number;
  prev_state?: number[];
  recent_error?: number;
}

interface BodyStateSnapshot {
  gaze_yaw?: number;
  gaze_pitch?: number;
  body_position?: number[];
  joint_angles?: number[];
  joint_velocities?: number[];
  touched_object_id?: number;
  step?: number;
  sandbox_attached?: boolean;
  n_total_actions?: number;
  n_perceptual_actions?: number;
}

interface EmbodimentMetadata {
  action?: number;
  action_name?: string;
  gaze_yaw?: number;
  gaze_pitch?: number;
  touched_object?: number;
  proprioception_error?: number;
  body_schema_confidence?: number;
  visual_prediction_error?: number;
  tactile_prediction_error?: number;
  sensorimotor?: SensorimotorSnapshot;
  body_schema?: BodySchemaSnapshot;
  body_snapshot?: BodyStateSnapshot;
  error?: string;
}

function readEmbodiment(snap: Snapshot | null): EmbodimentMetadata | undefined {
  if (!snap?.metadata) return undefined;
  const md = snap.metadata as { embodiment?: EmbodimentMetadata };
  return md.embodiment;
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

function actionColor(name: string | undefined): string {
  switch (name) {
    case 'physics':
      return '#60a5fa';
    case 'move_gaze':
      return '#fbbf24';
    case 'touch':
      return '#f472b6';
    case 'apply_force':
      return '#f87171';
    default:
      return '#9ca3af';
  }
}

function actionLabel(name: string | undefined): string {
  switch (name) {
    case 'physics':
      return '物理动作';
    case 'move_gaze':
      return '主动视觉';
    case 'touch':
      return '主动触觉';
    case 'apply_force':
      return '主动施力';
    default:
      return name ?? '—';
  }
}

// ------------------------------------------------------------------ //
// 子组件 1：动作 + 视线罗盘
// ------------------------------------------------------------------ //

function ActionCompass({
  actionName,
  actionIdx,
  gazeYaw,
  gazePitch,
  touchedObject,
}: {
  actionName: string;
  actionIdx: number;
  gazeYaw: number;
  gazePitch: number;
  touchedObject: number;
}) {
  const cx = 50;
  const cy = 50;
  const r = 36;
  // 将 yaw 投射到圆周方向（俯视：yaw→x, pitch→y）
  const yawRad = gazeYaw;
  const pitchRad = gazePitch;
  // 视线方向单位向量（限制在圆内）
  const dirLen = Math.min(Math.hypot(Math.sin(yawRad), Math.sin(pitchRad)), 1.0);
  const dx = Math.sin(yawRad) * r * dirLen;
  const dy = -Math.sin(pitchRad) * r * dirLen;

  return (
    <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
      <svg width="100" height="100" viewBox="0 0 100 100">
        {/* 罗盘外圈 */}
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill="none"
          stroke="#1a1a22"
          strokeWidth="0.8"
        />
        {/* 十字 */}
        <line x1={cx - r} y1={cy} x2={cx + r} y2={cy} stroke="#1a1a22" strokeWidth="0.4" />
        <line x1={cx} y1={cy - r} x2={cx} y2={cy + r} stroke="#1a1a22" strokeWidth="0.4" />
        {/* 方位标记 */}
        <text x={cx} y={cy - r - 2} fontSize="6" fill="#6b7280" textAnchor="middle">N (↑)</text>
        <text x={cx + r + 2} y={cy + 2} fontSize="6" fill="#6b7280">E</text>
        <text x={cx} y={cy + r + 6} fontSize="6" fill="#6b7280" textAnchor="middle">S</text>
        <text x={cx - r - 2} y={cy + 2} fontSize="6" fill="#6b7280" textAnchor="end">W</text>
        {/* 视线方向箭头 */}
        <line
          x1={cx}
          y1={cy}
          x2={cx + dx}
          y2={cy + dy}
          stroke={actionColor(actionName)}
          strokeWidth="1.5"
          markerEnd="url(#arrow-head)"
        />
        <defs>
          <marker
            id="arrow-head"
            markerWidth="6"
            markerHeight="6"
            refX="3"
            refY="3"
            orient="auto"
          >
            <polygon points="0,0 6,3 0,6" fill={actionColor(actionName)} />
          </marker>
        </defs>
        {/* 中心点 */}
        <circle cx={cx} cy={cy} r="2" fill="#fbbf24" />
      </svg>
      <div style={{ fontSize: '10px', color: '#9ca3af', flex: 1 }}>
        <div style={{ marginBottom: '3px' }}>
          动作:
          <span
            style={{
              color: actionColor(actionName),
              fontWeight: 600,
              marginLeft: '4px',
              padding: '1px 5px',
              background: `${actionColor(actionName)}22`,
              borderRadius: '2px',
              border: `1px solid ${actionColor(actionName)}55`,
            }}
          >
            {actionLabel(actionName)} #{actionIdx}
          </span>
        </div>
        <div style={{ fontFamily: 'monospace', fontSize: '9px', marginBottom: '2px' }}>
          yaw: <span style={{ color: '#d1d5db' }}>{gazeYaw.toFixed(3)}</span>
        </div>
        <div style={{ fontFamily: 'monospace', fontSize: '9px', marginBottom: '2px' }}>
          pitch: <span style={{ color: '#d1d5db' }}>{gazePitch.toFixed(3)}</span>
        </div>
        <div style={{ fontSize: '9px' }}>
          触摸物: <span style={{ color: touchedObject >= 0 ? '#f472b6' : '#6b7280' }}>
            {touchedObject >= 0 ? `#${touchedObject}` : '无'}
          </span>
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ //
// 子组件 2：误差双柱（视觉 vs 触觉）
// ------------------------------------------------------------------ //

function ErrorBars({
  visualError,
  tactileError,
  propError,
}: {
  visualError: number;
  tactileError: number;
  propError: number;
}) {
  const items: { label: string; value: number; color: string }[] = [
    { label: '视觉预测误差', value: visualError, color: '#60a5fa' },
    { label: '触觉预测误差', value: tactileError, color: '#f472b6' },
    { label: '本体感觉误差', value: propError, color: '#fbbf24' },
  ];
  const maxVal = Math.max(...items.map((i) => i.value), 1e-6);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      {items.map((it) => {
        const pct = Math.min(100, (it.value / maxVal) * 100);
        return (
          <div
            key={it.label}
            style={{
              display: 'grid',
              gridTemplateColumns: '110px 1fr 48px',
              gap: '6px',
              alignItems: 'center',
              fontSize: '10px',
            }}
          >
            <span style={{ color: '#9ca3af' }}>{it.label}</span>
            <div
              style={{
                background: '#1a1a22',
                height: '8px',
                borderRadius: '2px',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  width: `${pct}%`,
                  height: '100%',
                  background: it.color,
                  opacity: 0.85,
                }}
              />
            </div>
            <span style={{ color: '#d1d5db', fontFamily: 'monospace', textAlign: 'right' }}>
              {it.value.toFixed(3)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ------------------------------------------------------------------ //
// 子组件 3：谱半径仪表
// ------------------------------------------------------------------ //

function SpectralGauge({
  label,
  value,
  threshold,
  color,
}: {
  label: string;
  value: number;
  threshold: number;
  color: string;
}) {
  // 谱半径临界值通常 ~1.0；超出表示发散
  const pct = Math.min(100, (value / Math.max(threshold * 2, 1e-3)) * 100);
  const isStable = value < threshold;
  return (
    <div style={cardStyle}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '9px', color: '#6b7280' }}>{label}</span>
        <span
          style={{
            fontSize: '9px',
            padding: '1px 4px',
            background: isStable ? '#4ade8022' : '#f8717122',
            color: isStable ? '#4ade80' : '#f87171',
            borderRadius: '2px',
            border: `1px solid ${isStable ? '#4ade8044' : '#f8717144'}`,
          }}
        >
          {isStable ? '稳定' : '发散'}
        </span>
      </div>
      <div
        style={{
          background: '#1a1a22',
          height: '6px',
          borderRadius: '3px',
          overflow: 'hidden',
          marginTop: '4px',
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: '100%',
            background: color,
            opacity: 0.8,
          }}
        />
      </div>
      <div style={{ fontSize: '9px', color: '#d1d5db', fontFamily: 'monospace', marginTop: '2px' }}>
        ρ = {value.toFixed(4)} · 阈值 = {threshold}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ //
// 主组件
// ------------------------------------------------------------------ //

export default function EmbodiedBodyPanel() {
  const snapshot = useModelStore((s) => s.snapshot);
  const emb = readEmbodiment(snapshot);

  if (!snapshot || !emb) {
    return (
      <div style={wrapStyle}>
        <div style={titleStyle}>具身主动感知 · 身体与环境</div>
        <div style={emptyStyle}>
          {snapshot ? '具身模块未启用' : '等待数据...'}
        </div>
      </div>
    );
  }

  if (emb.error) {
    return (
      <div style={wrapStyle}>
        <div style={titleStyle}>具身主动感知 · 身体与环境</div>
        <div style={{ ...emptyStyle, color: '#ef4444' }}>{emb.error}</div>
      </div>
    );
  }

  const actionName = emb.action_name ?? 'physics';
  const actionIdx = emb.action ?? 0;
  const gazeYaw = emb.gaze_yaw ?? 0;
  const gazePitch = emb.gaze_pitch ?? 0;
  const touchedObject = emb.touched_object ?? -1;
  const propError = emb.proprioception_error ?? 0;
  const bodySchemaConf = emb.body_schema_confidence ?? 0;
  const visualError = emb.visual_prediction_error ?? 0;
  const tactileError = emb.tactile_prediction_error ?? 0;
  const sm = emb.sensorimotor;
  const bs = emb.body_schema;

  return (
    <div style={wrapStyle}>
      <div style={titleStyle}>具身主动感知 · 身体与环境</div>

      {/* 汇总 */}
      <div style={summaryStyle}>
        <span>
          动作: <span style={{ color: actionColor(actionName) }}>{actionLabel(actionName)}</span>
        </span>
        <span>
          身体图式置信度:{' '}
          <span style={{ color: bodySchemaConf > 0.5 ? '#4ade80' : '#fbbf24' }}>
            {(bodySchemaConf * 100).toFixed(1)}%
          </span>
        </span>
        {sm?.step != null && (
          <span>
            步数: <span style={{ color: '#d1d5db' }}>{sm.step}</span>
          </span>
        )}
      </div>

      {/* 1. 动作 + 视线罗盘 */}
      <div style={cardStyle}>
        <div style={cardTitleStyle}>当前动作 · 视线方向</div>
        <ActionCompass
          actionName={actionName}
          actionIdx={actionIdx}
          gazeYaw={gazeYaw}
          gazePitch={gazePitch}
          touchedObject={touchedObject}
        />
      </div>

      {/* 2. 感知-运动预测误差 */}
      <div style={cardStyle}>
        <div style={cardTitleStyle}>感知-运动预测误差</div>
        <ErrorBars
          visualError={visualError}
          tactileError={tactileError}
          propError={propError}
        />
        {sm && (
          <div
            style={{
              marginTop: '6px',
              fontSize: '9px',
              color: '#6b7280',
              display: 'flex',
              gap: '10px',
              flexWrap: 'wrap',
            }}
          >
            <span>
              平均视觉: <span style={{ color: '#60a5fa' }}>{(sm.avg_visual_error ?? 0).toFixed(4)}</span>
            </span>
            <span>
              平均触觉: <span style={{ color: '#f472b6' }}>{(sm.avg_tactile_error ?? 0).toFixed(4)}</span>
            </span>
            <span>
              近 20 步: <span style={{ color: '#d1d5db' }}>{(sm.recent_error ?? 0).toFixed(4)}</span>
            </span>
          </div>
        )}
      </div>

      {/* 3. S4 + 身体图式谱半径（稳定性指标） */}
      <div style={{ display: 'flex', gap: '6px', flexShrink: 0 }}>
        <div style={{ flex: 1 }}>
          <SpectralGauge
            label="S4 谱半径"
            value={sm?.s4_spectral_radius ?? 0}
            threshold={1.0}
            color="#a78bfa"
          />
        </div>
        <div style={{ flex: 1 }}>
          <SpectralGauge
            label="身体图式谱半径"
            value={bs?.schema_spectral_radius ?? 0}
            threshold={1.0}
            color="#fbbf24"
          />
        </div>
      </div>

      {/* 4. 感知-运动预测器状态 */}
      {sm && (
        <div style={cardStyle}>
          <div style={cardTitleStyle}>感知-运动预测器</div>
          <div
            style={{
              fontSize: '10px',
              color: '#9ca3af',
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '4px 12px',
            }}
          >
            <span>S4 状态范数: <span style={{ color: '#d1d5db', fontFamily: 'monospace' }}>{(sm.s4_state_norm ?? 0).toFixed(3)}</span></span>
            <span>W_vis 范数: <span style={{ color: '#d1d5db', fontFamily: 'monospace' }}>{(sm.w_vis_norm ?? 0).toFixed(3)}</span></span>
            <span>W_tac 范数: <span style={{ color: '#d1d5db', fontFamily: 'monospace' }}>{(sm.w_tac_norm ?? 0).toFixed(3)}</span></span>
            <span>步数: <span style={{ color: '#d1d5db', fontFamily: 'monospace' }}>{sm.step ?? 0}</span></span>
          </div>
        </div>
      )}

      {/* 5. 身体图式快照 */}
      {bs && (
        <div style={cardStyle}>
          <div style={cardTitleStyle}>身体图式快照</div>
          <div
            style={{
              fontSize: '10px',
              color: '#9ca3af',
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '4px 12px',
            }}
          >
            <span>平均预测误差: <span style={{ color: '#fbbf24', fontFamily: 'monospace' }}>{(bs.avg_prediction_error ?? 0).toFixed(4)}</span></span>
            <span>图式置信度: <span style={{ color: bodySchemaConf > 0.5 ? '#4ade80' : '#fbbf24', fontFamily: 'monospace' }}>{(bs.schema_confidence ?? 0).toFixed(3)}</span></span>
            <span>近 20 步误差: <span style={{ color: '#d1d5db', fontFamily: 'monospace' }}>{(bs.recent_error ?? 0).toFixed(4)}</span></span>
            <span>步数: <span style={{ color: '#d1d5db', fontFamily: 'monospace' }}>{bs.step ?? 0}</span></span>
          </div>
        </div>
      )}

      {/* 6. 本体感觉状态（prev_state 向量） */}
      {bs?.prev_state && bs.prev_state.length > 0 && (
        <div style={cardStyle}>
          <div style={cardTitleStyle}>本体感觉状态 · 身体内部信念</div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: `repeat(${Math.min(bs.prev_state.length, 8)}, 1fr)`,
              gap: '3px',
            }}
          >
            {bs.prev_state.slice(0, 8).map((v, i) => {
              const absV = Math.min(1, Math.abs(v));
              const color = v >= 0 ? '#4ade80' : '#f87171';
              return (
                <div
                  key={i}
                  style={{
                    background: '#0d0d12',
                    border: '1px solid #1a1a22',
                    borderRadius: '2px',
                    padding: '2px 3px',
                    fontSize: '8px',
                    fontFamily: 'monospace',
                    color: '#d1d5db',
                    textAlign: 'center',
                    position: 'relative',
                    overflow: 'hidden',
                  }}
                  title={`dim[${i}] = ${v.toFixed(4)}`}
                >
                  <div
                    style={{
                      position: 'absolute',
                      bottom: 0,
                      left: 0,
                      right: 0,
                      height: `${absV * 100}%`,
                      background: color,
                      opacity: 0.2,
                      pointerEvents: 'none',
                    }}
                  />
                  <span style={{ position: 'relative' }}>{v.toFixed(2)}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
