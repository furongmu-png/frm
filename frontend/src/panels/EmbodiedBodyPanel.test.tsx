import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import type { Snapshot } from '../types'
import { makeSnapshot } from '../test-setup'
import EmbodiedBodyPanel from './EmbodiedBodyPanel'

const { mockStore } = vi.hoisted(() => ({ mockStore: vi.fn() }))

vi.mock('../store/useModelStore', () => ({
  useModelStore: mockStore,
}))

function setStoreState(snapshot: Snapshot | null) {
  mockStore.mockImplementation((selector: (s: { snapshot: Snapshot | null }) => unknown) =>
    selector({ snapshot }),
  )
}

describe('EmbodiedBodyPanel 组件渲染', () => {
  beforeEach(() => {
    mockStore.mockReset()
  })

  it('snapshot 为 null 时显示等待数据', () => {
    setStoreState(null)
    render(<EmbodiedBodyPanel />)
    expect(screen.getByText('等待数据...')).toBeTruthy()
  })

  it('snapshot 无 embodiment 元数据时显示具身模块未启用', () => {
    setStoreState(makeSnapshot({ metadata: { cognitive_upgrades: {} } }))
    render(<EmbodiedBodyPanel />)
    expect(screen.getByText('具身模块未启用')).toBeTruthy()
  })

  it('embodiment.error 存在时显示错误信息', () => {
    setStoreState(
      makeSnapshot({
        metadata: { embodiment: { error: 'boom' } },
      }),
    )
    render(<EmbodiedBodyPanel />)
    expect(screen.getByText('boom')).toBeTruthy()
  })

  it('显示动作名、视线方向与触摸物索引', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          embodiment: {
            action: 5,
            action_name: 'touch',
            gaze_yaw: 0.523,
            gaze_pitch: -0.1,
            touched_object: 2,
            proprioception_error: 0.05,
            body_schema_confidence: 0.85,
            visual_prediction_error: 0.12,
            tactile_prediction_error: 0.03,
            sensorimotor: {
              step: 100,
              avg_visual_error: 0.10,
              avg_tactile_error: 0.04,
              s4_state_norm: 1.5,
              s4_spectral_radius: 0.92,
              w_vis_norm: 2.1,
              w_tac_norm: 1.8,
              recent_error: 0.08,
            },
            body_schema: {
              step: 100,
              avg_prediction_error: 0.06,
              schema_confidence: 0.85,
              schema_spectral_radius: 0.78,
              prev_state: [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8],
              recent_error: 0.05,
            },
          },
        },
      }),
    )
    render(<EmbodiedBodyPanel />)
    // 动作徽标
    expect(screen.getByText('主动触觉 #5')).toBeTruthy()
    // 触摸物索引
    expect(screen.getByText('#2')).toBeTruthy()
  })

  it('未触摸物体时显示无', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          embodiment: {
            action: 0,
            action_name: 'physics',
            gaze_yaw: 0,
            gaze_pitch: 0,
            touched_object: -1,
            proprioception_error: 0,
            body_schema_confidence: 0,
            visual_prediction_error: 0,
            tactile_prediction_error: 0,
          },
        },
      }),
    )
    render(<EmbodiedBodyPanel />)
    expect(screen.getByText('无')).toBeTruthy()
  })

  it('sensorimotor 子字段被正确渲染', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          embodiment: {
            action: 1,
            action_name: 'move_gaze',
            gaze_yaw: 0.1,
            gaze_pitch: 0.2,
            touched_object: -1,
            proprioception_error: 0.1,
            body_schema_confidence: 0.5,
            visual_prediction_error: 0.2,
            tactile_prediction_error: 0.1,
            sensorimotor: {
              step: 42,
              avg_visual_error: 0.15,
              avg_tactile_error: 0.05,
              s4_state_norm: 1.2,
              s4_spectral_radius: 0.95,
              w_vis_norm: 1.9,
              w_tac_norm: 1.5,
              recent_error: 0.12,
            },
            body_schema: {
              step: 42,
              avg_prediction_error: 0.08,
              schema_confidence: 0.6,
              schema_spectral_radius: 0.82,
              prev_state: [0.1, 0.2],
              recent_error: 0.07,
            },
          },
        },
      }),
    )
    render(<EmbodiedBodyPanel />)
    // S4 谱半径（嵌入在 "ρ = 0.9500 · 阈值 = 1" 中）
    expect(screen.getByText(/ρ = 0\.9500/)).toBeTruthy()
    // 身体图式谱半径
    expect(screen.getByText(/ρ = 0\.8200/)).toBeTruthy()
  })
})
