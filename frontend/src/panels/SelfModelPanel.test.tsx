import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import type { Snapshot } from '../types'
import { makeSnapshot } from '../test-setup'
import SelfModelPanel from './SelfModelPanel'

const { mockStore } = vi.hoisted(() => ({ mockStore: vi.fn() }))

vi.mock('../store/useModelStore', () => ({
  useModelStore: mockStore,
}))

function setStoreState(snapshot: Snapshot | null) {
  mockStore.mockImplementation((selector: (s: { snapshot: Snapshot | null }) => unknown) =>
    selector({ snapshot }),
  )
}

describe('SelfModelPanel 组件渲染', () => {
  beforeEach(() => {
    mockStore.mockReset()
  })

  it('snapshot 为 null 时显示等待数据', () => {
    setStoreState(null)
    render(<SelfModelPanel />)
    expect(screen.getByText('等待数据...')).toBeTruthy()
  })

  it('snapshot 无 iwsm 元数据时显示 IWSM 模块未启用', () => {
    setStoreState(makeSnapshot({ metadata: { cognitive_upgrades: {} } }))
    render(<SelfModelPanel />)
    expect(screen.getByText('IWSM 模块未启用')).toBeTruthy()
  })

  it('iwsm.error 存在时显示错误信息', () => {
    setStoreState(
      makeSnapshot({
        metadata: { iwsm: { error: 'kapow' } },
      }),
    )
    render(<SelfModelPanel />)
    expect(screen.getByText('kapow')).toBeTruthy()
  })

  it('显示 Φ_self 值、清醒状态与自我一致性', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          iwsm: {
            self_schema: {
              attn_error: 0.4,
              action_error: 0.5,
              affect_error: 0.6,
              attn_correct: true,
              action_correct: false,
              self_consistency: 0.78,
              step: 50,
            },
            self_schema_stats: {
              step: 50,
              attention_accuracy: 0.85,
              action_accuracy: 0.72,
              affect_error: 0.6,
              self_consistency: 0.78,
            },
            phi_self: 0.4231,
            phi_self_history: [0.1, 0.2, 0.3, 0.4231],
            is_sleeping: false,
            autobiographical: { n_episodes: 47 },
            counterfactual: {
              narratives_generated: 3,
              last_regret: 0.15,
              last_narrative: '如果当时选择向左移动，可能不会撞到墙壁',
            },
          },
        },
      }),
    )
    render(<SelfModelPanel />)
    // Φ_self 值
    expect(screen.getByText('0.4231')).toBeTruthy()
    // 状态：清醒
    expect(screen.getByText('清醒')).toBeTruthy()
    // 自我一致性百分比（78.0%）
    expect(screen.getByText('78.0%')).toBeTruthy()
    // 反事实叙述文本
    expect(screen.getByText(/如果当时选择向左移动/)).toBeTruthy()
    // 自传体记忆片段数
    expect(screen.getByText('47')).toBeTruthy()
  })

  it('is_sleeping=true 时显示睡眠状态', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          iwsm: {
            self_schema: { self_consistency: 0.3 },
            self_schema_stats: {
              attention_accuracy: 0.2,
              action_accuracy: 0.1,
              affect_error: 0.8,
              self_consistency: 0.3,
            },
            phi_self: 0.05,
            phi_self_history: [0.1, 0.05, 0.05, 0.05],
            is_sleeping: true,
            autobiographical: { n_episodes: 0 },
            counterfactual: {
              narratives_generated: 0,
              last_regret: 0.0,
              last_narrative: '',
            },
          },
        },
      }),
    )
    render(<SelfModelPanel />)
    // 状态：睡眠
    expect(screen.getByText('睡眠')).toBeTruthy()
    // 无叙述提示
    expect(screen.getByText(/暂无反事实叙述/)).toBeTruthy()
  })

  it('空 phi_self_history 时显示等待提示', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          iwsm: {
            self_schema: {},
            self_schema_stats: {},
            phi_self: 0,
            phi_self_history: [],
            is_sleeping: true,
            autobiographical: { n_episodes: 0 },
            counterfactual: {
              narratives_generated: 0,
              last_regret: 0.0,
              last_narrative: '',
            },
          },
        },
      }),
    )
    render(<SelfModelPanel />)
    expect(screen.getByText(/等待 Φ_self 历史/)).toBeTruthy()
  })

  it('未生成反事实叙述时显示提示', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          iwsm: {
            self_schema: {},
            self_schema_stats: {
              attention_accuracy: 0.5,
              action_accuracy: 0.5,
              affect_error: 0.5,
              self_consistency: 0.5,
            },
            phi_self: 0.3,
            phi_self_history: [0.2, 0.3, 0.3],
            is_sleeping: false,
            autobiographical: { n_episodes: 1 },
            counterfactual: {
              narratives_generated: 0,
              last_regret: 0.0,
              last_narrative: '',
            },
          },
        },
      }),
    )
    render(<SelfModelPanel />)
    // "暂无反事实叙述（每 20 步生成一次）"
    expect(screen.getByText(/暂无反事实叙述/)).toBeTruthy()
  })
})
