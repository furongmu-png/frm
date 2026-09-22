import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import type { Snapshot } from '../types'
import { makeSnapshot } from '../test-setup'
import ConsciousnessTheaterPanel from './ConsciousnessTheaterPanel'

const { mockStore } = vi.hoisted(() => ({ mockStore: vi.fn() }))

vi.mock('../store/useModelStore', () => ({
  useModelStore: mockStore,
}))

function setStoreState(snapshot: Snapshot | null) {
  mockStore.mockImplementation((selector: (s: { snapshot: Snapshot | null }) => unknown) =>
    selector({ snapshot }),
  )
}

describe('ConsciousnessTheaterPanel 组件渲染', () => {
  beforeEach(() => {
    mockStore.mockReset()
  })

  it('snapshot 为 null 时显示等待数据', () => {
    setStoreState(null)
    render(<ConsciousnessTheaterPanel />)
    expect(screen.getByText('等待数据...')).toBeTruthy()
  })

  it('snapshot 无 gwt 元数据时显示 GWT 模块未启用', () => {
    setStoreState(makeSnapshot({ metadata: { cognitive_upgrades: {} } }))
    render(<ConsciousnessTheaterPanel />)
    expect(screen.getByText('GWT 模块未启用')).toBeTruthy()
  })

  it('gwt.error 存在时显示错误信息', () => {
    setStoreState(
      makeSnapshot({
        metadata: { gwt: { error: 'boom' } },
      }),
    )
    render(<ConsciousnessTheaterPanel />)
    expect(screen.getByText('boom')).toBeTruthy()
  })

  it('显示 Φ 值、意识状态与时间步', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          gwt: {
            winner: 'consciousness',
            probabilities: { consciousness: 0.7, pcn_L0: 0.3 },
            broadcast_confidence: 0.85,
            global_timestamp: 42,
            phi: 0.2345,
            is_conscious: true,
            phi_history: [0.1, 0.15, 0.2, 0.2345],
            selector: {
              temperature: 1.0,
              step: 42,
              inhibited_modules: [],
              recent_winners: ['pcn_L0'],
              winner_distribution: { consciousness: 0.6, pcn_L0: 0.4 },
            },
            broadcaster: {
              alignment_lr: 0.1,
              n_receivers: 3,
              last_winner: 'consciousness',
              last_confidence: 0.85,
              last_broadcast_3d: [0.1, 0.2, 0.3],
            },
          },
        },
      }),
    )
    render(<ConsciousnessTheaterPanel />)
    // Φ 值
    expect(screen.getByText('0.2345')).toBeTruthy()
    // 意识状态：清醒
    expect(screen.getByText('清醒')).toBeTruthy()
    // 时间步
    expect(screen.getByText('42')).toBeTruthy()
    // 温度
    expect(screen.getByText('1.00')).toBeTruthy()
  })

  it('注意竞争条形图：胜者高亮且显示百分比', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          gwt: {
            winner: 'pcn_L1',
            probabilities: { pcn_L0: 0.2, pcn_L1: 0.5, consciousness: 0.3 },
            broadcast_confidence: 0.6,
            global_timestamp: 1,
            phi: 0.5,
            is_conscious: true,
            phi_history: [0.1, 0.5],
          },
        },
      }),
    )
    render(<ConsciousnessTheaterPanel />)
    // 各模块概率百分比
    expect(screen.getByText('50.0%')).toBeTruthy() // 胜者 pcn_L1 = 0.5
    expect(screen.getByText('30.0%')).toBeTruthy() // consciousness = 0.3
    expect(screen.getByText('20.0%')).toBeTruthy() // pcn_L0 = 0.2
    // 胜者标签含 ★ 标记（textContent 包含 ★ pcn_L1）
    const winnerSpans = screen.getAllByText(/pcn_L1/)
    const winnerMarked = winnerSpans.some((el) => el.textContent?.includes('★'))
    expect(winnerMarked).toBe(true)
  })

  it('is_conscious=false 时显示睡眠', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          gwt: {
            winner: null,
            probabilities: {},
            phi: 0.05,
            is_conscious: false,
            phi_history: [0.1, 0.05],
          },
        },
      }),
    )
    render(<ConsciousnessTheaterPanel />)
    expect(screen.getByText('睡眠')).toBeTruthy()
  })

  it('胜者分布与抑制模块区块渲染', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          gwt: {
            winner: 'pcn_L0',
            probabilities: { pcn_L0: 0.6, pcn_L1: 0.4 },
            broadcast_confidence: 0.5,
            global_timestamp: 5,
            phi: 0.3,
            is_conscious: true,
            phi_history: [0.3],
            selector: {
              recent_winners: ['pcn_L0'],
              winner_distribution: { pcn_L0: 0.8, pcn_L1: 0.2 },
              inhibited_modules: ['pcn_L1', 'consciousness'],
            },
            broadcaster: {
              alignment_lr: 0.1,
              n_receivers: 2,
              last_confidence: 0.5,
              last_broadcast_3d: [0, 0, 0],
            },
          },
        },
      }),
    )
    render(<ConsciousnessTheaterPanel />)
    // 胜者分布显示 80%
    expect(screen.getByText(/pcn_L0: 80%/)).toBeTruthy()
    // 抑制模块显示
    expect(screen.getByText(/pcn_L1, consciousness/)).toBeTruthy()
    // 广播器对齐率
    expect(screen.getByText('0.10')).toBeTruthy()
  })

  it('概率为空时显示无竞争候选', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          gwt: {
            winner: null,
            probabilities: {},
            phi: 0,
            is_conscious: false,
            phi_history: [],
          },
        },
      }),
    )
    render(<ConsciousnessTheaterPanel />)
    expect(screen.getByText('无竞争候选')).toBeTruthy()
  })

  it('广播内容雷达图渲染（含胜者与置信度）', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          gwt: {
            winner: 'consciousness',
            probabilities: { consciousness: 1.0 },
            broadcast_confidence: 0.85,
            phi: 0.4,
            is_conscious: true,
            phi_history: [0.4],
            broadcaster: {
              alignment_lr: 0.1,
              n_receivers: 4,
              last_confidence: 0.85,
              last_broadcast_3d: [0.5, 0.5, 0.5],
            },
          },
        },
      }),
    )
    const { container } = render(<ConsciousnessTheaterPanel />)
    // 雷达图区块含 polygon
    const polygons = container.querySelectorAll('polygon')
    expect(polygons.length).toBeGreaterThan(0)
    // 置信度百分比渲染（雷达图 + 广播器均显示，至少一处）
    expect(screen.getAllByText('85.0%').length).toBeGreaterThan(0)
  })
})
