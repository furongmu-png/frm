import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import type { Snapshot } from '../types'
import { makeSnapshot } from '../test-setup'
import JEPALatentPanel from './JEPALatentPanel'

const { mockStore } = vi.hoisted(() => ({ mockStore: vi.fn() }))

vi.mock('../store/useModelStore', () => ({
  useModelStore: mockStore,
}))

function setStoreState(snapshot: Snapshot | null) {
  mockStore.mockImplementation((selector: (s: { snapshot: Snapshot | null }) => unknown) =>
    selector({ snapshot }),
  )
}

describe('JEPALatentPanel 组件渲染', () => {
  beforeEach(() => {
    mockStore.mockReset()
  })

  it('snapshot 为 null 时显示等待数据', () => {
    setStoreState(null)
    render(<JEPALatentPanel />)
    expect(screen.getByText('等待数据...')).toBeTruthy()
  })

  it('snapshot 无 jepa 元数据时显示 JEPA 模块未启用', () => {
    setStoreState(makeSnapshot({ metadata: { cognitive_upgrades: {} } }))
    render(<JEPALatentPanel />)
    expect(screen.getByText('JEPA 模块未启用')).toBeTruthy()
  })

  it('显示 λ_jepa / jepa_error / 对齐度 / 合并自由能', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          jepa: {
            enabled: true,
            lambda_jepa: 0.5,
            jepa_error: 0.1234,
            alignment: 0.85,
            combined_free_energy: 1.5,
            online_3d: [0.2, 0.1, 0.0],
            target_3d: [0.3, 0.1, 0.0],
            error_trend: [0.5, 0.4, 0.3, 0.1234],
          },
        },
      }),
    )
    render(<JEPALatentPanel />)
    // lambda
    expect(screen.getByText('0.50')).toBeTruthy()
    // jepa_error
    expect(screen.getByText('0.1234')).toBeTruthy()
    // alignment
    expect(screen.getByText('0.8500')).toBeTruthy()
    // combined free energy
    expect(screen.getByText('1.5000')).toBeTruthy()
  })

  it('高对齐度时对齐度颜色为绿色', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          jepa: {
            lambda_jepa: 0.5,
            jepa_error: 0.01,
            alignment: 0.9,
            online_3d: [0.1, 0.1, 0.0],
            target_3d: [0.1, 0.1, 0.0],
          },
        },
      }),
    )
    render(<JEPALatentPanel />)
    const alignEl = screen.getByText('0.9000')
    // 高对齐（>0.7）→ 绿色
    expect(alignEl.style.color).toMatch(/4ade80|74.*222.*128/i)
  })

  it('jepa.error 存在时显示错误信息', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          jepa: { error: 'encoder failed' },
        },
      }),
    )
    render(<JEPALatentPanel />)
    expect(screen.getByText('encoder failed')).toBeTruthy()
  })

  it('渲染 3D 投影 SVG（含 target 圆与 online 方块）', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          jepa: {
            lambda_jepa: 0.5,
            jepa_error: 0.1,
            alignment: 0.5,
            online_3d: [0.3, 0.2, 0.1],
            target_3d: [0.2, 0.3, 0.1],
          },
        },
      }),
    )
    const { container } = render(<JEPALatentPanel />)
    // SVG 元素存在
    const svg = container.querySelector('svg')
    expect(svg).toBeTruthy()
    // 含 circle (target) 与 rect (online) 与 line (连接线)
    expect(svg!.querySelector('circle')).toBeTruthy()
    expect(svg!.querySelector('rect')).toBeTruthy()
    expect(svg!.querySelector('line')).toBeTruthy()
  })

  it('error_trend 多点时渲染趋势迷你图', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          jepa: {
            lambda_jepa: 0.5,
            jepa_error: 0.1,
            alignment: 0.5,
            error_trend: [0.5, 0.4, 0.3, 0.2, 0.1],
          },
        },
      }),
    )
    const { container } = render(<JEPALatentPanel />)
    // 趋势区有两个 svg：主投影 + 趋势图
    const svgs = container.querySelectorAll('svg')
    expect(svgs.length).toBeGreaterThanOrEqual(2)
    // 趋势图含 polyline
    const polyline = svgs[1].querySelector('polyline')
    expect(polyline).toBeTruthy()
  })
})
