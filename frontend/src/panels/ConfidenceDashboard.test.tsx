import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import type { Snapshot } from '../types'
import { makeSnapshot } from '../test-setup'
import ConfidenceDashboard from './ConfidenceDashboard'

const { mockStore } = vi.hoisted(() => ({ mockStore: vi.fn() }))

vi.mock('../store/useModelStore', () => ({
  useModelStore: mockStore,
}))

function setStoreState(snapshot: Snapshot | null) {
  mockStore.mockImplementation((selector: (s: { snapshot: Snapshot | null }) => unknown) =>
    selector({ snapshot }),
  )
}

describe('ConfidenceDashboard 数据格式化逻辑', () => {
  beforeEach(() => {
    mockStore.mockReset()
  })

  it('confidence 为有限数字时正常显示 (含 clamp)', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            meta_cognition: { confidence: 75, mode: 'explore', mean_uncertainty: 0.3 },
          },
        },
      }),
    )
    render(<ConfidenceDashboard />)
    expect(screen.getByText('75.0%')).toBeTruthy()
    expect(screen.getByText('0.300')).toBeTruthy()
    expect(screen.getByText('explore')).toBeTruthy()
  })

  it('confidence 超出 100 时 clamp 到 100', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            meta_cognition: { confidence: 150, mode: 'exploit' },
          },
        },
      }),
    )
    render(<ConfidenceDashboard />)
    expect(screen.getByText('100.0%')).toBeTruthy()
  })

  it('confidence 为负数时 clamp 到 0', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            meta_cognition: { confidence: -10, mode: 'safe' },
          },
        },
      }),
    )
    render(<ConfidenceDashboard />)
    expect(screen.getByText('0.0%')).toBeTruthy()
  })

  it('confidence 为 NaN 时回退到 0 (不显示 NaN%)', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            meta_cognition: { confidence: NaN, mode: 'balanced', mean_uncertainty: NaN },
          },
        },
      }),
    )
    render(<ConfidenceDashboard />)
    expect(screen.getByText('0.0%')).toBeTruthy()
    expect(screen.getByText('0.000')).toBeTruthy()
  })

  it('mean_uncertainty 超出 [0,1] 时 clamp', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            meta_cognition: { confidence: 50, mode: 'balanced', mean_uncertainty: 5 },
          },
        },
      }),
    )
    render(<ConfidenceDashboard />)
    expect(screen.getByText('1.000')).toBeTruthy()
  })
})

describe('ConfidenceDashboard 组件渲染', () => {
  beforeEach(() => {
    mockStore.mockReset()
  })

  it('snapshot 为 null 时显示等待数据', () => {
    setStoreState(null)
    render(<ConfidenceDashboard />)
    expect(screen.getByText('等待数据...')).toBeTruthy()
  })

  it('snapshot 无 meta_cognition 时显示未启用提示', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {},
        },
      }),
    )
    render(<ConfidenceDashboard />)
    expect(screen.getByText('元认知未启用')).toBeTruthy()
  })

  it('mode 缺失时默认为 balanced', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            meta_cognition: { confidence: 50 },
          },
        },
      }),
    )
    render(<ConfidenceDashboard />)
    expect(screen.getByText('balanced')).toBeTruthy()
  })
})
