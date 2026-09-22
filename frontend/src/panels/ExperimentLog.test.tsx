import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import type { Snapshot } from '../types'
import { makeSnapshot } from '../test-setup'
import ExperimentLog, { fmtErr } from './ExperimentLog'

const { mockStore } = vi.hoisted(() => ({ mockStore: vi.fn() }))

vi.mock('../store/useModelStore', () => ({
  useModelStore: mockStore,
}))

function setStoreState(snapshot: Snapshot | null) {
  mockStore.mockImplementation((selector: (s: { snapshot: Snapshot | null }) => unknown) =>
    selector({ snapshot }),
  )
}

describe('fmtErr', () => {
  it('正常有限数字 → 4 位小数', () => {
    expect(fmtErr(0.1234)).toBe('0.1234')
    expect(fmtErr(1.5)).toBe('1.5000')
    expect(fmtErr(0)).toBe('0.0000')
  })

  it('NaN → —', () => {
    expect(fmtErr(NaN)).toBe('—')
  })

  it('undefined → —', () => {
    expect(fmtErr(undefined)).toBe('—')
  })

  it('Infinity → —', () => {
    expect(fmtErr(Infinity)).toBe('—')
    expect(fmtErr(-Infinity)).toBe('—')
  })

  it('字符串 → —', () => {
    expect(fmtErr('abc')).toBe('—')
  })
})

describe('ExperimentLog 组件渲染', () => {
  beforeEach(() => {
    mockStore.mockReset()
  })

  it('snapshot 为 null 时显示等待数据', () => {
    setStoreState(null)
    render(<ExperimentLog />)
    expect(screen.getByText('等待数据...')).toBeTruthy()
  })

  it('snapshot 无 experiment 时显示未启用提示', () => {
    setStoreState(makeSnapshot({ metadata: { cognitive_upgrades: {} } }))
    render(<ExperimentLog />)
    expect(screen.getByText('实验规划器未启用')).toBeTruthy()
  })

  it('显示当前实验名称、干预和预期增益', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            experiment: {
              name: 'exp_1',
              intervention: 'increase_lr',
              predicted_gain: 0.5,
              param_uncertainty: 0.3,
            },
          },
        },
      }),
    )
    render(<ExperimentLog />)
    expect(screen.getByText('exp_1')).toBeTruthy()
    expect(screen.getByText('increase_lr')).toBeTruthy()
    expect(screen.getByText('0.5000')).toBeTruthy()
    expect(screen.getByText('0.300')).toBeTruthy()
  })

  it('predicted_gain 为 NaN 时显示 —', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            experiment: {
              name: 'exp_1',
              intervention: 'test',
              predicted_gain: NaN,
            },
          },
        },
      }),
    )
    render(<ExperimentLog />)
    expect(screen.getByText('—')).toBeTruthy()
  })

  it('渲染历史实验记录', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            experiment: {
              name: 'exp_current',
              intervention: 'current',
              predicted_gain: 0.5,
              history: [
                { name: 'exp_0', intervention: 'baseline', predicted_gain: 0.2 },
                { name: 'exp_1', intervention: 'variant_a', predicted_gain: 0.3 },
              ],
            },
          },
        },
      }),
    )
    render(<ExperimentLog />)
    expect(screen.getByText('baseline')).toBeTruthy()
    expect(screen.getByText('variant_a')).toBeTruthy()
    expect(screen.getByText('0.2000')).toBeTruthy()
    expect(screen.getByText('0.3000')).toBeTruthy()
  })

  it('无历史实验时显示占位', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            experiment: {
              name: 'exp_1',
              intervention: 'test',
              predicted_gain: 0.1,
            },
          },
        },
      }),
    )
    render(<ExperimentLog />)
    expect(screen.getByText('暂无历史实验')).toBeTruthy()
  })
})
