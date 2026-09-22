import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import type { Snapshot } from '../types'
import { makeSnapshot } from '../test-setup'
import LogicPanel, { fmtVal } from './LogicPanel'

const { mockStore } = vi.hoisted(() => ({ mockStore: vi.fn() }))

vi.mock('../store/useModelStore', () => ({
  useModelStore: mockStore,
}))

function setStoreState(snapshot: Snapshot | null) {
  mockStore.mockImplementation((selector: (s: { snapshot: Snapshot | null }) => unknown) =>
    selector({ snapshot }),
  )
}

describe('fmtVal', () => {
  it('number → 4 位小数', () => {
    expect(fmtVal(0.5)).toBe('0.5000')
    expect(fmtVal(0)).toBe('0.0000')
    expect(fmtVal(3.14159)).toBe('3.1416')
  })

  it('string → 原样返回', () => {
    expect(fmtVal('hello')).toBe('hello')
    expect(fmtVal('123')).toBe('123')
  })

  it('null → —', () => {
    expect(fmtVal(null)).toBe('—')
  })

  it('undefined → —', () => {
    expect(fmtVal(undefined)).toBe('—')
  })

  it('object → JSON.stringify', () => {
    expect(fmtVal({ a: 1 })).toBe('{"a":1}')
    expect(fmtVal([1, 2])).toBe('[1,2]')
  })

  it('NaN → —', () => {
    expect(fmtVal(NaN)).toBe('—')
  })

  it('Infinity → —', () => {
    expect(fmtVal(Infinity)).toBe('—')
    expect(fmtVal(-Infinity)).toBe('—')
  })
})

describe('LogicPanel 组件渲染', () => {
  beforeEach(() => {
    mockStore.mockReset()
  })

  it('snapshot 为 null 时显示等待数据', () => {
    setStoreState(null)
    render(<LogicPanel />)
    expect(screen.getByText('等待数据...')).toBeTruthy()
  })

  it('snapshot 无 logic_violations 时显示未启用提示', () => {
    setStoreState(makeSnapshot({ metadata: { cognitive_upgrades: {} } }))
    render(<LogicPanel />)
    expect(screen.getByText('逻辑约束层未启用')).toBeTruthy()
  })

  it('显示违反次数和总惩罚', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            logic_violations: {
              n_violations: 2,
              total_penalty: 0.8,
              violations: [],
            },
          },
        },
      }),
    )
    render(<LogicPanel />)
    expect(screen.getByText('2')).toBeTruthy()
    expect(screen.getByText('0.8000')).toBeTruthy()
  })

  it('无违反记录时显示占位', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            logic_violations: {
              n_violations: 0,
              total_penalty: 0,
              violations: [],
            },
          },
        },
      }),
    )
    render(<LogicPanel />)
    expect(screen.getByText('暂无违反记录')).toBeTruthy()
  })

  it('渲染违反记录 (规则名、推断、实际、penalty)', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            logic_violations: {
              n_violations: 2,
              total_penalty: 0.8,
              violations: [
                { rule: 'rule_a', inferred: 0.5, actual: 0.3, penalty: 0.6 },
                { rule: 'rule_b', inferred: 'yes', actual: 'no', penalty: 0.2 },
              ],
            },
          },
        },
      }),
    )
    render(<LogicPanel />)
    expect(screen.getByText('rule_a')).toBeTruthy()
    expect(screen.getByText('rule_b')).toBeTruthy()
    // fmtVal 集成: 数字格式化 (推断/实际 的值位于文本节点, 标签在子 span, 跨元素文本用行容器 textContent 匹配)
    const rowA = screen.getByText('rule_a').parentElement!
    expect(rowA).toHaveTextContent(/推断.*0\.5000/)
    expect(rowA).toHaveTextContent(/实际.*0\.3000/)
    // fmtVal 集成: 字符串原样
    const rowB = screen.getByText('rule_b').parentElement!
    expect(rowB).toHaveTextContent(/推断.*yes/)
    expect(rowB).toHaveTextContent(/实际.*no/)
    // penalty 值 (单元素文本)
    expect(screen.getByText('0.600')).toBeTruthy()
    expect(screen.getByText('0.200')).toBeTruthy()
  })

  it('inferred/actual 为 null 时显示 —', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            logic_violations: {
              n_violations: 1,
              total_penalty: 0,
              violations: [{ rule: 'rule_x', inferred: null, actual: undefined, penalty: 0 }],
            },
          },
        },
      }),
    )
    render(<LogicPanel />)
    // fmtVal(null)/fmtVal(undefined) → '—', 跨元素文本用行容器 textContent 匹配
    const row = screen.getByText('rule_x').parentElement!
    expect(row).toHaveTextContent(/推断.*—/)
    expect(row).toHaveTextContent(/实际.*—/)
  })

  it('inferred 为对象时显示 JSON', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            logic_violations: {
              n_violations: 1,
              total_penalty: 0,
              violations: [
                { rule: 'rule_o', inferred: { x: 1 }, actual: [1, 2], penalty: 0 },
              ],
            },
          },
        },
      }),
    )
    render(<LogicPanel />)
    // fmtVal(对象) → JSON.stringify, 跨元素文本用行容器 textContent 匹配
    const row = screen.getByText('rule_o').parentElement!
    expect(row).toHaveTextContent(/推断.*\{"x":1\}/)
    expect(row).toHaveTextContent(/实际.*\[1,2\]/)
  })

  it('penalty > 0.5 时 penalty 文本为红色', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            logic_violations: {
              n_violations: 1,
              total_penalty: 0.6,
              violations: [{ rule: 'rule_severe', inferred: 0.5, actual: 0.3, penalty: 0.6 }],
            },
          },
        },
      }),
    )
    render(<LogicPanel />)
    const penaltyEl = screen.getByText('0.600')
    // penalty > 0.5 → color '#ef4444' (jsdom 可能返回 hex 或 rgb)
    expect(penaltyEl.style.color).toMatch(/ef4444|239.*68.*68/i)
  })
})
