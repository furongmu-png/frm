import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import type { Snapshot } from '../types'
import { makeSnapshot } from '../test-setup'
import ModuleGraphPanel, { fmtErr } from './ModuleGraphPanel'

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
    expect(fmtErr(3.14159)).toBe('3.1416')
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
    expect(fmtErr('123')).toBe('—')
  })
})

describe('ModuleGraphPanel 组件渲染', () => {
  beforeEach(() => {
    mockStore.mockReset()
  })

  it('snapshot 为 null 时显示等待数据', () => {
    setStoreState(null)
    render(<ModuleGraphPanel />)
    expect(screen.getByText('等待数据...')).toBeTruthy()
  })

  it('显示 6 个核心模块名和格式化后的误差值', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          module_errors: {
            perception: 0.1,
            memory: 0.2,
            action: 0.3,
            value: 0.4,
            language: 0.5,
            meta: 0.6,
          },
        },
      }),
    )
    render(<ModuleGraphPanel />)
    expect(screen.getByText('perception')).toBeTruthy()
    expect(screen.getByText('memory')).toBeTruthy()
    expect(screen.getByText('action')).toBeTruthy()
    expect(screen.getByText('value')).toBeTruthy()
    expect(screen.getByText('language')).toBeTruthy()
    expect(screen.getByText('meta')).toBeTruthy()
    // fmtErr 集成: 数值格式化为 4 位小数 (值位于独立 span 内)
    expect(screen.getByText('0.1000')).toBeTruthy()
    expect(screen.getByText('0.2000')).toBeTruthy()
    expect(screen.getByText('0.6000')).toBeTruthy()
  })

  it('module_errors 为空时所有模块误差显示 —', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          module_errors: {},
        },
      }),
    )
    render(<ModuleGraphPanel />)
    // fmtErr(undefined) → '—', 值位于独立 span 内; 6 个模块各一个
    const dashes = screen.getAllByText('—')
    expect(dashes).toHaveLength(6)
  })

  it('显示架构升级的 dormant/split 计数', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          module_errors: { perception: 0 },
          cognitive_upgrades: {
            architecture: { dormant: 2, split: 1 },
          },
        },
      }),
    )
    // dormant/split 计数嵌套在带颜色的 span 内, 使用容器 textContent 匹配跨元素文本
    const { container } = render(<ModuleGraphPanel />)
    expect(container).toHaveTextContent(/dormant\s*=\s*2/)
    expect(container).toHaveTextContent(/split\s*=\s*1/)
  })

  it('无架构升级时不显示 dormant/split', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          module_errors: { perception: 0 },
        },
      }),
    )
    render(<ModuleGraphPanel />)
    expect(screen.queryByText(/架构升级/)).toBeNull()
  })
})
