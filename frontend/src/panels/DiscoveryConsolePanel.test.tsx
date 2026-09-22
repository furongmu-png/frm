import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import type { Snapshot } from '../types'
import { makeSnapshot } from '../test-setup'
import DiscoveryConsolePanel from './DiscoveryConsolePanel'

const { mockStore } = vi.hoisted(() => ({ mockStore: vi.fn() }))

vi.mock('../store/useModelStore', () => ({
  useModelStore: mockStore,
}))

function setStoreState(snapshot: Snapshot | null) {
  mockStore.mockImplementation((selector: (s: { snapshot: Snapshot | null }) => unknown) =>
    selector({ snapshot }),
  )
}

describe('DiscoveryConsolePanel 组件渲染', () => {
  beforeEach(() => {
    mockStore.mockReset()
  })

  it('snapshot 为 null 时显示等待数据', () => {
    setStoreState(null)
    render(<DiscoveryConsolePanel />)
    expect(screen.getByText('等待数据...')).toBeTruthy()
  })

  it('snapshot 无 discovery 元数据时显示引擎未启用', () => {
    setStoreState(makeSnapshot({ metadata: { cognitive_upgrades: {} } }))
    render(<DiscoveryConsolePanel />)
    expect(screen.getByText('科学发现引擎未启用')).toBeTruthy()
  })

  it('discovery.error 存在时显示错误信息', () => {
    setStoreState(
      makeSnapshot({
        metadata: { discovery: { error: 'sandbox exploded' } },
      }),
    )
    render(<DiscoveryConsolePanel />)
    expect(screen.getByText('sandbox exploded')).toBeTruthy()
  })

  it('enabled=false 时显示已禁用', () => {
    setStoreState(
      makeSnapshot({
        metadata: { discovery: { enabled: false } },
      }),
    )
    render(<DiscoveryConsolePanel />)
    expect(screen.getByText('科学发现引擎已禁用')).toBeTruthy()
  })

  it('显示汇总指标：循环数/假设/实验/接受/拒绝/论文', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          discovery: {
            enabled: true,
            trigger_interval: 5000,
            step: 100,
            cycle_count: 3,
            total_papers: 2,
            pending_approvals: 0,
            require_approval: false,
            hypothesis_generator: { max_hypotheses: 10, generated_count: 15 },
            experiment_designer: { designed_count: 8, sandbox_attached: true },
            result_analyzer: {
              total_analyses: 8,
              accepted: 3,
              rejected: 2,
              inconclusive: 3,
            },
            paper_writer: {
              paper_count: 2,
              recent_titles: ['发现报告：质量守恒...'],
              recent_decisions: ['accept'],
            },
            recent_cycles: [],
          },
        },
      }),
    )
    render(<DiscoveryConsolePanel />)
    // 汇总指标标签
    expect(screen.getByText('循环数')).toBeTruthy()
    expect(screen.getByText('假设')).toBeTruthy()
    expect(screen.getByText('实验')).toBeTruthy()
    expect(screen.getByText('论文')).toBeTruthy()
    // 数值（假设数 15 在汇总与子模块状态中均出现，用 getAllByText）
    expect(screen.getAllByText('15').length).toBeGreaterThan(0)
    expect(screen.getAllByText('2').length).toBeGreaterThan(0) // 论文数
  })

  it('触发进度条渲染剩余步数', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          discovery: {
            enabled: true,
            trigger_interval: 100,
            step: 30,
            cycle_count: 0,
            hypothesis_generator: {},
            experiment_designer: {},
            result_analyzer: {},
            paper_writer: {},
            recent_cycles: [],
          },
        },
      }),
    )
    render(<DiscoveryConsolePanel />)
    // 30 % 100 = 30, 剩 100-30=70 步
    expect(screen.getByText('剩 70 步')).toBeTruthy()
  })

  it('本次循环 last_cycle 高亮渲染与发现摘要', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          discovery: {
            enabled: true,
            trigger_interval: 1,
            step: 1,
            cycle_count: 1,
            last_cycle: {
              cycle_id: 0,
              n_hypotheses: 5,
              n_experiments: 3,
              n_accepted: 1,
              n_rejected: 1,
              papers_written: 2,
              findings: ['ACCEPT: 物体质量越大碰撞后速度变化越小'],
              error: null,
            },
            hypothesis_generator: {},
            experiment_designer: {},
            result_analyzer: {},
            paper_writer: {},
            recent_cycles: [],
          },
        },
      }),
    )
    render(<DiscoveryConsolePanel />)
    // 本次循环标题
    expect(screen.getByText(/本次循环/)).toBeTruthy()
    // 发现摘要
    expect(screen.getByText('ACCEPT: 物体质量越大碰撞后速度变化越小')).toBeTruthy()
  })

  it('近期循环列表渲染（按倒序）', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          discovery: {
            enabled: true,
            trigger_interval: 100,
            step: 50,
            cycle_count: 2,
            hypothesis_generator: {},
            experiment_designer: {},
            result_analyzer: {},
            paper_writer: {},
            recent_cycles: [
              {
                cycle_id: 0,
                n_hypotheses: 2,
                n_experiments: 1,
                n_accepted: 1,
                n_rejected: 0,
                papers_written: 1,
                findings: ['ACCEPT: 发现1'],
                error: null,
              },
              {
                cycle_id: 1,
                n_hypotheses: 3,
                n_experiments: 2,
                n_accepted: 0,
                n_rejected: 1,
                papers_written: 1,
                findings: ['REJECT: 拒绝1'],
                error: null,
              },
            ],
          },
        },
      }),
    )
    render(<DiscoveryConsolePanel />)
    // 列表项 id
    expect(screen.getByText('#0')).toBeTruthy()
    expect(screen.getByText('#1')).toBeTruthy()
    // 接受/拒绝计数
    expect(screen.getByText('✓1')).toBeTruthy() // cycle 0: 1 accepted
    expect(screen.getByText('✗1')).toBeTruthy() // cycle 1: 1 rejected
  })

  it('发现日志：论文列表与决策徽标', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          discovery: {
            enabled: true,
            trigger_interval: 100,
            step: 50,
            cycle_count: 1,
            hypothesis_generator: {},
            experiment_designer: {},
            result_analyzer: {},
            paper_writer: {
              paper_count: 2,
              recent_titles: ['发现报告：动量守恒...', '发现报告：摩擦无关...'],
              recent_decisions: ['accept', 'reject'],
            },
            recent_cycles: [],
          },
        },
      }),
    )
    render(<DiscoveryConsolePanel />)
    // 决策徽标（接受/拒绝 在指标标签与论文徽标中均出现，至少各一处）
    expect(screen.getAllByText('接受').length).toBeGreaterThan(0)
    expect(screen.getAllByText('拒绝').length).toBeGreaterThan(0)
    // 论文标题
    expect(screen.getByText('发现报告：动量守恒...')).toBeTruthy()
    expect(screen.getByText('发现报告：摩擦无关...')).toBeTruthy()
  })

  it('发现日志：暂无论文时显示占位', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          discovery: {
            enabled: true,
            trigger_interval: 100,
            step: 50,
            cycle_count: 0,
            hypothesis_generator: {},
            experiment_designer: {},
            result_analyzer: {},
            paper_writer: { paper_count: 0, recent_titles: [], recent_decisions: [] },
            recent_cycles: [],
          },
        },
      }),
    )
    render(<DiscoveryConsolePanel />)
    expect(screen.getByText('暂无论文')).toBeTruthy()
  })

  it('实验队列空时显示占位', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          discovery: {
            enabled: true,
            trigger_interval: 100,
            step: 10,
            cycle_count: 0,
            hypothesis_generator: {},
            experiment_designer: {},
            result_analyzer: {},
            paper_writer: {},
            recent_cycles: [],
          },
        },
      }),
    )
    render(<DiscoveryConsolePanel />)
    expect(screen.getByText('暂无循环记录')).toBeTruthy()
  })

  it('沙盒未挂载时显示未挂载标记', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          discovery: {
            enabled: true,
            trigger_interval: 100,
            step: 10,
            cycle_count: 0,
            hypothesis_generator: {},
            experiment_designer: { designed_count: 2, sandbox_attached: false },
            result_analyzer: {},
            paper_writer: {},
            recent_cycles: [],
          },
        },
      }),
    )
    render(<DiscoveryConsolePanel />)
    expect(screen.getByText('未挂载')).toBeTruthy()
  })

  it('require_approval 且有待审批实验时显示审批提示', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          discovery: {
            enabled: true,
            trigger_interval: 100,
            step: 50,
            cycle_count: 0,
            require_approval: true,
            pending_approvals: 3,
            hypothesis_generator: {},
            experiment_designer: {},
            result_analyzer: {},
            paper_writer: {},
            recent_cycles: [],
          },
        },
      }),
    )
    render(<DiscoveryConsolePanel />)
    expect(screen.getByText(/待审批实验: 3/)).toBeTruthy()
  })
})
