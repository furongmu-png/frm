import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import type { Snapshot } from '../types'
import { makeSnapshot } from '../test-setup'
import SkillsPanel from './SkillsPanel'

const { mockStore } = vi.hoisted(() => ({ mockStore: vi.fn() }))

vi.mock('../store/useModelStore', () => ({
  useModelStore: mockStore,
}))

function setStoreState(snapshot: Snapshot | null) {
  mockStore.mockImplementation((selector: (s: { snapshot: Snapshot | null }) => unknown) =>
    selector({ snapshot }),
  )
}

describe('SkillsPanel 组件渲染', () => {
  beforeEach(() => {
    mockStore.mockReset()
  })

  it('snapshot 为 null 时显示等待数据', () => {
    setStoreState(null)
    render(<SkillsPanel />)
    expect(screen.getByText('等待数据...')).toBeTruthy()
  })

  it('snapshot 无 skills 元数据时显示技能层未启用', () => {
    setStoreState(makeSnapshot({ metadata: { cognitive_upgrades: {} } }))
    render(<SkillsPanel />)
    expect(screen.getByText('技能层未启用')).toBeTruthy()
  })

  it('渲染汇总统计（总数/已启用/就绪/错误）', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          skills: {
            depth_estimator: { enabled: true, data: { depth_mean: 0.42 } },
            dialogue: { enabled: true, data: {} },
            theorem_proving: { enabled: true, data: {}, error: 'boom' },
            game_playing: { enabled: false, data: {} },
          },
        },
      }),
    )
    render(<SkillsPanel />)
    // 总数恒为 20（前端清单）
    expect(screen.getByText('20')).toBeTruthy()
    // 已启用 3（game_playing 被禁用）
    expect(screen.getByText('3')).toBeTruthy()
  })

  it('就绪技能显示关键指标，未就绪显示状态标签', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          skills: {
            depth_estimator: { enabled: true, data: { depth_mean: 0.5 } },
            dialogue: { enabled: false, data: {} },
          },
        },
      }),
    )
    render(<SkillsPanel />)
    // depth_estimator 标签存在
    expect(screen.getByText('深度估计')).toBeTruthy()
    // 关键指标渲染
    expect(screen.getByText('mean=0.500')).toBeTruthy()
    // dialogue 被禁用 → 显示"已禁用"
    expect(screen.getByText('已禁用')).toBeTruthy()
  })

  it('出错技能渲染错误状态标签', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          skills: {
            anomaly_detection: { enabled: true, data: { alert: true } },
            theorem_proving: { enabled: true, data: { proved: true }, error: 'fail' },
          },
        },
      }),
    )
    render(<SkillsPanel />)
    // anomaly_detection 告警指标
    expect(screen.getByText('⚠ ALERT')).toBeTruthy()
    // theorem_proving 标签
    expect(screen.getByText('定理证明')).toBeTruthy()
    // theorem_proving 有 error → 状态为"出错"
    expect(screen.getByText('出错')).toBeTruthy()
  })

  it('按 5 个维度分组渲染全部 20 项技能标签', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          skills: {},
        },
      }),
    )
    render(<SkillsPanel />)
    // 维度组标签
    expect(screen.getByText('感知扩展')).toBeTruthy()
    expect(screen.getByText('认知深化')).toBeTruthy()
    expect(screen.getByText('交互协作')).toBeTruthy()
    expect(screen.getByText('专业技能')).toBeTruthy()
    expect(screen.getByText('元技能')).toBeTruthy()
    // 抽样若干技能标签
    expect(screen.getByText('深度估计')).toBeTruthy()
    expect(screen.getByText('心理理论')).toBeTruthy()
    expect(screen.getByText('自然语言对话')).toBeTruthy()
    expect(screen.getByText('程序合成')).toBeTruthy()
    expect(screen.getByText('学习如何学习')).toBeTruthy()
  })
})
