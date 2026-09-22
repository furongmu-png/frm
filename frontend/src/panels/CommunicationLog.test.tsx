import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import type { Snapshot } from '../types'
import { makeSnapshot } from '../test-setup'
import CommunicationLog from './CommunicationLog'

const { mockStore } = vi.hoisted(() => ({ mockStore: vi.fn() }))

vi.mock('../store/useModelStore', () => ({
  useModelStore: mockStore,
}))

function setStoreState(snapshot: Snapshot | null, history: Snapshot[] = []) {
  mockStore.mockImplementation(
    (selector: (s: { snapshot: Snapshot | null; history: Snapshot[] }) => unknown) =>
      selector({ snapshot, history }),
  )
}

describe('CommunicationLog 组件渲染', () => {
  beforeEach(() => {
    mockStore.mockReset()
  })

  it('snapshot 为 null 时显示等待数据', () => {
    setStoreState(null)
    render(<CommunicationLog />)
    expect(screen.getByText('等待数据...')).toBeTruthy()
  })

  it('snapshot 无通信数据时显示占位提示', () => {
    setStoreState(makeSnapshot({ metadata: { cognitive_upgrades: {} } }))
    render(<CommunicationLog />)
    expect(screen.getByText('启用 MultiAgentWorld 后将显示通信记录')).toBeTruthy()
  })

  it('渲染表头 (step/sender/symbol/event)', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            communication: [
              { step: 1, sender: 'agent_a', symbol: 'MSG', event: 'hello' },
            ],
          },
        },
      }),
    )
    render(<CommunicationLog />)
    expect(screen.getByText('step')).toBeTruthy()
    expect(screen.getByText('sender')).toBeTruthy()
    expect(screen.getByText('symbol')).toBeTruthy()
    expect(screen.getByText('event')).toBeTruthy()
  })

  it('渲染 communication 数组条目', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            communication: [
              { step: 1, sender: 'agent_a', symbol: 'MSG', event: 'hello' },
              { step: 2, sender: 'agent_b', symbol: 'ACK', event: 'acknowledged' },
            ],
          },
        },
      }),
    )
    render(<CommunicationLog />)
    expect(screen.getByText('agent_a')).toBeTruthy()
    expect(screen.getByText('agent_b')).toBeTruthy()
    expect(screen.getByText('hello')).toBeTruthy()
    expect(screen.getByText('acknowledged')).toBeTruthy()
    expect(screen.getByText('MSG')).toBeTruthy()
    expect(screen.getByText('ACK')).toBeTruthy()
  })

  it('兼容 multiagent 字段', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            multiagent: [{ step: 5, sender: 'agent_c', symbol: 'SYNC', event: 'synced' }],
          },
        },
      }),
    )
    render(<CommunicationLog />)
    expect(screen.getByText('agent_c')).toBeTruthy()
    expect(screen.getByText('synced')).toBeTruthy()
  })

  it('兼容 { entries: [...] } 形态', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            communication: {
              entries: [{ step: 3, sender: 'agent_d', symbol: 'DATA', event: 'payload' }],
            },
          },
        },
      }),
    )
    render(<CommunicationLog />)
    expect(screen.getByText('agent_d')).toBeTruthy()
    expect(screen.getByText('payload')).toBeTruthy()
  })

  it('条目缺失字段时显示 -', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            communication: [{ step: 1 }],
          },
        },
      }),
    )
    render(<CommunicationLog />)
    // sender/symbol 缺失 → '-'
    expect(screen.getAllByText('-')).toHaveLength(2)
  })
})

describe('CommunicationLog auto-scroll 逻辑', () => {
  beforeEach(() => {
    mockStore.mockReset()
  })

  it('stickToBottomRef 初始为 true 时组件正常渲染日志容器', () => {
    // stickToBottomRef 初始化为 true, 首次渲染时 useEffect 尝试滚动到底部。
    // jsdom 不支持真实布局 (scrollHeight=0), 此测试验证 effect 不抛异常。
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            communication: [
              { step: 1, sender: 'agent_a', symbol: 'MSG', event: 'hello' },
            ],
          },
        },
      }),
      [makeSnapshot()],
    )
    render(<CommunicationLog />)
    expect(screen.getByText('agent_a')).toBeTruthy()
  })

  it('滚动事件不导致崩溃 (onScroll 更新 stickToBottomRef)', () => {
    setStoreState(
      makeSnapshot({
        metadata: {
          cognitive_upgrades: {
            communication: [
              { step: 1, sender: 'agent_a', symbol: 'MSG', event: 'hello' },
            ],
          },
        },
      }),
    )
    const { container } = render(<CommunicationLog />)
    // 找到日志容器 (带 onScroll 的 div, 样式含 overflow-y: auto)
    const logDiv = container.querySelector('[style*="overflow-y"]')
    expect(logDiv).toBeTruthy()
    // jsdom 中 scrollHeight/clientHeight 默认为 0 (只读, 无法通过 target 设置);
    // 直接派发 scroll 事件验证 onScroll handler 不抛异常
    fireEvent.scroll(logDiv!)
    // 验证滚动后条目仍在
    expect(screen.getByText('agent_a')).toBeTruthy()
  })
})
