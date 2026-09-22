import '@testing-library/jest-dom'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'
import type { Snapshot } from './types'

// Auto-unmount React components after each test to avoid cross-test leakage.
afterEach(() => {
  cleanup()
})

// Shared factory for constructing a minimal valid Snapshot in tests.
// Panels under test only read `snapshot` truthiness and `snapshot.metadata`,
// so most fields use inert defaults; tests override only what they need.
export function makeSnapshot(overrides: Partial<Snapshot> = {}): Snapshot {
  return {
    step: 0,
    timestamp: 0,
    modality: 'test',
    frame_b64: null,
    text_block: '',
    action: 0,
    free_energy: 0,
    prediction_error: 0,
    beta: 1,
    belief_state: [],
    causal_graph: { nodes: [], edges: [] },
    categories: {},
    kg_update: { new_nodes: [], new_edges: [] },
    self_authoring: null,
    event: '',
    ...overrides,
  }
}
