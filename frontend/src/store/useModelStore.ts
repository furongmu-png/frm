// frontend/src/store/useModelStore.ts
// Zustand store — central state for the Window of Consciousness.

import { create } from 'zustand';
import type { Snapshot, GraphData, Milestone, WSCommand, LatentPoint } from '../types';
import { apiUrl, wsUrl } from '../config/runtime';

const MAX_HISTORY = 2000;

interface ModelState {
  // Connection
  connected: boolean;
  ws: WebSocket | null;

  // Current state
  snapshot: Snapshot | null;
  history: Snapshot[];

  // Playback
  //   playbackIndex === -1  → real-time mode (follow latest snapshot)
  //   playbackIndex >= 0    → replay mode (render history[playbackIndex])
  playbackIndex: number;

  // Controls
  paused: boolean;
  speed: number;
  currentStep: number;
  beta: number;

  // Knowledge graph
  knowledgeGraph: GraphData;
  milestones: Milestone[];

  // Self-authoring
  authoringMessages: { role: 'model' | 'user'; text: string; step: number }[];

  // Actions
  connect: () => void;
  disconnect: () => void;
  sendCommand: (cmd: WSCommand) => void;
  addSnapshot: (snap: Snapshot) => void;
  setPlaybackIndex: (idx: number) => void;
  getLatentPoints: () => import('../types').LatentPoint[];
  requestHistory: (start: number, end?: number) => Promise<void>;
  requestGraph: () => Promise<void>;
  requestMilestones: () => Promise<void>;
  injectQuestion: (question: string) => void;
}

/**
 * Resolve the snapshot to render given the current playback state.
 * Returns the playback snapshot when in replay mode, otherwise the
 * latest live snapshot.
 */
export function selectActiveSnapshot(s: ModelState): Snapshot | null {
  if (s.playbackIndex >= 0 && s.playbackIndex < s.history.length) {
    return s.history[s.playbackIndex];
  }
  return s.snapshot;
}

export const useModelStore = create<ModelState>((set, get) => ({
  connected: false,
  ws: null,
  snapshot: null,
  history: [],
  playbackIndex: -1, // -1 = real-time
  paused: false,
  speed: 1.0,
  currentStep: 0,
  beta: 0,
  knowledgeGraph: { nodes: [], edges: [] },
  milestones: [],
  authoringMessages: [],

  connect: () => {
    const ws = new WebSocket(wsUrl());
    ws.onopen = () => {
      set({ connected: true, ws });
      console.log('[ws] connected');
      // Fetch initial data.
      get().requestGraph();
      get().requestMilestones();
    };
    ws.onclose = () => {
      set({ connected: false, ws: null });
      console.log('[ws] disconnected');
    };
    ws.onerror = (err) => {
      console.error('[ws] error', err);
    };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'history' && Array.isArray(msg.snapshots)) {
          // Initial history burst from the server (last 500 items).
          const snaps = msg.snapshots as Snapshot[];
          if (snaps.length > 0) {
            set((state) => {
              const merged = [...state.history, ...snaps].slice(-MAX_HISTORY);
              const latest = snaps[snaps.length - 1];
              return {
                history: merged,
                snapshot: state.snapshot ?? latest,
                currentStep: latest.step ?? state.currentStep,
                beta: latest.beta ?? state.beta,
              };
            });
          }
        } else if (msg.step !== undefined) {
          // It's a snapshot.
          get().addSnapshot(msg as Snapshot);
        } else if (msg.type === 'pause' || msg.type === 'resume') {
          set({ paused: msg.type === 'pause' });
        }
      } catch {
        // Non-JSON message; ignore.
      }
    };
  },

  disconnect: () => {
    const { ws } = get();
    if (ws) {
      ws.close();
      set({ ws: null, connected: false });
    }
  },

  addSnapshot: (snap) => {
    set((state) => {
      const history = [...state.history, snap].slice(-MAX_HISTORY);
      const messages = [...state.authoringMessages];
      if (snap.self_authoring) {
        messages.push({ role: 'model', text: snap.self_authoring, step: snap.step });
      }
      // In real-time mode (playbackIndex === -1), update the active
      // snapshot to the new arrival. In replay mode, leave the
      // active snapshot pinned to the replayed index.
      //
      // 回放陈旧修复策略 (Option A，已正确实现，无需改动):
      // 当新快照到达且用户处于回放模式 (playbackIndex >= 0) 时，仅将新快照
      // 追加到 history，但 *不* 改变 playbackIndex，也不更新当前展示的
      // snapshot / currentStep / beta —— 让用户继续浏览历史快照而不被打断。
      // 当用户点击“返回实时” (setPlaybackIndex(-1)) 时，再切回最新快照。
      // 实时模式 (playbackIndex === -1) 下，活动快照即新到达的最新快照。
      const inReplay = state.playbackIndex >= 0;
      return {
        history,
        snapshot: inReplay ? state.snapshot : snap,
        currentStep: inReplay ? state.currentStep : snap.step,
        beta: inReplay ? state.beta : (snap.beta ?? state.beta),
        authoringMessages: messages,
      };
    });
    // Update KG incrementally.
    if (snap.kg_update?.new_nodes?.length || snap.kg_update?.new_edges?.length) {
      set((state) => {
        const existingNodes = new Set(state.knowledgeGraph.nodes.map((n) => n.id));
        const newNodes = snap.kg_update.new_nodes
          .filter((id) => !existingNodes.has(id))
          .map((id) => ({ id, label: id, size: 10, group: 'default' }));
        const newEdges = snap.kg_update.new_edges.map((e) => ({
          source: e.source,
          target: e.target,
          weight: e.weight,
          type: 'link',
        }));
        return {
          knowledgeGraph: {
            nodes: [...state.knowledgeGraph.nodes, ...newNodes],
            edges: [...state.knowledgeGraph.edges, ...newEdges],
          },
        };
      });
    }
  },

  setPlaybackIndex: (idx) => {
    set((state) => {
      if (idx < 0 || idx >= state.history.length) {
        // Out of range → return to real-time mode.
        const latest = state.history[state.history.length - 1] ?? state.snapshot;
        return {
          playbackIndex: -1,
          snapshot: latest ?? null,
          currentStep: latest?.step ?? state.currentStep,
          beta: latest?.beta ?? state.beta,
        };
      }
      const snap = state.history[idx];
      return {
        playbackIndex: idx,
        snapshot: snap,
        currentStep: snap.step,
        beta: snap.beta ?? state.beta,
      };
    });
  },

  sendCommand: (cmd) => {
    const { ws } = get();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(cmd));
    }
    // Optimistic local state update.
    if (cmd.type === 'pause') set({ paused: true });
    if (cmd.type === 'resume') {
      set({ paused: false });
      // Resuming returns to real-time mode.
      get().setPlaybackIndex(-1);
    }
    if (cmd.type === 'set_speed' && cmd.speed) set({ speed: cmd.speed });
  },

  getLatentPoints: () => {
    const { history } = get();
    const pts: LatentPoint[] = [];
    for (const s of history) {
      const c = s.latent_3d;
      if (!c) continue;
      // Skip the default [0,0,0] sentinel when no belief was available.
      if (c[0] === 0 && c[1] === 0 && c[2] === 0) continue;
      pts.push({
        step: s.step,
        modality: s.modality,
        coords: [c[0], c[1], c[2]],
        freeEnergy: s.free_energy,
        predictionError: s.prediction_error,
      });
    }
    return pts;
  },

  requestHistory: async (start, end) => {
    try {
      const url = `${apiUrl('/history')}?start=${start}${end ? `&end=${end}` : ''}`;
      const res = await fetch(url);
      const data = await res.json();
      if (data.snapshots) {
        set({ history: data.snapshots });
      }
    } catch (err) {
      console.error('[api] history fetch failed', err);
    }
  },

  requestGraph: async () => {
    try {
      const res = await fetch(apiUrl('/knowledge-graph'));
      const data = await res.json();
      if (data.nodes) {
        set({
          knowledgeGraph: {
            nodes: data.nodes.map((n: string) => ({ id: n, label: n, size: 10 })),
            edges: data.edges || [],
          },
        });
      }
    } catch (err) {
      console.error('[api] graph fetch failed', err);
    }
  },

  requestMilestones: async () => {
    try {
      const res = await fetch(apiUrl('/story-milestones'));
      const data = await res.json();
      if (data.milestones) {
        set({ milestones: data.milestones });
      }
    } catch (err) {
      console.error('[api] milestones fetch failed', err);
    }
  },

  injectQuestion: (question) => {
    get().sendCommand({ type: 'inject_question', question });
    set((state) => ({
      authoringMessages: [...state.authoringMessages, { role: 'user', text: question, step: state.currentStep }],
    }));
  },
}));
