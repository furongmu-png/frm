// frontend/src/types.ts
// Type definitions for the "Window of Consciousness" protocol.

export interface Snapshot {
  step: number;
  timestamp: number;
  modality: string;
  frame_b64: string | null;
  text_block: string;
  action: number;
  free_energy: number;
  prediction_error: number;
  beta: number;
  belief_state: number[];
  causal_graph: CausalGraph;
  categories: Record<string, unknown>;
  kg_update: KGUpdate;
  self_authoring: string | null;
  event: string;
  // Phase-5 additions (optional for backwards compatibility with
  // snapshots produced by older backends).
  text_pos?: number;          // -1 when not reading text
  latent_3d?: [number, number, number];

  // Phase-G additions: cognitive upgrades (S4 / PCN / Hopfield).
  // All three are OPTIONAL so snapshots produced by older backends
  // (or when the upgrades are disabled) still typecheck. The Python
  // backend always emits them (as empty containers when off), but
  // clients must still treat them as optional for replay of historical
  // snapshots recorded before Phase G.
  s4_state?: number[];        // first ≤8 dims of the S4 hidden state
  layer_errors?: {            // per-layer PCN prediction-error norms
    L0?: number;
    L1?: number;
    L2?: number;
    [key: string]: number | undefined;
  };
  memory_retrieved?: {        // Hopfield retrieval summary
    n_memories?: number;
    top1_similarity?: number;
    [key: string]: unknown;
  };

  // Phase-6 additions: optional structured metadata produced by newer
  // backends. Kept permissive (index signature of `unknown`) so backend
  // additions don't break the type, and OPTIONAL so snapshots without
  // metadata still typecheck. Panels narrow specific sub-fields via
  // narrow `as` casts rather than the old `as unknown as` double-cast.
  metadata?: {
    module_errors?: Record<string, number>;
    cognitive_upgrades?: Record<string, unknown>;
    [key: string]: unknown;
  };
}

export interface CausalGraph {
  nodes: { id: number; label: string }[];
  edges: { source: number; target: number; strength: number }[];
}

export interface KGUpdate {
  new_nodes: string[];
  new_edges: { source: string; target: string; weight: number }[];
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface GraphNode {
  id: string;
  label: string;
  size?: number;
  group?: string;
  modality?: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight?: number;
  type?: string;
}

export interface Milestone {
  step: number;
  event: string;
  description: string;
}

/** A point in 3D latent space, derived from a Snapshot. */
export interface LatentPoint {
  step: number;
  modality: string;
  coords: [number, number, number];
  freeEnergy: number;
  predictionError: number;
}

export type CommandType = 'pause' | 'resume' | 'step' | 'set_speed' | 'intervention' | 'inject_question';

export interface WSCommand {
  type: CommandType;
  speed?: number;
  subtype?: string;
  params?: Record<string, unknown>;
  question?: string;
}
