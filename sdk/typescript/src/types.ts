/**
 * Type definitions for ZeroDataModel SDK requests and responses.
 *
 * The client returns loosely-typed payloads (`unknown` / `Record<string,
 * unknown>`) by default for maximum flexibility, but these interfaces are
 * provided for callers that want typed responses for the core endpoints.
 */

export interface HealthResponse {
  status: string;
  model_ready?: boolean;
}

export interface ThinkResponse {
  cycle?: number;
  output?: number[];
  confidence?: number;
  free_energy?: number;
  metadata?: Record<string, unknown>;
}

export interface KnowledgeGraphResponse {
  nodes: unknown[];
  edges: unknown[];
}

/** Generic JSON object returned by most phase-7 endpoints. */
export type JsonObject = Record<string, unknown>;

// --------------------------------------------------------------------------- //
// Phase-7 request payloads (subset with well-defined shapes)
// --------------------------------------------------------------------------- //

export interface ArchitectActionRequest {
  action_type: string;
  module_name: string;
}

export interface LayeredPredictorUpdateRequest {
  observation: number[];
}

export interface EpisodicGraphPlanRequest {
  start_id: number;
  goal_id: number;
}

export interface SemanticIndexSearchRequest {
  vector: number[];
  k: number;
}

export interface ExperimentResultRequest {
  candidate_id: string;
  outcome: number;
}

export interface HypothesisTestRequest {
  hypothesis_id: string;
  [key: string]: unknown;
}
