/**
 * ZeroDataModel TypeScript SDK client.
 *
 * A thin asynchronous client for the ZeroDataModel REST API covering the core
 * cognitive endpoints and the phase-7 cognitive-upgrade modules. Uses the
 * platform `fetch` (Node 18+ / browsers); a custom `fetch` may be injected via
 * the config for testing.
 */
import { ClientConfig, ClientConfigOptions } from './config';
import {
  AuthenticationError,
  NotFoundError,
  RateLimitError,
  ServerError,
  ValidationError,
  ZeroDataModelError,
} from './errors';
import type {
  HealthResponse,
  JsonObject,
  KnowledgeGraphResponse,
  ThinkResponse,
} from './types';

export class ZeroDataModelClient {
  private config: ClientConfig;

  constructor(config: ClientConfigOptions = {}) {
    this.config = new ClientConfig(config);
  }

  // ------------------------------------------------------------------ //
  // Internal helpers
  // ------------------------------------------------------------------ //

  private get baseUrl(): string {
    return this.config.baseUrl.replace(/\/$/, '');
  }

  private get headers(): Record<string, string> {
    const h: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.config.apiKey) h['X-API-Key'] = this.config.apiKey;
    return h;
  }

  private get fetchFn(): typeof fetch {
    return this.config.fetch ?? globalThis.fetch;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const init: RequestInit = {
      method,
      headers: this.headers,
    };
    if (body !== undefined) {
      init.body = JSON.stringify(body);
    }

    let resp: Response;
    try {
      resp = await this.fetchFn(url, init);
    } catch (e) {
      throw new ZeroDataModelError(`Request failed: ${e}`);
    }

    if (resp.status >= 400) {
      await this.handleError(resp);
    }

    return (await resp.json()) as T;
  }

  private async handleError(resp: Response): Promise<never> {
    let detail = '';
    try {
      const body = await resp.json();
      detail =
        (body && typeof body === 'object' && 'detail' in body
          ? String((body as { detail: unknown }).detail)
          : JSON.stringify(body)) ?? '';
    } catch {
      detail = await resp.text().catch(() => '');
    }

    if (resp.status === 401 || resp.status === 403) {
      throw new AuthenticationError(detail);
    } else if (resp.status === 404) {
      throw new NotFoundError(detail);
    } else if (resp.status === 422) {
      throw new ValidationError(detail);
    } else if (resp.status === 429) {
      const retryAfter = parseInt(resp.headers.get('Retry-After') ?? '60', 10);
      throw new RateLimitError(detail, retryAfter);
    } else if (resp.status >= 500) {
      throw new ServerError(detail);
    } else {
      throw new ZeroDataModelError(`HTTP ${resp.status}: ${detail}`);
    }
  }

  // ------------------------------------------------------------------ //
  // Core endpoints
  // ------------------------------------------------------------------ //

  health(): Promise<HealthResponse> {
    return this.request('GET', '/health');
  }

  ready(): Promise<HealthResponse> {
    return this.request('GET', '/ready');
  }

  think(cycles = 1, extra: Record<string, unknown> = {}): Promise<ThinkResponse> {
    return this.request('POST', '/think', { cycles, ...extra });
  }

  getKnowledgeGraph(): Promise<KnowledgeGraphResponse> {
    return this.request('GET', '/knowledge-graph');
  }

  getStoryMilestones(): Promise<JsonObject> {
    return this.request('GET', '/story-milestones');
  }

  // ------------------------------------------------------------------ //
  // Phase 7: Architecture
  // ------------------------------------------------------------------ //

  architectStats(): Promise<JsonObject> {
    return this.request('GET', '/architect/stats');
  }

  architectDormant(): Promise<JsonObject> {
    return this.request('GET', '/architect/dormant');
  }

  architectAction(actionType: string, moduleName: string): Promise<JsonObject> {
    return this.request('POST', '/architect/action', {
      action_type: actionType,
      module_name: moduleName,
    });
  }

  // ------------------------------------------------------------------ //
  // Phase 7: Temporal Memory
  // ------------------------------------------------------------------ //

  temporalMemoryState(): Promise<JsonObject> {
    return this.request('GET', '/temporal_memory/state');
  }

  temporalMemoryReset(): Promise<JsonObject> {
    return this.request('POST', '/temporal_memory/reset');
  }

  // ------------------------------------------------------------------ //
  // Phase 7: Layered Predictor
  // ------------------------------------------------------------------ //

  layeredPredictorBeliefs(): Promise<JsonObject> {
    return this.request('GET', '/layered_predictor/beliefs');
  }

  layeredPredictorUpdate(observation: number[]): Promise<JsonObject> {
    return this.request('POST', '/layered_predictor/update', { observation });
  }

  // ------------------------------------------------------------------ //
  // Phase 7: Episodic Graph
  // ------------------------------------------------------------------ //

  episodicGraphStats(): Promise<JsonObject> {
    return this.request('GET', '/episodic_graph/stats');
  }

  episodicGraphPlan(startId: number, goalId: number): Promise<JsonObject> {
    return this.request('POST', '/episodic_graph/plan', {
      start_id: startId,
      goal_id: goalId,
    });
  }

  episodicGraphInsert(extra: Record<string, unknown> = {}): Promise<JsonObject> {
    return this.request('POST', '/episodic_graph/insert', extra);
  }

  // ------------------------------------------------------------------ //
  // Phase 7: Semantic Index
  // ------------------------------------------------------------------ //

  semanticIndexSearch(vector: number[], k = 5): Promise<JsonObject> {
    return this.request('POST', '/semantic_index/search', { vector, k });
  }

  semanticIndexStats(): Promise<JsonObject> {
    return this.request('GET', '/semantic_index/stats');
  }

  // ------------------------------------------------------------------ //
  // Phase 7: Logic Layer
  // ------------------------------------------------------------------ //

  logicRules(): Promise<JsonObject> {
    return this.request('GET', '/logic_layer/rules');
  }

  logicAddRule(extra: Record<string, unknown> = {}): Promise<JsonObject> {
    return this.request('POST', '/logic_layer/add_rule', extra);
  }

  logicRemoveRule(ruleId: string): Promise<JsonObject> {
    return this.request('DELETE', `/logic_layer/rules/${ruleId}`);
  }

  logicEvaluate(): Promise<JsonObject> {
    return this.request('POST', '/logic_layer/evaluate');
  }

  logicViolations(): Promise<JsonObject> {
    return this.request('GET', '/logic_layer/violations');
  }

  // ------------------------------------------------------------------ //
  // Phase 7: Causal Inference
  // ------------------------------------------------------------------ //

  causalGraph(): Promise<JsonObject> {
    return this.request('GET', '/causal_inference/graph');
  }

  causalIntervene(extra: Record<string, unknown> = {}): Promise<JsonObject> {
    return this.request('POST', '/causal_inference/intervene', extra);
  }

  causalObserve(extra: Record<string, unknown> = {}): Promise<JsonObject> {
    return this.request('POST', '/causal_inference/observe', extra);
  }

  causalCounterfactual(extra: Record<string, unknown> = {}): Promise<JsonObject> {
    return this.request('POST', '/causal_inference/counterfactual', extra);
  }

  // ------------------------------------------------------------------ //
  // Phase 7: Meta-Cognition
  // ------------------------------------------------------------------ //

  metacogState(): Promise<JsonObject> {
    return this.request('GET', '/meta_cognition/state');
  }

  metacogConfidence(): Promise<JsonObject> {
    return this.request('GET', '/meta_cognition/confidence');
  }

  metacogUncertainty(): Promise<JsonObject> {
    return this.request('GET', '/meta_cognition/uncertainty');
  }

  metacogReport(): Promise<JsonObject> {
    return this.request('GET', '/meta_cognition/report');
  }

  // ------------------------------------------------------------------ //
  // Phase 7: Experiment Planner
  // ------------------------------------------------------------------ //

  experimentCandidates(): Promise<JsonObject> {
    return this.request('GET', '/experiment_planner/candidates');
  }

  experimentSelect(): Promise<JsonObject> {
    return this.request('POST', '/experiment_planner/select');
  }

  experimentResult(candidateId: string, outcome: number): Promise<JsonObject> {
    return this.request('POST', '/experiment_planner/result', {
      candidate_id: candidateId,
      outcome,
    });
  }

  experimentHistory(): Promise<JsonObject> {
    return this.request('GET', '/experiment_planner/history');
  }

  // ------------------------------------------------------------------ //
  // Phase 7: Hypothesis Tester
  // ------------------------------------------------------------------ //

  hypothesisList(): Promise<JsonObject> {
    return this.request('GET', '/hypothesis_tester/list');
  }

  hypothesisTest(
    hypothesisId: string,
    extra: Record<string, unknown> = {},
  ): Promise<JsonObject> {
    return this.request('POST', '/hypothesis_tester/test', {
      hypothesis_id: hypothesisId,
      ...extra,
    });
  }

  hypothesisSupported(): Promise<JsonObject> {
    return this.request('GET', '/hypothesis_tester/supported');
  }

  // ------------------------------------------------------------------ //
  // Phase 7: Multi-Agent (world / communication / culture)
  // ------------------------------------------------------------------ //

  worldStats(): Promise<JsonObject> {
    return this.request('GET', '/world/stats');
  }

  worldStep(): Promise<JsonObject> {
    return this.request('POST', '/world/step');
  }

  worldAgents(): Promise<JsonObject> {
    return this.request('GET', '/world/agents');
  }

  communicationChannels(): Promise<JsonObject> {
    return this.request('GET', '/communication/channels');
  }

  communicationSend(extra: Record<string, unknown> = {}): Promise<JsonObject> {
    return this.request('POST', '/communication/send', extra);
  }

  cultureGenerations(): Promise<JsonObject> {
    return this.request('GET', '/culture/generations');
  }

  cultureHistory(): Promise<JsonObject> {
    return this.request('GET', '/culture/history');
  }

  culturePropagate(extra: Record<string, unknown> = {}): Promise<JsonObject> {
    return this.request('POST', '/culture/propagate', extra);
  }
}
