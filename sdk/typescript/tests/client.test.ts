/**
 * Tests for the ZeroDataModel TypeScript SDK client.
 *
 * The suite injects a recording mock `fetch` so the client is exercised end to
 * end (real request serialization, header building, error mapping) without a
 * live server.
 */
import { describe, it, expect } from 'vitest';
import {
  ZeroDataModelClient,
  ZeroDataModelError,
  AuthenticationError,
  NotFoundError,
  RateLimitError,
  ValidationError,
  ServerError,
  type ClientConfigOptions,
} from '../src/index';

// --------------------------------------------------------------------------- //
// Test helpers
// --------------------------------------------------------------------------- //

interface RecordedCall {
  url: string;
  init: RequestInit;
}

type FetchHandler = (url: string, init: RequestInit) => Response | Promise<Response>;

function makeMockFetch(handler: FetchHandler): { fetchFn: typeof fetch; calls: RecordedCall[] } {
  const calls: RecordedCall[] = [];
  const fetchFn = (async (url: string | URL | Request, init?: RequestInit) => {
    const record: RecordedCall = { url: String(url), init: init ?? {} };
    calls.push(record);
    return handler(record.url, record.init);
  }) as typeof fetch;
  return { fetchFn, calls };
}

function jsonResponse(
  payload: unknown,
  status = 200,
  headers: Record<string, string> = {},
): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  });
}

function makeClient(
  handler: FetchHandler,
  opts: Partial<ClientConfigOptions> = {},
): { client: ZeroDataModelClient; calls: RecordedCall[] } {
  const { fetchFn, calls } = makeMockFetch(handler);
  const client = new ZeroDataModelClient({
    baseUrl: 'http://localhost:8000',
    apiKey: 'test-key',
    fetch: fetchFn,
    ...opts,
  });
  return { client, calls };
}

// --------------------------------------------------------------------------- //
// Core endpoints
// --------------------------------------------------------------------------- //

describe('core endpoints', () => {
  it('health returns payload via GET', async () => {
    const { client, calls } = makeClient(() => jsonResponse({ status: 'ok' }));
    const result = await client.health();
    expect(result).toEqual({ status: 'ok' });
    expect(calls[0].init.method).toBe('GET');
    expect(calls[0].url).toBe('http://localhost:8000/health');
  });

  it('ready returns payload', async () => {
    const { client } = makeClient(() =>
      jsonResponse({ status: 'ok', model_ready: true }),
    );
    expect(await client.ready()).toEqual({ status: 'ok', model_ready: true });
  });

  it('think sends cycles (and extra) in body', async () => {
    const { client, calls } = makeClient(() =>
      jsonResponse({ cycle: 1, free_energy: 0.42 }),
    );
    const result = await client.think(3, { input: [1, 2] });
    expect(result.free_energy).toBe(0.42);
    expect(calls[0].init.method).toBe('POST');
    expect(calls[0].url).toBe('http://localhost:8000/think');
    expect(JSON.parse(calls[0].init.body as string)).toEqual({
      cycles: 3,
      input: [1, 2],
    });
  });

  it('getKnowledgeGraph', async () => {
    const { client, calls } = makeClient(() =>
      jsonResponse({ nodes: [{ id: 1 }], edges: [] }),
    );
    const result = await client.getKnowledgeGraph();
    expect(result.nodes).toEqual([{ id: 1 }]);
    expect(calls[0].url).toBe('http://localhost:8000/knowledge-graph');
  });

  it('getStoryMilestones', async () => {
    const { client, calls } = makeClient(() => jsonResponse({ milestones: [] }));
    await client.getStoryMilestones();
    expect(calls[0].url).toBe('http://localhost:8000/story-milestones');
  });
});

// --------------------------------------------------------------------------- //
// Phase-7 endpoints
// --------------------------------------------------------------------------- //

describe('phase-7 endpoints', () => {
  it('architect endpoints', async () => {
    const { client, calls } = makeClient(() => jsonResponse({ ok: true }));
    await client.architectStats();
    await client.architectDormant();
    await client.architectAction('activate', 'memory');
    expect(calls.map((c) => c.url)).toEqual([
      'http://localhost:8000/architect/stats',
      'http://localhost:8000/architect/dormant',
      'http://localhost:8000/architect/action',
    ]);
    expect(calls[2].init.method).toBe('POST');
    expect(JSON.parse(calls[2].init.body as string)).toEqual({
      action_type: 'activate',
      module_name: 'memory',
    });
  });

  it('temporal memory endpoints', async () => {
    const { client, calls } = makeClient(() => jsonResponse({ ok: true }));
    await client.temporalMemoryState();
    await client.temporalMemoryReset();
    expect(calls.map((c) => c.url)).toEqual([
      'http://localhost:8000/temporal_memory/state',
      'http://localhost:8000/temporal_memory/reset',
    ]);
    expect(calls[1].init.method).toBe('POST');
  });

  it('layered predictor update sends observation', async () => {
    const { client, calls } = makeClient(() => jsonResponse({ ok: true }));
    await client.layeredPredictorBeliefs();
    await client.layeredPredictorUpdate([0.1, 0.2, 0.3]);
    expect(calls[1].url).toBe('http://localhost:8000/layered_predictor/update');
    expect(JSON.parse(calls[1].init.body as string)).toEqual({
      observation: [0.1, 0.2, 0.3],
    });
  });

  it('episodic graph plan and insert', async () => {
    const { client, calls } = makeClient(() => jsonResponse({ ok: true }));
    await client.episodicGraphStats();
    await client.episodicGraphPlan(1, 9);
    await client.episodicGraphInsert({ node: 'x', weight: 0.5 });
    expect(JSON.parse(calls[1].init.body as string)).toEqual({
      start_id: 1,
      goal_id: 9,
    });
    expect(JSON.parse(calls[2].init.body as string)).toEqual({
      node: 'x',
      weight: 0.5,
    });
  });

  it('semantic index search', async () => {
    const { client, calls } = makeClient(() => jsonResponse({ ok: true }));
    await client.semanticIndexStats();
    await client.semanticIndexSearch([1, 2], 3);
    expect(JSON.parse(calls[1].init.body as string)).toEqual({
      vector: [1, 2],
      k: 3,
    });
  });

  it('logic layer endpoints', async () => {
    const { client, calls } = makeClient(() => jsonResponse({ ok: true }));
    await client.logicRules();
    await client.logicAddRule({ name: 'r1', expr: 'x > 0' });
    await client.logicRemoveRule('r1');
    await client.logicEvaluate();
    await client.logicViolations();
    expect(calls.map((c) => `${c.init.method} ${c.url}`)).toEqual([
      'GET http://localhost:8000/logic_layer/rules',
      'POST http://localhost:8000/logic_layer/add_rule',
      'DELETE http://localhost:8000/logic_layer/rules/r1',
      'POST http://localhost:8000/logic_layer/evaluate',
      'GET http://localhost:8000/logic_layer/violations',
    ]);
    expect(JSON.parse(calls[1].init.body as string)).toEqual({
      name: 'r1',
      expr: 'x > 0',
    });
  });

  it('causal endpoints', async () => {
    const { client, calls } = makeClient(() => jsonResponse({ ok: true }));
    await client.causalGraph();
    await client.causalIntervene({ var: 'x', value: 1 });
    await client.causalObserve({ data: [1, 2] });
    await client.causalCounterfactual({ world: 'w' });
    expect(calls.map((c) => c.url)).toEqual([
      'http://localhost:8000/causal_inference/graph',
      'http://localhost:8000/causal_inference/intervene',
      'http://localhost:8000/causal_inference/observe',
      'http://localhost:8000/causal_inference/counterfactual',
    ]);
  });

  it('meta-cognition endpoints', async () => {
    const { client, calls } = makeClient(() => jsonResponse({ ok: true }));
    await client.metacogState();
    await client.metacogConfidence();
    await client.metacogUncertainty();
    await client.metacogReport();
    expect(calls.map((c) => c.url)).toEqual([
      'http://localhost:8000/meta_cognition/state',
      'http://localhost:8000/meta_cognition/confidence',
      'http://localhost:8000/meta_cognition/uncertainty',
      'http://localhost:8000/meta_cognition/report',
    ]);
  });

  it('experiment planner endpoints', async () => {
    const { client, calls } = makeClient(() => jsonResponse({ ok: true }));
    await client.experimentCandidates();
    await client.experimentSelect();
    await client.experimentResult('c1', 0.87);
    await client.experimentHistory();
    expect(JSON.parse(calls[2].init.body as string)).toEqual({
      candidate_id: 'c1',
      outcome: 0.87,
    });
  });

  it('hypothesis endpoints', async () => {
    const { client, calls } = makeClient(() => jsonResponse({ ok: true }));
    await client.hypothesisList();
    await client.hypothesisTest('h1', { threshold: 0.05 });
    await client.hypothesisSupported();
    expect(JSON.parse(calls[1].init.body as string)).toEqual({
      hypothesis_id: 'h1',
      threshold: 0.05,
    });
  });

  it('multi-agent endpoints', async () => {
    const { client, calls } = makeClient(() => jsonResponse({ ok: true }));
    await client.worldStats();
    await client.worldStep();
    await client.worldAgents();
    await client.communicationChannels();
    await client.communicationSend({ channel: 'ch1', message: 'hi' });
    await client.cultureGenerations();
    await client.cultureHistory();
    await client.culturePropagate({ meme: 'm1' });
    expect(calls.map((c) => c.url)).toEqual([
      'http://localhost:8000/world/stats',
      'http://localhost:8000/world/step',
      'http://localhost:8000/world/agents',
      'http://localhost:8000/communication/channels',
      'http://localhost:8000/communication/send',
      'http://localhost:8000/culture/generations',
      'http://localhost:8000/culture/history',
      'http://localhost:8000/culture/propagate',
    ]);
  });
});

// --------------------------------------------------------------------------- //
// API key handling
// --------------------------------------------------------------------------- //

describe('api key handling', () => {
  it('sends X-API-Key header when provided', async () => {
    const { client, calls } = makeClient(() => jsonResponse({ ok: true }), {
      apiKey: 'secret',
    });
    await client.health();
    const headers = new Headers(calls[0].init.headers);
    expect(headers.get('X-API-Key')).toBe('secret');
  });

  it('omits X-API-Key header when absent', async () => {
    const { client, calls } = makeClient(() => jsonResponse({ ok: true }), {
      apiKey: undefined,
    });
    await client.health();
    const headers = new Headers(calls[0].init.headers);
    expect(headers.get('X-API-Key')).toBeNull();
  });

  it('strips trailing slash from base url', async () => {
    const { client, calls } = makeClient(() => jsonResponse({ ok: true }), {
      baseUrl: 'http://localhost:8000/',
    });
    await client.health();
    expect(calls[0].url).toBe('http://localhost:8000/health');
  });
});

// --------------------------------------------------------------------------- //
// Error handling
// --------------------------------------------------------------------------- //

describe('error handling', () => {
  const cases: Array<[number, typeof Error]> = [
    [401, AuthenticationError],
    [403, AuthenticationError],
    [404, NotFoundError],
    [422, ValidationError],
    [500, ServerError],
    [503, ServerError],
  ];

  for (const [status, excType] of cases) {
    it(`maps ${status} to ${excType.name}`, async () => {
      const { client } = makeClient(() =>
        jsonResponse({ detail: 'boom' }, status),
      );
      await expect(client.health()).rejects.toBeInstanceOf(excType);
    });
  }

  it('rate limit error carries retryAfter from header', async () => {
    const { client } = makeClient(() =>
      jsonResponse({ detail: 'slow down' }, 429, { 'Retry-After': '30' }),
    );
    await expect(client.health()).rejects.toMatchObject({
      name: 'RateLimitError',
      retryAfter: 30,
    });
  });

  it('rate limit default retryAfter when header missing', async () => {
    const { client } = makeClient(() =>
      jsonResponse({ detail: 'slow down' }, 429),
    );
    await expect(client.health()).rejects.toMatchObject({
      name: 'RateLimitError',
      retryAfter: 60,
    });
  });

  it('unknown 4xx raises base error', async () => {
    const { client } = makeClient(() => jsonResponse({ detail: 'teapot' }, 418));
    await expect(client.health()).rejects.toBeInstanceOf(ZeroDataModelError);
    await expect(client.health()).rejects.toThrow(/418/);
  });

  it('wraps network errors', async () => {
    const { client } = makeClient(() => {
      throw new Error('connection refused');
    });
    await expect(client.health()).rejects.toBeInstanceOf(ZeroDataModelError);
    await expect(client.health()).rejects.toThrow(/Request failed/);
  });

  it('all errors subclass ZeroDataModelError', () => {
    for (const exc of [
      AuthenticationError,
      NotFoundError,
      RateLimitError,
      ValidationError,
      ServerError,
    ]) {
      expect(Object.getPrototypeOf(exc)).toBe(ZeroDataModelError);
    }
  });
});

// --------------------------------------------------------------------------- //
// Package surface
// --------------------------------------------------------------------------- //

describe('package surface', () => {
  it('exports client, errors, and VERSION', async () => {
    const mod = await import('../src/index');
    expect(mod.ZeroDataModelClient).toBeDefined();
    expect(mod.ZeroDataModelError).toBeDefined();
    expect(mod.VERSION).toBe('1.0.0');
  });
});
