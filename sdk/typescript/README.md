# ZeroDataModel TypeScript SDK

Official asynchronous TypeScript client for the ZeroDataModel REST API. Covers
the core cognitive endpoints (`/think`, `/health`, `/ready`, `/knowledge-graph`,
`/story-milestones`) and the phase-7 cognitive-upgrade modules (`/architect`,
`/temporal_memory`, `/layered_predictor`, `/episodic_graph`, `/semantic_index`,
`/logic_layer`, `/causal_inference`, `/meta_cognition`, `/experiment_planner`,
`/hypothesis_tester`, `/world`, `/communication`, `/culture`).

Built on the platform `fetch` (Node 18+ and modern browsers), so it has **zero
runtime dependencies**.

## Install

```bash
npm install @zero-data-model/client
# or
pnpm add @zero-data-model/client
```

## Quick start

```typescript
import { ZeroDataModelClient } from '@zero-data-model/client';

const client = new ZeroDataModelClient({
  baseUrl: 'http://localhost:8000',
  apiKey: '...',
});

const health = await client.health();
const result = await client.think(1);
console.log(result.free_energy);

// Phase-7 example
const beliefs = await client.layeredPredictorBeliefs();
await client.architectAction('activate', 'memory');
```

## Authentication

If `apiKey` is provided it is sent as the `X-API-Key` header on every request.

## Error handling

All failures derive from `ZeroDataModelError`:

| HTTP status | Error |
| ----------- | ----- |
| 401 / 403   | `AuthenticationError` |
| 404         | `NotFoundError` |
| 422         | `ValidationError` |
| 429         | `RateLimitError` (exposes `.retryAfter`) |
| 5xx         | `ServerError` |
| network     | `ZeroDataModelError` (wrapped) |

```typescript
import {
  ZeroDataModelClient,
  RateLimitError,
  ZeroDataModelError,
} from '@zero-data-model/client';

try {
  await client.think(1);
} catch (e) {
  if (e instanceof RateLimitError) {
    const wait = e.retryAfter;
  } else if (e instanceof ZeroDataModelError) {
    // ...
  }
}
```

## Configuration

| Option | Default | Description |
| ------ | ------- | ----------- |
| `baseUrl` | `http://localhost:8000` | API base URL |
| `apiKey` | `undefined` | Optional API key |
| `timeout` | `30000` | Per-request timeout (ms) |
| `fetch` | global `fetch` | Custom fetch (e.g. for testing) |

## Development

```bash
cd sdk/typescript
npm install
npm run build   # type-check + emit dist/
npm test        # vitest
```
