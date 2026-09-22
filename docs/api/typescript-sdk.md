# TypeScript SDK

ZeroDataModel does not ship a hand-written TypeScript SDK. Instead, the FastAPI
service exposes a live OpenAPI 3.1 schema at `/openapi.json` (when
`ZDM_ENV=development`), and you can generate a fully-typed TypeScript client
from it in one command. The browser UI under `frontend/` is itself a React +
Vite + TypeScript app that talks to the REST API directly over `fetch`.

## Generate a typed client from the OpenAPI spec

```bash
# 1. Start the API with docs enabled
export ZDM_ENV=development
PYTHONPATH=src uvicorn zero_data_model.api:app --port 8000

# 2. Generate the client (pick one tool)
npx openapi-typescript http://localhost:8000/openapi.json -o ./zdm-client.ts
# or, for a full fetch-based client:
npx openapi-fetch --input http://localhost:8000/openapi.json --output ./zdm-client
```

The generated `zdm-client.ts` exports typed interfaces for every request body,
response body, and query parameter — including the Pydantic `Field` bounds
(`min_length`, `pattern`, `ge`/`le`) that the API enforces server-side.

## Manual `fetch` usage

For quick integrations a hand-written client is short. All cognitive endpoints
take JSON bodies and return JSON.

```typescript
const API = "http://localhost:8000";
const API_KEY = process.env.ZDM_API_KEY;        // omit when auth disabled

const headers: Record<string, string> = {
  "Content-Type": "application/json",
};
if (API_KEY) headers["X-API-Key"] = API_KEY;

// Run a self-generated thought cycle
const thinkRes = await fetch(`${API}/think`, {
  method: "POST",
  headers,
  body: JSON.stringify({}),
});
const think = await thinkRes.json();
// { cycle: number, output: number[], confidence: number }

// Zero-shot classify
const classifyRes = await fetch(`${API}/classify`, {
  method: "POST",
  headers,
  body: JSON.stringify({ text: "the algorithm computes the network" }),
});
const classify = await classifyRes.json();
// { topic: string, confidence: number }
```

## The bundled React frontend

The repo ships a browser UI under `frontend/` (React 19 + Vite + TypeScript).
It is **not** a reusable SDK, but it demonstrates the integration patterns:

- **State management** — `frontend/src/store/useModelStore.ts` is a Zustand
  store that holds the model state and exposes actions wrapping the REST calls.
- **WebSocket stream** — connects to `ws://<host>:8765` for live belief-state,
  free-energy, and module-graph updates.
- **Panels** — `frontend/src/panels/` contains one component per dashboard
  panel (BeliefStateRaw, CausalGraph, ConfidenceDashboard, ExperimentLog,
  FreeEnergyChart, KnowledgeGraphView, LatentSpace3D, LogicPanel,
  ModuleGraphPanel, SandboxView, SelfAuthoring, StoryMode, TextExplorer,
  TextHeatmap).
- **Configurable API/WS paths** — `frontend/src/config/runtime.ts` lets the
  build pick `/api` and `/ws` prefixes for ingress compatibility (see
  [Kubernetes](../deployment/kubernetes.md)).

### Build the frontend

```bash
cd frontend
npm install
npm run build        # outputs to frontend/dist/
npm run dev          # Vite dev server with HMR
npm run test         # vitest unit tests
npm run e2e           # Playwright end-to-end tests
```

Node.js **18+** is required. See `frontend/package.json` for the full list of
dependencies (React, Zustand, Recharts, react-three/fiber, Cytoscape, etc.).

## When to use which

| Approach | Use when |
| --- | --- |
| `openapi-typescript` generated client | You want full type safety and auto-refresh when the API changes. |
| Hand-written `fetch` | Quick prototype, single endpoint, minimal dependencies. |
| The bundled React frontend | You want the existing dashboard UI without writing code. |
| The [Python SDK](python-sdk.md) | You are in Python and want zero HTTP overhead. |
