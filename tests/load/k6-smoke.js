import http from 'k6/http';
import { check, sleep } from 'k6';

// =============================================================================
// k6-smoke.js — Smoke test: verify the API is up and answers /health, /ready,
// and /think with the expected shape.
//
// Uses a fixed iteration count (not a duration) so the 5 /think requests stay
// well under the endpoint's 10/minute rate limit (see api.py @_limit("10/minute")).
// A 30s duration with 1 VU would issue ~60 think requests and trip the limiter,
// turning the smoke test into a 429 flood.
// =============================================================================

export const options = {
  vus: 1,
  iterations: 5,
};

const BASE = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
  const health = http.get(`${BASE}/health`);
  check(health, {
    'health 200': (r) => r.status === 200,
  });

  const ready = http.get(`${BASE}/ready`);
  check(ready, {
    'ready 200': (r) => r.status === 200,
  });

  const think = http.post(`${BASE}/think`, JSON.stringify({ cycles: 1 }), {
    headers: { 'Content-Type': 'application/json' },
  });
  check(think, {
    'think 200': (r) => r.status === 200,
    'think has free_energy': (r) => r.json('free_energy') !== undefined,
  });

  sleep(0.5);
}
