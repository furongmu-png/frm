import http from 'k6/http';
import { check, sleep, group } from 'k6';

// =============================================================================
// k6-load.js — Sustained load test.
//
// Ramps to 10 VUs, holds 2 minutes, ramps down. Hits the always-available
// health endpoints (/, /health, /ready — rate-limit-exempt) plus /think
// (accepts 429: the endpoint is capped at 10/minute, so under load a 429 is
// the limiter doing its job, not a failure) and a phase7 read endpoint
// (accepts 503: phase7 modules return 503 when their feature flag is off).
// =============================================================================

export const options = {
  stages: [
    { duration: '30s', target: 10 },   // ramp up to 10 VUs
    { duration: '2m', target: 10 },     // hold at 10 VUs
    { duration: '30s', target: 0 },     // ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],   // 95% of requests < 500ms
    http_req_failed: ['rate<0.01'],     // error rate < 1%
  },
};

const BASE = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
  group('health', () => {
    const r = http.get(`${BASE}/health`);
    check(r, { '200': (r) => r.status === 200 });
  });

  group('ready', () => {
    const r = http.get(`${BASE}/ready`);
    check(r, { '200': (r) => r.status === 200 });
  });

  group('think', () => {
    const r = http.post(`${BASE}/think`, JSON.stringify({ cycles: 1 }), {
      headers: { 'Content-Type': 'application/json' },
      expectedStatuses: [200, 429], // 429 = rate limiter engaged (expected under load)
    });
    check(r, {
      '200 or 429': (r) => r.status === 200 || r.status === 429,
      '200 has metadata': (r) => r.status === 200 ? r.json('metadata') !== undefined : true,
    });
  });

  group('phase7 architect', () => {
    const r = http.get(`${BASE}/architect/stats`, {
      expectedStatuses: [200, 503], // 503 = module not enabled (feature flag off)
    });
    check(r, { '200 or 503': (r) => r.status === 200 || r.status === 503 });
  });

  sleep(0.2);
}
