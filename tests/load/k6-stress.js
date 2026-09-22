import http from 'k6/http';
import { check, sleep, group } from 'k6';

// =============================================================================
// k6-stress.js — Stress test: push the API past normal capacity to find its
// breaking point.
//
// Ramps to 50 VUs, holds, ramps to 100 VUs, holds, ramps down. Same endpoint
// mix as k6-load.js but with tighter pacing (sleep 0.1) and looser thresholds
// (p(99) < 2s, error rate < 5%) to reflect the higher error tolerance under
// stress. /think accepts 429 (rate limiter) and phase7 endpoints accept 503
// (module not enabled) so those expected responses do not inflate the error rate.
// =============================================================================

export const options = {
  stages: [
    { duration: '1m', target: 50 },    // ramp up to 50
    { duration: '2m', target: 50 },     // hold
    { duration: '1m', target: 100 },    // ramp to 100
    { duration: '2m', target: 100 },    // hold
    { duration: '30s', target: 0 },     // ramp down
  ],
  thresholds: {
    http_req_duration: ['p(99)<2000'],   // 99% < 2s
    http_req_failed: ['rate<0.05'],      // error rate < 5%
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
      expectedStatuses: [200, 429], // 429 = rate limiter engaged (expected under stress)
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

  sleep(0.1);
}
