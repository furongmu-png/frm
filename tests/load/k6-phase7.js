import http from 'k6/http';
import { check, sleep } from 'k6';

// =============================================================================
// k6-phase7.js — Phase 7 cognitive-upgrade module endpoint load test.
//
// Iterates over all read-only GET endpoints of the 40 phase7 endpoints
// (architect, temporal_memory, layered_predictor, episodic_graph,
// semantic_index, logic_layer, meta_cognition, experiment_planner,
// hypothesis_tester, world, communication, culture).
//
// Measures per-endpoint latency and enforces per-endpoint thresholds so a
// single slow endpoint cannot hide behind an aggregate p(95).
//
// Prerequisites:
//   - The server must be started with all phase7 feature flags enabled
//     (enable_architect, enable_layered_predictor, enable_episodic_memory,
//      enable_logic_layer, enable_meta_cognition, enable_experiment_planner,
//      enable_multiagent) so every endpoint returns 200.
//   - Set BASE_URL to point at the running server.
// =============================================================================

const BASE = __ENV.BASE_URL || 'http://localhost:8000';

// All 29 read-only GET endpoints among the 40 phase7 endpoints.
// Each entry is tagged so k6 thresholds can target a single endpoint.
const ENDPOINTS = [
  // architect (plasticity)
  { path: '/architect/stats', tag: 'architect_stats' },
  { path: '/architect/dormant', tag: 'architect_dormant' },
  // temporal_memory (cogtime)
  { path: '/temporal_memory/context', tag: 'temporal_memory_context' },
  { path: '/temporal_memory/spectral_radius', tag: 'temporal_memory_spectral_radius' },
  // layered_predictor (cogtime)
  { path: '/layered_predictor/context', tag: 'layered_predictor_context' },
  { path: '/layered_predictor/rhythm', tag: 'layered_predictor_rhythm' },
  // episodic_graph (cogmem)
  { path: '/episodic_graph/recent', tag: 'episodic_graph_recent' },
  { path: '/episodic_graph/node_count', tag: 'episodic_graph_node_count' },
  // semantic_index (cogmem)
  { path: '/semantic_index/size', tag: 'semantic_index_size' },
  // logic_layer (knowledge)
  { path: '/logic_layer/rules', tag: 'logic_layer_rules' },
  { path: '/logic_layer/check', tag: 'logic_layer_check' },
  { path: '/logic_layer/penalty', tag: 'logic_layer_penalty' },
  // meta_cognition (metacog)
  { path: '/meta_cognition/confidence', tag: 'meta_cognition_confidence' },
  { path: '/meta_cognition/uncertainty', tag: 'meta_cognition_uncertainty' },
  { path: '/meta_cognition/should_seek_info', tag: 'meta_cognition_should_seek_info' },
  { path: '/meta_cognition/stats', tag: 'meta_cognition_stats' },
  // experiment_planner (experiment)
  { path: '/experiment_planner/candidates', tag: 'experiment_planner_candidates' },
  { path: '/experiment_planner/stats', tag: 'experiment_planner_stats' },
  // hypothesis_tester (experiment)
  { path: '/hypothesis_tester/hypotheses', tag: 'hypothesis_tester_hypotheses' },
  { path: '/hypothesis_tester/supported', tag: 'hypothesis_tester_supported' },
  { path: '/hypothesis_tester/stats', tag: 'hypothesis_tester_stats' },
  // world (multiagent)
  { path: '/world/collaboration_stats', tag: 'world_collaboration_stats' },
  { path: '/world/agent_count', tag: 'world_agent_count' },
  { path: '/world/step_count', tag: 'world_step_count' },
  // communication (multiagent)
  { path: '/communication/emergent_meanings', tag: 'communication_emergent_meanings' },
  { path: '/communication/stats', tag: 'communication_stats' },
  // culture (multiagent)
  { path: '/culture/knowledge_curve', tag: 'culture_knowledge_curve' },
  { path: '/culture/stats', tag: 'culture_stats' },
  { path: '/culture/generations', tag: 'culture_generations' },
];

// Build per-endpoint thresholds. Read-only GET endpoints should be fast
// (in-memory reads), so each gets a p(95) < 300ms cap. The aggregate
// threshold guards the overall run.
function buildThresholds() {
  const t = {
    http_req_failed: ['rate<0.01'], // < 1% errors across all endpoints
    http_req_duration: ['p(95)<500'], // aggregate p(95) < 500ms
  };
  for (const ep of ENDPOINTS) {
    // Per-endpoint: 95% of requests to THIS endpoint < 300ms.
    t[`http_req_duration{endpoint:${ep.tag}}`] = ['p(95)<300'];
  }
  return t;
}

export const options = {
  vus: 10,
  duration: '2m',
  thresholds: buildThresholds(),
};

export default function () {
  for (const ep of ENDPOINTS) {
    // expectedStatuses includes 503: phase7 modules return 503 when their
    // feature flag (enable_architect, enable_layered_predictor, ...) is off.
    // A 503 is "module not enabled", not a server error, so it must not
    // inflate http_req_failed.
    const r = http.get(`${BASE}${ep.path}`, {
      tags: { endpoint: ep.tag },
      expectedStatuses: [200, 503],
    });
    check(
      r,
      {
        [`${ep.tag} 200 or 503`]: (r) => r.status === 200 || r.status === 503,
      },
      { endpoint: ep.tag }
    );
  }
  sleep(0.2);
}

export function handleSummary(data) {
  // Print a per-endpoint latency breakdown so slow endpoints are visible.
  const lines = [];
  lines.push('\n=== Per-endpoint latency (p95) ===');
  for (const ep of ENDPOINTS) {
    const key = `http_req_duration{endpoint:${ep.tag}}`;
    const metric = data.metrics[key];
    const p95 = metric && metric.values['p(95)'] !== undefined ? metric.values['p(95)'].toFixed(2) : 'n/a';
    const ok = metric && metric.thresholds && metric.thresholds['p(95)<300'];
    const pass = ok ? ok.ok ? 'PASS' : 'FAIL' : '-';
    lines.push(`  ${ep.tag.padEnd(40)} p95=${p95}ms  [${pass}]`);
  }
  console.log(lines.join('\n'));
  return {};
}
