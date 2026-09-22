# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 7-stage cognitive upgrade modules (architect, temporal_memory, layered_predictor, episodic_graph, semantic_index, logic_layer, causal_inference, meta_cognition, experiment_planner, hypothesis_tester, multiagent)
- 40 REST API endpoints + 40 MCP tools for cognitive modules
- Prometheus metrics (24 metrics) + Grafana dashboards
- K8s/Helm deployment configuration
- Playwright E2E tests
- API security middleware (rate limiting, API key auth, CORS, security headers)
- CI/CD release pipeline

### Changed
- Frontend now uses configurable API/WS paths (/api, /ws) for ingress compatibility

### Fixed
- Multiagent module now integrated into think() cycle
