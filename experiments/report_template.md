# Cognitive Emergence Evaluation Report

> **Template**. Copy this file as `REPORT_<date>.md` and fill in the
> placeholders (`«...»`) with the values from your run. Most fields
> can be auto-filled from `experiments/output/replay_run/FULL_REPORT.md`
> after running `python experiments/run_full_evaluation.py`.

## 0. Run Metadata

| Field | Value |
|---|---|
| Date | «YYYY-MM-DD» |
| Researcher | «name» |
| Commit hash | «git rev-parse HEAD» |
| Run command | «e.g. `python experiments/run_full_evaluation.py --long`» |
| Total elapsed | «e.g. 312.4s» |
| Random seed | «e.g. 42» |
| Machine | «e.g. Intel i7-12700K, 32GB RAM, Linux 6.5» |

## 1. Experiment Configuration

| Parameter | Value |
|---|---|
| Model dim | «64» |
| Initial sandbox bodies | «3» |
| Total steps | «10000» |
| Snapshot interval | every «50» steps |
| Total snapshots saved | «200» |
| Disappearance step | «5000» |
| Disappearance object index | «1» |
| Sleep interval | every «200» steps |
| Sleep rounds per phase | «3» |
| Buffer capacity | «2000» |
| Curiosity β schedule | «linear, 1.0 → 0.01 over T=max_steps» |

### Modules exercised
- [x] ActiveInferenceEngine (with Phase-E curiosity)
- [x] ConsciousnessCore (GlobalWorkspace)
- [x] CategoryTheoryEngine (Topos)
- [x] MathUniverse (Fractal)
- [x] CausalEmergenceEngine (CausalInferenceEngine.discover offline)
- [ ] QuantumHybrid (not exercised in this evaluation)
- [ ] BiologicalSubstrate (not exercised in this evaluation)

## 2. Disappearance Intervention (Object Permanence)

### Hypothesis
If the model has built an internal representation of the disappeared
object, removing it should produce:
1. A prediction-error spike at the disappearance step (surprise).
2. A state-vec displacement in the belief/workspace vectors exceeding
   the baseline step-to-step drift.
3. A slow recovery of FE back to the pre-event baseline.

### Results

| Metric | Value |
|---|---|
| FE baseline (pre-event mean) | «x.xxxx» |
| FE at event | «x.xxxx» |
| FE Δ at event | «±x.xxxx» |
| FE Δ / baseline (%) | «±x.x%» |
| FE recovery slope (per step) | «±x.xxxxx» |
| Manifest ΔFE (before → after) | «x.xxxx → x.xxxx (±x.xxxx)» |

**State vector displacement (L2 from pre-event baseline):**

| State vector | Baseline drift | At event | Ratio |
|---|---|---|---|
| belief_state | «x.xxxxx» | «x.xxxxx» | «x.xx x» |
| workspace_attention | «x.xxxxx» | «x.xxxxx» | «x.xx x» |
| workspace_sum | «x.xxxxx» | «x.xxxxx» | «x.xx x» |
| topos_diag | «x.xxxxx» | «x.xxxxx» | «x.xx x» |

### Plots
- `disappearance_fe_curve.png`
- `disappearance_state_drift.png`

### Interpretation
«Paste the `interpretation` field from `disappearance_summary.json`,
then add 1–3 sentences of your own analysis.»

## 3. Causal Graph (Internal State Time Series)

### Hypothesis
Internal-state time series should exhibit CAUSAL structure beyond
trivial autocorrelation. Examples of meaningful edges:
- `belief_pc_k → fe_t` (uncertainty drives surprise)
- `fe_t → expl_step` (high error extends exploration)
- `topos_pc_k → fe_t` (category judgements drive error)

### Results

| Feature set | Samples | Variables | Edges | Method | Acyclic? |
|---|---|---|---|---|---|
| Snapshots | «200» | «19» | «N» | «pc» | «true/false» |
| CSV (dense) | «1000» | «9» | «N» | «pc» | «true/false» |

### Edges discovered (snapshot features)
| Cause | Effect | Weight |
|---|---|---|
| «var1» | «var2» | «±0.xxxx» |
| ... | ... | ... |

### Plots
- `causal_graph_snapshots.png`
- `causal_graph_csv.png`

### Interpretation
«Paste the `interpretation` field from `causal_graph_summary.json`,
then add your own analysis. Note which (if any) edges match the
predefined 'meaningful edge' templates above.»

## 4. Category Structure (Topos + Object Counts)

### Hypothesis
If the model has formed pre-linguistic object concepts:
1. Topos classifier should accumulate structured changes
   (cumulative drift > 3× mean step drift).
2. Truth-value entropy should increase (using a broader truth spectrum).
3. Per-category object counts should track the actual body count
   (decrease at disappearance).

### Results

| Metric | Value |
|---|---|
| Mean step-to-step classifier drift | «x.xxxxx» |
| Max step-to-step classifier drift | «x.xxxxx» |
| Final cumulative classifier drift | «x.xxxxx» |
| Truth-value entropy (first → last) | «x.xxx → x.xxx» |
| Truth-value L2 from baseline (last) | «x.xxxx» |
| Total category objects (pre-event mean) | «x.x» |
| Total category objects (post-event mean) | «x.x» |
| Total category objects (Δ at event) | «±N» |

### Plots
- `categories_classifier_drift.png`
- `categories_truth_values.png`
- `categories_object_counts.png`

### Interpretation
«Paste the `interpretation` field from `categories_summary.json`.»

## 5. Action-Effect Intent

### Hypothesis
If the model has learned a body schema + action-effect model:
1. Action distribution should depend on horizontal position
   (MI > 0.02 nats; chi-squared p < 0.05).
2. Boundary-aware pushing: P(push-away | near-wall) > baseline.
3. Velocity-action correlation: r < 0 (counter-impulse against drift)
   = classic active-inference homeostatic signature.

### Results

| Metric | Value |
|---|---|
| Mutual information I(action; position) | «x.xxxxx nats» |
| Chi-squared statistic | «x.xx» |
| Chi-squared p-value | «x.xxxx» |
| Pearson r (action_x_impulse, agent_vx) | «±0.xxxx» |
| P(push right | near left wall) | «x.xxxx» |
| P(push left | near right wall) | «x.xxxx» |
| P(push right) overall baseline | «x.xxxx» |
| P(push left) overall baseline | «x.xxxx» |
| Boundary intent score | «±0.xxxx» |

### Plots
- `action_intent_distribution.png`

### Interpretation
«Paste the `interpretation` field from `action_intent_summary.json`.»

## 6. Overall Verdict

### Signal tally

| Axis | Signal detected? | Strength |
|---|---|---|
| Disappearance (object permanence) | «yes/no» | «weak/moderate/strong» |
| Causal graph (internal structure) | «yes/no» | «weak/moderate/strong» |
| Categories (topos + object coupling) | «yes/no» | «weak/moderate/strong» |
| Action intent (position coupling) | «yes/no» | «weak/moderate/strong» |

### Conclusion
«1–3 paragraphs synthesising the findings. State which hypotheses
were supported, which were not, and what the next experiment would
be (e.g. longer run, different seed, more objects, vary curiosity
β).»

### Caveats and known limitations
- The default `CategoryTheoryEngine` starts with 3 hard-coded
  categories (NLP, CV, Analytics) with random concept objects —
  these are NOT physical-object categories. The emergence hypothesis
  is that the engine RE-USES these slots to encode physical objects.
- PC algorithm needs many samples; 1000-step runs produce only 20
  snapshots — use `--long` (10000 steps → 200 snapshots) for
  reliable causal graphs.
- All "interpretation" strings are heuristic — they should not be
  read as statistical conclusions, only as decision-theoretic
  summaries. Always inspect the underlying JSON for the raw numbers.

## 7. Reproducibility

```bash
# Reproduce this run:
cd /workspace
git checkout «commit_hash»
python experiments/run_full_evaluation.py --max-steps «N» \
    --disappear-step «D» --seed «S» --output-dir experiments/output/replay_run
```

The deterministic seed contract (see `examples/physics_sandbox.py`
and `run_sandbox_curious.py:make_curious_model`) guarantees that
running the same command twice on the same machine produces
byte-identical frames and snapshot arrays.

## 8. Files Produced

- `manifest.json` — run configuration + paths.
- `replay_log.csv` — dense per-step log.
- `snapshot_step*.pkl` — pickled cognitive snapshots.
- `disappearance_fe_curve.png` / `disappearance_state_drift.png`
- `disappearance_summary.json`
- `causal_graph_snapshots.png` / `causal_graph_csv.png`
- `causal_graph_summary.json`
- `categories_classifier_drift.png` / `categories_truth_values.png` / `categories_object_counts.png`
- `categories_summary.json`
- `action_intent_distribution.png`
- `action_intent_summary.json`
- `FULL_REPORT.md` — auto-generated report (basis for this template).
