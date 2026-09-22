# Active Experiment

Phase 6 lets the model select which experiment to run next to maximally reduce
uncertainty (`BayesianExperimentPlanner`) and test hypotheses with Bayes factors
(`HypothesisTester`).

## Theory basis

- **Bayesian optimal experimental design** — pick the candidate whose expected
  information gain (mutual information between outcome and parameter) is
  largest. Implemented as a Dijkstra-style search over a candidate graph
  weighted by expected uncertainty reduction.
- **Bayes factors** — `HypothesisTester` uses
  `BF = P(data|H1) / P(data|H0)` to decide support/reject, avoiding the
  arbitrary p-value threshold problem.

## Feature flag

`enable_experiment_planner` (default `False`).

## `think()` integration

1. `experiment_planner.evaluate(current_uncertainty=mean_uncertainty,
   step=cycle)` → `upgrade_meta["experiment"]`.
2. `hypothesis_tester.get_supported()` →
   `upgrade_meta["supported_hypotheses"]`.

## Thread safety

- `BayesianExperimentPlanner` — lock around `evaluate` / `record_result`.
- `HypothesisTester` — lock around `get_supported` / `record_result`.

## REST endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/experiment/select-best` | Pick the next best experiment (Bayesian). |
| `POST` | `/experiment/record` | Record an experiment outcome. |

## Auto-generated reference

::: zero_data_model.experiment.experiment_planner.BayesianExperimentPlanner

::: zero_data_model.experiment.experiment_planner.CandidateExperiment

::: zero_data_model.experiment.hypothesis_tester.HypothesisTester

::: zero_data_model.experiment.hypothesis_tester.Hypothesis
