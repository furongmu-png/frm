# Active Inference

`ActiveInferenceEngine` (`src/zero_data_model/active_inference.py`) implements
the Free Energy Principle. It owns a `GenerativeModel`, a `MarkovBlanket`
separating internal state from observations, and a `HomeostaticController`
that drives actions to minimise free energy.

## Theory basis

- **Free Energy Principle (FEP)** — a self-organising system maintains itself
  in a low-surprise regime by minimising variational free energy, an upper
  bound on the negative log-likelihood of observations under its generative
  model.
- **Generative model + Markov blanket** — internal states are insulated from
  the environment by a statistical boundary (the blanket); actions and
  perception jointly minimise free energy.
- **Homeostatic controller** — drives the action vector toward set-points that
  keep the system within viable bounds.

## Role in `think()`

- `process(signal)` — updates the generative model with the new observation.
- `predict(signal)` — returns a `Prediction` whose `uncertainty` contributes
  to the model-wide `mean_err`.
- `update(mean_err)` — nudges internal parameters by a prediction-error-scaled
  noise term, closing the predictive-processing loop.

`free_energy_history` is appended every cycle. The last value is exposed via:

- The `zdm_free_energy_last` Prometheus gauge.
- The `last_fe` field returned by `POST /think` (read under `model._lock` so
  it is consistent with the cycle count).

## Auto-generated reference

::: zero_data_model.active_inference.ActiveInferenceEngine

::: zero_data_model.active_inference.GenerativeModel

::: zero_data_model.active_inference.MarkovBlanket

::: zero_data_model.active_inference.HomeostaticController
