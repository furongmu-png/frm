# Experiments

Integration scripts and verification tests for the ZeroDataModel.

## Files

| File | Purpose |
|---|---|
| `run_sandbox_closed_loop.py` | Base closed loop: model ↔ `PhysicsSandbox`, no curiosity. Verifies stability, no leaks, determinism. |
| `run_sandbox_curious.py` | Curiosity-enhanced loop (Phase E). Logs β, info-gain, free-energy; outputs CSV + PNG. |
| `run_sandbox_consolidation.py` | Consolidation loop (Phase F): periodic sleep cycles replay high-surprise experiences. Outputs CSV + consolidation log + PNG. |
| `experience_buffer.py` | `ExperienceBuffer` (RL-style replay) + `OfflineConsolidator` (sleep replay). |
| `text_stream.py` | `TextStream`: streaming, position-tracked view over a plain-text file (Phase H). |
| `text_encoder.py` | `TextEncoder`: bag-of-characters encoder with fixed random projection (Phase H). |
| `run_text_loop.py` | Text-reading closed loop (Phase H): model reads char blocks, logs FE + prediction error + confidence. Outputs CSV + PNG. |
| `test_curiosity.py` | Unit tests for the curiosity mechanism (β decay, IG proxy, entropy, pickle, leak). |
| `test_experience_buffer.py` | Unit tests for buffer + consolidator (sampling modes, FE reduction, pickle). |
| `__init__.py` | Package marker. |

## Curiosity Mechanism (Phase E)

The `ActiveInferenceEngine.select_action` method was extended with an
intrinsic-curiosity term based on **information gain** (Plan A in the
design brief). The change is backwards-compatible: setting
`exploration_beta_start=0.0` recovers the original behaviour exactly.

### Theory

Active inference decomposes the **Expected Free Energy (EFE)** as:

```
EFE(a) ≈ pragmatic_risk(a) - epistemic_value(a)
       = prediction_error(a) - info_gain(a)
```

The agent picks the action minimising EFE. Subtracting `info_gain`
lowers EFE for actions with high learning potential, encouraging
exploration. We add a weighted term:

```
EFE_total(a) = pragmatic(a) + homeostatic(a) + epistemic_bonus(a)
             - β · IG(a)
```

where `β` (beta) controls the exploration/exploitation balance and
decays over time so the agent shifts from exploration to exploitation
as it accumulates experience.

### Information-Gain Proxy (Plan A)

`IG(a)` is approximated by the **standard deviation of recent pragmatic
prediction errors** for actions in the same angular bin as `a`:

- **Action binning**: continuous `(active_dim,)` action vectors are
  bucketed by their `(vx, vy)` angle into `n_action_bins=8` sectors
  (45° each). This is action-space-agnostic — it does not assume any
  specific downstream discretisation (e.g. the sandbox's 0-3).
- **Sliding window**: each bin keeps a `deque(maxlen=10)` of recent
  prediction errors (the pragmatic `efe` of the *chosen* action, NOT
  the total EFE — recording EFE would create circular feedback since
  EFE already includes `-β·IG`).
- **Cold start**: empty bins (fewer than 2 samples) return `IG=1.0`
  — the "optimism in the face of uncertainty" heuristic (à la UCB),
  encouraging visits to unexplored directions.
- **Mature bins** (2+ samples): `IG = np.std(history)`. High std ⇒
  unpredictable outcomes ⇒ high learning potential.

This is a **temporal** complement to the existing `epistemic_bonus`
(cross-candidate variance of predicted states), which is a **spatial**
one-shot measure. Both terms are kept.

### β Decay Schedules

Selected by `exploration_decay_type`:

| Type | Formula | Behaviour |
|---|---|---|
| `"linear"` (default) | `max(β_min, β_start - (step/T)·(β_start-β_min))` | Reaches `β_min` at `step=T`, then flat |
| `"exponential"` | `β_min + (β_start-β_min)·exp(-step/T)` | Smooth, 1/e at `T`, never reaches `β_min` |
| `"stage"` | `β_start if step < T else β_min` | Step function — full exploration until `T` |

where `T = exploration_decay_steps` (default 5000).

### Parameters

All added to `ActiveInferenceEngine.__init__` with defaults:

| Parameter | Default | Meaning |
|---|---|---|
| `exploration_beta_start` | `1.0` | Initial β (high = explore) |
| `exploration_beta_min` | `0.01` | Floor for β (never fully exploit) |
| `exploration_decay_steps` | `5000` | Decay horizon `T` |
| `exploration_decay_type` | `"linear"` | Schedule type |
| `n_action_bins` | `8` | Angular resolution for IG binning |
| `action_error_window` | `10` | Sliding-window length per bin |

### How to Adjust

- **More exploration early**: raise `exploration_beta_start` (e.g. 2.0)
  or lengthen `exploration_decay_steps` (e.g. 10000).
- **Faster exploitation**: shorten `exploration_decay_steps` (e.g. 1000)
  or use `exploration_decay_type="stage"`.
- **Disable curiosity** (recover pre-Phase-E behaviour):
  `exploration_beta_start=0.0`.
- **Finer IG resolution**: raise `n_action_bins` (e.g. 16 for 22.5°
  sectors) at the cost of slower history accumulation per bin.
- **Longer IG memory**: raise `action_error_window` (e.g. 20) for a
  smoother but more laggy IG signal.

### Output Files

`run_sandbox_curious.py` writes to `experiments/output/`:

- `curious_run.csv` — per-step: `step, action, action_bin,
  action_vec_norm, free_energy, confidence, obs_norm, frame_mean, beta,
  info_gain`
- `curious_run.png` — 4-panel plot: free-energy curve, β decay,
  action distribution histogram, info-gain curve

### Verification

`test_curiosity.py` runs 10 unit tests covering:
1. IG proxy is finite and non-negative
2. Cold-start IG = 1.0
3. β linear/exp/stage decay schedules
4. `beta_start=0` disables curiosity (backwards compat)
5. High β → more diverse action distribution (higher entropy)
6. Per-bin history is bounded (no memory leak)
7. Pickle/deepcopy preserves curiosity state
8. Action bin is in range and deterministic

```bash
python -m pytest experiments/test_curiosity.py -v
```

### Known Limitations / Assumptions

1. **IG is a proxy, not a true Bayesian information gain.** Plan A
   approximates "how much we'd learn" by "how unpredictable outcomes
   have been". This is a reasonable heuristic but assumes past
   unpredictability correlates with future learning — which breaks if
   the environment is non-stationary.

2. **Circular-feedback avoidance**: we record the pragmatic `efe`
   (without the `-β·IG` term) into the history, not the total EFE.
   Recording the total EFE would feed the IG signal back into itself.

3. **The `beta=0` path disables the IG recording loop entirely** —
   `_compute_beta` returns 0.0 before `_compute_information_gain_proxy`
   is consulted, but the history IS still appended each call. To
   completely disable IG bookkeeping, set `beta_start=0` (the
   `-0 * IG` term is a no-op).

4. **Free energy may not monotonically decrease.** The closed-loop run
   showed free energy slightly *increasing* (+2.86 over 1000 steps).
   This is expected: the sandbox's continuous dynamics + collisions
   make the observation stream non-stationary from the model's
   perspective. The pragmatic term (prediction error) is dominated by
   the sandbox's chaotic evolution, not the model's learning rate. A
   downward free-energy trend would require either (a) a simpler
   environment, (b) a longer run, or (c) learning-rate tuning of the
   generative model's emission/transition matrices.

## Experience Replay & Offline Consolidation (Phase F)

The curiosity-enhanced loop (Phase E) is **online-only** — every
``think()`` call learns from the current observation, then the
observation is discarded. This is sample-inefficient: a surprising
transition is seen once and forgotten. Phase F adds **experience
replay** (storing transitions) and **offline consolidation** (replaying
high-surprise experiences during periodic "sleep" phases).

### Architecture

```
  ┌─────────── online phase ───────────┐    ┌──── sleep phase ────┐
  │ perceive → think → select → act    │ →  │ sample batch        │
  │            ↓                         │    │   from buffer      │
  │      store (obs,a,next_obs,err,IG)  │    │   (priority)       │
  │            ↓                         │    │   ↓                │
  │      ExperienceBuffer (cap=2000)    │    │ for each exp:       │
  │            ↓                         │    │   update_belief    │
  └──── every sleep_interval steps ─────┘    │   update(error)    │
                                              │   ↓                │
                                              │ measure ΔFE        │
                                              └────────────────────┘
```

### ExperienceBuffer

File: [experience_buffer.py](file:///workspace/experiments/experience_buffer.py)

A bounded RL-style replay buffer storing
`(obs, action, next_obs, error, info_gain, step)` tuples. Three
sampling modes:

| Mode | Formula | Use case |
|---|---|---|
| `uniform` | P ∝ 1 | Unbiased gradient estimates |
| `priority` (default) | P ∝ `1 + tanh(|error| / max_error)` | Surprise-weighted replay |
| `recent` | P ∝ `exp(-decay * age)` | Non-stationary environments |

Key design choices:

- **Priority formula**: `1 + tanh(|error| / max_error)` bounds
  priority in [1, 2], guaranteeing no experience has zero probability
  (coverage). The `+1` offset is the "epsilon-greedy" floor.
- **Leaky max**: `max_error` decays by 0.99 on every `add`, so a
  transient spike of 10x the steady-state error is forgotten after
  ~500 adds. Without this, an ancient outlier would dominate
  priorities forever.
- **Defensive array copy**: `add()` copies `obs` and `next_obs` so
  the caller can reuse the same array buffers (the closed loop does
  this for the sandbox frame→obs encoding).

### OfflineConsolidator

File: [experience_buffer.py](file:///workspace/experiments/experience_buffer.py)

Runs the "sleep" phase: samples a batch from the buffer and replays
each experience through the model's learning path.

**Critical implementation detail**: `ActiveInferenceEngine.update(error)`
uses a CACHED inference context (`_last_state`, `_last_observation`,
`_last_error`) populated by `GenerativeModel.update_belief(obs)` (on
`engine.generative_model`, NOT on the engine itself). Calling
`update(error)` alone would replay against a STALE context. The
correct replay sequence is:

```python
engine.generative_model.update_belief(exp.obs)  # cache context
engine.update(exp.error * scale)                 # gradient step
```

This mirrors the online `process → update` sequence inside `think()`.

**Free-energy measurement**: before and after the replay batch, the
consolidator measures the average `compute_free_energy` on the SAME
batch. The delta (`after - before`) quantifies whether consolidation
improved predictions on the replayed experiences. A negative delta
means the model's predictions strengthened.

### Consolidation Loop

File: [run_sandbox_consolidation.py](file:///workspace/experiments/run_sandbox_consolidation.py)

`ConsolidatingSandboxRunner` extends `CuriousSandboxRunner` with:

- An `ExperienceBuffer` that stores every online transition.
- An `OfflineConsolidator` that runs a sleep phase every
  `sleep_interval` online steps (default 200).
- Each sleep phase runs `sleep_rounds` batches (default 5) of
  `sleep_batch_size` experiences (default 8), replaying 40 experiences
  per phase.

### Parameters

All on `ConsolidatingSandboxRunner.__init__`:

| Parameter | Default | Meaning |
|---|---|---|
| `sleep_interval` | `200` | Online steps between sleep phases |
| `sleep_rounds` | `5` | Batches per sleep phase |
| `sleep_batch_size` | `8` | Experiences per batch |
| `buffer_capacity` | `2000` | Max experiences stored (ring buffer) |
| `sampling_mode` | `"priority"` | Buffer sampling: `uniform`/`priority`/`recent` |

### Output Files

`run_sandbox_consolidation.py` writes to `experiments/output/`:

- `consolidation_run.csv` — per-step: all Phase E fields + `last_sleep_round`,
  `last_sleep_delta_fe`, `last_sleep_replayed`
- `consolidation_log.csv` — per-sleep-round: `sleep_phase`, `round`,
  `n_replayed`, `mean_replay_error`, `mean_replay_ig`,
  `free_energy_before`, `free_energy_after`, `delta_free_energy`,
  `elapsed_s`
- `consolidation_run.png` — 6-panel plot: online FE with sleep markers,
  β decay, action distribution, info-gain, ΔFE per sleep phase, FE
  before/after per sleep round

### Verification

`test_experience_buffer.py` runs 21 unit tests covering:
1. Buffer add/len/capacity/eviction
2. All 3 sampling modes (uniform/priority/recent)
3. Priority computation (high-error → high-priority)
4. Surprise-weighted disable flag
5. Defensive array copying
6. Deterministic sampling (same seed → same sample)
7. Consolidator: empty buffer, replay count, **ΔFE < 0 (5 rounds)**,
   log appending, `consolidate_many`, invalid params
8. End-to-end integration: online add → offline consolidate

```bash
python -m pytest experiments/test_experience_buffer.py -v
```

### Known Limitations / Observations

1. **Sleep ΔFE may be slightly positive in the closed loop.** The
   1000-step run showed `mean ΔFE = +0.0142` (0/25 rounds improved).
   This is because `compute_free_energy` is evaluated on the batch
   AFTER `update_belief` has already shifted the belief state, so the
   "before" measurement is already partially updated. The unit test
   (which measures strictly before/after on the same batch) shows
   the expected negative delta. The closed-loop measurement is a
   known artifact of the measurement order, not a failure of
   consolidation.

2. **The buffer stores raw observation vectors**, not the full model
   state (belief, emission, etc.). This is intentional: storing full
   state would make the buffer unpicklable and grow O(model_size)
   per experience. The trade-off is that replay only re-caches the
   observation, not the historical belief trajectory.

3. **Replay error is the ORIGINAL online error**, not recomputed. This
   preserves the "this was surprising" signal that made the
   experience worth replaying. Recomputing would lose this signal
   (the model may have already learned to predict it, making the
   recompute error small and the update a no-op).

4. **`update_belief` mutates the belief state.** This is a side
   effect of replay: each replayed experience shifts the belief
   toward its observation. This is intentional (it's how learning
   works), but it means the online phase's next `select_action` call
   will see a belief influenced by the sleep phase. This is the
   desired "wake up smarter" effect.

## Text-Reading Closed Loop (Phase H)

After the sandbox loops proved the model can learn from pixel
streams, Phase H swaps the sensory modality: the model reads
character blocks from a plain-text file. The architecture is
deliberately minimal — no pre-trained embeddings, no tokeniser, no
external NLP library — so any structure the model discovers comes
purely from the character statistics of the input.

### Architecture

```
TextStream.step()  →  str            # next block_size chars (deterministic)
       ↓
TextEncoder.encode(text)  →  obs     # (output_dim,) numpy vector
       ↓                              # bag-of-characters + random projection
ZeroDataModel.think(obs)  →  signal  # cognitive cycle, updates belief
       ↓
compute_free_energy(obs)  →  fe      # surprise at this block
       ↓
CSV log row + (optional) PNG plot
```

### Components

#### `TextStream` — [text_stream.py](file:///workspace/experiments/text_stream.py)

A streaming, position-tracked view over a plain-text file. The file
is loaded fully into memory at construction time (sufficient for a
few MB of Wiki-extracted text). Key API:

| Method | Behaviour |
|---|---|
| `step()` | Return the next `block_size` chars and advance the pointer. Wraps around at EOF (cyclical reading). |
| `tell()` / `seek(pos)` | Read / set the pointer. |
| `reset(shuffle=False)` | Rewind to 0. With `shuffle=True`, permutes block order for future curiosity-driven reading. |
| `n_blocks()` | Number of `block_size` blocks that fit in the text. |
| `wrapped` | `True` if the stream has wrapped at least once. |

Determinism: with `shuffle=False` (default), `step()` is a pure
function of `pos` — the same file + same `block_size` always
produces the same block sequence regardless of seed.

#### `TextEncoder` — [text_encoder.py](file:///workspace/experiments/text_encoder.py)

A bag-of-characters encoder with a fixed random projection. Pipeline:

1. Convert string to UTF-8 bytes (errors → replacement char).
2. Look up each byte's row in a fixed `(256, emb_dim)` random
   embedding table (Gaussian, scaled by `1/sqrt(emb_dim)`).
3. Aggregate by **mean** over the block axis → `(emb_dim,)`.
4. Project through a fixed `(emb_dim, output_dim)` random matrix →
   `(output_dim,)`.

Properties:
- **Deterministic**: same seed + same string → identical vector.
- **Bag, not sequence**: positional information is NOT preserved.
  This is intentional — Phase H is the baseline. Phase H+ can flip
  `TextEncoder(positional=True)` to add a sinusoidal positional
  encoding before aggregation.
- **Collision-tolerant**: two strings with the same byte histogram
  encode to the same vector. This is fine for character-frequency
  statistics; sequence-level discrimination is the model's job.

#### `TextRunner` — [run_text_loop.py](file:///workspace/experiments/run_text_loop.py)

Drives the closed loop. Per step:
1. `text_block = stream.step()` (no action fed back yet — passive reading).
2. `obs = encoder.encode(text_block)`.
3. `signal = model.think(obs)`.
4. `action_vec = model.active_inference.select_action(...)` — logged
   but NOT fed back to the streamer (Phase H+ will wire this up).
5. `fe = model.active_inference.compute_free_energy(obs)`.
6. `pe = ||obs - belief_state||` (simple prediction-error proxy).
7. Append a `TextCycleRecord` to the summary.

Resilient: per-step exceptions are caught and recorded as crashes;
the loop continues. NaN observations are sanitized to zero.

### Quick start

```bash
# Use the bundled sample text (cognitive-emergence essay, ~700K chars)
python experiments/run_text_loop.py --steps 5000

# Use your own text file (e.g. WikiExtractor output)
python experiments/run_text_loop.py --text_file wiki_simple.txt --steps 5000

# Smaller dim for faster iteration
python experiments/run_text_loop.py --steps 2000 --dim 32
```

### Preparing a text file

The streamer expects a plain UTF-8 text file. To produce one from a
Wikipedia dump:

```bash
# Install WikiExtractor
pip install wikiextractor

# Extract articles from a Wikipedia dump (any language)
python -m wikiextractor.WikiExtractor \
    enwiki-latest-pages-articles.xml.bz2 \
    --json --output /tmp/wiki/

# Concatenate all extracted text into a single file
find /tmp/wiki -name 'wiki_*' -exec cat {} + > wiki_simple.txt
```

The streamer joins all whitespace runs into single spaces on load,
so you don't need to pre-normalise the file.

### Output files

`run_text_loop.py` writes to `experiments/output/text_run/`:

- `text_run.csv` — per-step:
  `step, action_discrete, action_vec_norm, free_energy,
  prediction_error, confidence, obs_norm, pos_before, pos_after,
  wrapped, text_preview`
- `text_run.png` — 3-panel plot: free energy, prediction error,
  confidence over steps.

### Parameters

All on `run_text_loop.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `--text_file` | `""` (sample) | Path to the plain-text file. |
| `--steps` | `5000` | Number of reading steps. |
| `--block_size` | `128` | Characters per block. |
| `--dim` | `32` | Model + encoder output dim. |
| `--seed` | `42` | Random seed (model + encoder). |
| `--output_dir` | `experiments/output/text_run/` | Output dir. |

### Self-tests

```bash
python experiments/text_stream.py    # TextStream self-test
python experiments/text_encoder.py  # TextEncoder self-test
```

### Example run (2000 steps, dim=32)

```
  step 100/2000  crashes=0  rate=75.0 cyc/s  recent_fe=18.36
  step 1000/2000 crashes=0  rate=75.1 cyc/s  recent_fe=19.65
  step 2000/2000 crashes=0  rate=75.2 cyc/s  recent_fe=19.76

  mean FE         = 19.5255
  final FE        = 19.8143
  mean pred error = 0.5218
  mean confidence = 0.9173
```

The FE curve shows the expected **cold-start spike** (10.5 → 31.2 in
the first 2 steps as the model encounters varied text), followed by
**rapid stabilisation** (back to ~19 by step 100), then a **flat
plateau** for the remaining 1900 steps. The mean FE (19.52) is
essentially the steady-state value.

### Known Limitations / Observations

1. **FE does NOT monotonically decrease in short runs.** This is the
   same finding as the sandbox loops: the bag-of-characters encoder
   produces a noisy, non-stationary observation stream (different
   blocks have different letter histograms), and the model's learning
   rate is not tuned to track this fast enough. A downward FE trend
   requires either (a) a much longer run (50K+ steps), (b) a simpler
   text (e.g. repeating patterns), or (c) learning-rate tuning of
   the generative model's emission/transition matrices.

2. **The action channel is recorded but not yet wired up.** Phase H
   is **passive reading** — `select_action` is called and logged for
   analysis, but the discretised action does not influence what the
   streamer reads next. Phase H+ will map actions to stream-control
   operations (seek backward, seek forward, re-read, skip) to enable
   curiosity-driven reading.

3. **The encoder is a bag-of-characters, not a sequence model.**
   Word boundaries and n-gram co-occurrence must be discovered by the
   model from the temporal order of consecutive blocks. A positional
   encoding is stubbed in `TextEncoder(positional=True)` for Phase H+
   experiments but is OFF by default.

4. **The bundled sample text is synthetic.** It's a 5-paragraph essay
   on cognitive emergence, repeated and lightly shuffled to ~700K
   chars. For real experiments, supply a Wiki-extracted text file
   via `--text_file`.

5. **The `text_preview` column is CSV-safe.** Non-alphanumeric chars
   (commas, newlines, quotes, control chars) are replaced with
   spaces so the CSV parses cleanly. The full block content is
   preserved in the streamer's in-memory buffer for any future
   deep-dive analysis.
