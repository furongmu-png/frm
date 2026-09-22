# Cross-Modal Retrieval Evaluation

- Test samples: 200
- Corpus size: 500
- Unique event labels: 4 (action, collision, movement, wall_hit)

## Exact Match (Top-1 / Top-5)

Retrieved corpus index must equal the test index.

| Direction | Top-1 | Top-5 | Mean Rank |
|---|---|---|---|
| Physics → Text | 0.0000 | 0.0050 | 244.8 |
| Text → Physics | 0.0050 | 0.0150 | 260.5 |
| Random baseline | 0.0020 | 0.0100 | 250.5 |
| Modality-mean baseline | 0.0050 | — | — |

## Event-Category Match (Top-1 / Top-5)

Retrieved corpus item must share the same EVENT LABEL as the query (collision / movement / wall_hit / action / stationary).
This is the semantic test — does the bridge know what KIND of event it is seeing?

| Direction | Top-1 | Top-5 |
|---|---|---|
| Physics → Text | 0.2450 | 0.2450 |
| Text → Physics | 0.5300 | 0.5300 |
| Random baseline | 0.4043 | 1.0000 |

## Verdict: WEAK ALIGNMENT

- Best event-category Top-1 = 0.5300 vs random 0.4043 (ratio = 1.31x)
- Best exact Top-1 vs modality-mean baseline = 0.0050

The bridge shows weak cross-modal alignment at the event-category level — retrieval is slightly above chance. Longer training or richer text may help.
