# Outlier comparison: runs 205, 206, 230, and 262

## Scope and definitions

This report uses the four `outliers.csv` files in this directory.  Each row is
one outlying `(trial, neuron, token)` record: its trial-level MAE is strictly
greater than `0.300` for token 1 (time), or strictly greater than `0.485` for
token 2 (recorded spike amplitude).  All four runs use the same dataset and
the same saved trial split: 115 train trials and 13 validation trials.

For overlap results, **neuron** means `neuron_id`, **trial** means
`trial_index`, and **neuron--trial** means the pair of those identifiers.  The
reported overlap percentage is the Jaccard percentage: `common / union`.
The two following percentages state the fraction of the unique entities in
each run that are shared, in run order.  For example, `24.0% (31.7% / 50.0%)`
means 24.0% of the union is shared; the shared entities constitute 31.7% of
the first run's entities and 50.0% of the second run's entities.

The additional **outlier-record coverage** tables below answer a subtly
different question: for each token's outlier records, what fraction has an
entity also present in the comparison set?  For example, a token-1 outlier
record counts as a "same neuron" match when its `neuron_id` appears in *any*
token-2 outlier record.  "Same trial--neuron" requires the exact
`(trial_index, neuron_id)` pair, but does not require the same token.  Since
each CSV contains at most one record for a given token, trial, and neuron,
the exact-pair count is also the count of shared outlier records after token
identity is ignored.

## Parameter sets

Runs 205 and 206 share parameter set A; runs 230 and 262 share parameter
set B.  Within each pair, the meaningful data/model/training settings are the
same.  The saved files differ only in execution/output details (`device` and,
for set A, the cluster path used for `data_root_path` and `output.dir`).

| Setting | Set A: runs 205 and 206 | Set B: runs 230 and 262 |
|---|---:|---:|
| Dataset | `2025-07-04_04_ESPM154_vol_000_subset_1024_target_random_order` | same |
| Validation selection / split | `run_179` / 0.10 | same |
| Model | `TAE_v1_01_1` | same |
| Input / latent dimension | 64 / 64 | 64 / 64 |
| Latent tokens | **512** | **64** |
| Transformer layers / heads | 4 / 4 | 4 / 4 |
| Input / transformer / head dropout | 0.0 / 0.10 / 0.0 | same |
| Epochs | **350** | **500** |
| Learning rate / weight decay / gradient clip | 1e-4 / 1e-4 / 1.0 | same |
| Loss | combined: ID CE 0.1; time masked-MSE 10; recording masked-MSE 10 | same |
| Data batch size / workers | 8 / 16 | 8 / 16 |

Thus the substantive differences between the two sets are the latent-token
count (512 versus 64) and training duration (350 versus 500 epochs).

## Train versus validation outliers

Percentages below are fractions of each run's complete outlier-record total,
not fractions of all possible trial--neuron combinations.

| Run | Total outliers | Train, n (%) | Validation, n (%) | Token 1: train / validation | Token 2: train / validation |
|---|---:|---:|---:|---:|---:|
| 205 | 139 | 41 (29.5%) | 98 (70.5%) | 41 / 98 | 0 / 0 |
| 206 | 88 | 10 (11.4%) | 78 (88.6%) | 10 / 78 | 0 / 0 |
| 230 | 348 | 218 (62.6%) | 130 (37.4%) | 23 / 108 | 195 / 22 |
| 262 | 170 | 50 (29.4%) | 120 (70.6%) | 31 / 99 | 19 / 21 |

## Same-parameter run overlap

### Set A: run 205 versus run 206

All records are token-1 outliers, so the all-token and token-1 results are
identical.  Neither run has a token-2 outlier.

| Entity compared | Run 205 unique | Run 206 unique | Common | Union | Shared: Jaccard (of 205 / of 206) |
|---|---:|---:|---:|---:|---:|
| Neuron | 99 | 72 | 38 | 133 | 28.6% (38.4% / 52.8%) |
| Trial | 34 | 17 | 16 | 35 | 45.7% (47.1% / 94.1%) |
| Neuron--trial | 139 | 88 | 44 | 183 | 24.0% (31.7% / 50.0%) |

Interpretation: 38 individual neurons and 16 trials generate outliers in both
runs.  There are 44 exact neuron--trial combinations in common (out of 183
seen in either run).

### Set B: run 230 versus run 262

#### Ignoring token identity

| Entity compared | Run 230 unique | Run 262 unique | Common | Union | Shared: Jaccard (of 230 / of 262) |
|---|---:|---:|---:|---:|---:|
| Neuron | 113 | 117 | 84 | 146 | 57.5% (74.3% / 71.8%) |
| Trial | 127 | 50 | 50 | 127 | 39.4% (39.4% / 100.0%) |
| Neuron--trial | 348 | 170 | 96 | 422 | 22.7% (27.6% / 56.5%) |

All 50 trials containing an outlier in run 262 also occur among run 230's 127
outlier trials.  At neuron level, 84 neurons are shared; 96 exact
neuron--trial pairs are shared.

#### Matching the same token in both runs

| Token | Entity compared | Run 230 unique | Run 262 unique | Common | Union | Shared: Jaccard (of 230 / of 262) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Neuron | 108 | 108 | 84 | 132 | 63.6% (77.8% / 77.8%) |
| 1 | Trial | 24 | 30 | 18 | 36 | 50.0% (75.0% / 60.0%) |
| 1 | Neuron--trial | 131 | 130 | 96 | 165 | 58.2% (73.3% / 73.8%) |
| 2 | Neuron | 5 | 11 | 0 | 16 | 0.0% (0.0% / 0.0%) |
| 2 | Trial | 125 | 28 | 28 | 125 | 22.4% (22.4% / 100.0%) |
| 2 | Neuron--trial | 217 | 40 | 0 | 257 | 0.0% (0.0% / 0.0%) |

The 96 shared neuron--trial pairs for set B are all token-1 pairs.  For token
2, the runs share the same 28 affected trials, but no individual neuron and
no exact neuron--trial combination is shared.

## Do token 1 and token 2 identify the same sources within a run?

This checks overlap *within* each run between its token-1 and token-2 outlier
sets.  It answers whether a neuron (or the same neuron in the same trial)
causes outliers in both reconstructed quantities.

| Run | Token-1 neurons | Token-2 neurons | Shared neurons | Neuron Jaccard (of token 1 / of token 2) | Shared neuron--trial pairs |
|---|---:|---:|---:|---:|---:|
| 205 | 99 | 0 | 0 | N/A: no token-2 outliers | 0 |
| 206 | 72 | 0 | 0 | N/A: no token-2 outliers | 0 |
| 230 | 108 | 5 | 0 | 0.0% (0.0% / 0.0%) | 0 |
| 262 | 108 | 11 | 2 | 1.7% (1.9% / 18.2%) | 0 |

Therefore, no run contains the same **neuron--trial** pair as an outlier in
both token 1 and token 2.  Runs 205 and 206 have no token-2 outliers.  Run
230 has no neuron shared across its token-1 and token-2 sets.  Run 262 has
two neurons shared across the two token sets, but each is outlying on different
trials for the two tokens.

### Requested within-run record coverage: runs 230 and 262

This makes the preceding within-run result explicit as percentages of the
token-1 and token-2 *outlier records*.  The first percentage is relative to
token 1; the second is relative to token 2.  Exact trial--neuron matches are
zero in both runs, so no individual outlying neuron occurs on the same trial
for both reconstructed quantities.

| Run | Token-1 / token-2 outlier records | Same neuron: n (% of token 1 / % of token 2) | Same trial: n (% of token 1 / % of token 2) | Same trial--neuron: n (% of token 1 / % of token 2) |
|---|---:|---:|---:|---:|
| 230 | 131 / 217 | 0 (0.0% / 0.0%) | 100 / 40 (76.3% / 18.4%) | 0 (0.0% / 0.0%) |
| 262 | 130 / 40 | 2 / 5 (1.5% / 12.5%) | 76 / 17 (58.5% / 42.5%) | 0 (0.0% / 0.0%) |

For run 262, the two token-1 matches and five token-2 matches for “same
neuron” arise from the same two neuron IDs, but on different trials; this is
why the same-trial--neuron column remains zero.

### Requested across-run record coverage: run 230 versus run 262

Here a run-230 record is counted when its entity occurs anywhere in the
corresponding run-262 comparison set, and conversely.  The token-specific
rows compare like tokens; the all-token row ignores token identity.  This
separates broad recurrence of a neuron or trial from a recurrence of the
exact affected neuron in the exact affected trial.

| Comparison | Outlier records: run 230 / run 262 | Same neuron: n (% of 230 / % of 262) | Same trial: n (% of 230 / % of 262) | Same trial--neuron: n (% of 230 / % of 262) |
|---|---:|---:|---:|---:|
| All tokens (token identity ignored) | 348 / 170 | 104 / 103 (29.9% / 60.6%) | 210 / 170 (60.3% / 100.0%) | 96 / 96 (27.6% / 56.5%) |
| Token 1 only | 131 / 130 | 104 / 98 (79.4% / 75.4%) | 119 / 113 (90.8% / 86.9%) | 96 / 96 (73.3% / 73.8%) |
| Token 2 only | 217 / 40 | 0 / 0 (0.0% / 0.0%) | 54 / 40 (24.9% / 100.0%) | 0 / 0 (0.0% / 0.0%) |

Thus all 96 exact cross-run trial--neuron matches are token-1 outliers.
Token-2 outliers recur at 28 trials, but never for the same neuron ID across
the two runs.

## Source artifacts

- `run_205/outliers.csv`, `run_206/outliers.csv`, `run_230/outliers.csv`, and
  `run_262/outliers.csv` provide the per-record evidence.
- The corresponding `summary.json` files provide the split counts.
- `outputs/neural_autoencoder/run_{205,206,230,262}/config.toml` provide the
  saved parameter settings.
