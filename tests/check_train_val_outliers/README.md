# Train-versus-validation outlier check

Run the utility with a saved neural-autoencoder run name:

```bash
/home/albertestop/miniconda3/envs/sci_albert/bin/python \
  tests/check_train_val_outliers/analyze_run_outliers.py --run run_193 --device cuda:0
```

It evaluates every complete train and validation trial from the run's saved
dataset split. An outlier is the existing high-MAE definition: trial-level MAE
strictly greater than `0.3` for token 1 or `0.485` for token 2. It writes
`tests/run_193/` by default (substituting the requested run name), with:

- `summary.json`: train-versus-validation outlier counts by token;
- `outliers.csv`: split, trial, global dataset row range, neuron index/ID,
  token, threshold, and MAE for every outlier;
- `README.md`: definition of the generated records.

Use `--data-root` when the dataset directory recorded in a run configuration
is unavailable locally, and `--output-dir` to choose a different destination.
