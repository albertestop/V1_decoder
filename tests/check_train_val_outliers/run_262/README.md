# Train vs validation outliers

`outliers.csv` has one record per outlying `(split, trial, neuron, token)`. An outlier has trial-level MAE strictly above 0.3 for token 1 or 0.485 for token 2. `row_start:row_end_exclusive` is the complete dataset-row range used for that MAE.
