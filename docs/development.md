# Development Notes

## Coding conventions

- Keep model code under `src/v1tovideo/<module>/`.
- Keep runnable scripts in `scripts/` with argparse CLI.
- Keep prototype or exploratory scripts in `legacy/`.
- Avoid hardcoded absolute dataset paths in reusable modules.

## Data and outputs

- Put local source data under `data/` (git-ignored).
- Put generated files under `outputs/` (git-ignored).
- Keep only small, representative assets under version control in `assets/`.

## Reproducibility

- Use explicit random seeds in scripts.
- Save summary metrics to text or JSON in `outputs/`.
