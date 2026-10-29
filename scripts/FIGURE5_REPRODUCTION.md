# Historical SHAP reproduction

This file documents the earlier SHAP analysis that was numbered Figure 5.
Its source tables remain in `data/figure5/`; the plotting entry point is
`python scripts/plot_figure5_grouped_shap.py --top 20`.

The historical contract uses 21,434 records from 4,085 publication groups,
278 raw predictors, and 6,263 encoded model inputs. Its numerical dimensions
and source hashes are retained for reproducibility and must not be relabeled
as the repaired manuscript analysis.

The current manuscript uses Figure 4b for SHAP, with 21,426 records from 4,080
publication groups and 6,262 encoded inputs. Its exact source tables, editable
Origin projects, and plotting scripts are under `manuscript/`. Follow
`manuscript/docs/REPRODUCIBILITY.md` for current figures.

Both analyses explain a final fitted model using exact interventional
TreeSHAP. In-sample attributions do not measure performance on unseen papers
and must not be interpreted as causal effects. The current main stability
figure is Figure 5.
