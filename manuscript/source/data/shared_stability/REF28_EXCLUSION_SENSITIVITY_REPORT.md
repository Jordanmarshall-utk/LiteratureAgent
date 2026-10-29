# Stability eligibility and dependency analysis

Analysis version: `2026-10-28_controlled_repaired_completion`.

The completed analysis applies the Ref_ID 28 target exclusions, repaired
publication identities, and repaired predicted-PCE features to every retained
stability population. The dependency table contains 66 Original/Integrated
estimates. Forty-four affected fits were recomputed under the retained model
contract; the remaining 22 retain their unchanged predictions.

The contract uses 300-tree ExtraTrees estimators, five deterministic
publication-grouped folds, and the retained preprocessing and feature sets.
Intervals use 2,000 normalized-DOI cluster bootstrap resamples of the
out-of-fold predictions. Predicted PCE is excluded from the main Figure 5
feature sets. Its supplementary alternatives use repaired PCE predictions
joined to stable record identities.

All 24 requested alternative estimates are present in
`requested_24_updated_stability_estimates.csv`. `TableS10_updated.csv` includes
the full alternative-model table, confidence intervals, eligible row counts,
publication-group counts, fold summaries, and prediction references.

Compared with the earlier Ref_ID 28 exclusion table, eight of the 24 selected
Figure 5 values change beyond numerical roundoff, all in the Integrated
series. The exact changes are in
`Figure5_dependency_completion_comparison.csv`. Original-series values agree
within `1e-12`. The current plot tables and editable SVG panels use the
completed values and intervals.

`metric_comparison_vs_frozen.csv` compares the pre-exclusion estimates with
the completed repaired values. Its `*_current` columns identify the earlier
comparison population; its `*_repaired` columns identify the current analysis.
The `delta` is repaired minus earlier estimate.

`refitted_endpoint_provenance.csv` identifies the predictions and folds for
each recomputed fit. Carried predictions and fold metrics are retained under
`analysis/stability/fits/` relative to the manuscript
directory. All source references in current estimate tables use these
released paths.
