# Stability figure sources

`primary_repaired_stability_metrics.csv` contains the 24 estimates selected
for Figure 5. The panel-specific `plot_data.csv` tables contain the same
values, 95% confidence limits, and sample sizes, plus rendering columns.
`Figure5_stability_originpro_replot_source.csv` presents the selected estimates
in a wide worksheet layout.

`all_stability_metrics_and_sensitivity.csv` contains all 66 retained endpoint
and feature-mode estimates. `TableS10_updated.csv` presents the supplementary
comparisons. `stability_dependency_change_table.csv` records eligibility,
publication-group, and predicted-PCE dependencies for each estimate.

Current out-of-fold prediction paths are relative to the manuscript directory.
Forty-four recomputed estimates reference `analysis/stability/fits/`;
22 carried estimates reference `analysis/stability/fits/`.
`refitted_endpoint_provenance.csv` supplies the recomputed source references.

To regenerate the two individual panels from the repository root:

```powershell
python manuscript/scripts/plot_stability.py all --out-dir manuscript/build/stability
```

The released editable format for these two panels is SVG. Each panel is
85 × 72 mm with Times New Roman 12 pt body text and 13 pt axis labels.
The navy square denotes Original and the orange circle denotes Integrated;
intervals are 95% normalized-publication bootstrap limits. The shared legend
is supplied separately with the other manuscript figure assets.
