# Figure 5 and companion interpretability figures

Figure 5 explains a post-evaluation Extra Trees PCE refit after the final
Figure 4 predictor order, 113/165 numeric/categorical typing, categorical-schema
harmonizer, and estimator settings are frozen and applied to every eligible
training record. This all-row interpretability refit uses the same model
contract as Figure 4, but it is not one of the held-out estimators used to
calculate Figure 4 performance. The current canonical cohort contains 21,434
device records from 4,085 DOI/publication groups. Exact interventional TreeSHAP
is evaluated for every eligible record; no explanation rows are omitted.

These are in-sample explanations of the final all-data refit. They are not
out-of-fold predictions and are not evidence of performance on unseen
publications. DOI-aware grouped cross-validation remains the model-performance
evaluation. SHAP values describe how the fitted model used correlated database
features and must not be interpreted as causal effects of changing a material
or process.

The release creates four independent single-panel figures:

1. `figure_pce_choice_stratified_shap`: the main Figure 5 distribution. It
   resolves six prespecified scientific bundles into concrete recorded states.
   The source table contains `N x 6` rows, where `N` is the validated number of
   explained devices.
2. `figure_pce_grouped_shap_beeswarm`: the global top-20 figure.
   It displays the 20 highest-ranked source-level scientific groups while
   retaining every device contribution in the plotted distributions.
3. `figure_pce_solvent_formulation_shap`: the formulation-stratified companion.
   It combines precursor-solvent identity with its paired mixing ratio and
   reports the number of device records and DOI/publication groups supporting
   each displayed formulation.
4. `figure_pce_grouped_shap_local_random`: a single-record explanation selected
   with a fixed random seed. Seed 42 selects one record deterministically from the complete
   explanation table before group contributions are ranked for that record.
   Its header and selection JSON identify the selected publication group and
   the number of explained device records sharing that identifier. The base
   value is the expected model output under the governed interventional
   background; the base plus all grouped contributions reconstructs predicted
   PCE, not observed PCE. `All other scientific groups` is the signed net sum
   of undisplayed contributions, so positive and negative values can cancel.
   When a displayed group has a direct recorded state, the label reports that
   state from row metadata (including full/short composition, inorganic and
   lead-free flags, A/B/X-site composition, substrate stack, deposition,
   additives, and synthesis atmosphere). `ADD_DESC_text_length` is explicitly
   identified as an engineered text-metadata proxy rather than a material or
   processing choice.

Each figure is written independently as PNG, PDF, and SVG. The output is not a
multipanel composite.

## Canonical attribution method

The frozen preprocessing pipeline expands 278 raw predictors to 6,263 encoded
model inputs. Downstream code cross-checks that fitted count against both the
frozen contract and `pce_grouped_shap_validation.json`; a changed one-hot
dimension is rejected rather than silently relabeled as the final model. The
canonical analysis then:

1. refits the frozen Extra Trees specification to all eligible training rows;
2. selects a deterministic 100-record interventional background by taking
   DOI/publication-group-diverse records spanning the observed PCE range;
3. computes exact TreeSHAP with
   `feature_perturbation="interventional"` and `model_output="raw"` for every
   fitted row in deterministic batches;
4. sums signed encoded-input contributions within each of 218 governed
   source-level scientific groups for each device; and
5. ranks groups by the device-weighted mean absolute value of those aggregated
   contributions.

The default SHAP batch size is 256 records. Batching limits working memory; it
does not reduce explanation coverage. The background size and selected row
identities are recorded in `pce_grouped_shap_validation.json`. The 100-record
background is part of the canonical attribution definition and should not be
changed when reproducing manuscript results.

The main global figure shows the top 20 groups from the device-weighted ranking.
The complete 218-group ranking remains in
`pce_grouped_shap_importance.csv`. Because publications contribute different
numbers of device records, the release also provides
`pce_grouped_shap_importance_doi_balanced.csv`. That sensitivity analysis first
averages absolute contributions within each publication and then gives each
publication equal weight. The validation JSON records rank correlations and
top-20 overlap between the device-weighted and DOI-balanced results.

Summing signed contributions before taking their absolute magnitude preserves
the net local contribution and prediction reconstruction. These values are
more precisely described as source-group-aggregated SHAP contributions; they
are not coalition-level group Shapley values.

### Numerical additivity disclosure

The released tables retain the raw exact-interventional TreeSHAP contributions;
no residual is redistributed among encoded inputs or scientific groups. The
validation JSON records the strict-pass count, any retained float32 tree
split-boundary exceptions, cohort mean absolute residual, maximum residual,
governed tolerances, and exception-row lineage for the completed harmonized
run. Captions and audit summaries read those values from the validation record
instead of carrying forward numerical results from an earlier fit.

## What the main Figure 5 represents

For every explained device, signed encoded-input contributions are summed only
within each of these six prespecified bundles:

- precursor solvent formulation: solvent identity plus its paired mixing ratio;
- ETL stack and deposition: exactly `ETL_stack_sequence` plus
  `ETL_deposition_procedure`;
- HTL stack and deposition: exactly `HTL_stack_sequence` plus
  `HTL_deposition_procedure`;
- absorber composition identity: recorded ion identities, aligned
  coefficients, short and long composition forms, and inorganic/lead-free
  indicators;
- quenching-induced crystallization status; and
- back-contact material.

The choice-value table therefore contains six rows per explained device. Its
expected row count is read from the validation file as `explained_rows × 6`,
rather than from a fixed plotting constant.

States are selected by prevalence, independently of SHAP magnitude. A state
must be reported, valid, and present in at least five device records; at most
the three most prevalent states in each bundle are retained. Figure labels
state both the number of devices and the number of distinct DOI/publication
groups. Each dot is one device's signed bundle contribution to fitted PCE; the
diamond and horizontal line show the median and interquartile range.

Blue contributions lower the fitted prediction relative to the explainer's
background expectation, and orange contributions raise it. Color is not a
low-to-high feature-value scale because a grouped categorical bundle has no
single defensible scalar feature value.

`Perovskite precursor solvent formulation` contains only solvent identity and
its paired mixing ratio. Supplier and purity remain separate scientific groups.
The display label `Not reported in source (imputed for model)` denotes absent
source metadata handled by the frozen model preprocessing; it is not an
explicitly learned missing category.
The formulation panel reports fitted-model associations, not the PCE change
expected from experimentally replacing one solvent with another.

## Rebuild the complete attribution audit

Install `scripts/requirements-figure5.txt`. The explicit canonical command is:

```powershell
python scripts/audit_pce_feature_interpretability.py `
  --final-harmonized-contract `
  --figure5-only `
  --shap-method exact-interventional `
  --shap-batch-size 256 `
  --interventional-background-rows 100 `
  --data path\to\final_integrated_database.csv `
  --manifest path\to\pce_feature_columns_278.csv `
  --out path\to\pce_feature_interpretability_harmonized_full_refit_v1
```

`exact-interventional`, 256 records per batch, and 100 background records are
the command defaults, so the three SHAP options may be omitted without changing
the canonical method. `--final-harmonized-contract` is mandatory for manuscript
Figure 5 and automatically selects the bundled, hash-pinned final model snapshot
and 113/165 typing inventory unless explicit overrides are supplied. Progress is reported batch
by batch. Final-harmonized runs atomically checkpoint every completed batch in
`<out>/_shap_checkpoints`. If the process or command wrapper is interrupted,
rerun the identical command: resume is enabled by default, the manifest and
every shard are strictly revalidated, and completed batches are skipped. The
manifest pins the fitted population and tree model, all frozen source hashes,
encoded feature names and mappings, group matrices, interventional background,
SHAP method, library version, and batch size; a changed contract is rejected
rather than mixed with stale shards. Use `--shap-checkpoint-dir` only to choose
another checkpoint location and keep `--no-resume-shap` for an intentionally
new, empty checkpoint directory. The archived `pce_feature_interpretability_audit` directory is retained
for historical feature-representation/ablation provenance and must never be
used as a fallback Figure 5 source.

Before the long TreeSHAP computation, validate the final contract with the same
data, manifest, and versioned output directory:

```powershell
python scripts/audit_pce_feature_interpretability.py `
  --final-harmonized-contract `
  --preflight-only `
  --data path\to\final_integrated_database.csv `
  --manifest path\to\pce_feature_columns_278.csv `
  --out path\to\pce_feature_interpretability_harmonized_full_refit_v1
```

The preflight records the fitted encoded-input count, ordered feature-name and
mapping fingerprints, population fingerprint, model/data/manifest/typing/
harmonizer hashes, and a seeded repeat-fit numerical-equivalence check. The
full validation must reproduce this frozen contract before figures are built.

Within the full project, `build_curated_publication_figures.py` reads this
versioned directory by default. Use `--shap-audit-dir` to make an override
explicit; the builder does not fall back to the archived pre-harmonizer audit.

The audit writes the following canonical Figure 5 source files:

- `pce_grouped_shap_values.csv`
- `pce_grouped_shap_importance.csv`
- `pce_grouped_shap_importance_doi_balanced.csv`
- `pce_grouped_shap_row_metadata.csv`
- `pce_grouped_shap_validation.json`
- `pce_choice_stratified_shap_values.csv`
- `pce_choice_stratified_shap_summary.csv`
- `pce_solvent_formulation_shap_values.csv`
- `pce_solvent_formulation_shap_summary.csv`
- `pce_grouped_shap_local_random_selection.json`
- `pce_encoded_feature_dictionary.csv`

If complete exact-interventional contribution tables already exist but their
validation/report finalization was interrupted, use the same command with
`--finalize-existing-shap`. This mode rejects a changed or reordered cohort,
feature mapping, model prediction vector, governed background, importance
table, or choice table; it also recomputes exact SHAP for all strict-threshold
exceptions plus a deterministic DOI-diverse spot check. It writes fresh
validation and audit reports and refreshes source-derived row context without
recomputing or modifying the full contribution table. It is not a shortcut for
partial, sampled, stale, or otherwise misaligned SHAP results.

The governed integrated modeling table and frozen 278-feature manifest are not
duplicated in the public source release. The exact frozen model snapshot and
feature-typing inventory are included because they are small contract files;
the command still requires the user-supplied governed database and matching
278-feature manifest shown above. See
`scripts/figure5_reproduction_inventory.csv` for the included inputs and entry
points.

## Reproduce the single-panel figures

After placing the canonical source tables under `data/figure5/`, run from the
repository root:

```powershell
python scripts/plot_figure5_grouped_shap.py --top 20
```

Twenty is the plotting default. The attribution audit validates the complete
encoded-input-to-218-group mapping and complete explanation coverage. The plotting command
then validates row alignment, correspondence between all 218 groups and the
importance table, six-bundle coverage, and prediction reconstruction before
writing figures and derived tables under `results/LiteratureAgent/figure5/`. Figure annotations
read the cohort size and explanation scope from
`pce_grouped_shap_validation.json`; they do not assume a fixed row count.

The primary ranked table is `pce_grouped_shap_top20.csv`. The complete global
values and all 218 importance rows remain available in the source files listed
above.

## Troubleshooting and noncanonical diagnostics

- If memory is limited, reduce `--shap-batch-size`. This changes only how many
  records are processed at once and does not authorize partial coverage. A
  smaller value such as 32 also creates durable checkpoints more frequently;
  use the same batch size on every resume because it is part of the checkpoint
  manifest.
- `--shap-method exact-path-dependent` is retained only for diagnostic work.
  Exact path-dependent attribution is numerically unsuitable for this fitted
  ensemble and is not the manuscript method.
- `--shap-method approximate-path-dependent` is also noncanonical. Its
  additivity reconstruction does not make the path approximation exact, so it
  must not replace exact interventional TreeSHAP in released manuscript
  figures.
