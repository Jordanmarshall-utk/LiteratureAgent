# Analysis definitions

## PCE

The historical fixed test contains 4,092 records, 744 stored group labels,
and 741 normalized DOI identities. Normalized identities govern publication
exclusion and cluster resampling. The separate LiteratureAgent-domain test
retains 125 records from 80 normalized publication groups.

One adjudicated PCE target changes from 10.9% to 13.32%. That record remains in
the primary 125-record test. The unchanged-target sensitivity contains 124
records from 79 groups. Its expansion-minus-original bootstrap median deltas
are R2 +0.991712 (95% interval +0.573774 to +1.672944), RMSE -2.238294 PCE
percentage points (-2.837476 to -1.543085), and MAE -2.159698 points
(-2.801068 to -1.446664).

Five repeated publication splits retain seeds 0 through 4 and the established
20% publication-group test selection. The row-random diagnostic retains its
4,159-row test membership. Neither result replaces the primary test.

The repaired-minus-frozen main out-of-fold comparison covers 21,425 common
unchanged-target records. Paired intervals are calculated on the joint
publication resamples, not from marginal confidence bounds.

The repaired SHAP and PCE fitting population is 21,426 records from 4,080
publication groups, distinct from each evaluation test.

## SHAP

Figure 4b and Figures S1-S3 explain a final full-cohort refit using 21,426
records from 4,080 publication groups. The frozen preprocessing contract uses
278 predictors (113 numeric, 165 categorical), 6,262 encoded inputs, and 218
scientific groups. Exact interventional TreeSHAP uses the retained 100-record
background. These are in-sample model attributions, not held-out performance
or causal effects.

The historical 21,434-record SHAP files in the repository's `data/figure5/`
retain their original contract. They are distinct from the current repaired
manuscript analysis.

## Stability

The completed table contains 66 estimates. The 24 specifically affected
alternatives cover the prespecified design-only, predicted-PCE, and
measured-initial-performance modes. The source-context rule quarantines six
derived targets for Ref_ID 28; eligibility of other targets is evaluated
separately. DOI repairs are applied before publication-group splitting.

Predicted-PCE alternatives use stable joins to repaired out-of-fold values for
the PCE training population and the documented full-fit fallback outside that
population. Predicted PCE is excluded from every main Figure 5 feature set.
The source tables retain the distinction between measured initial performance,
physical-condition features, and condition-normalized targets.

Figure 5 contains 24 selected Original/Integrated estimates. Eight Integrated
point estimates change relative to the preceding completed source. The largest
absolute changes are 0.014805 ROC-AUC and 0.014621 R2. Exact changes, intervals,
counts, and feature modes are in
`analysis/stability/Figure5_dependency_completion_comparison.csv`.

Use the label "Calculated ln(k), k in h^-1". The internal variable remains
`target_phys_log_k_obs` to preserve the original data contract.

## Evidence and graphs

The final 173-field adjudication contains 55 values with retained historical
evidence, 55 with reconstructed sample-specific evidence, 24 wrong field/layer
assignments, 20 incorrect values/sample assignments, nine unsupported or
ambiguous values, five metadata completions, and five unavailable sources.
This is a targeted evidence review rather than a new broad human-validation
study. The typed ledger contains 232 operations; record identifiers and native
missing values are preserved.

The integrated graph contains 91,134 nodes and 173,911 relationships. Its class
counts reconcile the historical `label` key and the `type` key on 142 appended
Evidence nodes. Excluded modeling records may remain in the graph for
provenance. DOI metadata changes do not replace stable entity identifiers.

## Figures and supplementary data

Figure 3 reports PCE evaluation; Figure 4a is the five-paper graph; Figure 4b
is SHAP; Figure 5 is stability; Figure 6 is the integrated graph. Figures S4a
and S4b retain their normalization distributions and parameter-scenario inputs;
they are distinct from the corrected S10 model estimates.

The supplementary-data manifest maps Data S1 and S1a-S1d to benchmark values,
the response workbook, scoring-reference assignments, prompt template, and run
manifest. Data S2 contains the 278-predictor inventory. Model names such as
ChatGPT are benchmark identities and remain part of the scientific record.
