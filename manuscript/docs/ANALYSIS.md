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
4,159-row test membership, order and targets. Canonical publication identity
and unchanged sample-key components exclude the same sample from every
training condition. Training counts are 16,634 Original, 16,741 curated and
17,267 expanded after removing two DOI-prefix aliases. Original and curated
saved predictions are reused; only the affected expanded diagnostic was
refitted with 300 Extra Trees, minimum leaf size 2, max_features=sqrt and seed 42.
Neither secondary result replaces the primary test.

The row-random test has 2,248 raw labels but 2,243 bootstrap clusters: 2,242
identified publications plus one retained NO_GROUP cluster containing 48
records without DOI identity. That cluster is not a verified publication.
In each training condition, 2,161 identified test publications are also
represented in training, plus NO_GROUP; publication overlap is intentional
for this diagnostic, but same-sample overlap is zero. Marginal and paired
intervals share 2,000 multinomial cluster draws, seed 42 and first-seen group
order. Direct metric differences are distinct from median bootstrap deltas.
Historical paired differences use the separate validated 741-group contract;
the nine correct primary Figure 3/Table S6A marginal intervals are unchanged.

The repaired-minus-frozen main out-of-fold comparison covers 21,425 common
unchanged-target records. Paired intervals are calculated on the joint
publication resamples, not from marginal confidence bounds.

The repaired SHAP and PCE fitting population is 21,426 records from 4,080
publication groups, distinct from each evaluation test.
This final eligible cohort contains 51 unknown publication years, not the 85
count from an earlier input-filter stage. The 2018 cutoff is retained.

## Predictor membership and lineage

The frozen contract includes 248 raw fields and 30 engineered descriptors:
113 numeric and 165 categorical inputs, with all 16 Boolean inputs categorical.
Data S2 covers the union of original/repaired raw headers plus engineered
inputs: 461 fields, 278 included and 183 directly excluded. Fourteen frozen
type decisions are retained where the historical dynamic selector differs.

All 44 Stability_ and 40 Outdoor_ fields are excluded as direct predictors by
the recorded code rule. A retained author note gives a general downstream-
measurement intention, not a field-specific scientific explanation for every
protocol, condition or location field; that author rationale remains pending.
The broad descriptor-name selector reads 21 raw fields, including six directly
excluded fields: Ref_publication_date, JV_certification_institute and the four
Outdoor_location_ country/city/coordinates/climate_zone fields. These fields
contribute to concatenated descriptor text, so direct exclusion does not prove
absence of indirect contributions. This is static lineage evidence, not a
measured predictive effect or a new same-sample leakage claim. The frozen
selector and model features are unchanged.

The retained completeness thresholds are 0.5% column coverage and 15% row
completeness. Input-availability eligibility is assessed before the publication
split; fitted imputation, scaling and encoding remain training-only. The
targeted sample-identity correction does not establish that all preprocessing
choices are holdout-independent or that all inputs are pre-experiment.

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
separately. The recorded DOI repairs in the 28 October completion preceded its
affected fits. Final case/format normalization is applied to group metadata
and bootstrap clusters after verifying the retained historical partitions;
it does not claim that those historical models were fitted with new folds.

Predicted-PCE alternatives use stable joins to repaired out-of-fold values for
the PCE training population and the documented full-fit fallback outside that
population. Predicted PCE is excluded from every main Figure 5 feature set.
The source tables retain the distinction between measured initial performance,
physical-condition features, and condition-normalized targets.

Figure 5 contains 24 selected Original/Integrated estimates. The earlier
28 October dependency completion changed eight Integrated point estimates;
that completed point-estimate snapshot is retained. The final grouping check
reproduces every saved per-fold result from the exact source order and frozen
implementation for all 66 estimates. No canonical publication crosses verified
historical test folds. The two case-alias records for 10.1039/c8se00314a occupy
the same exported fold ID 2 in all seven affected Original populations, each
with 4,986 rows and 1,821 canonical identities rather than 1,822 raw labels.
Consequently no stability refit is needed.

All intervals use 2,000 draws over sorted canonical publication identities
with the retained deterministic seed rule. A leading U+FEFF format marker is
removed only from an already-recorded DOI; this is not a new DOI assignment
or an additional publication merge. Canonical sorting changes the deterministic
finite Monte Carlo draws even in populations whose partition is unchanged.
In total, 64 of 66 intervals change, including all 24 primary Figure 5 intervals,
while all point estimates remain unchanged. No interval changes its relation
to ROC-AUC 0.5 or R2 zero. Table S10 retains 33 outcome/input-set comparisons.
Original and Integrated eligible populations generally differ; their point
differences are descriptive, not paired same-record effects.

Final canonical prediction, fold and metric exports are authoritative for the
current uncertainty estimates. The older `fits/` files remain identified as
the original raw-label prediction sources; they are not the canonical plotting
source. Final tables carry source paths and hashes for both stages.

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

DatasetSnapshot.field_evidence_count=37,663 is the historical builder-input
subtotal deduplicated by sample identity, normalized DOI, field and cleaned
supporting text. The current graph has 37,805 field-evidence links: 31,116
observation-to-evidence SUPPORTED_BY plus 6,689 HAS_FIELD_EVIDENCE, equal to the
historical subtotal plus 142 retained repair links. The 2,694 scientific-claim
links included in 33,810 total SUPPORTED_BY relationships are excluded from
that field subtotal. A companion metadata clarification preserves historical
scope and records current subtotals without changing node IDs, relationships
or printed totals. Link counts are not independent validation events or
estimates of extraction accuracy.

## Figures and supplementary data

Figure 3 reports PCE evaluation; Figure 4a is the five-paper graph; Figure 4b
is SHAP; Figure 5 is stability; Figure 6 is the integrated graph. Figures S4a
and S4b retain their normalization distributions and parameter-scenario inputs;
they are distinct from the corrected S10 model estimates.

The supplementary-data manifest maps Data S1 and S1a-S1d to benchmark values,
the response workbook, scoring-reference assignments, prompt template, and run
manifest. Data S2 contains the included/excluded inventory with 278 retained
predictors among 461 fields. Data S1 remains the archived comparator benchmark;
this release does not update comparator-model versions or rerun that benchmark.
Model names such as
ChatGPT are benchmark identities and remain part of the scientific record.
