# Record adjudication and controlled corrections

The final evidence ledger covers 173 specifically reviewed fields. This review
resolves source identity, sample assignment, field/layer roles, units, and exact
supporting passages. It does not estimate extraction accuracy over the entire
corpus and is not an independent broad human-validation study.

| Final disposition | Fields |
|---|---:|
| Correct, retained sample-specific historical evidence | 55 |
| Correct, reconstructed sample-specific evidence | 55 |
| Wrong field or layer role | 24 |
| Incorrect value or sample assignment | 20 |
| Unsupported or ambiguous | 9 |
| Metadata completed from source | 5 |
| Exact source evidence unavailable | 5 |
| Total | 173 |

`original_and_adjudicated_evidence_classifications.csv` retains the stored
values, original classification, final classification, scientific definitions,
typed corrections, source hashes, precise excerpts/locations, and sample/test
context. Native missing values and immutable device/observation identifiers
remain part of the audit trail.

The typed applied-operation ledger has 232 rows: 142 evidence links, 52 value
assignments, 31 value clearances, five modeling exclusions, and two display-label
updates. These are operations, not 232 independent validation decisions.
The 90-row data patch preserves the implementation record of the corrections.

`eligibility_and_exclusion_ledger.csv` distinguishes experimental-model
eligibility from provenance retention, including the review-derived record.
`doi_verification_and_duplicate_decisions.csv` documents the ten verified DOI
decisions. Canonical DOI metadata supplements stable record/source identities;
it does not replace those identifiers.

The final integrated data contain 45,241 parsed CSV records. Modeling excludes
five records, leaving 45,236. `repaired_data_hashes.csv` reports parsed CSV row
counts rather than physical newline counts, because quoted fields can span
several lines. The original source datasets are identified by their SHA-256
hashes; restricted source-document locators are not published here.

The primary LiteratureAgent PCE test retains all 125 identities, including the
adjudicated target change from 10.9% to 13.32%. Its separate unchanged-target
sensitivity excludes that one record and retains 124 identities. The main
repaired/frozen out-of-fold comparison uses 21,425 common unchanged-target
records. Row keys, publication groups, fold assignments, and saved predictions
are included for tracing these populations.

Stability eligibility is target-specific. The source-context exclusion for
Ref_ID 28 applies to six derived endpoints. The completed dependency table
also traces corrected DOI groups and the repaired predicted-PCE feature into
each retained alternative. Predicted PCE is excluded from the main Figure 5
models. Modeling exclusions do not require deleting graph provenance nodes.

Source paths have been converted to relative included paths or archival
`source_records/` identifiers. Input checksums and release transformation
records are retained under `../provenance/`. An archival identifier does not
imply that the underlying source publication is redistributed.
