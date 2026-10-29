# Stability source context: DOI 10.1039/c9ta01893j

The questioned record belongs to the original Perovskite Database and is
identified by `Ref_ID=28`, `Ref_ID_temp=28`, composition
`Cs0.05FA0.788MA0.162PbBr0.51I2.49`, and device PCE 18.45%. DOI alone does
not uniquely identify the record because six adjacent records share the paper.

The retained database states an unencapsulated ISOS-L-1 device test with
Xenon AM 1.5 illumination, MPPT, constant 25 °C, and ambient atmosphere.
Its raw fields encode 5 h exposure, 75% of initial PCE at the endpoint, and
T80 of 200 h. Initial stability PCE, relative humidity, and the raw trace link
are missing.

The exposure and retention pair implies that the 80% threshold was passed
by 5 h, which conflicts with a 200 h observed T80 for the same trace. The
separate extrapolated-lifetime field is empty. No retained main-paper trace
or sample-specific passage resolves whether 200 represents a unit error,
another device, or another test. The available supporting information
describes film storage and does not verify the device-PCE stability numbers.

The source check supports an eligibility restriction rather than choosing
a replacement measurement. The six incompatible derived targets are omitted
for this record in the primary and alternative analyses. The record remains
available to unaffected targets, and the original raw assertions remain in
`REF28_FIELD_TO_SOURCE.csv`.

The endpoint-derived first-order rate is calculated as
`k = -ln(retention_fraction) / exposure_hours`, followed by `ln(k)`.
It is displayed as “Calculated ln(k), k in h^-1”; this calculation does not
establish a source-verified degradation mechanism or reconcile the T80 value.

The current model comparison is Figure 5. Model inputs, eligible row counts,
publication-group counts, and intervals are recorded in
`primary_repaired_stability_metrics.csv` and `TableS10_updated.csv`.
