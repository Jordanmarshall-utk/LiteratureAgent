# Reproduction

Use Python 3.11 or later. Core validation and statistical comparisons do not
require OriginPro. Install the manuscript dependencies from the repository root:

```powershell
python -m pip install -r manuscript/requirements.txt
python manuscript/scripts/validate_release.py
python manuscript/scripts/reproduce_comparisons.py --check
```

Rebuild the two current stability panels as individual SVG, PDF, and 600 dpi
PNG files:

```powershell
python manuscript/scripts/plot_stability.py --help
python manuscript/scripts/plot_stability.py
```

The figures contain six endpoints and two series each. Their plotting tables
include estimates, asymmetric uncertainty bounds, feature modes, target keys,
eligible-record counts, canonical publication counts and exact prediction/fold
source hashes. SVG text and graphical elements remain editable. These are
individual 85 by 72 mm panels, not an assembled manuscript figure. All 24
intervals reflect the final canonical-publication bootstrap correction; point
locations and the existing style are retained. The SVG/PDF metadata records
the exact plotted values and input-table SHA-256.

## Origin panels

Origin builds require Windows, a licensed OriginPro installation, the
`originpro` Python bridge, Times New Roman, and Edge or Chrome for export.
Install the additional bridge on that machine:

```powershell
python -m pip install -r manuscript/requirements-origin.txt
python manuscript/scripts/build_manuscript_plot.py all --dry-run
python manuscript/scripts/build_fig2_campaign_origin.py all --dry-run
python manuscript/scripts/build_figs4_origin_no_grid.py all --dry-run
python manuscript/scripts/build_figs6_benchmark_origin.py all --dry-run
```

Run `--help` on the selected builder before requesting an Origin export. The
builders read `manuscript/source/` and write to `manuscript/build/` by default.
The publication panels under `manuscript/figures/` are retained release assets;
review a fresh rendering before replacing them. Desktop release validation
checks sources and supplied assets; it does not claim to have reopened Origin
projects on a machine without OriginPro.

The shared S4 legend specification is in `manuscript/legend_inputs/`. Individual
legends remain in `manuscript/figures/legends/`. Panels are exported separately;
the scripts do not assemble a manuscript layout.

## Scientific scope

`models/pce_stability.py` and the byte-preserved snapshot under `models/frozen/`
provide the model implementation. The 278-feature inventory, 113 numeric and
165 categorical predictor typing, fold assignments, target definitions, and
saved predictions are retained. Evaluation uses publication groups; the
row-random result is a diagnostic with publication overlap, not the primary
generalization estimate.

Exact paired comparisons can be checked from saved predictions without
rerunning extraction or fitting estimators. The 124-record comparison script
uses 2,000 normalized-DOI cluster bootstrap replicates and seed 42. It excludes
only the adjudicated target correction and leaves all 125 primary identities
in the original evaluation files.

Acquisition cannot be reproduced from these files alone: access to source
publications and external services is separate. The record-level graph and
private source-document locators are not public inputs. Graph aggregate tables
and five-paper relationship records are sufficient for the supplied graph
figures. Source identifiers under `source_records/` are archival references,
not a claim that the underlying private document is included.
