# LiteratureAgent standalone Windows platform

## Install and start

The supported Windows bundle includes a portable Python 3.12 runtime and the
unchanged official Ollama Windows installer.

1. Extract the complete ZIP to a local folder.
2. Run `Setup_LiteratureAgent.cmd`.
3. Complete the Ollama installer if it opens.
4. Wait while setup installs Python packages, downloads both Qwen models, runs a
   local model smoke test, and saves matching application settings.
5. Run `Launch_LiteratureAgent.cmd`.

The normal laptop profile installs `qwen2.5:3b` and `qwen2.5vl:3b` (about
5.1 GB of model downloads). `Setup_Quality_Models.cmd` switches to
`qwen2.5:7b` and `qwen2.5vl:7b-q4_K_M` (about 10.7 GB). Later launches remember
the selected profile. Qwen weights are downloaded by Ollama and are not
redistributed in the ZIP.

The interface runs locally at `http://127.0.0.1:8501`. Settings, logs, the
private Python environment, and run history are stored under
`%LOCALAPPDATA%\LiteratureAgent`, so the extracted application folder contains
no personal configuration.

The production package opens the working application directly. It does not
start a prerecorded or video-demonstration interface.

## Domain profiles

Every new campaign starts by selecting a scientific domain:

| Profile | Structured output | Knowledge graph | PCE/stability model |
|---|---|---|---|
| General science | Study design, system/subject, entities, variables, methods, conditions, measurements, findings, mechanisms, limitations, and evidence | Yes | No |
| Oxide perovskites | General fields plus composition, A/B sites, substitutions, oxygen stoichiometry, oxidation states, phases/symmetry, processing, defects, microstructure, characterization, and measured properties | Yes | No |
| Halide-perovskite photovoltaics | Original database-compatible device, fabrication, PCE, and stability schema | Yes | Yes |

General and oxide campaigns write the compatibility table `csv/all_records.csv`
and the easier-to-read `csv/domain_records.csv`. Values are retained only when
the model supplies a supporting paper-text excerpt. Paper summary JSON/text,
raw extraction evidence, and figure/vision artifacts are written alongside the
tables.

Knowledge graphs are available for every profile. Cross-domain graphs use
paper, studied-system, entity, method, condition, measurement, claim, and
evidence nodes. Oxide graphs add composition/site/structure/defect relations.
They do not create placeholder photovoltaic device nodes. The same graph build
also produces audit reports, plots, derived views, and Neo4j import CSV files.

Only `halide_pv` may enter the baseline database merge or PCE/stability model.
The controller rejects that stage for other profiles instead of silently
mapping unrelated science into photovoltaic fields.

The bundled `data/Perovskite_database_content_all_data.csv` and the script under
`models/` are halide-PV compatibility resources. Do not overwrite the baseline
CSV to create a database for another field. General and custom-domain database
work should begin with `csv/domain_records.csv` and the provenance artifacts in
the same run folder.

## Before using a new scientific domain

Read `docs/CUSTOMIZING_FOR_A_NEW_DOMAIN.md` before a full campaign. It separates
four decisions that must stay aligned:

1. the ontology's `canonical_terms` and synonyms;
2. the domain pack's `structured_output.fields` and evidence contract;
3. the eventual database's row grain, identifiers, types, units, and missing
   values; and
4. the knowledge-graph node and relationship mapping.

Changing ontology vocabulary alone does not add structured-output columns;
adding `structured_output.fields` does. Custom columns receive generic graph
nodes and evidence links automatically, while richer typed graph relations
remain a separate design choice. Run the required pilot on one to three
familiar papers and review both the records and graph before scaling up.

## Editing domain terminology

The shipped editable ontology files are:

- `config/domain_packs/general_science.json`
- `config/domain_packs/oxide_perovskite.json`
- `config/domain_packs/custom_domain_template.json`
- `config/perovskite_ontology_library_v19.json`

For a field outside oxide perovskites, copy and edit
`custom_domain_template.json`, then choose **General science** and select the
copy under **Advanced options > Ontology / domain-pack JSON**. Top-level
`canonical_terms` entries define preferred graph names and aliases.
`structured_output.fields` adds evidence-gated fields that are exported as
`Domain_<field_name>` columns without editing the controller. Extra `Domain_*`
fields receive a safe generic graph mapping; the customization guide explains
the JSON contract and when richer typed relationships are needed.

## Workflows

- **New literature run:** web search, a Google Drive collection, or a local PDF.
- **Analyze outputs:** post-hoc vision, summary sanitization, knowledge graph,
  and (halide-PV only) dataset integration/modeling.
- **Summary library:** search and download readable paper summaries.
- **Evidence packets:** answer a focused question from locally extracted
  evidence.
- **Knowledge graphs:** build one reasoning-policy graph or compare policies.
- **Results:** inspect and download CSV, JSON, text, image, spreadsheet, and PDF
  artifacts.

The user-supplied search query is preserved. General/oxide runs disable the old
photovoltaic candidate ranking and PCE schema-family calls. Vision prompts also
follow the selected domain. The legacy inline-vision path is intentionally
rejected for non-halide profiles; use the domain-aware **Analyze outputs >
Vision pass** workflow instead.

## Google Drive and privacy

No API keys, OAuth files, prior papers, prior runs, or model weights are in the
distribution. Optional Google OAuth files must be supplied by the recipient.
Local model requests go to Ollama on that computer. Web literature retrieval
still contacts the selected scholarly/web services. LiteratureAgent does not
ship a personal Unpaywall identity; set `UNPAYWALL_EMAIL` (or pass
`--unpaywall_email`) if that retrieval source should be used.

## Building the supported bundle

From the project root:

```powershell
python .\scripts\build_literatureagent_coworker_package.py
```

The builder requires the local portable runtime at
`polaris_full_stack/.tools/Python312` and
`third_party/installers/OllamaSetup.exe`. These large prerequisites are not
ordinary source-control payloads. The builder audits the staged package for
credentials, generated files, machine-specific paths, required customization
resources, and video-demo code; creates a per-file SHA-256 manifest; and writes
the ZIP plus a ZIP SHA-256 file to `deliverables/distribution`.
