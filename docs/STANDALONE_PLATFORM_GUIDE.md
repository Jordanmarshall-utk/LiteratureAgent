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

## Editing or adding domain terminology

The shipped editable ontology files are:

- `config/domain_packs/general_science.json`
- `config/domain_packs/oxide_perovskite.json`
- `config/perovskite_ontology_library_v19.json`

Changing an ontology extends canonical terms and synonyms used by graph and
retrieval logic. The selected domain profile still determines the structured
field contract. For a field outside oxide perovskites, start with General
science and edit `general_science.json`; the domain-neutral record and graph
schema remain available without code changes.

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
follow the selected domain.

## Google Drive and privacy

No API keys, OAuth files, prior papers, prior runs, or model weights are in the
distribution. Optional Google OAuth files must be supplied by the recipient.
Local model requests go to Ollama on that computer. Web literature retrieval
still contacts the selected scholarly/web services.

## Building the supported bundle

From the project root:

```powershell
python .\scripts\build_literatureagent_coworker_package.py
```

The builder requires `third_party/installers/OllamaSetup.exe`, audits the staged
package for credentials/generated files/machine-specific paths, creates a
per-file SHA-256 manifest, and writes the ZIP plus a ZIP SHA-256 file to
`deliverables/distribution`.
