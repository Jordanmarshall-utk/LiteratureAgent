# LiteratureAgent

LiteratureAgent is an evidence-linked scientific literature mining platform. It
collects papers, extracts ontology-guided structured records with field-level
provenance, applies quality gates, builds directed heterogeneous knowledge
graphs, and retains plain-language summaries and evidence for later review.
The original halide-perovskite photovoltaic workflow also prepares data for PCE
and stability modeling.

## Main capabilities

- Web, Google Drive, and local-PDF literature workflows.
- Schema-guided extraction with retained evidence and DOI provenance.
- Completeness, identity, duplication, and source-alignment audits.
- PCE and stability database integration and predictive modeling.
- Evidence-linked property graphs plus bounded directional or acyclic views.
- Policy-aware scientific reasoning modes for task-appropriate analysis.
- Streamlit interface for extraction, later processing passes, graphs, evidence
  packets, run monitoring, results, and endpoint settings.
- Domain profiles for general science, oxide perovskites, and halide-perovskite
  photovoltaics. General science is the starting point for other fields.
- Structured extraction and knowledge-graph generation for every shipped
  profile; only halide-PV enables the specialized PCE/stability model.

## Windows quick start

1. Extract the complete Windows release ZIP to a normal local folder.
2. Double-click `Setup_LiteratureAgent.cmd` once.
3. Complete the bundled official Ollama installer if prompted; setup then
   downloads the selected local Qwen text and vision models.
4. Double-click `Launch_LiteratureAgent.cmd` to open the local application.

The release includes an embedded Python runtime and the original Ollama Windows
installer. Qwen model weights are downloaded by Ollama during setup and are not
stored in the ZIP. The default laptop profile uses the 3B text and vision models;
`Setup_Quality_Models.cmd` selects the larger 7B profile.

For a source checkout, the setup script can use an installed Python or Windows
Package Manager when the release-only runtime and installer are absent.

For a non-perovskite project, read
[`docs/CUSTOMIZING_FOR_A_NEW_DOMAIN.md`](docs/CUSTOMIZING_FOR_A_NEW_DOMAIN.md)
before the first campaign. Copy
`config/domain_packs/custom_domain_template.json`, select it under **General
science > Advanced options**, and pilot it on one to three known papers.
`canonical_terms` controls graph normalization, while
`structured_output.fields` adds evidence-gated `Domain_<field>` columns.

## Manual setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-platform.txt
.venv\Scripts\python.exe -m streamlit run literature_agent_platform\app.py
```

## Tests and manuscript-figure reproduction

Install the development and figure dependencies, then run the repository tests:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest tests -q
```

The four released Figure 5 panels can be regenerated from the included source
tables with `python scripts/plot_figure5_grouped_shap.py --top 20`. Recomputing
the complete 21,434-row SHAP audit is a separate, long-running workflow that
requires the governed integrated database and frozen 278-predictor manifest;
see `scripts/FIGURE5_REPRODUCTION.md`. The validated numerical environment used
scikit-learn 1.8.0 and SHAP 0.51.0.

## Repository layout

- `literature_agent_full_end_to_end_v21_3_english_sanitizer.py`: pipeline controller.
- `literature_agent_platform/`: standalone application and run manager.
- `literature_agent/reasoning/`: reasoning policies and philosophy manager.
- `config/`: extraction ontology, graph schema, and source priorities.
- `config/domain_packs/custom_domain_template.json`: editable vocabulary and
  structured-output template for another scientific field.
- `data/`: baseline perovskite dataset required by the default workflow.
- `models/`: PCE and stability modeling implementation.
- `scripts/`: reusable campaign, QA, analysis, graph, and plotting tools.
- `scripts/build_literatureagent_coworker_package.py`: audited Windows release
  builder; its local build prerequisites are documented in the standalone guide.
- `docs/CUSTOMIZING_FOR_A_NEW_DOMAIN.md`: ontology, structured-output,
  database-design, graph-mapping, and pilot-validation checklist.
- `scripts/FIGURE5_REPRODUCTION.md`: choice-stratified main, global audit, solvent-formulation, and fixed-seed local SHAP inputs and commands.
- `third_party/licenses/`: Ollama license and Qwen model notice.
- `tests/`: pipeline, model, integration, reasoning, and platform checks.

## Local state and credentials

Runs are written under `platform_workspace/`. Credentials belong under
`secrets/` or are selected in Settings. Both locations are ignored by Git.
Never commit API keys, OAuth files, extracted copyrighted PDFs, or local run
outputs.

## Data note

The bundled baseline CSV and the model under `models/` support only the original
halide-perovskite photovoltaic workflow. They are not the destination database
for a new domain. General science runs write `csv/domain_records.csv`, while
knowledge-graph builds write Neo4j import CSVs under
`knowledge_graph/neo4j_import/`. Confirm the baseline dataset's redistribution
and citation requirements before changing this repository from private to
public.
