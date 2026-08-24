# LiteratureAgent

LiteratureAgent is an evidence-linked scientific literature mining platform for
perovskite research. It collects papers, extracts structured device and sample
records with field-level provenance, applies quality gates, builds directed
heterogeneous knowledge graphs, and prepares data for downstream PCE and
stability modeling.

## Main capabilities

- Web, Google Drive, and local-PDF literature workflows.
- Schema-guided extraction with retained evidence and DOI provenance.
- Completeness, identity, duplication, and source-alignment audits.
- PCE and stability database integration and predictive modeling.
- Evidence-linked property graphs plus bounded directional or acyclic views.
- Policy-aware scientific reasoning modes for task-appropriate analysis.
- Streamlit interface for extraction, later processing passes, graphs, evidence
  packets, run monitoring, results, and endpoint settings.

## Windows quick start

1. Install Python 3.11 or 3.12.
2. Double-click `Setup_LiteratureAgent.cmd` once.
3. Double-click `Launch_LiteratureAgent.cmd` to open the local application.
4. Configure an OpenAI-compatible endpoint or local Ollama models in Settings.

The setup script creates a private `.venv` and installs `requirements-platform.txt`.

## Manual setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-platform.txt
.venv\Scripts\python.exe -m streamlit run literature_agent_platformpp.py
```

## Repository layout

- `literature_agent_full_end_to_end_v21_3_english_sanitizer.py`: pipeline controller.
- `literature_agent_platform/`: standalone application and run manager.
- `literature_agent/reasoning/`: reasoning policies and philosophy manager.
- `config/`: extraction ontology, graph schema, and source priorities.
- `data/`: baseline perovskite dataset required by the default workflow.
- `models/`: PCE and stability modeling implementation.
- `scripts/`: reusable campaign, QA, analysis, graph, and plotting tools.
- `tests/`: pipeline, model, integration, reasoning, and platform checks.

## Local state and credentials

Runs are written under `platform_workspace/`. Credentials belong under
`secrets/` or are selected in Settings. Both locations are ignored by Git.
Never commit API keys, OAuth files, extracted copyrighted PDFs, or local run
outputs.

## Data note

The bundled baseline CSV supports the default perovskite workflow. Confirm the
underlying dataset's redistribution and citation requirements before changing
this repository from private to public.
