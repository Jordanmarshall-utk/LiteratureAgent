# LiteratureAgent

LiteratureAgent collects scientific papers, extracts structured records with
source evidence, checks record quality, and builds directed knowledge graphs.
The original perovskite photovoltaic workflow also supports power conversion
efficiency (PCE) and stability modeling. Domain profiles support other fields
without applying photovoltaic models to unrelated records.

## Manuscript materials

The [manuscript directory](manuscript/README.md) contains the current figure
panels, editable assets, exact plotting tables, supplementary data, repair
evidence, and publication-grouped evaluation results. See the
[reproduction instructions](manuscript/docs/REPRODUCIBILITY.md) and
[analysis description](manuscript/docs/ANALYSIS.md).

Current numbering is Figure 3 for PCE evaluation, Figure 4a for the five-paper
graph, Figure 4b for SHAP, Figure 5 for stability, and Figure 6 for the
integrated graph. The repaired SHAP cohort contains 21,426 records from 4,080
publication groups. SHAP explains the final fitted model; it does not measure
held-out performance or establish causal effects.

The final 29 October checks correct the secondary row-random sample exclusion,
canonical stability intervals, predictor inclusion/exclusion documentation and
graph evidence-count definitions. Figure 5a and Figure 5b are supplied as
separate editable panels for manual manuscript assembly. Licensing and approved
manuscript citation metadata remain explicitly pending in this public release.

## Run the application

For the Windows distribution, run `Setup_LiteratureAgent.cmd` once, then
`Launch_LiteratureAgent.cmd`. Setup installs dependencies and downloads the
selected Qwen models through Ollama. Model weights are not stored here.

For a source checkout:

```powershell
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements-platform.txt
./.venv/Scripts/python.exe -m streamlit run literature_agent_platform/app.py
```

Use `python literature_agent_pipeline.py --help` for the command-line pipeline.
For another scientific field, start with the General science profile and follow
[the domain customization guide](docs/CUSTOMIZING_FOR_A_NEW_DOMAIN.md).
Credentials are supplied locally; run data are written to `platform_workspace/`.

## Code and tests

- `literature_agent_pipeline.py`: retrieval and extraction controller.
- `literature_agent/`: domain contracts and scientific reasoning policies.
- `literature_agent_platform/`: local application and run management.
- `models/pce_stability.py`: PCE and stability models.
- `config/`: ontologies, domain profiles, and graph schema.
- `scripts/`: analysis tools and `build_windows_release.py`.
- `manuscript/`: current manuscript figures, data, and reproducibility tools.
- `third_party/licenses/`: retained dependency notices.

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python manuscript/scripts/validate_release.py
```

The earlier 21,434-record SHAP analysis remains under `data/figure5/` for
historical reproducibility. It is not the current manuscript figure source;
[its documentation](scripts/FIGURE5_REPRODUCTION.md) identifies that scope.

## Citation and use

See [citation information](docs/CITATION.md) and
[software and data rights](docs/RIGHTS.md). No software license has been
designated by the authors. This repository does not grant a new license to
third-party software, model weights, source publications, or underlying data.
