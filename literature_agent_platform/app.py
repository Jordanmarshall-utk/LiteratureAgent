from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from literature_agent_platform.health import platform_health
from literature_agent_platform.run_manager import (
    DEFAULT_BASE_CSV,
    DEFAULT_MODEL,
    DEFAULT_ONTOLOGY,
    PLATFORM_ROOT,
    PROJECT_ROOT,
    RUNS_DIR,
    RunStore,
    campaign_command,
    common_controller_args,
    controller_command,
    csv_preview,
    discover_outputs,
    load_settings,
    save_settings,
    tail_text,
)


st.set_page_config(page_title="LiteratureAgent", page_icon="LA", layout="wide")
st.markdown(
    """
    <style>
    :root { --accent:#176B5B; --ink:#1D2528; --line:#D7DEDC; }
    .block-container {padding-top:1.4rem; padding-bottom:2rem; max-width:1500px;}
    h1 {font-size:1.85rem !important; letter-spacing:0 !important;}
    h2 {font-size:1.25rem !important; letter-spacing:0 !important;}
    h3 {font-size:1.02rem !important; letter-spacing:0 !important;}
    [data-testid="stMetric"] {border-top:3px solid var(--accent); padding:0.65rem 0.2rem;}
    [data-testid="stForm"] {border:1px solid var(--line); border-radius:6px; padding:1rem;}
    .status-ok {color:#176B5B; font-weight:650;}
    .status-bad {color:#B23A2B; font-weight:650;}
    .muted {color:#5F6B6D; font-size:0.9rem;}
    code {white-space:pre-wrap !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return cleaned[:60] or "run"


def new_run_dir(name: str) -> Path:
    return RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slug(name)}"


def path_setting(label: str, default: Path, key: str) -> str:
    return st.text_input(label, value=str(default), key=key)


def subprocess_display(command: list[str]) -> str:
    def quote(value: str) -> str:
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"' if any(ch.isspace() for ch in value) else value

    return " ".join(quote(value) for value in command)


def launch(name: str, workflow: str, command: list[str], run_dir: Path) -> None:
    try:
        run = store.launch(name, workflow, command, run_dir)
    except Exception as exc:
        st.error(f"Could not start the run: {exc}")
        return
    st.session_state["selected_run"] = run.id
    st.success(f"Started {run.name}. You can leave this page; the run continues in the background.")


def output_roots_for_run(run) -> list[Path]:
    roots = [Path(run.run_dir)]
    path_flags = {
        "--work_dir", "--work-dir", "--integration_out_dir", "--integration-out-dir",
        "--model_out_dir", "--model-out-dir", "--knowledge_graph_out_dir",
        "--evidence_out_dir", "--campaign-root",
    }
    for index, value in enumerate(run.command[:-1]):
        if value in path_flags:
            candidate = Path(run.command[index + 1])
            if candidate.exists() and candidate not in roots:
                roots.append(candidate)
    return roots


def extraction_dir_for_run(run) -> Path:
    root = Path(run.run_dir)
    candidates = [root / "outputs", root]
    for candidate in candidates:
        if (candidate / "csv").exists() or (candidate / "paper_summaries_json").exists():
            return candidate
    return root / "outputs"


def configured_common_args(*, work_dir: str, integration_dir: str, model_dir: str, reasoning_mode: str) -> list[str]:
    return common_controller_args(
        base_csv=base_csv,
        ontology=ontology,
        work_dir=work_dir,
        integration_dir=integration_dir,
        model_dir=model_dir,
        reasoning_mode=reasoning_mode,
        llm_api_url=runtime_settings["llm_api_url"],
        llm_model=runtime_settings["llm_model"],
        vision_api_url=runtime_settings["vision_api_url"],
        vision_model=runtime_settings["vision_model"],
    )


def render_health(compact: bool = False) -> None:
    health = platform_health()
    checks = health["checks"]
    if compact:
        st.sidebar.markdown(
            '<span class="status-ok">System ready</span>' if health["ready"] else '<span class="status-bad">Setup needs attention</span>',
            unsafe_allow_html=True,
        )
        return
    rows = [
        {"Component": name, "Status": "Ready" if item["ok"] else "Needs attention", "Detail": item["detail"]}
        for name, item in checks.items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


store = RunStore()
PLATFORM_ROOT.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
bundled_root = PROJECT_ROOT / "bundled_collections"
if bundled_root.exists():
    for bundled in bundled_root.iterdir():
        if bundled.is_dir() and (bundled / "csv" / "all_records.csv").exists():
            store.register_existing(bundled.name.replace("_", " "), bundled)

st.sidebar.title("LiteratureAgent")
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Home"
page = st.sidebar.radio(
    "Workspace",
    ["Home", "New literature run", "Analyze outputs", "Knowledge graphs", "Evidence packets", "Runs", "Results", "Settings"],
    key="nav_page",
)
render_health(compact=True)
st.sidebar.caption("Standalone scientific literature mining")

runtime_settings = load_settings()
base_csv = st.session_state.get("base_csv", runtime_settings["base_csv"])
ontology = st.session_state.get("ontology", runtime_settings["ontology"])

runs = store.list_runs()

if page == "Home":
    st.title("LiteratureAgent")
    st.caption("Choose the scientific outcome you need.")
    active = [run for run in runs if run.status == "running"]
    completed = [run for run in runs if run.status == "finished"]
    failed = [run for run in runs if run.status in {"failed", "stopped"}]
    cols = st.columns(4)
    cols[0].metric("Active runs", len(active))
    cols[1].metric("Completed runs", len(completed))
    cols[2].metric("Stopped or failed", len(failed))
    cols[3].metric("Total runs", len(runs))

    st.subheader("Start")
    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown("**Mine a research topic**")
        st.caption("Search the web or process a curated paper collection.")
        if st.button("New literature run", key="home_new", width="stretch"):
            st.session_state["nav_page"] = "New literature run"
            st.rerun()
    with q2:
        st.markdown("**Ground a hypothesis**")
        st.caption("Retrieve focused evidence from an existing collection.")
        if st.button("Build evidence packet", key="home_evidence", width="stretch"):
            st.session_state["nav_page"] = "Evidence packets"
            st.rerun()
    with q3:
        st.markdown("**Continue a completed run**")
        st.caption("Run vision, integration, modeling, or a knowledge graph.")
        if st.button("Analyze outputs", key="home_analyze", width="stretch"):
            st.session_state["nav_page"] = "Analyze outputs"
            st.rerun()
    if st.button("Open knowledge graphs", key="home_graphs"):
        st.session_state["nav_page"] = "Knowledge graphs"
        st.rerun()

    with st.expander("System checks"):
        render_health()
    st.subheader("Recent activity")
    if not runs:
        st.info("No platform runs yet. Start a new literature run.")
    else:
        st.dataframe(
            pd.DataFrame([
                {"Name": run.name, "Workflow": run.workflow, "Status": run.status, "Started": run.started_at, "Folder": run.run_dir}
                for run in runs[:10]
            ]),
            hide_index=True,
            width="stretch",
        )

elif page == "New literature run":
    st.title("New literature run")
    st.caption("Set the research goal, source, and topic. LiteratureAgent selects the matching pipeline preset.")
    with st.form("extract_form"):
        purpose = st.selectbox(
            "Research goal",
            ["General literature collection", "Evidence for a hypothesis", "PCE and stability dataset"],
            help="General and hypothesis modes prioritize summaries and evidence. PCE/stability mode also produces database-ready device records.",
        )
        source = st.segmented_control("Paper source", ["Web search", "Google Drive", "Local PDF"], default="Web search")
        default_name = {
            "General literature collection": "general_literature",
            "Evidence for a hypothesis": "hypothesis_evidence",
            "PCE and stability dataset": "pce_stability_expansion",
        }[purpose]
        run_name = st.text_input("Run name", value=default_name)
        query = ""
        folder_id = ""
        pdf_path = ""
        if source == "Google Drive":
            folder_id = st.text_input("Google Drive folder ID or link", help="Leave blank to use the configured default folder.")
            process_all = st.toggle("Process every file in this folder", value=True)
            max_papers = st.number_input("Maximum papers", min_value=1, value=25, disabled=process_all)
        elif source == "Local PDF":
            pdf_path = st.text_input("PDF path on this computer")
            uploaded = st.file_uploader("Or upload one PDF", type=["pdf"])
            process_all = False
            max_papers = 1
        else:
            query = st.text_area(
                "Search query",
                placeholder="Describe the exact topic, relationship, material, method, or phenomenon to investigate.",
                height=110,
                help="This query is never replaced by a fixed PCE/stability query.",
            )
            process_all = False
            max_papers = st.number_input("Papers to process", min_value=1, max_value=5000, value=25)

        purpose_profile = {
            "General literature collection": "general_literature",
            "Evidence for a hypothesis": "hypothesis_support",
            "PCE and stability dataset": "pce_stability_modeling",
        }
        with st.expander("Advanced options"):
            c1, c2, c3 = st.columns(3)
            profiles = list(dict.fromkeys([purpose_profile[purpose], "general_literature", "hypothesis_support", "pce_stability_modeling"]))
            profile = c1.selectbox("Extraction profile", profiles)
            gating_default = 0 if purpose == "PCE and stability dataset" else 1
            gating = c2.selectbox("Paper-family gating", ["strict", "moderate", "off"], index=gating_default)
            reasoning = c3.selectbox("Reasoning policy", ["auto", "multi", "single", "off"])
            figure_reports = st.toggle("Create text-based figure reports", value=True)
        run_full_pipeline = st.toggle(
            "Integrate records and run PCE/stability modeling after extraction",
            value=purpose == "PCE and stability dataset",
            disabled=purpose != "PCE and stability dataset",
        )
        submitted = st.form_submit_button("Start extraction", type="primary")

    if submitted:
        run_dir = new_run_dir(run_name)
        work = run_dir / "outputs"
        is_modeling = purpose == "PCE and stability dataset"
        if source == "Web search" and is_modeling and int(max_papers) > 10:
            if not query.strip():
                st.error("Enter a search query before starting.")
                st.stop()
            candidate_filter = "stability_target_ready" if "stability" in query.lower() and "pce" not in query.lower() else "model_ready_strict"
            args = [
                "--stage", "pilot" if int(max_papers) <= 100 else "scale",
                "--batch-size", str(max_papers), "--target-total", str(max_papers),
                "--target-processed-papers", str(max_papers), "--max-candidate-attempts", str(max(500, int(max_papers) * 12)),
                "--target-mode", "pce_stability", "--search-query", query.strip(),
                "--campaign-root", str(run_dir / "campaign"), "--work-dir", str(work),
                "--base-csv", base_csv, "--ontology-path", ontology,
                "--family-gating", gating, "--expansion-candidate-filter", candidate_filter,
                "--expansion-require-oa", "0", "--reasoning-policy-mode", reasoning,
                "--figure-report-enable", "1" if figure_reports else "0",
                "--use-reasoning-layer" if reasoning != "off" else "--no-use-reasoning-layer",
            ]
            if run_full_pipeline:
                args += ["--run-model-check"]
            launch(run_name, "expansion_campaign", campaign_command(*args), run_dir)
            st.stop()

        args = configured_common_args(
            work_dir=str(work), integration_dir=str(run_dir / "integration"),
            model_dir=str(run_dir / "models"), reasoning_mode=reasoning,
        )
        args += [
            "--pipeline_stage", "full" if run_full_pipeline else "extract_batch",
            "--run_mode", "initial" if source != "Web search" else "expand",
            "--extraction_profile", profile, "--family_gating", gating,
            "--figure_report_enable", "1" if figure_reports else "0", "--vision_enable", "0",
            "--no_require_doi", "--llm_max_retries", "1",
        ]
        if run_full_pipeline:
            args += ["--run_model"]
        if source == "Google Drive":
            args += ["--full_literature_run", "--drive_process_all_files", "1" if process_all else "0", "--max_papers", str(max_papers)]
            if Path(runtime_settings["oauth_client"]).exists():
                args += ["--google_drive_oauth_client_secrets", runtime_settings["oauth_client"]]
            if Path(runtime_settings["oauth_token"]).exists():
                args += ["--google_drive_oauth_token_file", runtime_settings["oauth_token"]]
            if folder_id.strip():
                args += ["--google_drive_folder_id", folder_id.strip()]
        elif source == "Local PDF":
            if uploaded is not None:
                upload_dir = run_dir / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                saved = upload_dir / uploaded.name
                saved.write_bytes(uploaded.getbuffer())
                pdf_path = str(saved)
            if not pdf_path.strip():
                st.error("Choose or enter a PDF before starting.")
                st.stop()
            args += ["--test_one_paper", "--single_paper_source", "pdf", "--single_paper_pdf_path", pdf_path.strip()]
        else:
            if not query.strip():
                st.error("Enter a research query before starting.")
                st.stop()
            args += [
                "--full_literature_run", "--disable_google_drive", "1",
                "--search_query", query.strip(), "--max_papers", str(max_papers),
                "--expansion_candidate_filter", "target_ready" if is_modeling else "off",
            ]
        launch(run_name, "extract_batch", controller_command(*args), run_dir)

elif page == "Analyze outputs":
    st.title("Process existing outputs")
    st.caption("Run later passes on an extraction workspace without repeating paper extraction.")
    with st.expander("Attach an existing LiteratureAgent collection"):
        with st.form("attach_existing_form"):
            existing_name = st.text_input("Collection name", value="existing_literature_collection")
            existing_path = st.text_input(
                "Existing output folder",
                placeholder="Choose a LiteratureAgent output folder",
            )
            attach = st.form_submit_button("Attach collection")
        if attach:
            try:
                attached = store.register_existing(existing_name, Path(existing_path))
                st.session_state["selected_run"] = attached.id
                st.success("Collection attached. Existing papers will not be re-extracted.")
                st.rerun()
            except (OSError, ValueError) as exc:
                st.error(f"Could not attach that folder: {exc}")
    output_runs = [run for run in runs if Path(run.run_dir).exists()]
    if not output_runs:
        st.info("Complete or start an extraction run first.")
    else:
        labels = {f"{run.name} | {run.created_at}": run for run in output_runs}
        with st.form("process_form"):
            selected_label = st.selectbox("Source run", labels)
            stage = st.segmented_control("Pass", ["Vision", "Sanitize summaries", "Integrate & model", "Knowledge graph"], default="Vision")
            run_model = st.toggle("Train PCE and stability models", value=True, disabled=stage != "Integrate & model")
            reasoning = st.selectbox("Reasoning policy", ["auto", "multi", "single", "off"])
            submitted = st.form_submit_button("Start pass", type="primary")
        if submitted:
            source_run = labels[selected_label]
            source_root = Path(source_run.run_dir)
            work = extraction_dir_for_run(source_run)
            followup_dir = new_run_dir(f"{source_run.name}_{slug(stage)}")
            stage_map = {"Vision": "vision_pass", "Sanitize summaries": "sanitize_summaries", "Integrate & model": "integrate_and_model", "Knowledge graph": "knowledge_graph"}
            args = configured_common_args(
                work_dir=str(work), integration_dir=str(source_root / "integration"),
                model_dir=str(source_root / "models"), reasoning_mode=reasoning,
            )
            args += ["--pipeline_stage", stage_map[stage], "--run_mode", "initial", "--skip_literature_agent"]
            if stage == "Vision":
                args += ["--vision_enable", "1", "--vision_only_max_papers", "0"]
            elif stage == "Integrate & model":
                args += ["--no_require_doi"]
                if run_model:
                    args += ["--run_model"]
            elif stage == "Knowledge graph":
                args += ["--knowledge_graph_out_dir", str(source_root / "knowledge_graph")]
            launch(f"{source_run.name}: {stage}", stage_map[stage], controller_command(*args), followup_dir)

elif page == "Knowledge graphs":
    st.title("Knowledge graphs")
    st.caption("Build an evidence-grounded graph from an extracted literature collection.")
    graph_runs = [run for run in runs if (extraction_dir_for_run(run) / "csv" / "all_records.csv").exists()]
    if not graph_runs:
        st.info("Attach or create a literature collection first.")
    else:
        labels = {f"{run.name} | {run.created_at}": run for run in graph_runs}
        with st.form("knowledge_graph_form"):
            selected_label = st.selectbox("Literature collection", labels)
            graph_mode = st.segmented_control("Graph build", ["Single policy", "Compare policies"], default="Single policy")
            policies = [
                "current", "humean_evidence", "kantian_constraints", "socratic_uncertainty",
                "cartesian_decomposition", "aristotelian_classification",
                "hegelian_conflict_resolution", "platonic_abstraction",
            ]
            if graph_mode == "Single policy":
                selected_policy = st.selectbox("Reasoning policy", policies)
                selected_policies = []
            else:
                selected_policy = "current"
                selected_policies = st.multiselect("Reasoning policies", policies, default=policies)
            max_records = st.number_input("Maximum records", min_value=0, value=0, help="Use 0 for the full collection.")
            submitted = st.form_submit_button("Build knowledge graph", type="primary")
        if submitted:
            source_run = labels[selected_label]
            work = extraction_dir_for_run(source_run)
            records = work / "csv" / "all_records.csv"
            run_dir = new_run_dir("knowledge_graph")
            if graph_mode == "Single policy":
                command = [
                    os.sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "build_literature_knowledge_graph.py"),
                    "--records", str(records), "--ontology", ontology, "--work-dir", str(work),
                    "--out", str(run_dir / "knowledge_graph"), "--max-records", str(max_records),
                    "--reasoning-policy", selected_policy,
                ]
                workflow = "knowledge_graph"
            else:
                if not selected_policies:
                    st.error("Select at least one reasoning policy.")
                    st.stop()
                command = [
                    os.sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "build_reasoning_policy_knowledge_graph_variants.py"),
                    "--records", str(records), "--ontology", ontology, "--work-dir", str(work),
                    "--out-root", str(run_dir / "knowledge_graph_comparison"), "--max-records", str(max_records),
                    "--policies", *selected_policies,
                ]
                workflow = "knowledge_graph_comparison"
            launch(f"{source_run.name}: {graph_mode}", workflow, command, run_dir)

    prior_graphs = PROJECT_ROOT / "artifacts" / "kg_all_reasoning_policy_comparison_expansion_5000"
    bundled_graphs = PROJECT_ROOT / "bundled_results" / "kg_all_reasoning_policy_comparison_expansion_5000"
    if not prior_graphs.exists() and bundled_graphs.exists():
        prior_graphs = bundled_graphs
    if prior_graphs.exists():
        with st.expander("Existing reasoning-policy comparison"):
            summary_csv = prior_graphs / "reasoning_policy_kg_comparison.csv"
            if summary_csv.exists():
                st.dataframe(pd.read_csv(summary_csv), hide_index=True, width="stretch")
            st.caption(str(prior_graphs))

elif page == "Evidence packets":
    st.title("Evidence search")
    st.caption("Build a compact, source-linked evidence packet from previously extracted literature.")
    output_runs = [run for run in runs if extraction_dir_for_run(run).exists()]
    if not output_runs:
        st.info("Extract literature first so there is a local evidence collection to search.")
    else:
        labels = {f"{run.name} | {run.created_at}": run for run in output_runs}
        with st.form("evidence_form"):
            source_label = st.selectbox("Literature collection", labels)
            question = st.text_area("Scientific question or hypothesis", height=120)
            limit = st.slider("Maximum supporting papers", 1, 20, 5)
            submitted = st.form_submit_button("Build evidence packet", type="primary")
        if submitted:
            if not question.strip():
                st.error("Enter a scientific question or hypothesis.")
                st.stop()
            source_run = labels[source_label]
            source_root = Path(source_run.run_dir)
            run_dir = new_run_dir("evidence_packet")
            args = configured_common_args(
                work_dir=str(extraction_dir_for_run(source_run)), integration_dir=str(source_root / "integration"),
                model_dir=str(source_root / "models"), reasoning_mode="auto",
            )
            args += [
                "--pipeline_stage", "evidence_packet", "--skip_literature_agent",
                "--evidence_query", question.strip(), "--evidence_limit", str(limit),
                "--evidence_out_dir", str(run_dir / "evidence_packets"),
            ]
            launch("Evidence packet", "evidence_packet", controller_command(*args), run_dir)

elif page == "Runs":
    st.title("Runs")
    top = st.columns([1, 4])
    if top[0].button("Refresh", width="stretch"):
        st.rerun()
    top[1].caption("Runs continue in the background when this browser page is closed.")
    if not runs:
        st.info("No runs have been launched from this platform.")
    else:
        labels = {f"{run.name} | {run.status} | {run.created_at}": run for run in runs}
        default_index = 0
        selected_id = st.session_state.get("selected_run")
        if selected_id:
            default_index = next((i for i, run in enumerate(labels.values()) if run.id == selected_id), 0)
        label = st.selectbox("Run", list(labels), index=default_index)
        run = labels[label]
        st.session_state["selected_run"] = run.id
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status", run.status)
        c2.metric("Workflow", run.workflow)
        c3.metric("Process ID", run.pid or "-")
        c4.metric("Started", run.started_at or "-")
        if run.status == "running" and st.button("Stop run"):
            store.stop(run.id)
            st.rerun()
        elif run.command and run.status in {"stopped", "failed", "finished"} and st.button("Resume with saved outputs"):
            launch(f"{run.name} (resumed)", run.workflow, run.command, Path(run.run_dir))
            st.rerun()
        st.caption(run.run_dir)
        tab1, tab2, tab3 = st.tabs(["Live log", "Errors", "Command"])
        with tab1:
            st.code(tail_text(run.stdout_path, 250) or "Waiting for output...", language="text")
        with tab2:
            st.code(tail_text(run.stderr_path, 150) or "No error output.", language="text")
        with tab3:
            st.code(subprocess_display(run.command), language="powershell")

elif page == "Results":
    st.title("Results")
    st.caption("Browse tables, reports, plots, summaries, and model artifacts produced by a run.")
    if not runs:
        st.info("No platform runs are available.")
    else:
        labels = {f"{run.name} | {run.created_at}": run for run in runs}
        label = st.selectbox("Run", labels)
        run = labels[label]
        files_by_path = {}
        for root in output_roots_for_run(run):
            for item in discover_outputs(root):
                files_by_path[item["path"]] = item
        files = sorted(files_by_path.values(), key=lambda item: item["modified"], reverse=True)
        st.metric("Discovered artifacts", len(files))
        if not files:
            st.info("No result artifacts have been written yet.")
        else:
            frame = pd.DataFrame(files)
            st.dataframe(frame, hide_index=True, width="stretch")
            selected_path = st.selectbox("Preview artifact", [item["path"] for item in files])
            path = Path(selected_path)
            if path.suffix.lower() == ".csv":
                _, rows = csv_preview(path)
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            elif path.suffix.lower() in {".json", ".txt", ".md"}:
                text = path.read_text(encoding="utf-8", errors="replace")
                st.code(text[:100_000], language="json" if path.suffix.lower() == ".json" else "text")
            elif path.suffix.lower() == ".png":
                st.image(str(path), width="stretch")
            with path.open("rb") as handle:
                st.download_button("Download selected file", data=handle.read(), file_name=path.name)

elif page == "Settings":
    st.title("Settings")
    st.caption("Configure local project resources. Changes apply to new runs.")
    with st.form("settings_form"):
        configured_base = st.text_input("Baseline database", value=base_csv)
        configured_ontology = st.text_input("Ontology file", value=ontology)
        c1, c2 = st.columns(2)
        configured_llm_url = c1.text_input("Text-model endpoint", value=runtime_settings["llm_api_url"])
        configured_llm_model = c2.text_input("Text model", value=runtime_settings["llm_model"])
        c3, c4 = st.columns(2)
        configured_vision_url = c3.text_input("Vision-model endpoint", value=runtime_settings["vision_api_url"])
        configured_vision_model = c4.text_input("Vision model", value=runtime_settings["vision_model"])
        configured_oauth_client = st.text_input("Google OAuth client file", value=runtime_settings["oauth_client"])
        configured_oauth_token = st.text_input("Google OAuth token file", value=runtime_settings["oauth_token"])
        st.text_input("Platform workspace", value=str(PLATFORM_ROOT), disabled=True)
        saved = st.form_submit_button("Save settings", type="primary")
    if saved:
        missing = [path for path in [configured_base, configured_ontology] if not Path(path).exists()]
        if missing:
            st.error("These files were not found: " + ", ".join(missing))
        else:
            st.session_state["base_csv"] = configured_base
            st.session_state["ontology"] = configured_ontology
            save_settings({
                "base_csv": configured_base,
                "ontology": configured_ontology,
                "llm_api_url": configured_llm_url,
                "llm_model": configured_llm_model,
                "vision_api_url": configured_vision_url,
                "vision_model": configured_vision_model,
                "oauth_client": configured_oauth_client,
                "oauth_token": configured_oauth_token,
            })
            st.success("Settings saved.")
    st.subheader("System checks")
    render_health()
    st.subheader("Distribution")
    st.code(
        "powershell -ExecutionPolicy Bypass -File .\\scripts\\build_standalone_platform_package.ps1 -IncludeBaselineDatabase",
        language="powershell",
    )
