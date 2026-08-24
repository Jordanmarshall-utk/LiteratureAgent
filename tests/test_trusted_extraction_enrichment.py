import importlib.util
import os
import sys
from types import SimpleNamespace

import literature_agent_full_end_to_end_v21_3_english_sanitizer as controller


def test_pce_profile_preserves_explicit_google_drive_options():
    args = SimpleNamespace(
        extraction_profile="pce_stability_modeling",
        google_drive_folder_id="drive-folder-id",
        run_mode="initial",
        disable_google_drive=0,
        family_gating="strict",
        expansion_candidate_filter="target_ready",
        expansion_require_oa=1,
        target_finalization_enable=0,
        target_recovery_enable=0,
        figure_report_enable=1,
        use_reasoning_layer=1,
        no_require_doi=False,
    )
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "controller.py",
            "--run_mode", "initial",
            "--google_drive_folder_id", "drive-folder-id",
            "--figure_report_enable", "1",
            "--use_reasoning_layer", "1",
        ]
        controller._apply_extraction_profile(args)
    finally:
        sys.argv = old_argv

    assert args.run_mode == "initial"
    assert args.disable_google_drive == 0
    assert args.figure_report_enable == 1
    assert args.use_reasoning_layer == 1


def test_strict_gate_rejects_generic_device_language_without_metrics(tmp_path):
    module = _load_runtime(tmp_path)
    module.FAMILY_GATING_MODE = "strict"
    text = (
        "This thermodynamics study analyzes phase transitions in layered perovskites. "
        "Device performance and perovskite solar cells are discussed as possible applications. "
        "Density functional theory and synchrotron diffraction are used. "
    ) * 12

    paper_type, scores = module._v21_classify_paper_type(
        {"title": "Thermodynamics of two-dimensional perovskites"},
        text,
    )

    assert scores["metric_score"] == 0
    assert scores["fabrication_score"] == 0
    assert scores["architecture_score"] == 0
    assert scores["explicit_new_device"] is False
    assert paper_type != "device_experimental"
    assert not module._v21_should_run_family("performance", paper_type, scores, text)
    assert not module._v21_should_run_family("etl", paper_type, scores, text)


def test_strict_gate_accepts_device_with_metric_and_architecture(tmp_path):
    module = _load_runtime(tmp_path)
    module.FAMILY_GATING_MODE = "strict"
    text = (
        "We fabricated inverted perovskite solar cells with the architecture "
        "ITO/PEDOT:PSS/perovskite/PCBM/BCP/Ag. The champion device achieved "
        "a power conversion efficiency of 17.21%, Jsc of 21.64 mA cm-2, and "
        "a fill factor of 80.19%."
    )

    paper_type, scores = module._v21_classify_paper_type(
        {"title": "Mixed-spacer two-dimensional perovskite solar cells"},
        text,
    )

    assert scores["metric_score"] >= 2
    assert scores["explicit_new_device"] is True
    assert paper_type == "device_experimental"
    assert module._v21_should_run_family("performance", paper_type, scores, text)


def test_reasoning_layer_preserves_original_extraction_prompt(tmp_path, monkeypatch=None):
    calls = []

    class FakeExtractor:
        def _call(self, system_prompt, user_payload, paper_slug, stage):
            calls.append(system_prompt)
            return {"records": []}

    class FakePolicy:
        @staticmethod
        def apply_literature_reasoning_policy(**kwargs):
            return {"prompt": "REPLACEMENT PHILOSOPHY TEMPLATE", "reasoning_metadata": {}}

    old_policy_loader = controller._load_scientific_reasoning_policy_module
    old_adapter_loader = controller._load_polaris_reasoning_adapter_module
    old_env = os.environ.get("USE_REASONING_LAYER")
    try:
        os.environ["USE_REASONING_LAYER"] = "1"
        controller._load_scientific_reasoning_policy_module = lambda: (FakePolicy, tmp_path / "policy.py")
        controller._load_polaris_reasoning_adapter_module = lambda: (None, None)
        module = SimpleNamespace(HuggingFaceQwenExtractor=FakeExtractor)
        args = SimpleNamespace(
            use_reasoning_layer=1,
            work_dir=str(tmp_path),
            reasoning_policy_mode="multi",
        )
        controller._install_scientific_reasoning_policy_layer(module, args)
        module.HuggingFaceQwenExtractor()._call(
            "ORIGINAL EXTRACTION SCHEMA PROMPT",
            {"output_schema": {"records": []}},
            "paper",
            "performance_partial_1",
        )
    finally:
        controller._load_scientific_reasoning_policy_module = old_policy_loader
        controller._load_polaris_reasoning_adapter_module = old_adapter_loader
        if old_env is None:
            os.environ.pop("USE_REASONING_LAYER", None)
        else:
            os.environ["USE_REASONING_LAYER"] = old_env

    assert "ORIGINAL EXTRACTION SCHEMA PROMPT" in calls[0]
    assert "REPLACEMENT PHILOSOPHY TEMPLATE" not in calls[0]
    assert "place extracted values in their corresponding data fields" in calls[0]


def _load_runtime(tmp_path):
    previous_work_dir = os.environ.get("WORK_DIR")
    os.environ["WORK_DIR"] = str(tmp_path / "lit_outputs")
    try:
        literature_path, _ = controller._write_embedded_modules(tmp_path)
        spec = importlib.util.spec_from_file_location(f"test_literature_runtime_{tmp_path.name}", literature_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_work_dir is None:
            os.environ.pop("WORK_DIR", None)
        else:
            os.environ["WORK_DIR"] = previous_work_dir


def test_broad_full_text_is_not_used_for_deterministic_enrichment():
    meta = {
        "title": "Bottom interfacial engineering for efficient perovskite solar cells",
        "doi": "10.1002/solr.202100285",
        "abstract": (
            "The champion perovskite solar cell reached a power conversion efficiency "
            "of 21.15% and retained 91.5% after 1500 hours."
        ),
    }
    unrelated_or_comparison_text = (
        "A cited comparison used CH3NH3PbI3 with TiO2, SnO2, Spiro-OMeTAD, and Au. "
        "The plotted axis also includes Voc = -0.2 V."
    )

    records = controller._ctrl_v21_10_enrich_extraction_records(
        meta,
        unrelated_or_comparison_text,
        unrelated_or_comparison_text,
        [{}],
    )
    record = records[0]

    assert record["JV_default_PCE"] == 21.15
    assert record["Ref_DOI_number"] == "10.1002/solr.202100285"
    assert "Perovskite_composition_short_form" not in record
    assert "ETL_stack_sequence" not in record
    assert "HTL_stack_sequence" not in record
    assert "Backcontact_stack_sequence" not in record
    assert "JV_default_Voc" not in record


def test_explicit_field_evidence_can_support_enrichment():
    records = controller._ctrl_v21_10_enrich_extraction_records(
        {"title": "CsPbI3 perovskite solar cells", "doi": "10.1000/test"},
        "",
        "",
        [{"_evidence_Perovskite_composition_short_form": "The absorber was CsPbI3."}],
    )

    assert records[0]["Perovskite_composition_short_form"] == "CsPbI3"


def test_parameterized_family_formula_resolves_champion_composition():
    source = (
        "Figure 5 shows XRD patterns of (BDA)1-a(PEA2)aMA4Pb5X16 films. "
        "For mixed spacer cations, the device with (BDA)0.8(PEA2)0.2 film "
        "achieves the champion PCE of 17.21%."
    )
    records = controller._ctrl_v21_10_enrich_extraction_records(
        {"title": "Mixed spacer 2D perovskite solar cells"},
        source,
        source,
        [{}],
    )

    record = records[0]
    assert record["Perovskite_composition_short_form"] == "(BDA)0.8(PEA2)0.2MA4Pb5X16"
    assert record["Perovskite_composition_a_ions"] == "MA"
    assert record["Perovskite_composition_b_ions"] == "Pb"
    assert record["Perovskite_composition_c_ions"] == "X"
    assert "substituting" in record["Perovskite_composition_assumption"]
    assert "(BDA)1-a(PEA2)aMA4Pb5X16" in record["_evidence_Perovskite_composition_short_form"]


def test_deterministic_preextract_receives_trusted_composition(tmp_path):
    module = _load_runtime(tmp_path)
    module.WORK_DIR = str(tmp_path)
    args = SimpleNamespace(
        work_dir=str(tmp_path),
        disable_google_drive=1,
        run_mode="expand",
    )
    controller._install_web_expansion_fulltext_fallback(module, args)
    source = (
        "Figure 5 shows XRD patterns of (BDA)1-a(PEA2)aMA4Pb5X16 films. "
        "The device with (BDA)0.8(PEA2)0.2 film achieves the champion PCE "
        "of 17.21%."
    )

    record = module.deterministic_preextract_record(
        {"title": "Mixed spacer 2D perovskite solar cells"},
        source,
    )

    assert record["Perovskite_composition_short_form"] == "(BDA)0.8(PEA2)0.2MA4Pb5X16"
    assert record["Perovskite_composition_long_form"] == "(BDA)0.8(PEA2)0.2MA4Pb5X16"
    assert record["Perovskite_composition_a_ions"] == "MA"
    assert record["Perovskite_composition_b_ions"] == "Pb"
    assert record["Perovskite_composition_c_ions"] == "X"
    assert record["_lit_agent_composition_source"] == "v21.12_trusted_deterministic_preextract"


def test_final_record_wrapper_overrides_malformed_llm_composition(tmp_path):
    module = _load_runtime(tmp_path)
    module.WORK_DIR = str(tmp_path)
    module.extract_records_with_llm = lambda meta, extraction_text, extractor: [{
        "Perovskite_composition_c_ions": "PbI2",
        "JV_default_PCE": 17.21,
        "_evidence_JV_default_PCE": (
            "The device with (BDA)0.8(PEA2)0.2 film achieves the champion PCE of 17.21%."
        ),
    }]
    controller._install_web_expansion_fulltext_fallback(
        module,
        SimpleNamespace(work_dir=str(tmp_path), disable_google_drive=0, run_mode="initial"),
    )
    source = (
        "Figure 5 shows XRD patterns of (BDA)1-a(PEA2)aMA4Pb5X16 films. "
        "The device with (BDA)0.8(PEA2)0.2 film achieves the champion PCE "
        "of 17.21%."
    )

    records = module.extract_records_with_llm(
        {"title": "Mixed spacer 2D perovskite solar cells"},
        source,
        object(),
    )

    assert len(records) == 1
    assert records[0]["Perovskite_composition_short_form"] == "(BDA)0.8(PEA2)0.2MA4Pb5X16"
    assert records[0]["Perovskite_composition_c_ions"] == "X"
    assert records[0]["JV_default_PCE"] == 17.21


def test_expansion_record_wrapper_uses_runtime_identity_assigner(tmp_path):
    module = _load_runtime(tmp_path)
    module.WORK_DIR = str(tmp_path)
    assigned = []

    module.extract_records_with_llm = lambda meta, extraction_text, extractor: [{
        "JV_default_PCE": 20.27,
    }]

    def assign_distinct_ids(records):
        assigned.append(True)
        records = [dict(record) for record in records]
        records[0]["Ref_internal_sample_id"] = "runtime_assigned_device"
        return records

    module._v21_17_assign_distinct_sample_ids = assign_distinct_ids
    controller._install_web_expansion_fulltext_fallback(
        module,
        SimpleNamespace(
            work_dir=str(tmp_path),
            disable_google_drive=1,
            run_mode="expand",
            automated_retrieval_enable=0,
        ),
    )

    records = module.extract_records_with_llm(
        {"title": "Expansion device paper"},
        "The champion device achieved a PCE of 20.27%.",
        object(),
    )

    assert assigned == [True]
    assert records[0]["Ref_internal_sample_id"] == "runtime_assigned_device"


def test_drive_url_does_not_overwrite_recovered_doi():
    records = controller._ctrl_v21_10_enrich_extraction_records(
        {
            "title": "Device paper",
            "doi": None,
            "landing_page": "https://drive.google.com/file/d/abc123/view",
        },
        "",
        "",
        [{"Ref_DOI_number": "10.1002/advs.202004510"}],
    )

    assert controller._ctrl_v21_10_normalize_doi(
        "https://drive.google.com/file/d/abc123/view"
    ) is None
    assert records[0]["Ref_DOI_number"] == "10.1002/advs.202004510"


def test_grounded_pce_recovers_colocated_jsc_and_ff_from_unicode_units():
    source = (
        "The control device reached a PCE of 14.02%. "
        "The device with (BDA)0.8(PEA2)0.2 film achieves the champion PCE of "
        "17.21% with a short-circuit current density (JSC) of 21.64 mA cm−2 "
        "and fill factor (FF) of 80.19%."
    )
    records = controller._ctrl_v21_10_enrich_extraction_records(
        {"title": "Thermal and humidity stability of mixed spacer devices"},
        source,
        source,
        [{
            "JV_default_PCE": 17.21,
            "_evidence_JV_default_PCE": "The champion device achieved a PCE of 17.21%.",
        }],
    )

    assert records[0]["JV_default_PCE"] == 17.21
    assert records[0]["JV_default_Jsc"] == 21.64
    assert records[0]["JV_default_FF"] == 0.8019
    assert "21.64" in records[0]["_evidence_JV_default_Jsc"]
    assert "80.19" in records[0]["_evidence_JV_default_FF"]


def test_drive_metadata_recovers_publication_year_from_filename(tmp_path):
    module = _load_runtime(tmp_path)
    meta = module._v21_enrich_meta_from_text(
        {
            "title": "2021_Adv. Sci._Thermal Stability__gdrive_example",
            "year": 2026,
            "doi": "10.1002/advs.202004510",
        },
        "Adv. Sci. 2021, 8, 2004510",
        "thermal_stability",
    )

    assert meta["year"] == 2021
    assert meta["_v21_reference_metadata_recovered"]["publication_year"] == 2021


def test_deterministic_backcontact_does_not_treat_front_ito_as_back_contact():
    fields = controller._ctrl_v21_10_extract_feature_fields(
        "Device architecture: Glass/ITO/PEDOT:PSS/Perovskite/PCBM/BCP/Ag."
    )

    assert fields["Backcontact_stack_sequence"] == "Ag"
    assert fields["Cell_stack_sequence"] == "PEDOT:PSS / perovskite / PCBM; BCP / Ag"


def test_summary_performance_is_reconciled_to_grounded_record(tmp_path):
    module = _load_runtime(tmp_path)
    extractor = module.HuggingFaceQwenExtractor.__new__(module.HuggingFaceQwenExtractor)
    extractor._call = lambda *args, **kwargs: {
        "paper_summary": {
            "best_reported_performance": {
                "pce": 19.54,
                "voc": 0.863,
                "jsc": 27.62,
                "ff": 78.2,
                "context": "unsupported comparison",
            }
        }
    }
    records = [{
        "Ref_internal_sample_id": "champion_device",
        "JV_default_PCE": 17.21,
        "_evidence_JV_default_PCE": "The champion device achieved a PCE of 17.21%.",
        "JV_default_Jsc": 21.64,
        "_evidence_JV_default_Jsc": "JSC of 21.64 mA cm-2.",
        "JV_default_FF": 80.19,
        "_evidence_JV_default_FF": "fill factor of 80.19%.",
    }]

    result = extractor.summarize_paper({}, records, "paper text", "paper")
    performance = result["paper_summary"]["best_reported_performance"]

    assert performance["pce"] == 17.21
    assert performance["voc"] is None
    assert performance["jsc"] == 21.64
    assert performance["ff"] == 80.19
    assert result["_summary_performance_reconciliation"]["status"] == "grounded_record_applied"


def test_background_formula_is_not_selected_as_device_composition():
    source = (
        "For example, prior literature reports the general formula (GA)(MA)nPbnI3n+1. "
        "This paper discusses morphology without reporting an experimental absorber formula."
    )
    records = controller._ctrl_v21_10_enrich_extraction_records(
        {"title": "Perovskite morphology"},
        source,
        source,
        [{}],
    )

    assert "Perovskite_composition_short_form" not in records[0]


def test_embedded_csv_export_preserves_evidence_columns():
    patch_source = controller._v21_controller_embedded_quality_patch_source()

    assert 'startswith(("_evidence_", "_source_", "_lit_agent_"))' in patch_source
    assert "pd.DataFrame(rows, columns=columns).to_csv" in patch_source


def test_final_extractor_uses_real_field_evidence_schema(tmp_path):
    module = _load_runtime(tmp_path)

    extractor = module.HuggingFaceQwenExtractor.__new__(module.HuggingFaceQwenExtractor)
    extractor._call = lambda system, payload, **kwargs: payload
    payload = extractor.extract_family_records(
        metadata={"title": "test"},
        chunk_text="The device reached a PCE of 20.0%.",
        family_name="performance",
        family_columns=["JV_default_PCE"],
        alias_map={"JV_default_PCE": ["PCE"]},
        paper_slug="test",
        chunk_idx=1,
    )

    schema = payload["output_schema"]["records"][0]
    assert schema == {"JV_default_PCE": None}
    assert "allowed_fields" not in payload
    assert payload["requirements"]["evidence_is_attached_by_downstream_source_validator"] is True
    assert "example_column" not in str(payload)

    gated = module.validate_record_keys(
        [
            {"JV_default_PCE": 20.0},
            {"JV_default_PCE": 21.0, "_evidence_JV_default_PCE": "PCE of 21.0%."},
        ],
        ["JV_default_PCE"],
    )
    assert gated[0]["JV_default_PCE"] is None
    assert gated[1]["JV_default_PCE"] == 21.0


def test_chunk_evidence_replaces_value_placeholder_with_source_quote(tmp_path):
    module = _load_runtime(tmp_path)
    payload = {
        "records": [
            {
                "JV_default_PCE": 17.21,
                "_evidence_JV_default_PCE": 17.21,
            }
        ]
    }
    text = "The champion device achieved a power conversion efficiency (PCE) of 17.21%."

    result = module._v21_12_attach_chunk_evidence(
        payload,
        text,
        ["JV_default_PCE"],
        {"JV_default_PCE": ["PCE", "power conversion efficiency"]},
    )

    record = result["records"][0]
    assert record["JV_default_PCE"] == 17.21
    assert "champion device" in record["_evidence_JV_default_PCE"]
    assert "17.21" in record["_evidence_JV_default_PCE"]


def test_chunk_evidence_rejects_value_not_supported_by_source_chunk(tmp_path):
    module = _load_runtime(tmp_path)
    payload = {
        "records": [
            {
                "JV_default_PCE": 31.7,
                "_evidence_JV_default_PCE": 31.7,
            }
        ]
    }
    text = "The champion device achieved a PCE of 17.21%."

    result = module._v21_12_attach_chunk_evidence(
        payload,
        text,
        ["JV_default_PCE"],
        {"JV_default_PCE": ["PCE"]},
    )

    record = result["records"][0]
    assert record["JV_default_PCE"] is None
    assert "_evidence_JV_default_PCE" not in record


def test_chunk_evidence_unwraps_reasoning_value_objects(tmp_path):
    module = _load_runtime(tmp_path)
    payload = {
        "records": [
            {
                "Perovskite_thickness": {
                    "value": 200,
                    "unit": "nm",
                    "description": "The film thickness was 200 nm.",
                },
                "_evidence_Perovskite_thickness": {
                    "value": True,
                    "description": "The authors report a 200 nm film.",
                },
            }
        ]
    }
    text = "The perovskite film thickness measured by profilometry was approximately 200 nm."

    result = module._v21_12_attach_chunk_evidence(
        payload,
        text,
        ["Perovskite_thickness"],
        {"Perovskite_thickness": ["film thickness"]},
    )

    record = result["records"][0]
    assert record["Perovskite_thickness"] == 200
    assert "profilometry" in record["_evidence_Perovskite_thickness"]


def test_chunk_evidence_rejects_partially_unsupported_ion_list(tmp_path):
    module = _load_runtime(tmp_path)
    payload = {
        "records": [
            {
                "Perovskite_composition_b_ions": {
                    "value": ["Pb", "Sn"],
                    "description": "Lead and tin are present.",
                }
            }
        ]
    }
    text = "The perovskite precursor contained PbI2 and MAI."

    result = module._v21_12_attach_chunk_evidence(
        payload,
        text,
        ["Perovskite_composition_b_ions"],
        {"Perovskite_composition_b_ions": ["B-site ions"]},
    )

    assert result["records"][0]["Perovskite_composition_b_ions"] is None


def test_chunk_evidence_prefers_explanatory_sentence_over_table_cell(tmp_path):
    module = _load_runtime(tmp_path)
    payload = {"records": [{"JV_default_PCE": 17.21}]}
    text = (
        "PCE\n17.21\nAverage\n"
        "The mixed-spacer champion device achieved a PCE of 17.21% with a fill factor of 80.19%."
    )

    result = module._v21_12_attach_chunk_evidence(
        payload,
        text,
        ["JV_default_PCE"],
        {"JV_default_PCE": ["PCE", "power conversion efficiency"]},
    )

    evidence = result["records"][0]["_evidence_JV_default_PCE"]
    assert "champion device" in evidence
    assert evidence != "17.21"


def test_role_validation_rejects_absorber_spacers_as_etl(tmp_path):
    module = _load_runtime(tmp_path)
    payload = {"records": [{"ETL_stack_sequence": ["BDA", "PEA2"]}]}
    text = "The perovskite films used the spacer compositions BDA and PEA2."

    result = module._v21_12_attach_chunk_evidence(
        payload,
        text,
        ["ETL_stack_sequence"],
        {"ETL_stack_sequence": ["ETL", "electron transport layer"]},
    )

    assert result["records"][0]["ETL_stack_sequence"] is None


def test_role_validation_accepts_ontology_etl_and_explicit_novel_htl(tmp_path):
    module = _load_runtime(tmp_path)
    etl = module._v21_12_attach_chunk_evidence(
        {"records": [{"ETL_stack_sequence": ["PCBM"]}]},
        "The inverted devices used PCBM as the electron transport layer.",
        ["ETL_stack_sequence"],
        {"ETL_stack_sequence": ["ETL", "electron transport layer"]},
    )
    novel_htl = module._v21_12_attach_chunk_evidence(
        {"records": [{"HTL_stack_sequence": ["NovelHTM-7"]}]},
        "NovelHTM-7 was deposited as the hole transport layer in the device.",
        ["HTL_stack_sequence"],
        {"HTL_stack_sequence": ["HTL", "hole transport layer"]},
    )

    assert etl["records"][0]["ETL_stack_sequence"] == ["PCBM"]
    assert novel_htl["records"][0]["HTL_stack_sequence"] == ["NovelHTM-7"]


def test_extraction_uses_source_text_without_generated_appendices(tmp_path):
    module = _load_runtime(tmp_path)
    captured = {}

    def fake_extract(self, meta, full_text, extraction_text):
        captured["text"] = extraction_text
        return {"records": []}

    module._v16_original_extraction_extract = fake_extract
    agent = module.ExtractionAgent.__new__(module.ExtractionAgent)
    source = "ACTUAL PAPER SENTENCE: The device retained 95% after 500 h."
    module._v21_12_extraction_extract(agent, {"title": "test"}, source, source)

    assert captured["text"] == source
    assert "COMPACT SOURCE-DERIVED EVIDENCE INDEX" not in captured["text"]


def test_stability_context_rejects_unrelated_device_count(tmp_path):
    module = _load_runtime(tmp_path)
    result = module._v21_12_attach_chunk_evidence(
        {"records": [{"Stability_average_over_n_number_of_cells": 50}]},
        "The averages and standard deviations were calculated from 50 photovoltaic devices.",
        ["Stability_average_over_n_number_of_cells"],
        {"Stability_average_over_n_number_of_cells": ["number of cells"]},
    )

    assert result["records"][0]["Stability_average_over_n_number_of_cells"] is None


def test_semantic_guards_reject_misassigned_schema_values(tmp_path):
    module = _load_runtime(tmp_path)

    assert not module._v21_12_field_context_is_supported(
        "Stability_PCE_after_1000_h",
        95,
        "The device retained 95% after 500 h in ambient air.",
    )
    assert not module._v21_12_field_context_is_supported(
        "Perovskite_deposition_procedure",
        "(BDA)0.8(PEA2)0.2",
        "The film composition was (BDA)0.8(PEA2)0.2.",
    )
    assert not module._v21_12_field_context_is_supported(
        "Perovskite_composition_a_ions",
        ["MA", "FA", "Cs"],
        "For example, the A site can be MA, FA, or Cs in the general formula.",
    )
    assert not module._v21_12_role_value_is_supported(
        "ETL_stack_sequence",
        "ITO/PEDOT:PSS/perovskite/PCBM/BCP/Ag",
        "The device stack was ITO/PEDOT:PSS/perovskite/PCBM/BCP/Ag.",
    )


def test_deterministic_partial_merge_keeps_grounded_values_and_splits_conflicts(tmp_path):
    module = _load_runtime(tmp_path)
    partials = [
        {
            "records": [
                {
                    "JV_default_PCE": 17.21,
                    "_evidence_JV_default_PCE": "The champion device achieved a PCE of 17.21%.",
                }
            ]
        },
        {
            "records": [
                {
                    "JV_default_Jsc": 21.64,
                    "_evidence_JV_default_Jsc": "The short-circuit current density was 21.64 mA cm-2.",
                }
            ]
        },
        {
            "records": [
                {
                    "JV_default_PCE": 12.4,
                    "_evidence_JV_default_PCE": "The control device achieved a PCE of 12.4%.",
                }
            ]
        },
    ]

    records = module._v21_12_merge_grounded_partial_records(
        partials,
        ["JV_default_PCE", "JV_default_Jsc"],
    )

    assert len(records) == 2
    champion = next(record for record in records if record.get("JV_default_PCE") == 17.21)
    assert champion["JV_default_Jsc"] == 21.64
    assert "short-circuit" in champion["_evidence_JV_default_Jsc"]


def test_final_coalescing_copies_context_to_target_rows_without_dropping_conflicts(tmp_path):
    module = _load_runtime(tmp_path)
    records = [
        {
            "Perovskite_dimension_2D": True,
            "_evidence_Perovskite_dimension_2D": "The absorber was a two-dimensional perovskite film.",
        },
        {
            "ETL_stack_sequence": ["PCBM", "BCP"],
            "_evidence_ETL_stack_sequence": "PCBM and BCP were deposited as electron-transport layers.",
        },
        {
            "JV_reverse_scan_PCE": 17.21,
            "_evidence_JV_reverse_scan_PCE": "The champion device achieved a PCE of 17.21%.",
        },
        {
            "JV_reverse_scan_PCE": 12.4,
            "_evidence_JV_reverse_scan_PCE": "The control device achieved a PCE of 12.4%.",
        },
    ]

    coalesced = module._v21_12_coalesce_final_records(records)

    assert len(coalesced) == 2
    assert {record["JV_reverse_scan_PCE"] for record in coalesced} == {17.21, 12.4}
    assert all(record["Perovskite_dimension_2D"] is True for record in coalesced)
    assert all(record["ETL_stack_sequence"] == ["PCBM", "BCP"] for record in coalesced)


def test_final_coalescing_merges_same_device_context_conflicts(tmp_path):
    module = _load_runtime(tmp_path)
    records = [
        {
            "Ref_DOI_number": "10.1002/advs.202004510",
            "Ref_internal_sample_id": "device_1",
            "Cell_stack_sequence": "ITO/PEDOT:PSS/Perovskite/PCBM/BCP/Ag",
            "JV_default_PCE": 17.21,
            "JV_default_Jsc": 21.64,
            "JV_default_FF": 0.8019,
            "Stability_time_total_exposure": 2000,
            "_evidence_Cell_stack_sequence": "Devices used ITO/PEDOT:PSS/perovskite/PCBM/BCP/Ag.",
        },
        {
            "Ref_DOI_number": "10.1002/advs.202004510",
            "Ref_internal_sample_id": "device_1",
            "Cell_stack_sequence": "PCBM; BCP / perovskite / PEDOT:PSS / Ag; ITO",
            "JV_default_PCE": 17.21,
            "JV_default_Jsc": 21.64,
            "JV_default_FF": 0.8019,
        },
    ]

    coalesced = module._v21_12_coalesce_final_records(records)

    assert len(coalesced) == 1
    assert coalesced[0]["Cell_stack_sequence"] == "ITO/PEDOT:PSS/Perovskite/PCBM/BCP/Ag"
    assert coalesced[0]["Stability_time_total_exposure"] == 2000


def test_final_coalescing_preserves_distinct_stability_conditions(tmp_path):
    module = _load_runtime(tmp_path)
    records = [
        {
            "Ref_DOI_number": "10.1002/advs.202004510",
            "Ref_internal_sample_id": "device_1",
            "Stability_atmosphere": "Nitrogen",
            "Stability_time_total_exposure": 2000,
        },
        {
            "Ref_DOI_number": "10.1002/advs.202004510",
            "Ref_internal_sample_id": "device_1",
            "Stability_atmosphere": "Ambient air",
            "Stability_relative_humidity_load_conditions": "40 +/- 5%",
        },
    ]

    coalesced = module._v21_12_coalesce_final_records(records)

    assert len(coalesced) == 2
    assert {record["Stability_atmosphere"] for record in coalesced} == {
        "Nitrogen",
        "Ambient air",
    }


def test_humidity_parser_uses_central_value_not_uncertainty():
    parsed = controller._ctrl_v21_10_extract_stability(
        "The devices were stored in ambient air with 40 +/- 5% relative humidity."
    )

    assert parsed["Stability_relative_humidity_average_value"] == 40.0


def test_summary_renderer_uses_reconciled_publication_year(tmp_path):
    module = _load_runtime(tmp_path)
    payload = {
        "paper_summary": {
            "one_sentence_summary": "A device study.",
            "paper_type": "experimental",
        },
        "_summary_metadata_reconciliation": {"year": 2021},
    }

    rendered = module.render_paper_summary_text(
        payload,
        {"title": "Paper", "journal": "Manual Google Drive Import", "year": 2026},
    )

    assert "Year: 2021" in rendered
    assert "Year: 2026" not in rendered


def test_stability_enrichment_does_not_leak_humidity_between_rows():
    records = [
        {
            "Stability_atmosphere": "Nitrogen",
            "Stability_time_total_exposure": 2000,
            "_evidence_Stability_time_total_exposure": (
                "The device was stored in nitrogen for about 2000 h in the dark."
            ),
        },
        {
            "Stability_atmosphere": "Ambient air",
            "Stability_relative_humidity_load_conditions": "40 +/- 5%",
            "_evidence_Stability_relative_humidity_load_conditions": (
                "The device was stored in ambient air with 40 +/- 5% relative humidity."
            ),
        },
    ]

    enriched = controller._ctrl_v21_10_enrich_extraction_records(
        {
            "title": "Thermal and humidity stability",
            "abstract": "Devices were compared under 40 +/- 5% relative humidity.",
        },
        "",
        "",
        records,
    )

    assert enriched[0].get("Stability_relative_humidity_average_value") is None
    assert enriched[1]["Stability_relative_humidity_average_value"] == 40.0


def test_final_target_sanitizer_normalizes_database_units_and_rejects_cross_metric_values():
    module = controller
    record = {
        "JV_default_PCE": 20.8,
        "_evidence_JV_default_PCE": "The device reached a power conversion efficiency of 20.8%.",
        "JV_default_Voc": 1137,
        "_evidence_JV_default_Voc": "The cell had a Voc of 1137 mV.",
        "JV_default_Jsc": 22.3,
        "_evidence_JV_default_Jsc": "The Jsc was 22.3 mA cm-2.",
        "JV_default_FF": 82.2,
        "_evidence_JV_default_FF": "The fill factor was 82.2%.",
    }

    changes = module._ctrl_v21_13_sanitize_record_targets(record)

    assert record["JV_default_PCE"] == 20.8
    assert record["JV_default_Voc"] == 1.137
    assert record["JV_default_Jsc"] == 22.3
    assert record["JV_default_FF"] == 0.822
    assert any("millivolts_to_volts" in change for change in changes)
    assert any("percent_to_fraction" in change for change in changes)


def test_final_target_sanitizer_clears_retention_and_wrong_metric_assignments():
    module = controller
    record = {
        "JV_default_PCE": 80,
        "_evidence_JV_default_PCE": "The device retained 80% of its initial PCE after 1000 h.",
        "JV_default_Voc": 8.4,
        "_evidence_JV_default_Voc": "The HA device showed a PCE of 8.4%.",
        "JV_default_Jsc": 90,
        "_evidence_JV_default_Jsc": "The device retained 90% of its initial efficiency.",
    }

    module._ctrl_v21_13_sanitize_record_targets(record)

    assert record["JV_default_PCE"] is None
    assert record["JV_default_Voc"] is None
    assert record["JV_default_Jsc"] is None


def test_final_target_sanitizer_clears_unscoped_humidity_from_nitrogen_row():
    module = controller
    record = {
        "Stability_atmosphere": "Nitrogen",
        "Stability_time_total_exposure": 2000,
        "Stability_relative_humidity_average_value": 40,
        "_evidence_Stability_relative_humidity_average_value": (
            "Stored in ambient air at 40 +/- 5% RH or in a nitrogen glove box."
        ),
    }

    module._ctrl_v21_13_sanitize_record_targets(record)

    assert record["Stability_relative_humidity_average_value"] is None


def test_final_target_sanitizer_enforces_stability_target_semantics():
    record = {
        "Stability_PCE_initial_value": 97.02,
        "_evidence_Stability_PCE_initial_value": (
            "The device retained 97.02% of its initial PCE after aging."
        ),
        "Stability_PCE_end_of_experiment": 18,
        "_evidence_Stability_PCE_end_of_experiment": (
            "The control showed an 18% PCE degradation."
        ),
        "Stability_PCE_after_1000_h": 80,
        "_evidence_Stability_PCE_after_1000_h": "T80 was more than 1000 hours.",
    }

    controller._ctrl_v21_13_sanitize_record_targets(record)

    assert record["Stability_PCE_initial_value"] is None
    assert record["Stability_PCE_end_of_experiment"] is None
    assert record["Stability_PCE_after_1000_h"] is None


def test_final_target_sanitizer_preserves_grounded_retention_and_converts_days():
    record = {
        "Stability_PCE_initial_value": 13.32,
        "_evidence_Stability_PCE_initial_value": "The initial PCE was 13.32%.",
        "Stability_PCE_end_of_experiment": 82,
        "_evidence_Stability_PCE_end_of_experiment": (
            "The device retained 82% of its initial PCE after 30 days."
        ),
        "Stability_time_total_exposure": 30,
        "_evidence_Stability_time_total_exposure": (
            "The device retained 82% of its initial PCE after 30 days."
        ),
    }

    controller._ctrl_v21_13_sanitize_record_targets(record)

    assert record["Stability_PCE_initial_value"] == 13.32
    assert record["Stability_PCE_end_of_experiment"] == 82
    assert record["Stability_time_total_exposure"] == 720
