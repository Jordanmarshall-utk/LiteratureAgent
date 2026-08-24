import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd

import literature_agent_full_end_to_end_v21_3_english_sanitizer as controller
from scripts import audit_expansion_batch
from scripts import run_expansion_campaign

BRIDGE_MODULE = ModuleType("integration_bridge_test")
sys.modules[BRIDGE_MODULE.__name__] = BRIDGE_MODULE
exec(controller._INTEGRATION_BRIDGE_SOURCE, BRIDGE_MODULE.__dict__)
exec(controller._v21_sparse_aware_integration_patch_source(), BRIDGE_MODULE.__dict__)
exec(controller._v21_8_expansion_backfill_patch_source(), BRIDGE_MODULE.__dict__)
BRIDGE = BRIDGE_MODULE.__dict__


def _args(work_dir: Path, *, expand: bool = True):
    return SimpleNamespace(
        disable_google_drive=1 if expand else 0,
        run_mode="expand" if expand else "initial",
        automated_retrieval_enable=1,
        retrieval_max_urls_per_candidate=2,
        retrieval_supplementary_enable=1,
        retrieval_max_supplementary_files=2,
        retrieval_max_pdf_bytes=25_000_000,
        retrieval_max_pdf_pages=80,
        expansion_min_fulltext_chars=1000,
        expansion_min_extraction_text_chars=1000,
        web_request_timeout=10,
        work_dir=str(work_dir),
    )


def _fake_runtime(work_dir: Path):
    def merge_record_lists(existing, new):
        return [{"legacy_merge": True}]

    def merge_deterministic_with_llm_records(deterministic, records, meta):
        return [{"legacy_deterministic_merge": True}]

    def process_one_paper_api(meta, force_reprocess=False, run_mode=None):
        return {
            "status": "ok",
            "records": [
                {
                    "Ref_internal_sample_id": "target",
                    "JV_default_PCE": 22.1,
                    "Perovskite_composition_short_form": "FAPbI3",
                    "Cell_stack_sequence": "ITO/SnO2/FAPbI3/Spiro/Au",
                    "Perovskite_deposition_procedure": "spin coating",
                },
                {
                    "Ref_internal_sample_id": "control",
                    "JV_default_PCE": 19.2,
                },
            ],
        }

    return SimpleNamespace(
        WORK_DIR=str(work_dir),
        merge_record_lists=merge_record_lists,
        merge_deterministic_with_llm_records=merge_deterministic_with_llm_records,
        process_one_paper_api=process_one_paper_api,
        try_full_text=lambda meta, slug: {"full_text": "", "text_source": "none"},
        slugify=lambda value: "paper_slug",
    )


class ExpansionQualityControlTests(unittest.TestCase):
    def test_known_candidate_loader_can_exclude_legacy_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work_dir = root / "work"
            csv_dir = work_dir / "csv"
            csv_dir.mkdir(parents=True)
            base_csv = root / "base.csv"
            registry_csv = root / "processed.csv"
            pd.DataFrame([{
                "Ref_DOI_number": "10.1/legacy",
                "Ref_original_filename_data_upload": "Legacy stability paper in baseline",
            }]).to_csv(base_csv, index=False)
            pd.DataFrame([{
                "Ref_DOI_number": "10.1/completed",
                "Ref_original_filename_data_upload": "Completed LiteratureAgent paper",
            }]).to_csv(registry_csv, index=False)

            args = SimpleNamespace(
                base_csv=str(base_csv),
                expansion_processed_registry=str(registry_csv),
                expansion_skip_base_known_candidates=0,
            )
            dois, titles = controller._load_expansion_known_candidate_keys(args, work_dir)

            self.assertNotIn("10.1/legacy", dois)
            self.assertIn("10.1/completed", dois)
            self.assertNotIn("legacy stability paper in baseline", titles)
            self.assertIn("completed literatureagent paper", titles)

    def test_stability_query_sweep_stays_target_focused(self):
        plan = run_expansion_campaign.build_query_plan(
            "model_ready_stability_table",
            search_query=None,
            query_sweep=True,
            max_rounds=8,
        )

        self.assertEqual(plan[0][0], "model_ready_stability_table")
        self.assertEqual(plan[1][0], "model_ready_mpp")
        self.assertTrue(all("pce_table" not in mode for mode, _ in plan))
        self.assertTrue(plan[-1][0].startswith("expanded_stability_"))

    def test_stability_target_gate_defers_numeric_values_to_full_text(self):
        item = {
            "title": ["Operational stability of perovskite solar cell devices"],
            "abstract": "Devices were evaluated under continuous illumination.",
            "type": "journal-article",
            "DOI": "10.1/stability",
        }

        allowed, reason, _, _ = controller._score_expansion_candidate(
            item,
            mode="stability_target_ready",
        )

        self.assertTrue(allowed)
        self.assertIn("stability_target_candidate", reason)

    def test_stability_target_gate_still_blocks_reviews(self):
        item = {
            "title": ["Review of operational stability in perovskite solar cells"],
            "abstract": "A review of device lifetime measurements.",
            "type": "journal-article",
            "DOI": "10.1/review",
        }

        allowed, reason, _, _ = controller._score_expansion_candidate(
            item,
            mode="stability_target_ready",
        )

        self.assertFalse(allowed)
        self.assertIn("blocked_title_term:review", reason)

    def test_improved_metrics_do_not_use_pce_endpoint_as_fill_factor(self):
        text = (
            "Upon introducing SBS at the bottom interface, the Voc and FF of "
            "the perovskite devices increase from 1.14 V and 81.01% to "
            "1.18 V and 82.31%, respectively, while the PCE increases from "
            "18.99% to 20.27%."
        )

        metrics = controller._ctrl_v21_14_extract_improved_device_metrics(text)

        self.assertEqual(metrics["JV_default_PCE"], 20.27)
        self.assertAlmostEqual(metrics["JV_default_FF"], 0.8231)

    def test_expansion_extraction_wrapper_uses_runtime_identity_assigner(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _fake_runtime(Path(tmp))
            runtime.extract_records_with_llm = (
                lambda meta, extraction_text, extractor: [{
                    "JV_default_PCE": 20.27,
                }]
            )
            assigned = []

            def assign_distinct_ids(records):
                assigned.append(True)
                records = [dict(record) for record in records]
                records[0]["Ref_internal_sample_id"] = "runtime_assigned_device"
                return records

            runtime._v21_17_assign_distinct_sample_ids = assign_distinct_ids
            controller._install_web_expansion_fulltext_fallback(
                runtime,
                _args(Path(tmp)),
            )

            records = runtime.extract_records_with_llm(
                {"title": "Expansion device paper"},
                "The champion device achieved a PCE of 20.27%.",
                object(),
            )

            self.assertEqual(assigned, [True])
            self.assertEqual(
                records[0]["Ref_internal_sample_id"],
                "runtime_assigned_device",
            )

    def test_cross_metric_gate_clears_pce_fraction_misassigned_as_ff(self):
        frame = pd.DataFrame([{
            "Ref_DOI_number": "10.1/example",
            "JV_default_PCE": 15.5,
            "JV_default_FF": 0.155,
        }])

        cleaned, report = BRIDGE["_v21_18_sanitize_cross_metric_roles"](frame)

        self.assertTrue(pd.isna(cleaned.loc[0, "JV_default_FF"]))
        self.assertEqual(
            cleaned.loc[0, "_lit_agent_metric_consistency_warning"],
            "ff_duplicates_pce_fraction",
        )
        self.assertEqual(len(report), 1)

    def test_final_identity_reconciliation_merges_sparse_same_pce_rows(self):
        frame = pd.DataFrame([
            {
                "Ref_DOI_number": "10.1/example",
                "Ref_internal_sample_id": "paper_device_1_control_pce_14p82",
                "JV_default_PCE": 15.44,
                "JV_default_Voc": 1.15,
                "Cell_stack_sequence": "ITO/perovskite/Ag",
            },
            {
                "Ref_DOI_number": "10.1/example",
                "Ref_internal_sample_id": "paper_device_1_device",
                "JV_default_PCE": 15.44,
                "JV_default_Voc": 1.22,
                "JV_default_Jsc": 16.34,
                "Cell_stack_sequence": "ITO/perovskite/Ag",
            },
        ])

        cleaned, report = BRIDGE["_v21_18_reconcile_final_device_identities"](frame)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.loc[0, "Ref_internal_sample_id"], "paper_device_1")
        self.assertTrue(pd.isna(cleaned.loc[0, "JV_default_Voc"]))
        self.assertEqual(cleaned.loc[0, "JV_default_Jsc"], 16.34)
        self.assertEqual(len(report), 1)

    def test_performance_prefers_certified_device_over_theoretical_limit(self):
        text = (
            "MAPbI3 can reach a theoretical maximum PCE of 31% in the "
            "radiative limit under the Shockley-Queisser approximation. "
            "In this work, a certified steady-state PCE of 15.5% was obtained "
            "for the HSL-free carbon-electrode perovskite solar cell."
        )

        performance = controller._ctrl_v21_10_extract_performance(text)

        self.assertEqual(performance["JV_default_PCE"], 15.5)

    def test_performance_accepts_pce_with_uncertainty(self):
        performance = controller._ctrl_v21_10_extract_performance(
            "Mixed-cation solar cells exhibited the highest PCE "
            "(21.0 ± 0.3%) in the measured devices."
        )

        self.assertEqual(performance["JV_default_PCE"], 21.0)

    def test_performance_accepts_improved_from_to_pce_wording(self):
        performance = controller._ctrl_v21_10_extract_performance(
            "The champion device PCE is improved from 19.22% to 21.76%."
        )

        self.assertEqual(performance["JV_default_PCE"], 21.76)

    def test_stability_accepts_efficiency_retention_wording(self):
        stability = controller._ctrl_v21_10_extract_stability(
            "The devices showed 86.7% efficiency retention after 500 hours "
            "of stability testing."
        )

        self.assertEqual(stability["Stability_PCE_end_of_experiment"], 86.7)
        self.assertEqual(stability["Stability_time_total_exposure"], 500.0)

    def test_stability_accepts_remains_wording(self):
        stability = controller._ctrl_v21_10_extract_stability(
            "The champion device PCE remains 87% of its initial value after "
            "illumination for 1000 hours."
        )

        self.assertEqual(stability["Stability_PCE_end_of_experiment"], 87.0)
        self.assertEqual(stability["Stability_time_total_exposure"], 1000.0)

    def test_abstract_stability_is_scoped_to_matching_pce_row(self):
        abstract = (
            "Mixed-cation devices showed the highest PCE (21.0 ± 0.3%) "
            "and 86.7% efficiency retention after 500 hours."
        )
        records = [
            {
                "JV_default_PCE": 21.0,
                "_evidence_JV_default_PCE": "highest PCE (21.0 ± 0.3%)",
            },
            {
                "JV_default_PCE": 18.0,
                "_evidence_JV_default_PCE": "control PCE of 18.0%",
            },
        ]

        enriched = controller._ctrl_v21_10_enrich_extraction_records(
            {"title": "Experimental device study", "abstract": abstract},
            "",
            "",
            records,
        )

        self.assertEqual(
            enriched[0]["Stability_PCE_end_of_experiment"],
            86.7,
        )
        self.assertEqual(
            enriched[0]["Stability_time_total_exposure"],
            500.0,
        )
        self.assertNotIn("Stability_PCE_end_of_experiment", enriched[1])

    def test_target_sanitizer_removes_theoretical_pce(self):
        record = {
            "JV_default_PCE": 31.0,
            "_evidence_JV_default_PCE": (
                "The theoretical maximum PCE of 31% occurs in the radiative "
                "limit under the Shockley-Queisser approximation."
            ),
        }

        changes = controller._ctrl_v21_13_sanitize_record_targets(record)

        self.assertIsNone(record["JV_default_PCE"])
        self.assertTrue(any("theoretical_or_background" in item for item in changes))

    def test_target_sanitizer_rejects_pce_not_present_in_evidence(self):
        record = {
            "JV_default_PCE": 31.0,
            "_evidence_JV_default_PCE": (
                "The device achieved a certified power conversion efficiency "
                "of 15.5%."
            ),
        }

        changes = controller._ctrl_v21_13_sanitize_record_targets(record)

        self.assertIsNone(record["JV_default_PCE"])
        self.assertTrue(any("not_supported_by_evidence" in item for item in changes))

    def test_target_sanitizer_keeps_ocr_clipped_efficiency_evidence(self):
        record = {
            "JV_default_PCE": 15.5,
            "_evidence_JV_default_PCE": (
                "ciency of 15.5% with a steady-state maximum power point"
            ),
        }

        changes = controller._ctrl_v21_13_sanitize_record_targets(record)

        self.assertEqual(record["JV_default_PCE"], 15.5)
        self.assertEqual(changes, [])

    def test_target_sanitizer_rejects_generic_pce_in_stabilized_field(self):
        record = {
            "Stabilised_performance_PCE": 20.06,
            "_evidence_Stabilised_performance_PCE": (
                "The highest PCE of 20.06% was achieved by the treated device."
            ),
        }

        changes = controller._ctrl_v21_13_sanitize_record_targets(record)

        self.assertIsNone(record["Stabilised_performance_PCE"])
        self.assertTrue(any("stabilized_measurement_cue" in item for item in changes))

    def test_target_sanitizer_excludes_computational_device_targets(self):
        record = {
            "_lit_agent_paper_type": "computational_theory",
            "JV_default_PCE": 27.84,
            "_evidence_JV_default_PCE": (
                "The SCAPS-1D simulated device achieved a PCE of 27.84%."
            ),
            "JV_default_Voc": 1.27,
            "_evidence_JV_default_Voc": (
                "The simulated device achieved VOC = 1.27 V."
            ),
        }

        changes = controller._ctrl_v21_13_sanitize_record_targets(record)

        self.assertIsNone(record["JV_default_PCE"])
        self.assertIsNone(record["JV_default_Voc"])
        self.assertEqual(
            record["_lit_agent_model_exclusion_reason"],
            "paper_type_is_computational_or_review",
        )
        self.assertTrue(any("nonexperimental_paper" in item for item in changes))

    def test_enrichment_repairs_cross_device_metric_pairing(self):
        abstract = (
            "The best MAPbI3 perovskite solar cell exhibits a PCE of 7.72% "
            "with JSC of 17.26 mA cm-2, VOC = 0.94 V, and FF of 47.6%."
        )
        records = [{
            "JV_default_PCE": 7.72,
            "_evidence_JV_default_PCE": abstract,
            "JV_default_Jsc": 14.13,
            "_evidence_JV_default_Jsc": (
                "A control device had PCE of 5.17% with JSC of "
                "14.13 mA cm-2."
            ),
            "JV_default_Voc": 0.84,
            "_evidence_JV_default_Voc": (
                "A control device had PCE of 5.17% and VOC = 0.84 V."
            ),
            "JV_default_FF": 0.476,
            "_evidence_JV_default_FF": abstract,
        }]

        enriched = controller._ctrl_v21_10_enrich_extraction_records(
            {"title": "Experimental device study", "abstract": abstract},
            abstract,
            abstract,
            records,
        )

        self.assertEqual(enriched[0]["JV_default_PCE"], 7.72)
        self.assertEqual(enriched[0]["JV_default_Jsc"], 17.26)
        self.assertEqual(enriched[0]["JV_default_Voc"], 0.94)
        self.assertEqual(enriched[0]["JV_default_FF"], 0.476)

    def test_enrichment_selects_improved_grouped_device_endpoints(self):
        pce_evidence = (
            "The device without Co3O4 only achieved a PCE of 9.87%, "
            "whereas the modified device PCE significantly increased to 11.13%."
        )
        metric_evidence = (
            "The short-circuit current density (JSC), open-circuit voltage "
            "(VOC), and fill factor (FF) increased from 13.93 mA cm-2, "
            "1.07 V, and 66.28% to 14.05 mA cm-2, 1.11 V, and 71.26%, "
            "respectively."
        )
        records = [{
            "JV_default_PCE": 9.87,
            "_evidence_JV_default_PCE": pce_evidence,
            "JV_default_Jsc": 13.93,
            "_evidence_JV_default_Jsc": metric_evidence,
            "JV_default_Voc": 1.07,
            "_evidence_JV_default_Voc": metric_evidence,
            "JV_default_FF": 0.6628,
            "_evidence_JV_default_FF": metric_evidence,
        }]

        enriched = controller._ctrl_v21_10_enrich_extraction_records(
            {"title": "Experimental device study", "abstract": ""},
            pce_evidence + " " + metric_evidence,
            pce_evidence + " " + metric_evidence,
            records,
        )

        self.assertEqual(enriched[0]["JV_default_PCE"], 11.13)
        self.assertEqual(enriched[0]["JV_default_Jsc"], 14.05)
        self.assertEqual(enriched[0]["JV_default_Voc"], 1.11)
        self.assertEqual(enriched[0]["JV_default_FF"], 0.7126)

    def test_directional_metrics_select_separate_from_to_endpoints(self):
        evidence = (
            "The VOC and JSC showed enhancement, with an increase from "
            "1.121 V to 1.136 V and from 25.07 mA/cm2 to 25.30 mA/cm2. "
            "The FF rose from 80.59% to 83.88%."
        )

        metrics = controller._ctrl_v21_14_extract_improved_device_metrics(evidence)

        self.assertEqual(metrics["JV_default_Voc"], 1.136)
        self.assertEqual(metrics["JV_default_Jsc"], 25.30)
        self.assertEqual(metrics["JV_default_FF"], 0.8388)

    def test_voc_endpoint_is_retained_when_evidence_also_reports_gain(self):
        record = {
            "JV_default_Voc": 1.136,
            "_evidence_JV_default_Voc": (
                "VOC increased from 1.121 V to 1.136 V, an enhancement "
                "of 1.3%."
            ),
        }

        changes = controller._ctrl_v21_13_sanitize_record_targets(record)

        self.assertEqual(record["JV_default_Voc"], 1.136)
        self.assertFalse(any("voc_difference" in item for item in changes))

    def test_spaced_mixed_cation_formula_is_normalized(self):
        text = (
            "Perovskite Precursor Solution: The absorber "
            "Rb 0.05Cs0.1FA0.85PbI 3 was prepared and deposited in the "
            "best-performing device."
        )

        formula, evidence, _ = controller._ctrl_v21_12_extract_device_composition(text)

        self.assertEqual(formula, "Rb0.05Cs0.1FA0.85PbI3")
        self.assertIn("Precursor Solution", evidence)

    def test_candidate_preview_is_not_trusted_source_text(self):
        records = [{
            "_source_title": (
                "Fill Factor Assessment with 15.5% Certified Power "
                "Conversion Efficiency"
            ),
            "_evidence_v19_deterministic_candidates": (
                "theoretical maximum PCE of 31% in the radiative limit"
            ),
        }]

        source = controller._ctrl_v21_10_source_text(
            {},
            "",
            "",
            records,
        )

        self.assertIn("15.5%", source)
        self.assertNotIn("31%", source)

    def test_source_alignment_rejects_unrelated_domain_paper(self):
        meta = {
            "title": (
                "Bifunctional Dimethyldichlorosilane Assisted Air-Processed "
                "Perovskite Solar Cell with Enhanced Stability and Low Voltage Loss"
            ),
            "doi": "10.1002/solr.202201067",
        }
        unrelated = (
            "CH3NH3PbI3/GeSe bilayer heterojunction solar cell with high performance\n"
            "This computational paper studies a perovskite bilayer. "
            "Perovskite devices can have voltage loss and assisted transport. "
            "The perovskite absorber is modeled theoretically. "
            + ("generic photovoltaic discussion " * 800)
            + "bifunctional stability processed"
        )

        aligned, reason, _, head_coverage, doi_found, _ = (
            controller._expansion_source_alignment(meta, unrelated)
        )

        self.assertFalse(aligned)
        self.assertEqual(reason, "doi_title_head_mismatch")
        self.assertEqual(doi_found, 0)
        self.assertLess(head_coverage, 0.70)

    def test_source_alignment_accepts_matching_front_matter_title(self):
        title = (
            "Bifunctional Dimethyldichlorosilane Assisted Air-Processed "
            "Perovskite Solar Cell with Enhanced Stability and Low Voltage Loss"
        )
        matching = (
            f"{title}\n"
            "Abstract. This perovskite solar-cell study reports an air-processed "
            "perovskite absorber and evaluates perovskite device performance."
        )

        aligned, reason, _, head_coverage, doi_found, _ = (
            controller._expansion_source_alignment(
                {"title": title, "doi": "10.1002/solr.202201067"},
                matching,
            )
        )

        self.assertTrue(aligned)
        self.assertEqual(reason, "title_head_and_domain_match")
        self.assertEqual(doi_found, 0)
        self.assertGreaterEqual(head_coverage, 0.70)

    def test_source_alignment_rejects_near_title_without_doi(self):
        title = (
            "Interface Engineering of 2D 3D Perovskite Heterojunction "
            "Improves Photovoltaic Efficiency and Stability"
        )
        near_match = (
            "Interface Engineering of 2D 3D Perovskite Bilayer Improves "
            "Photovoltaic Efficiency and Stability\n"
            + ("perovskite device passivation performance " * 500)
        )

        aligned, reason, _, head_coverage, doi_found, _ = (
            controller._expansion_source_alignment(
                {"title": title, "doi": "10.1002/solr.202100072"},
                near_match,
            )
        )

        self.assertFalse(aligned)
        self.assertEqual(reason, "doi_title_head_mismatch")
        self.assertEqual(doi_found, 0)
        self.assertLess(head_coverage, 0.90)

    def test_pdf_preflight_rejects_oversized_content_before_reader(self):
        class ReaderMustNotRun:
            def __init__(self, _stream):
                raise AssertionError("reader should not run for oversized content")

        reason = controller._ctrl_v21_15_pdf_preflight_reason(
            b"x" * 101,
            ReaderMustNotRun,
            max_pdf_bytes=100,
            max_pdf_pages=80,
        )

        self.assertEqual(reason, "pdf_too_large:101>100")

    def test_arxiv_feed_rejects_topically_similar_wrong_title(self):
        xml = """
        <feed>
          <entry>
            <title>Generic Stability Analysis of Thin Film Solar Cells</title>
            <link href="https://arxiv.org/pdf/1234.56789v1" type="application/pdf"/>
          </entry>
        </feed>
        """

        urls = controller._ctrl_v21_15_arxiv_pdf_urls_from_feed(
            "Surface Management for Carbon Based CsPbI2Br Perovskite Solar Cell",
            xml,
        )

        self.assertEqual(urls, [])

    def test_arxiv_feed_keeps_near_exact_title_before_download(self):
        title = (
            "Surface Management for Carbon Based CsPbI2Br "
            "Perovskite Solar Cell"
        )
        xml = f"""
        <feed>
          <entry>
            <title>{title}</title>
            <link href="https://arxiv.org/pdf/1234.56789v1" type="application/pdf"/>
          </entry>
        </feed>
        """

        urls = controller._ctrl_v21_15_arxiv_pdf_urls_from_feed(title, xml)

        self.assertEqual(
            urls,
            ["https://arxiv.org/pdf/1234.56789v1"],
        )

    def test_landing_page_discovers_pdf_and_dspace_bitstream_links(self):
        html = """
        <html><head>
          <meta name="citation_pdf_url" content="/files/article.pdf">
        </head><body>
          <a href="/download/manuscript.pdf">Download PDF</a>
          <script>
            &q;https://repo.example.org/server/api/core/bitstreams/
            d723511d-83a7-4541-b736-bb18ca0ed498&q;
          </script>
        </body></html>
        """

        urls = controller._ctrl_v21_15_landing_fulltext_urls(
            html,
            "https://repo.example.org/item/123",
        )

        self.assertIn("https://repo.example.org/files/article.pdf", urls)
        self.assertIn(
            "https://repo.example.org/download/manuscript.pdf",
            urls,
        )
        self.assertIn(
            "https://repo.example.org/server/api/core/bitstreams/"
            "d723511d-83a7-4541-b736-bb18ca0ed498/content",
            urls,
        )

    def test_landing_page_discovers_repository_docx_manuscript(self):
        html = """
        <html><body>
          <a href="/filesets/abc-123/download">
            Surface Stoichiometric.docx
          </a>
          <script type="application/ld+json">
          {"distribution":[{
            "contentUrl":"https://repo.example.org/filesets/abc-123/download",
            "encodingFormat":"application/vnd.openxmlformats-officedocument.wordprocessingml.document"
          }]}
          </script>
        </body></html>
        """

        urls = controller._ctrl_v21_15_landing_fulltext_urls(
            html,
            "https://repo.example.org/datasets/123",
        )

        self.assertEqual(
            urls,
            ["https://repo.example.org/filesets/abc-123/download"],
        )

    def test_landing_page_discovers_html_xml_and_plain_text_fulltext(self):
        html = """
        <html><head>
          <meta name="citation_fulltext_html_url" content="/article/full.html">
          <meta name="citation_xml_url" content="/article/full.xml">
        </head><body>
          <a href="/article/full.txt">Download plain text</a>
        </body></html>
        """

        urls = controller._ctrl_v21_15_landing_fulltext_urls(
            html,
            "https://repository.example/item/123",
        )

        self.assertEqual(
            urls,
            [
                "https://repository.example/article/full.html",
                "https://repository.example/article/full.xml",
                "https://repository.example/article/full.txt",
            ],
        )

    def test_docx_author_manuscript_text_is_extracted(self):
        import io
        import zipfile

        document_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body>
            <w:p><w:r><w:t>Surface Stoichiometric Tuning</w:t></w:r></w:p>
            <w:tbl><w:tr><w:tc><w:p><w:r><w:t>ITO/SnO2/perovskite/Au</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
          </w:body>
        </w:document>
        """
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("word/document.xml", document_xml)

        text = controller._ctrl_v21_17_extract_docx_text(stream.getvalue())

        self.assertIn("Surface Stoichiometric Tuning", text)
        self.assertIn("ITO/SnO2/perovskite/Au", text)

    def test_aligned_landing_page_recovers_linked_docx_manuscript(self):
        import io
        import zipfile
        from urllib.parse import urljoin

        from bs4 import BeautifulSoup

        title = "Surface Stoichiometric Tuning for Perovskite Solar Cells"
        body = (
            title
            + "\nDOI 10.1002/example.123\n"
            + (
                "FASnI3 perovskite device FTO ETL HTL gold PCE stability "
                "light soaking nitrogen encapsulated. "
            )
            * 30
        )
        document_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body><w:p><w:r><w:t>{body}</w:t></w:r></w:p></w:body></w:document>"
        ).encode("utf-8")
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("word/document.xml", document_xml)
        docx = stream.getvalue()
        landing_url = "https://repo.example.org/item/123"
        manuscript_url = "https://repo.example.org/filesets/abc/download"
        landing = (
            f"<html><head><title>{title}</title></head><body>"
            f"<p>{title} DOI 10.1002/example.123 perovskite solar cells.</p>"
            f'<a href="{manuscript_url}">Author manuscript.docx</a>'
            "</body></html>"
        ).encode("utf-8")

        class FakeResponse:
            def __init__(self, content, content_type):
                self.status_code = 200
                self.content = content
                self.headers = {"content-type": content_type}
                self.encoding = "utf-8"
                self.text = content.decode("utf-8", errors="ignore")

        class FakeRequests:
            @staticmethod
            def get(url, **_kwargs):
                if url == landing_url:
                    return FakeResponse(landing, "text/html; charset=utf-8")
                if url == manuscript_url:
                    return FakeResponse(
                        docx,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                return SimpleNamespace(
                    status_code=404,
                    content=b"",
                    headers={"content-type": "text/plain"},
                    encoding="utf-8",
                    text="",
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = _fake_runtime(root)
            runtime.requests = FakeRequests()
            runtime.BeautifulSoup = BeautifulSoup
            runtime.urljoin = urljoin
            runtime.HEADERS = {}
            runtime.HTML_DIR = root / "html"
            controller._install_web_expansion_fulltext_fallback(
                runtime,
                _args(root),
            )

            result = runtime.try_full_text(
                {
                    "title": title,
                    "doi": "10.1002/example.123",
                    "landing_page": landing_url,
                },
                "docx_slug",
            )

            self.assertGreater(len(result["full_text"]), 1000)
            self.assertIn("FASnI3", result["full_text"])
            self.assertTrue(
                str(result["retrieval_saved_path"]).endswith(".docx")
            )

    def test_crossref_text_mining_links_are_preserved(self):
        urls = controller._ctrl_v21_15_crossref_fulltext_urls({
            "link": [
                {
                    "URL": "https://publisher.example/article.pdf",
                    "content-type": "application/pdf",
                    "intended-application": "text-mining",
                },
                {
                    "URL": "https://publisher.example/article.xml",
                    "content-type": "application/xml",
                    "intended-application": "text-mining",
                },
                {
                    "URL": "https://publisher.example/citations",
                    "content-type": "text/html",
                    "intended-application": "citation-list",
                },
            ],
        })

        self.assertEqual(
            urls,
            [
                "https://publisher.example/article.pdf",
                "https://publisher.example/article.xml",
            ],
        )

    def test_pdf_preflight_rejects_excessive_page_count(self):
        class FakeReader:
            def __init__(self, _stream):
                self.pages = [object()] * 81

        reason = controller._ctrl_v21_15_pdf_preflight_reason(
            b"%PDF-small",
            FakeReader,
            max_pdf_bytes=1000,
            max_pdf_pages=80,
        )

        self.assertEqual(reason, "pdf_too_many_pages:81>80")

    def test_pdf_preflight_allows_normal_extractor_after_reader_error(self):
        class BrokenReader:
            def __init__(self, _stream):
                raise ValueError("unsupported PDF feature")

        reason = controller._ctrl_v21_15_pdf_preflight_reason(
            b"%PDF-unusual",
            BrokenReader,
            max_pdf_bytes=1000,
            max_pdf_pages=80,
        )

        self.assertEqual(reason, "")

    def test_automated_retrieval_skips_duplicate_downloaded_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = _fake_runtime(root)
            runtime.PDF_DIR = root / "pdf"
            runtime.HTML_DIR = root / "html"
            runtime.PdfReader = lambda _stream: SimpleNamespace(pages=[object()])
            extraction_calls = []

            def extract_pdf_text_native(path):
                extraction_calls.append(str(path))
                return "unrelated source text " * 100

            class FakeResponse:
                status_code = 200
                content = b"%PDF-identical-content"
                headers = {"content-type": "application/pdf"}
                encoding = "utf-8"
                text = ""

            class FakeRequests:
                @staticmethod
                def get(_url, **_kwargs):
                    return FakeResponse()

            runtime.extract_pdf_text_native = extract_pdf_text_native
            runtime.requests = FakeRequests()
            controller._install_web_expansion_fulltext_fallback(
                runtime,
                _args(root),
            )

            runtime.try_full_text(
                {"title": "First", "pdf_url": "https://example.org/a.pdf"},
                "first_slug",
            )
            runtime.try_full_text(
                {"title": "Second", "pdf_url": "https://example.org/b.pdf"},
                "second_slug",
            )

            self.assertEqual(len(extraction_calls), 1)
            report = root / "retrieval_reports" / "retrieval_candidate_report.csv"
            with report.open("r", encoding="utf-8-sig", newline="") as handle:
                statuses = [
                    row["status"]
                    for row in csv.DictReader(handle)
                ]
            self.assertIn("duplicate_source_content_skipped", statuses)

    def test_expansion_merge_preserves_family_candidates_and_single_fallback_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _fake_runtime(Path(tmp))
            controller._install_web_expansion_fulltext_fallback(
                runtime,
                _args(Path(tmp)),
            )

            records = runtime.merge_record_lists(
                [{"JV_default_PCE": 20.0}],
                [{"JV_default_PCE": 22.0}],
            )
            self.assertEqual([record["JV_default_PCE"] for record in records], [20.0, 22.0])

            merged = runtime.merge_deterministic_with_llm_records(
                {
                    "JV_default_PCE": 21.5,
                    "Perovskite_composition_short_form": "FAPbI3",
                },
                [{"ETL_stack_sequence": "SnO2"}, {"HTL_stack_sequence": "Spiro"}],
                {},
            )
            self.assertEqual(merged[0]["JV_default_PCE"], 21.5)
            self.assertNotIn("JV_default_PCE", merged[1])
            self.assertEqual(
                merged[1]["Perovskite_composition_short_form"],
                "FAPbI3",
            )

    def test_distinct_grounded_devices_receive_unique_sample_ids(self):
        records = [
            {
                "Ref_DOI_number": "10.1002/example",
                "Ref_internal_sample_id": "paper_device_1",
                "JV_default_PCE": 21.61,
                "_evidence_JV_default_PCE": (
                    "The target PSC showed a PCE of 21.61%."
                ),
            },
            {
                "Ref_DOI_number": "10.1002/example",
                "Ref_internal_sample_id": "paper_device_1",
                "JV_default_PCE": 21.13,
                "_evidence_JV_default_PCE": (
                    "The control PSC showed a PCE of 21.13%."
                ),
            },
            {
                "Ref_DOI_number": "10.1002/example",
                "Ref_internal_sample_id": "paper_device_1",
                "JV_default_PCE": 19.12,
                "_evidence_JV_default_PCE": (
                    "The independently certified PCE was 19.12%."
                ),
            },
        ]

        assigned = controller._v21_17_assign_distinct_sample_ids(records)
        sample_ids = [row["Ref_internal_sample_id"] for row in assigned]

        self.assertEqual(len(set(sample_ids)), 3)
        self.assertTrue(any("_target_" in value for value in sample_ids))
        self.assertTrue(any("_control_" in value for value in sample_ids))
        self.assertTrue(any("_certified_" in value for value in sample_ids))

    def test_drive_mode_does_not_replace_legacy_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _fake_runtime(Path(tmp))
            original = runtime.merge_record_lists
            controller._install_web_expansion_fulltext_fallback(
                runtime,
                _args(Path(tmp), expand=False),
            )
            self.assertIs(runtime.merge_record_lists, original)

    def test_per_paper_eligibility_is_written_without_changing_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = _fake_runtime(root)
            controller._install_web_expansion_fulltext_fallback(runtime, _args(root))
            result = runtime.process_one_paper_api(
                {"title": "Device paper", "doi": "10.1234/example"},
                run_mode="expand",
            )

            self.assertEqual(len(result["records"]), 2)
            report = root / "model_eligibility" / "paper_model_eligibility.jsonl"
            payload = json.loads(report.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(payload["pce_target_rows"], 2)
            self.assertEqual(payload["pce_strict_rows"], 1)

    def test_candidate_accounting_deduplicates_doi(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "candidate_report.csv"
            with report.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["doi", "title", "allowed"])
                writer.writeheader()
                writer.writerows([
                    {"doi": "10.1/A", "title": "First", "allowed": 1},
                    {"doi": "https://doi.org/10.1/a", "title": "First again", "allowed": 1},
                    {"doi": "10.1/B", "title": "Second", "allowed": 0},
                ])
            all_keys = run_expansion_campaign._report_candidate_keys(report)
            allowed = run_expansion_campaign._report_candidate_keys(
                report,
                require_allowed=True,
            )
            self.assertEqual(len(all_keys), 2)
            self.assertEqual(len(allowed), 1)

    def test_batch_attempt_registry_deduplicates_routes_and_seeds_prior_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = root / "processed.csv"
            report = root / "retrieval.csv"
            registry = root / "batch_attempts.csv"
            registry_fields = [
                "doi",
                "title_key",
                "title",
                "paper_slug",
                "source_label",
                "first_seen",
                "last_seen",
            ]

            with seed.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=registry_fields)
                writer.writeheader()
                writer.writerow({
                    "doi": "10.1000/already",
                    "title_key": "already processed perovskite device",
                    "title": "Already processed perovskite device",
                    "paper_slug": "already",
                })

            with report.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["doi", "title", "slug", "source", "url", "ok"],
                )
                writer.writeheader()
                writer.writerow({
                    "doi": "10.1000/new",
                    "title": "A newly attempted perovskite solar cell",
                    "slug": "new-paper",
                    "source": "candidate_landing",
                    "url": "https://doi.org/10.1000/new",
                    "ok": "0",
                })
                writer.writerow({
                    "doi": "10.1000/new",
                    "title": "A newly attempted perovskite solar cell",
                    "slug": "new-paper",
                    "source": "crossref_fulltext",
                    "url": "https://example.org/new.pdf",
                    "ok": "0",
                })

            added = run_expansion_campaign.sync_batch_attempt_registry(
                retrieval_report=report,
                registry_csv=registry,
                seed_registry=seed,
                source_label="pilot:round_1",
            )

            with registry.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(added, 1)
            self.assertEqual(
                {row["doi"] for row in rows},
                {"10.1000/already", "10.1000/new"},
            )

    def test_openalex_locations_prioritize_repository_copy(self):
        urls = controller._ctrl_v21_18_openalex_location_urls({
            "primary_location": {
                "source": {"type": "journal"},
                "pdf_url": "https://publisher.example/article.pdf",
                "landing_page_url": "https://publisher.example/article",
            },
            "locations": [
                {
                    "source": {"type": "repository"},
                    "pdf_url": "https://repository.example/article.pdf",
                    "landing_page_url": "https://repository.example/article",
                },
            ],
        })

        self.assertEqual(urls[0][0], "openalex_repository_pdf")
        self.assertEqual(urls[1][0], "openalex_repository_landing")
        self.assertIn(("openalex_publisher_pdf", "https://publisher.example/article.pdf"), urls)

    def test_unpaywall_uses_all_locations_and_prioritizes_repository(self):
        urls = controller._ctrl_v21_18_unpaywall_location_urls({
            "best_oa_location": {
                "host_type": "publisher",
                "url_for_pdf": "https://publisher.example/article.pdf",
            },
            "oa_locations": [
                {
                    "host_type": "repository",
                    "url_for_pdf": "https://repository.example/manuscript.pdf",
                    "url_for_landing_page": "https://repository.example/item",
                },
            ],
        })

        self.assertEqual(urls[0][0], "unpaywall_repository_pdf")
        self.assertEqual(urls[1][0], "unpaywall_repository_landing")
        self.assertIn(("unpaywall_publisher_pdf", "https://publisher.example/article.pdf"), urls)

    def test_resolver_priority_prefers_repository_and_machine_readable_routes(self):
        routes = [
            ("doi_landing", "https://doi.org/10.1000/test"),
            ("crossref_fulltext", "https://publisher.example/article"),
            (
                "europepmc_repository_fulltext_xml",
                "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1/fullTextXML",
            ),
            ("openalex_repository_pdf", "https://repository.example/article.pdf"),
        ]

        ordered = sorted(
            routes,
            key=lambda item: controller._ctrl_v21_18_resolver_priority(
                item[0],
                item[1],
            ),
        )

        self.assertEqual(ordered[0][0], "openalex_repository_pdf")
        self.assertEqual(ordered[1][0], "europepmc_repository_fulltext_xml")
        self.assertEqual(ordered[-1][0], "doi_landing")

    def test_query_sweep_includes_broad_scale_variants_without_duplicates(self):
        plan = run_expansion_campaign.build_query_plan(
            target_mode="pce_stability",
            search_query=None,
            query_sweep=True,
            max_rounds=0,
        )

        normalized = [" ".join(query.lower().split()) for _, query in plan]
        self.assertGreater(len(plan), len(run_expansion_campaign.SWEEP_TARGET_MODES))
        self.assertEqual(len(normalized), len(set(normalized)))
        self.assertTrue(any(mode == "expanded_pce_inverted" for mode, _ in plan))
        self.assertTrue(any(mode == "expanded_stability_t80" for mode, _ in plan))

    def test_query_sweep_round_limit_applies_after_expansion(self):
        plan = run_expansion_campaign.build_query_plan(
            target_mode="pce_stability",
            search_query=None,
            query_sweep=True,
            max_rounds=17,
        )

        self.assertEqual(len(plan), 17)

    def test_sample_identity_collapses_duplicate_final_jv_rows(self):
        base = "paper_device_1"
        records = [
            {
                "Ref_DOI_number": "10.1000/device",
                "Ref_internal_sample_id": f"{base}_control_pce_14p82",
                "Perovskite_composition_long_form": "CsPbI2Br",
                "Cell_stack_sequence": "ITO/NiOx/CsPbI2Br/PCBM/Ag",
                "JV_default_PCE": 15.44,
                "JV_default_Voc": 1.15,
                "JV_default_Jsc": 16.34,
                "JV_default_FF": 0.8215,
                "JV_hysteresis_index": 9.84,
                "_evidence_JV_hysteresis_index": (
                    "The control device exhibits a hysteresis index of 9.84%."
                ),
                "_evidence_JV_default_PCE": (
                    "The optimized passivated device achieves a champion PCE of 15.44%."
                ),
            },
            {
                "Ref_DOI_number": "10.1000/device",
                "Ref_internal_sample_id": f"{base}_control_pce_15p86",
                "Perovskite_composition_long_form": "CsPbI2Br",
                "Cell_stack_sequence": "ITO/NiOx/CsPbI2Br/PCBM/Ag",
                "JV_default_PCE": 15.44,
                "JV_default_Voc": 1.15,
                "JV_default_Jsc": 16.34,
                "JV_default_FF": 0.8215,
                "JV_hysteresis_index": 1.55,
                "_evidence_JV_hysteresis_index": (
                    "Upon asymmetric dual-interface passivation the index is 1.55%."
                ),
                "_evidence_JV_default_PCE": (
                    "The optimized passivated device achieves a champion PCE of 15.44%."
                ),
            },
            {
                "Ref_DOI_number": "10.1000/device",
                "Ref_internal_sample_id": f"{base}_control_pce_13p48",
                "Perovskite_composition_long_form": "CsPbI2Br",
                "Cell_stack_sequence": "ITO/NiOx/PEAI/CsPbI2Br/PCBM/Ag",
                "JV_default_PCE": 13.48,
                "_evidence_JV_default_PCE": (
                    "The bottom-interface passivated device reached a PCE of 13.48%."
                ),
            },
        ]

        cleaned = controller._v21_17_assign_distinct_sample_ids(records)

        self.assertEqual(len(cleaned), 2)
        champion = next(row for row in cleaned if row["JV_default_PCE"] == 15.44)
        self.assertEqual(champion["JV_hysteresis_index"], 1.55)
        self.assertIn("_target_pce_15p44", champion["Ref_internal_sample_id"])
        self.assertIn("Collapsed 2 duplicate", champion["_lit_agent_sample_identity_warning"])

    def test_batch_audit_rejects_sample_id_pce_mismatch_and_duplicate_identity(self):
        frame = pd.DataFrame([
            {
                "Ref_DOI_number": "10.1000/device",
                "Ref_internal_sample_id": "paper_device_1_target_pce_14p82",
                "JV_default_PCE": 15.44,
                "JV_default_Voc": 1.15,
                "JV_default_Jsc": 16.34,
                "JV_default_FF": 0.8215,
                "Perovskite_composition_long_form": "CsPbI2Br",
                "Cell_stack_sequence": "ITO/NiOx/CsPbI2Br/PCBM/Ag",
            },
            {
                "Ref_DOI_number": "10.1000/device",
                "Ref_internal_sample_id": "paper_device_1_target_pce_15p86",
                "JV_default_PCE": 15.44,
                "JV_default_Voc": 1.15,
                "JV_default_Jsc": 16.34,
                "JV_default_FF": 0.8215,
                "Perovskite_composition_long_form": "CsPbI2Br",
                "Cell_stack_sequence": "ITO/NiOx/CsPbI2Br/PCBM/Ag",
            },
        ])

        issues = audit_expansion_batch.identity_consistency_issues(frame)

        self.assertEqual(
            set(issues["issue"]),
            {"sample_id_pce_mismatch", "duplicate_final_device_identity"},
        )
        self.assertEqual(
            int((issues["issue"] == "sample_id_pce_mismatch").sum()),
            2,
        )


if __name__ == "__main__":
    unittest.main()
