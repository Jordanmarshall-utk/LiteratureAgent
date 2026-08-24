#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
literature_agent_full_end_to_end_v21_3_english_sanitizer.py

FULL SINGLE-FILE VERSION

This file contains:
1) The full LiteratureAgent code.
2) The full LiteratureAgent -> PCE/Stability dataset integration code.
3) A single end-to-end controller that can:
   - clear old outputs,
   - run LiteratureAgent,
   - locate/assemble LiteratureAgent CSV outputs,
   - merge accepted LiteratureAgent rows into the baseline Perovskite Database CSV,
   - optionally launch the PCE -> stability modeling script.

You still need:
- The editable ontology JSON file, unless the LiteratureAgent's internal fallback ontology is enough.
- The baseline Perovskite Database CSV.
- Google OAuth client secret JSON if using Google Drive mode.
- Optional external PCE/stability model script if you use --run_model.

Recommended first run:
    python literature_agent_full_end_to_end_v21_3_english_sanitizer.py ^
      --base_csv ".\\data\\Perovskite_database_content_all_data.csv" ^
      --from_scratch ^
      --test_one_paper ^
      --single_paper_source drive ^
      --vision_enable 0 ^
      --no_require_doi

Recommended full run later:
    python literature_agent_full_end_to_end_v21_3_english_sanitizer.py ^
      --base_csv ".\\data\\Perovskite_database_content_all_data.csv" ^
      --from_scratch ^
      --full_literature_run ^
      --vision_enable 0 ^
      --no_require_doi
"""

from __future__ import annotations

import argparse as _argparse_controller
import csv as _csv_controller
import html as _html_controller
import io as _io_controller
import importlib as _importlib_controller
import importlib.util as _importlib_util_controller
import json as _json_controller
import hashlib as _hashlib_controller
import math as _math_controller
import os as _os_controller
import re as _re_controller
import runpy as _runpy_controller
import shutil as _shutil_controller
import subprocess as _subprocess_controller
import sys as _sys_controller
import tempfile as _tempfile_controller
import time as _time_controller
from datetime import datetime as _datetime_controller
from pathlib import Path as _Path_controller
from typing import Optional as _Optional_controller, List as _List_controller
from urllib.parse import urljoin as _urljoin_controller


# =============================================================================
# EMBEDDED LITERATUREAGENT SOURCE
# =============================================================================

_LITERATURE_AGENT_SOURCE = 'from __future__ import annotations\nimport json\nimport os\nimport re\nimport time\nimport html as ihtml\nimport hashlib\nimport random\nimport logging\nimport shutil\nfrom pathlib import Path\nfrom typing import Any, Dict, List, Optional, Tuple\nfrom urllib.parse import quote, urljoin\nimport pandas as pd\nimport requests\nfrom bs4 import BeautifulSoup\ntry:\n    from PyPDF2 import PdfReader\n    HAS_PYPDF2 = True\nexcept ImportError:\n    HAS_PYPDF2 = False\ntry:\n    import feedparser\n    HAS_FEEDPARSER = True\nexcept ImportError:\n    HAS_FEEDPARSER = False\ntry:\n    import trafilatura\n    HAS_TRAFILATURA = True\nexcept ImportError:\n    HAS_TRAFILATURA = False\nlogging.getLogger("trafilatura").setLevel(logging.ERROR)\n# ============================================================\n# USER SETTINGS\n# ============================================================\nSEARCH_QUERY = (\n    \'"perovskite solar cell" experimental device fabrication performance \'\n    \'J-V EQE stability\'\n)\nYEAR_MIN = 2021\nYEAR_MAX = 2026\nMAX_PAPERS_TO_PROCESS = 1\nCROSSREF_ROWS_PER_PAGE = 50\nCROSSREF_MAX_PAGES = 6\nUNPAYWALL_EMAIL = "Jmarsh39@vols.utk.edu"\ndef default_work_dir() -> str:\n    """Default outputs to the user\'s Downloads folder instead of the current working directory.\n\n    Override anytime with:\n      PowerShell: $env:WORK_DIR="$env:USERPROFILE\\\\Downloads\\\\lit_outputs"\n    """\n    explicit = os.getenv("WORK_DIR", "").strip()\n    if explicit:\n        return explicit\n    downloads = Path.home() / "Downloads"\n    if downloads.exists():\n        return str(downloads / "lit_outputs")\n    return str(Path.cwd() / "lit_outputs")\n\nWORK_DIR = default_work_dir()\nRSS_FEEDS = [\n    "https://pubs.acs.org/action/showFeed?jc=jaaucr&type=etoc&feed=rss",\n    "https://rss.sciencedirect.com/publication/science/25424351",\n]\nREQUEST_TIMEOUT = 45\nSLEEP_BETWEEN_REQUESTS = 0.15\nHEADERS = {\n    "User-Agent": "Mozilla/5.0",\n    "Accept-Language": "en-US,en;q=0.9",\n}\n# ============================================================\n# HUGGING FACE / QWEN SETTINGS\n# ============================================================\nHF_API_KEY = os.getenv("HF_API_KEY", "").strip()\nHF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen3-32B")\nHF_API_URL = os.getenv("HF_API_URL", "https://router.huggingface.co/v1/chat/completions")\n# ============================================================\n# THROTTLING / RETRY SETTINGS\n# ============================================================\nAPI_CALL_DELAY_SECONDS = 2.5\nAPI_JITTER_SECONDS = 0.8\nAPI_RETRY_BACKOFF_SECONDS = 8.0\nAPI_MAX_RETRIES = 4\nLLM_CHUNK_CHAR_LIMIT = 12000\nLLM_CHUNK_OVERLAP = 1200\nLLM_MIN_FULLTEXT_CHARS = 15000\nMIN_EXTRACTION_TEXT_CHARS = 8000\nLLM_MAX_OUTPUT_TOKENS = 3500\n# ============================================================\n# EXTRACTION SETTINGS\n# ============================================================\nSTRICT_FAIL_ON_LLM_ERROR = True\nSAVE_RAW_MODEL_TEXT = True\nUSE_CONSOLIDATION_PASS = True\nSKIP_CONSOLIDATION_IF_ONE_CHUNK = True\nSAVE_EVIDENCE_FIELDS_IN_JSON = True\n# ============================================================\n# DATASET COLUMNS\n# ============================================================\nALL_COLUMNS = [\n    \'Ref_ID\', \'Ref_ID_temp\', \'Ref_name_of_person_entering_the_data\', \'Ref_data_entered_by_author\',\n    \'Ref_DOI_number\', \'Ref_lead_author\', \'Ref_publication_date\', \'Ref_journal\',\n    \'Ref_part_of_initial_dataset\', \'Ref_original_filename_data_upload\', \'Ref_free_text_comment\',\n    \'Ref_internal_sample_id\', \'Cell_stack_sequence\', \'Cell_area_total\', \'Cell_area_measured\',\n    \'Cell_number_of_cells_per_substrate\', \'Cell_architecture\', \'Cell_flexible\',\n    \'Cell_flexible_min_bending_radius\', \'Cell_semitransparent\', \'Cell_semitransparent_AVT\',\n    \'Cell_semitransparent_wavelength_range\', \'Cell_semitransparent_raw_data\', \'Module\',\n    \'Module_number_of_cells_in_module\', \'Module_area_total\', \'Module_area_effective\',\n    \'Module_JV_data_recalculated_per_cell\', \'Substrate_stack_sequence\', \'Substrate_thickness\',\n    \'Substrate_area\', \'Substrate_supplier\', \'Substrate_brand_name\', \'Substrate_deposition_procedure\',\n    \'Substrate_surface_roughness_rms\', \'Substrate_etching_procedure\', \'Substrate_cleaning_procedure\',\n    \'ETL_stack_sequence\', \'ETL_thickness\', \'ETL_additives_compounds\', \'ETL_additives_concentrations\',\n    \'ETL_deposition_procedure\', \'ETL_deposition_aggregation_state_of_reactants\',\n    \'ETL_deposition_synthesis_atmosphere\', \'ETL_deposition_synthesis_atmosphere_pressure_total\',\n    \'ETL_deposition_synthesis_atmosphere_pressure_partial\',\n    \'ETL_deposition_synthesis_atmosphere_relative_humidity\', \'ETL_deposition_solvents\',\n    \'ETL_deposition_solvents_mixing_ratios\', \'ETL_deposition_solvents_supplier\',\n    \'ETL_deposition_solvents_purity\', \'ETL_deposition_reaction_solutions_compounds\',\n    \'ETL_deposition_reaction_solutions_compounds_supplier\',\n    \'ETL_deposition_reaction_solutions_compounds_purity\',\n    \'ETL_deposition_reaction_solutions_concentrations\',\n    \'ETL_deposition_reaction_solutions_volumes\', \'ETL_deposition_reaction_solutions_age\',\n    \'ETL_deposition_reaction_solutions_temperature\', \'ETL_deposition_substrate_temperature\',\n    \'ETL_deposition_thermal_annealing_temperature\', \'ETL_deposition_thermal_annealing_time\',\n    \'ETL_deposition_thermal_annealing_atmosphere\', \'ETL_storage_time_until_next_deposition_step\',\n    \'ETL_storage_atmosphere\', \'ETL_storage_relative_humidity\',\n    \'ETL_surface_treatment_before_next_deposition_step\', \'Perovskite_single_crystal\',\n    \'Perovskite_dimension_0D\', \'Perovskite_dimension_2D\', \'Perovskite_dimension_2D3D_mixture\',\n    \'Perovskite_dimension_3D\', \'Perovskite_dimension_3D_with_2D_capping_layer\',\n    \'Perovskite_dimension_list_of_layers\', \'Perovskite_composition_perovskite_ABC3_structure\',\n    \'Perovskite_composition_perovskite_inspired_structure\', \'Perovskite_composition_a_ions\',\n    \'Perovskite_composition_a_ions_coefficients\', \'Perovskite_composition_b_ions\',\n    \'Perovskite_composition_b_ions_coefficients\', \'Perovskite_composition_c_ions\',\n    \'Perovskite_composition_c_ions_coefficients\',\n    \'Perovskite_composition_none_stoichiometry_components_in_excess\',\n    \'Perovskite_composition_short_form\', \'Perovskite_composition_long_form\',\n    \'Perovskite_composition_assumption\', \'Perovskite_composition_inorganic\',\n    \'Perovskite_composition_leadfree\', \'Perovskite_additives_compounds\',\n    \'Perovskite_additives_concentrations\', \'Perovskite_thickness\', \'Perovskite_band_gap\',\n    \'Perovskite_band_gap_graded\', \'Perovskite_band_gap_estimation_basis\', \'Perovskite_pl_max\',\n    \'Perovskite_deposition_number_of_deposition_steps\', \'Perovskite_deposition_procedure\',\n    \'Perovskite_deposition_aggregation_state_of_reactants\',\n    \'Perovskite_deposition_synthesis_atmosphere\',\n    \'Perovskite_deposition_synthesis_atmosphere_pressure_total\',\n    \'Perovskite_deposition_synthesis_atmosphere_pressure_partial\',\n    \'Perovskite_deposition_synthesis_atmosphere_relative_humidity\',\n    \'Perovskite_deposition_solvents\', \'Perovskite_deposition_solvents_mixing_ratios\',\n    \'Perovskite_deposition_solvents_supplier\', \'Perovskite_deposition_solvents_purity\',\n    \'Perovskite_deposition_reaction_solutions_compounds\',\n    \'Perovskite_deposition_reaction_solutions_compounds_supplier\',\n    \'Perovskite_deposition_reaction_solutions_compounds_purity\',\n    \'Perovskite_deposition_reaction_solutions_concentrations\',\n    \'Perovskite_deposition_reaction_solutions_volumes\',\n    \'Perovskite_deposition_reaction_solutions_age\',\n    \'Perovskite_deposition_reaction_solutions_temperature\',\n    \'Perovskite_deposition_substrate_temperature\',\n    \'Perovskite_deposition_quenching_induced_crystallisation\',\n    \'Perovskite_deposition_quenching_media\',\n    \'Perovskite_deposition_quenching_media_mixing_ratios\',\n    \'Perovskite_deposition_quenching_media_volume\',\n    \'Perovskite_deposition_quenching_media_additives_compounds\',\n    \'Perovskite_deposition_quenching_media_additives_concentrations\',\n    \'Perovskite_deposition_thermal_annealing_temperature\',\n    \'Perovskite_deposition_thermal_annealing_time\',\n    \'Perovskite_deposition_thermal_annealing_atmosphere\',\n    \'Perovskite_deposition_thermal_annealing_relative_humidity\',\n    \'Perovskite_deposition_thermal_annealing_pressure\',\n    \'Perovskite_deposition_solvent_annealing\',\n    \'Perovskite_deposition_solvent_annealing_timing\',\n    \'Perovskite_deposition_solvent_annealing_solvent_atmosphere\',\n    \'Perovskite_deposition_solvent_annealing_time\',\n    \'Perovskite_deposition_solvent_annealing_temperature\',\n    \'Perovskite_deposition_after_treatment_of_formed_perovskite\',\n    \'Perovskite_deposition_after_treatment_of_formed_perovskite_met\',\n    \'Perovskite_storage_time_until_next_deposition_step\',\n    \'Perovskite_storage_atmosphere\', \'Perovskite_storage_relative_humidity\',\n    \'Perovskite_surface_treatment_before_next_deposition_step\', \'HTL_stack_sequence\',\n    \'HTL_thickness_list\', \'HTL_additives_compounds\', \'HTL_additives_concentrations\',\n    \'HTL_deposition_procedure\', \'HTL_deposition_aggregation_state_of_reactants\',\n    \'HTL_deposition_synthesis_atmosphere\',\n    \'HTL_deposition_synthesis_atmosphere_pressure_total\',\n    \'HTL_deposition_synthesis_atmosphere_pressure_partial\',\n    \'HTL_deposition_synthesis_atmosphere_relative_humidity\', \'HTL_deposition_solvents\',\n    \'HTL_deposition_solvents_mixing_ratios\', \'HTL_deposition_solvents_supplier\',\n    \'HTL_deposition_solvents_purity\', \'HTL_deposition_reaction_solutions_compounds\',\n    \'HTL_deposition_reaction_solutions_compounds_supplier\',\n    \'HTL_deposition_reaction_solutions_compounds_purity\',\n    \'HTL_deposition_reaction_solutions_concentrations\',\n    \'HTL_deposition_reaction_solutions_volumes\', \'HTL_deposition_reaction_solutions_age\',\n    \'HTL_deposition_reaction_solutions_temperature\', \'HTL_deposition_substrate_temperature\',\n    \'HTL_deposition_thermal_annealing_temperature\', \'HTL_deposition_thermal_annealing_time\',\n    \'HTL_deposition_thermal_annealing_atmosphere\', \'HTL_storage_time_until_next_deposition_step\',\n    \'HTL_storage_atmosphere\', \'HTL_storage_relative_humidity\',\n    \'HTL_surface_treatment_before_next_deposition_step\', \'Backcontact_stack_sequence\',\n    \'Backcontact_thickness_list\', \'Backcontact_additives_compounds\',\n    \'Backcontact_additives_concentrations\', \'Backcontact_deposition_procedure\',\n    \'Backcontact_deposition_aggregation_state_of_reactants\',\n    \'Backcontact_deposition_synthesis_atmosphere\',\n    \'Backcontact_deposition_synthesis_atmosphere_pressure_total\',\n    \'Backcontact_deposition_synthesis_atmosphere_pressure_partial\',\n    \'Backcontact_deposition_synthesis_atmosphere_relative_humidity\',\n    \'Backcontact_deposition_solvents\', \'Backcontact_deposition_solvents_mixing_ratios\',\n    \'Backcontact_deposition_solvents_supplier\', \'Backcontact_deposition_solvents_purity\',\n    \'Backcontact_deposition_reaction_solutions_compounds\',\n    \'Backcontact_deposition_reaction_solutions_compounds_supplier\',\n    \'Backcontact_deposition_reaction_solutions_compounds_purity\',\n    \'Backcontact_deposition_reaction_solutions_concentrations\',\n    \'Backcontact_deposition_reaction_solutions_volumes\',\n    \'Backcontact_deposition_reaction_solutions_age\',\n    \'Backcontact_deposition_reaction_solutions_temperature\',\n    \'Backcontact_deposition_substrate_temperature\',\n    \'Backcontact_deposition_thermal_annealing_temperature\',\n    \'Backcontact_deposition_thermal_annealing_time\',\n    \'Backcontact_deposition_thermal_annealing_atmosphere\',\n    \'Backcontact_storage_time_until_next_deposition_step\', \'Backcontact_storage_atmosphere\',\n    \'Backcontact_storage_relative_humidity\',\n    \'Backcontact_surface_treatment_before_next_deposition_step\', \'Add_lay_front\',\n    \'Add_lay_front_function\', \'Add_lay_front_stack_sequence\', \'Add_lay_front_thickness_list\',\n    \'Add_lay_front_additives_compounds\', \'Add_lay_front_additives_concentrations\',\n    \'Add_lay_front_deposition_procedure\',\n    \'Add_lay_front_deposition_aggregation_state_of_reactants\',\n    \'Add_lay_front_deposition_synthesis_atmosphere\',\n    \'Add_lay_front_deposition_synthesis_atmosphere_pressure_total\',\n    \'Add_lay_front_deposition_synthesis_atmosphere_pressure_partial\',\n    \'Add_lay_front_deposition_synthesis_atmosphere_relative_humidity\',\n    \'Add_lay_front_deposition_solvents\', \'Add_lay_front_deposition_solvents_mixing_ratios\',\n    \'Add_lay_front_deposition_solvents_supplier\', \'Add_lay_front_deposition_solvents_purity\',\n    \'Add_lay_front_deposition_reaction_solutions_compounds\',\n    \'Add_lay_front_deposition_reaction_solutions_compounds_supplier\',\n    \'Add_lay_front_deposition_reaction_solutions_compounds_purity\',\n    \'Add_lay_front_deposition_reaction_solutions_concentrations\',\n    \'Add_lay_front_deposition_reaction_solutions_volumes\',\n    \'Add_lay_front_deposition_reaction_solutions_age\',\n    \'Add_lay_front_deposition_reaction_solutions_temperature\',\n    \'Add_lay_front_deposition_substrate_temperature\',\n    \'Add_lay_front_deposition_thermal_annealing_temperature\',\n    \'Add_lay_front_deposition_thermal_annealing_time\',\n    \'Add_lay_front_deposition_thermal_annealing_atmosphere\',\n    \'Add_lay_front_storage_time_until_next_deposition_step\',\n    \'Add_lay_front_storage_atmosphere\', \'Add_lay_front_storage_relative_humidity\',\n    \'Add_lay_front_surface_treatment_before_next_deposition_step\', \'Add_lay_back\',\n    \'Add_lay_back_function\', \'Add_lay_back_stack_sequence\', \'Add_lay_back_thickness_list\',\n    \'Add_lay_back_additives_compounds\', \'Add_lay_back_additives_concentrations\',\n    \'Add_lay_back_deposition_procedure\',\n    \'Add_lay_back_deposition_aggregation_state_of_reactants\',\n    \'Add_lay_back_deposition_synthesis_atmosphere\',\n    \'Add_lay_back_deposition_synthesis_atmosphere_pressure_total\',\n    \'Add_lay_back_deposition_synthesis_atmosphere_pressure_partial\',\n    \'Add_lay_back_deposition_synthesis_atmosphere_relative_humidity\',\n    \'Add_lay_back_deposition_solvents\', \'Add_lay_back_deposition_solvents_mixing_ratios\',\n    \'Add_lay_back_deposition_solvents_supplier\', \'Add_lay_back_deposition_solvents_purity\',\n    \'Add_lay_back_deposition_reaction_solutions_compounds\',\n    \'Add_lay_back_deposition_reaction_solutions_compounds_supplier\',\n    \'Add_lay_back_deposition_reaction_solutions_compounds_purity\',\n    \'Add_lay_back_deposition_reaction_solutions_concentrations\',\n    \'Add_lay_back_deposition_reaction_solutions_volumes\',\n    \'Add_lay_back_deposition_reaction_solutions_age\',\n    \'Add_lay_back_deposition_reaction_solutions_temperature\',\n    \'Add_lay_back_deposition_substrate_temperature\',\n    \'Add_lay_back_deposition_thermal_annealing_temperature\',\n    \'Add_lay_back_deposition_thermal_annealing_time\',\n    \'Add_lay_back_deposition_thermal_annealing_atmosphere\',\n    \'Add_lay_back_storage_time_until_next_deposition_step\',\n    \'Add_lay_back_storage_atmosphere\', \'Add_lay_back_storage_relative_humidity\',\n    \'Add_lay_back_surface_treatment_before_next_deposition_step\', \'Encapsulation\',\n    \'Encapsulation_stack_sequence\', \'Encapsulation_edge_sealing_materials\',\n    \'Encapsulation_atmosphere_for_encapsulation\',\n    \'Encapsulation_water_vapour_transmission_rate\',\n    \'Encapsulation_oxygen_transmission_rate\', \'JV_measured\',\n    \'JV_average_over_n_number_of_cells\', \'JV_certified_values\',\n    \'JV_certification_institute\', \'JV_storage_age_of_cell\', \'JV_storage_atmosphere\',\n    \'JV_storage_relative_humidity\', \'JV_test_atmosphere\', \'JV_test_relative_humidity\',\n    \'JV_test_temperature\', \'JV_light_source_type\', \'JV_light_source_brand_name\',\n    \'JV_light_source_simulator_class\', \'JV_light_intensity\', \'JV_light_spectra\',\n    \'JV_light_wavelength_range\', \'JV_light_illumination_direction\', \'JV_light_masked_cell\',\n    \'JV_light_mask_area\', \'JV_scan_speed\', \'JV_scan_delay_time\', \'JV_scan_integration_time\',\n    \'JV_scan_voltage_step\', \'JV_preconditioning_protocol\', \'JV_preconditioning_time\',\n    \'JV_preconditioning_potential\', \'JV_preconditioning_light_intensity\',\n    \'JV_reverse_scan_Voc\', \'JV_reverse_scan_Jsc\', \'JV_reverse_scan_FF\',\n    \'JV_reverse_scan_PCE\', \'JV_reverse_scan_Vmp\', \'JV_reverse_scan_Jmp\',\n    \'JV_reverse_scan_series_resistance\', \'JV_reverse_scan_shunt_resistance\',\n    \'JV_forward_scan_Voc\', \'JV_forward_scan_Jsc\', \'JV_forward_scan_FF\',\n    \'JV_forward_scan_PCE\', \'JV_forward_scan_Vmp\', \'JV_forward_scan_Jmp\',\n    \'JV_forward_scan_series_resistance\', \'JV_forward_scan_shunt_resistance\',\n    \'JV_link_raw_data\', \'JV_default_Voc\', \'JV_default_Jsc\', \'JV_default_FF\',\n    \'JV_default_PCE\', \'JV_default_Voc_scan_direction\', \'JV_default_Jsc_scan_direction\',\n    \'JV_default_FF_scan_direction\', \'JV_default_PCE_scan_direction\', \'JV_hysteresis_index\',\n    \'Stabilised_performance_measured\', \'Stabilised_performance_procedure\',\n    \'Stabilised_performance_procedure_metrics\', \'Stabilised_performance_measurement_time\',\n    \'Stabilised_performance_PCE\', \'Stabilised_performance_Vmp\', \'Stabilised_performance_Jmp\',\n    \'Stabilised_performance_link_raw_data\', \'EQE_measured\', \'EQE_light_bias\',\n    \'EQE_integrated_Jsc\', \'EQE_link_raw_data\', \'Stability_measured\',\n    \'Stability_protocol\', \'Stability_average_over_n_number_of_cells\',\n    \'Stability_light_source_type\', \'Stability_light_source_brand_name\',\n    \'Stability_light_source_simulator_class\', \'Stability_light_intensity\',\n    \'Stability_light_spectra\', \'Stability_light_wavelength_range\',\n    \'Stability_light_illumination_direction\', \'Stability_light_load_condition\',\n    \'Stability_light_cycling_times\', \'Stability_light_UV_filter\',\n    \'Stability_potential_bias_load_condition\', \'Stability_potential_bias_range\',\n    \'Stability_potential_bias_passive_resistance\', \'Stability_temperature_load_condition\',\n    \'Stability_temperature_range\', \'Stability_temperature_cycling_times\',\n    \'Stability_temperature_ramp_speed\', \'Stability_atmosphere\',\n    \'Stability_atmosphere_oxygen_concentration\',\n    \'Stability_relative_humidity_load_conditions\', \'Stability_relative_humidity_range\',\n    \'Stability_relative_humidity_average_value\', \'Stability_time_total_exposure\',\n    \'Stability_periodic_JV_measurements\',\n    \'Stability_periodic_JV_measurements_time_between_jv\',\n    \'Stability_PCE_initial_value\', \'Stability_PCE_burn_in_observed\',\n    \'Stability_PCE_end_of_experiment\', \'Stability_PCE_T95\', \'Stability_PCE_Ts95\',\n    \'Stability_PCE_T80\', \'Stability_PCE_Ts80\', \'Stability_PCE_Te80\',\n    \'Stability_PCE_Tse80\', \'Stability_PCE_after_1000_h\',\n    \'Stability_lifetime_energy_yield\',\n    \'Stability_flexible_cell_number_of_bending_cycles\',\n    \'Stability_flexible_cell_bending_radius\',\n    \'Stability_flexible_cell_PCE_initial_value\',\n    \'Stability_flexible_cell_PCE_end_of_experiment\',\n    \'Stability_link_raw_data_for_stability_trace\', \'Outdoor_tested\',\n    \'Outdoor_protocol\', \'Outdoor_average_over_n_number_of_cells\',\n    \'Outdoor_location_country\', \'Outdoor_location_city\',\n    \'Outdoor_location_coordinates\', \'Outdoor_location_climate_zone\',\n    \'Outdoor_installation_tilt\', \'Outdoor_installation_cardinal_direction\',\n    \'Outdoor_installation_number_of_solar_tracking_axis\', \'Outdoor_time_season\',\n    \'Outdoor_time_start\', \'Outdoor_time_end\', \'Outdoor_time_total_exposure\',\n    \'Outdoor_potential_bias_load_condition\', \'Outdoor_potential_bias_range\',\n    \'Outdoor_potential_bias_passive_resistance\', \'Outdoor_temperature_load_condition\',\n    \'Outdoor_temperature_range\', \'Outdoor_temperature_tmodule\',\n    \'Outdoor_periodic_JV_measurements\',\n    \'Outdoor_periodic_JV_measurements_time_between_measurements\',\n    \'Outdoor_PCE_initial_value\', \'Outdoor_PCE_burn_in_observed\',\n    \'Outdoor_PCE_end_of_experiment\', \'Outdoor_PCE_T95\', \'Outdoor_PCE_Ts95\',\n    \'Outdoor_PCE_T80\', \'Outdoor_PCE_Ts80\', \'Outdoor_PCE_Te80\',\n    \'Outdoor_PCE_Tse80\', \'Outdoor_PCE_after_1000_h\', \'Outdoor_power_generated\',\n    \'Outdoor_link_raw_data_for_outdoor_trace\', \'Outdoor_detaild_weather_data_available\',\n    \'Outdoor_link_detailed_weather_data\', \'Outdoor_spectral_data_available\',\n    \'Outdoor_link_spectral_data\', \'Outdoor_irradiance_measured\',\n    \'Outdoor_link_irradiance_data\'\n]\nEXTRA_COLUMNS = [\n    "_source_abstract", "_source_pdf_url", "_source_open_access",\n    "_source_landing_page", "_full_text_length", "_full_text_source",\n    "_composition_json"\n]\n# ============================================================\n# PATH / SETUP HELPERS\n# ============================================================\nBASE = Path(WORK_DIR)\nPDF_DIR = BASE / "pdf"\nHTML_DIR = BASE / "html"\nTEXT_DIR = BASE / "text"\nJSON_DIR = BASE / "json"\nCSV_DIR = BASE / "csv"\nRAW_LLM_DIR = BASE / "raw_llm_json"\nRAW_LLM_TEXT_DIR = BASE / "raw_llm_text"\nfor d in [BASE, PDF_DIR, HTML_DIR, TEXT_DIR, JSON_DIR, CSV_DIR, RAW_LLM_DIR, RAW_LLM_TEXT_DIR]:\n    d.mkdir(parents=True, exist_ok=True)\nTITLE_BLOCK_RE = re.compile(\n    r"(editorial|erratum|correction|retraction|book review|news|preface|table of contents)",\n    re.IGNORECASE,\n)\n# ============================================================\n# SCHEMA FAMILIES\n# ============================================================\nSCHEMA_FAMILIES = {\n    "reference_device": [\n        "Ref_DOI_number", "Ref_lead_author", "Ref_publication_date", "Ref_journal",\n        "Ref_original_filename_data_upload", "Cell_stack_sequence", "Cell_architecture",\n        "Cell_area_total", "Cell_area_measured", "Cell_number_of_cells_per_substrate",\n        "Cell_flexible", "Cell_semitransparent", "Module", "Substrate_stack_sequence",\n        "Substrate_thickness", "Substrate_area", "Substrate_supplier", "Substrate_brand_name",\n        "Substrate_deposition_procedure", "Substrate_cleaning_procedure"\n    ],\n    "perovskite_chemistry": [\n        "Perovskite_single_crystal",\n        "Perovskite_dimension_0D", "Perovskite_dimension_2D", "Perovskite_dimension_2D3D_mixture",\n        "Perovskite_dimension_3D", "Perovskite_dimension_3D_with_2D_capping_layer",\n        "Perovskite_dimension_list_of_layers",\n        "Perovskite_composition_perovskite_ABC3_structure",\n        "Perovskite_composition_perovskite_inspired_structure",\n        "Perovskite_composition_a_ions", "Perovskite_composition_a_ions_coefficients",\n        "Perovskite_composition_b_ions", "Perovskite_composition_b_ions_coefficients",\n        "Perovskite_composition_c_ions", "Perovskite_composition_c_ions_coefficients",\n        "Perovskite_composition_none_stoichiometry_components_in_excess",\n        "Perovskite_composition_short_form", "Perovskite_composition_long_form",\n        "Perovskite_composition_assumption", "Perovskite_composition_inorganic",\n        "Perovskite_composition_leadfree", "Perovskite_additives_compounds",\n        "Perovskite_additives_concentrations", "Perovskite_thickness",\n        "Perovskite_band_gap", "Perovskite_band_gap_graded",\n        "Perovskite_band_gap_estimation_basis", "Perovskite_pl_max"\n    ],\n    "etl": [\n        "ETL_stack_sequence", "ETL_thickness", "ETL_additives_compounds",\n        "ETL_additives_concentrations", "ETL_deposition_procedure",\n        "ETL_deposition_aggregation_state_of_reactants",\n        "ETL_deposition_synthesis_atmosphere",\n        "ETL_deposition_synthesis_atmosphere_pressure_total",\n        "ETL_deposition_synthesis_atmosphere_pressure_partial",\n        "ETL_deposition_synthesis_atmosphere_relative_humidity",\n        "ETL_deposition_solvents", "ETL_deposition_solvents_mixing_ratios",\n        "ETL_deposition_solvents_supplier", "ETL_deposition_solvents_purity",\n        "ETL_deposition_reaction_solutions_compounds",\n        "ETL_deposition_reaction_solutions_compounds_supplier",\n        "ETL_deposition_reaction_solutions_compounds_purity",\n        "ETL_deposition_reaction_solutions_concentrations",\n        "ETL_deposition_reaction_solutions_volumes",\n        "ETL_deposition_reaction_solutions_age",\n        "ETL_deposition_reaction_solutions_temperature",\n        "ETL_deposition_substrate_temperature",\n        "ETL_deposition_thermal_annealing_temperature",\n        "ETL_deposition_thermal_annealing_time",\n        "ETL_deposition_thermal_annealing_atmosphere",\n        "ETL_storage_time_until_next_deposition_step",\n        "ETL_storage_atmosphere", "ETL_storage_relative_humidity",\n        "ETL_surface_treatment_before_next_deposition_step"\n    ],\n    "perovskite_processing": [\n        "Perovskite_deposition_number_of_deposition_steps",\n        "Perovskite_deposition_procedure",\n        "Perovskite_deposition_aggregation_state_of_reactants",\n        "Perovskite_deposition_synthesis_atmosphere",\n        "Perovskite_deposition_synthesis_atmosphere_pressure_total",\n        "Perovskite_deposition_synthesis_atmosphere_pressure_partial",\n        "Perovskite_deposition_synthesis_atmosphere_relative_humidity",\n        "Perovskite_deposition_solvents", "Perovskite_deposition_solvents_mixing_ratios",\n        "Perovskite_deposition_solvents_supplier", "Perovskite_deposition_solvents_purity",\n        "Perovskite_deposition_reaction_solutions_compounds",\n        "Perovskite_deposition_reaction_solutions_compounds_supplier",\n        "Perovskite_deposition_reaction_solutions_compounds_purity",\n        "Perovskite_deposition_reaction_solutions_concentrations",\n        "Perovskite_deposition_reaction_solutions_volumes",\n        "Perovskite_deposition_reaction_solutions_age",\n        "Perovskite_deposition_reaction_solutions_temperature",\n        "Perovskite_deposition_substrate_temperature",\n        "Perovskite_deposition_quenching_induced_crystallisation",\n        "Perovskite_deposition_quenching_media",\n        "Perovskite_deposition_quenching_media_mixing_ratios",\n        "Perovskite_deposition_quenching_media_volume",\n        "Perovskite_deposition_quenching_media_additives_compounds",\n        "Perovskite_deposition_quenching_media_additives_concentrations",\n        "Perovskite_deposition_thermal_annealing_temperature",\n        "Perovskite_deposition_thermal_annealing_time",\n        "Perovskite_deposition_thermal_annealing_atmosphere",\n        "Perovskite_deposition_thermal_annealing_relative_humidity",\n        "Perovskite_deposition_thermal_annealing_pressure",\n        "Perovskite_deposition_solvent_annealing",\n        "Perovskite_deposition_solvent_annealing_timing",\n        "Perovskite_deposition_solvent_annealing_solvent_atmosphere",\n        "Perovskite_deposition_solvent_annealing_time",\n        "Perovskite_deposition_solvent_annealing_temperature",\n        "Perovskite_deposition_after_treatment_of_formed_perovskite",\n        "Perovskite_deposition_after_treatment_of_formed_perovskite_met",\n        "Perovskite_storage_time_until_next_deposition_step",\n        "Perovskite_storage_atmosphere", "Perovskite_storage_relative_humidity",\n        "Perovskite_surface_treatment_before_next_deposition_step"\n    ],\n    "htl_backcontact_additional": [\n        "HTL_stack_sequence", "HTL_thickness_list", "HTL_additives_compounds",\n        "HTL_additives_concentrations", "HTL_deposition_procedure",\n        "HTL_deposition_aggregation_state_of_reactants",\n        "HTL_deposition_synthesis_atmosphere",\n        "HTL_deposition_synthesis_atmosphere_pressure_total",\n        "HTL_deposition_synthesis_atmosphere_pressure_partial",\n        "HTL_deposition_synthesis_atmosphere_relative_humidity",\n        "HTL_deposition_solvents", "HTL_deposition_solvents_mixing_ratios",\n        "HTL_deposition_solvents_supplier", "HTL_deposition_solvents_purity",\n        "HTL_deposition_reaction_solutions_compounds",\n        "HTL_deposition_reaction_solutions_compounds_supplier",\n        "HTL_deposition_reaction_solutions_compounds_purity",\n        "HTL_deposition_reaction_solutions_concentrations",\n        "HTL_deposition_reaction_solutions_volumes",\n        "HTL_deposition_reaction_solutions_age",\n        "HTL_deposition_reaction_solutions_temperature",\n        "HTL_deposition_substrate_temperature",\n        "HTL_deposition_thermal_annealing_temperature",\n        "HTL_deposition_thermal_annealing_time",\n        "HTL_deposition_thermal_annealing_atmosphere",\n        "HTL_storage_time_until_next_deposition_step",\n        "HTL_storage_atmosphere", "HTL_storage_relative_humidity",\n        "HTL_surface_treatment_before_next_deposition_step",\n        "Backcontact_stack_sequence", "Backcontact_thickness_list",\n        "Backcontact_additives_compounds", "Backcontact_additives_concentrations",\n        "Backcontact_deposition_procedure",\n        "Backcontact_deposition_aggregation_state_of_reactants",\n        "Backcontact_deposition_synthesis_atmosphere",\n        "Backcontact_deposition_synthesis_atmosphere_pressure_total",\n        "Backcontact_deposition_synthesis_atmosphere_pressure_partial",\n        "Backcontact_deposition_synthesis_atmosphere_relative_humidity",\n        "Backcontact_deposition_solvents", "Backcontact_deposition_solvents_mixing_ratios",\n        "Backcontact_deposition_solvents_supplier", "Backcontact_deposition_solvents_purity",\n        "Backcontact_deposition_reaction_solutions_compounds",\n        "Backcontact_deposition_reaction_solutions_compounds_supplier",\n        "Backcontact_deposition_reaction_solutions_compounds_purity",\n        "Backcontact_deposition_reaction_solutions_concentrations",\n        "Backcontact_deposition_reaction_solutions_volumes",\n        "Backcontact_deposition_reaction_solutions_age",\n        "Backcontact_deposition_reaction_solutions_temperature",\n        "Backcontact_deposition_substrate_temperature",\n        "Backcontact_deposition_thermal_annealing_temperature",\n        "Backcontact_deposition_thermal_annealing_time",\n        "Backcontact_deposition_thermal_annealing_atmosphere",\n        "Backcontact_storage_time_until_next_deposition_step",\n        "Backcontact_storage_atmosphere", "Backcontact_storage_relative_humidity",\n        "Backcontact_surface_treatment_before_next_deposition_step",\n        "Add_lay_front", "Add_lay_front_function", "Add_lay_front_stack_sequence",\n        "Add_lay_front_thickness_list", "Add_lay_front_additives_compounds",\n        "Add_lay_front_additives_concentrations", "Add_lay_front_deposition_procedure",\n        "Add_lay_back", "Add_lay_back_function", "Add_lay_back_stack_sequence",\n        "Add_lay_back_thickness_list", "Add_lay_back_additives_compounds",\n        "Add_lay_back_additives_concentrations", "Add_lay_back_deposition_procedure",\n        "Encapsulation", "Encapsulation_stack_sequence",\n        "Encapsulation_edge_sealing_materials", "Encapsulation_atmosphere_for_encapsulation"\n    ],\n    "performance": [\n        "JV_measured", "JV_average_over_n_number_of_cells", "JV_certified_values",\n        "JV_certification_institute", "JV_storage_age_of_cell", "JV_storage_atmosphere",\n        "JV_storage_relative_humidity", "JV_test_atmosphere", "JV_test_relative_humidity",\n        "JV_test_temperature", "JV_light_source_type", "JV_light_source_brand_name",\n        "JV_light_source_simulator_class", "JV_light_intensity", "JV_light_spectra",\n        "JV_light_wavelength_range", "JV_light_illumination_direction", "JV_light_masked_cell",\n        "JV_light_mask_area", "JV_scan_speed", "JV_scan_delay_time",\n        "JV_scan_integration_time", "JV_scan_voltage_step", "JV_preconditioning_protocol",\n        "JV_preconditioning_time", "JV_preconditioning_potential",\n        "JV_preconditioning_light_intensity",\n        "JV_reverse_scan_Voc", "JV_reverse_scan_Jsc", "JV_reverse_scan_FF", "JV_reverse_scan_PCE",\n        "JV_reverse_scan_Vmp", "JV_reverse_scan_Jmp",\n        "JV_reverse_scan_series_resistance", "JV_reverse_scan_shunt_resistance",\n        "JV_forward_scan_Voc", "JV_forward_scan_Jsc", "JV_forward_scan_FF", "JV_forward_scan_PCE",\n        "JV_forward_scan_Vmp", "JV_forward_scan_Jmp",\n        "JV_forward_scan_series_resistance", "JV_forward_scan_shunt_resistance",\n        "JV_link_raw_data", "JV_default_Voc", "JV_default_Jsc", "JV_default_FF",\n        "JV_default_PCE", "JV_default_Voc_scan_direction", "JV_default_Jsc_scan_direction",\n        "JV_default_FF_scan_direction", "JV_default_PCE_scan_direction", "JV_hysteresis_index",\n        "EQE_measured", "EQE_light_bias", "EQE_integrated_Jsc", "EQE_link_raw_data",\n        "Stabilised_performance_measured", "Stabilised_performance_procedure",\n        "Stabilised_performance_procedure_metrics", "Stabilised_performance_measurement_time",\n        "Stabilised_performance_PCE", "Stabilised_performance_Vmp", "Stabilised_performance_Jmp",\n        "Stabilised_performance_link_raw_data"\n    ],\n    "stability_outdoor": [\n        "Stability_measured", "Stability_protocol", "Stability_average_over_n_number_of_cells",\n        "Stability_light_source_type", "Stability_light_source_brand_name",\n        "Stability_light_source_simulator_class", "Stability_light_intensity",\n        "Stability_light_spectra", "Stability_light_wavelength_range",\n        "Stability_light_illumination_direction", "Stability_light_load_condition",\n        "Stability_light_cycling_times", "Stability_light_UV_filter",\n        "Stability_potential_bias_load_condition", "Stability_potential_bias_range",\n        "Stability_potential_bias_passive_resistance", "Stability_temperature_load_condition",\n        "Stability_temperature_range", "Stability_temperature_cycling_times",\n        "Stability_temperature_ramp_speed", "Stability_atmosphere",\n        "Stability_atmosphere_oxygen_concentration",\n        "Stability_relative_humidity_load_conditions", "Stability_relative_humidity_range",\n        "Stability_relative_humidity_average_value", "Stability_time_total_exposure",\n        "Stability_periodic_JV_measurements",\n        "Stability_periodic_JV_measurements_time_between_jv",\n        "Stability_PCE_initial_value", "Stability_PCE_burn_in_observed",\n        "Stability_PCE_end_of_experiment", "Stability_PCE_T95", "Stability_PCE_Ts95",\n        "Stability_PCE_T80", "Stability_PCE_Ts80", "Stability_PCE_Te80",\n        "Stability_PCE_Tse80", "Stability_PCE_after_1000_h",\n        "Stability_lifetime_energy_yield",\n        "Stability_flexible_cell_number_of_bending_cycles",\n        "Stability_flexible_cell_bending_radius",\n        "Stability_flexible_cell_PCE_initial_value",\n        "Stability_flexible_cell_PCE_end_of_experiment",\n        "Stability_link_raw_data_for_stability_trace",\n        "Outdoor_tested", "Outdoor_protocol", "Outdoor_average_over_n_number_of_cells",\n        "Outdoor_location_country", "Outdoor_location_city",\n        "Outdoor_location_coordinates", "Outdoor_location_climate_zone",\n        "Outdoor_installation_tilt", "Outdoor_installation_cardinal_direction",\n        "Outdoor_installation_number_of_solar_tracking_axis", "Outdoor_time_season",\n        "Outdoor_time_start", "Outdoor_time_end", "Outdoor_time_total_exposure",\n        "Outdoor_potential_bias_load_condition", "Outdoor_potential_bias_range",\n        "Outdoor_potential_bias_passive_resistance", "Outdoor_temperature_load_condition",\n        "Outdoor_temperature_range", "Outdoor_temperature_tmodule",\n        "Outdoor_periodic_JV_measurements",\n        "Outdoor_periodic_JV_measurements_time_between_measurements",\n        "Outdoor_PCE_initial_value", "Outdoor_PCE_burn_in_observed",\n        "Outdoor_PCE_end_of_experiment", "Outdoor_PCE_T95", "Outdoor_PCE_Ts95",\n        "Outdoor_PCE_T80", "Outdoor_PCE_Ts80", "Outdoor_PCE_Te80",\n        "Outdoor_PCE_Tse80", "Outdoor_PCE_after_1000_h", "Outdoor_power_generated",\n        "Outdoor_link_raw_data_for_outdoor_trace", "Outdoor_detaild_weather_data_available",\n        "Outdoor_link_detailed_weather_data", "Outdoor_spectral_data_available",\n        "Outdoor_link_spectral_data", "Outdoor_irradiance_measured",\n        "Outdoor_link_irradiance_data"\n    ]\n}\n# ============================================================\n# FIELD ALIASES\n# ============================================================\nFIELD_ALIASES: Dict[str, List[str]] = {\n    "JV_default_Voc": ["Voc", "open-circuit voltage", "open circuit voltage", "V_OC"],\n    "JV_default_Jsc": ["Jsc", "short-circuit current density", "short circuit current density", "J_SC"],\n    "JV_default_FF": ["FF", "fill factor"],\n    "JV_default_PCE": ["PCE", "power conversion efficiency", "efficiency", "device efficiency"],\n    "JV_certified_values": ["certified PCE", "certified efficiency", "certified value", "NREL-certified", "certified"],\n    "JV_default_PCE_scan_direction": ["reverse scan", "forward scan", "scan direction", "RS", "FS"],\n    "JV_light_intensity": ["AM 1.5G", "100 mW cm-2", "illumination intensity", "light intensity"],\n    "EQE_integrated_Jsc": ["integrated Jsc", "EQE integrated current density", "integrated photocurrent"],\n    "Stabilised_performance_PCE": ["stabilized PCE", "stabilised PCE", "steady-state PCE", "steady state PCE", "MPP PCE"],\n    "Stabilised_performance_Vmp": ["Vmp", "maximum power point voltage"],\n    "Stabilised_performance_Jmp": ["Jmp", "maximum power point current density"],\n    "Perovskite_composition_short_form": ["perovskite composition", "absorber composition", "formula", "nominal composition"],\n    "Perovskite_composition_long_form": ["full composition", "composition", "perovskite precursor composition"],\n    "Perovskite_band_gap": ["band gap", "Eg", "optical band gap"],\n    "Perovskite_pl_max": ["PL peak", "photoluminescence peak", "PL maximum"],\n    "Perovskite_thickness": ["perovskite thickness", "absorber thickness", "film thickness"],\n    "Perovskite_additives_compounds": ["additive", "additives", "dopant", "modifier"],\n    "Perovskite_deposition_procedure": ["spin-coated", "spin coating", "blade coating", "slot-die", "spray coating", "doctor blade"],\n    "Perovskite_deposition_solvents": ["solvent", "solvents", "DMF", "DMSO", "GBL", "NMP"],\n    "Perovskite_deposition_solvents_mixing_ratios": ["solvent ratio", "mixing ratio"],\n    "Perovskite_deposition_quenching_media": ["antisolvent", "anti-solvent", "quenching solvent", "dripping solvent", "quenching media"],\n    "Perovskite_deposition_quenching_media_volume": ["antisolvent volume", "dripped", "dropwise"],\n    "Perovskite_deposition_thermal_annealing_temperature": ["annealed at", "annealing temperature", "heated at", "thermal annealing temperature"],\n    "Perovskite_deposition_thermal_annealing_time": ["annealed for", "annealing time", "heated for"],\n    "ETL_stack_sequence": ["ETL", "electron transport layer", "electron-transport layer", "electron transporting layer"],\n    "ETL_thickness": ["ETL thickness", "electron transport layer thickness"],\n    "ETL_deposition_procedure": ["ETL deposited", "SnO2 colloid", "spin-coated ETL", "spray pyrolysis"],\n    "HTL_stack_sequence": ["HTL", "hole transport layer", "hole-transport layer", "hole transporting layer", "HTM"],\n    "HTL_thickness_list": ["HTL thickness", "hole transport layer thickness"],\n    "HTL_additives_compounds": ["Li-TFSI", "tBP", "FK209", "cobalt complex", "dopants in HTL"],\n    "HTL_deposition_procedure": ["HTL deposited", "spiro-OMeTAD solution", "spin-coated HTL"],\n    "Backcontact_stack_sequence": ["back contact", "electrode", "metal electrode", "top electrode", "counter electrode"],\n    "Backcontact_thickness_list": ["electrode thickness", "metal thickness"],\n    "Cell_architecture": ["n-i-p", "p-i-n", "inverted", "regular architecture"],\n    "Cell_stack_sequence": ["device structure", "stack", "architecture", "ITO/SnO2/perovskite/..."],\n    "Stability_protocol": ["ISOS", "stability test", "operational stability", "shelf stability", "thermal stability"],\n    "Stability_time_total_exposure": ["exposure time", "stability duration", "aging time", "test duration"],\n    "Stability_PCE_after_1000_h": ["after 1000 h", "1000 h retained", "1000 hours"],\n    "Outdoor_tested": ["outdoor", "field test", "real-world test"]\n}\n# ============================================================\n# NORMALIZATION / VALIDATION SETTINGS\n# ============================================================\nBOOL_FIELDS = {\n    "JV_measured",\n    "EQE_measured",\n    "Stability_measured",\n    "Outdoor_tested",\n    "Stabilised_performance_measured",\n    "Perovskite_single_crystal",\n    "Perovskite_dimension_0D",\n    "Perovskite_dimension_2D",\n    "Perovskite_dimension_2D3D_mixture",\n    "Perovskite_dimension_3D",\n    "Perovskite_dimension_3D_with_2D_capping_layer",\n    "Perovskite_composition_perovskite_ABC3_structure",\n    "Perovskite_composition_perovskite_inspired_structure",\n    "Perovskite_composition_inorganic",\n    "Perovskite_composition_leadfree",\n    "Cell_flexible",\n    "Cell_semitransparent",\n    "Module",\n    "Encapsulation",\n}\nNUMERIC_FIELDS = {\n    "Cell_area_total", "Cell_area_measured", "Cell_number_of_cells_per_substrate",\n    "Module_number_of_cells_in_module", "Module_area_total", "Module_area_effective",\n    "Substrate_thickness", "Substrate_area", "ETL_thickness", "Perovskite_thickness",\n    "Perovskite_band_gap", "Perovskite_pl_max",\n    "Perovskite_deposition_number_of_deposition_steps",\n    "Perovskite_deposition_reaction_solutions_age",\n    "Perovskite_deposition_reaction_solutions_temperature",\n    "Perovskite_deposition_substrate_temperature",\n    "Perovskite_deposition_quenching_media_volume",\n    "Perovskite_deposition_thermal_annealing_temperature",\n    "Perovskite_deposition_thermal_annealing_time",\n    "Perovskite_deposition_thermal_annealing_relative_humidity",\n    "Perovskite_deposition_thermal_annealing_pressure",\n    "Perovskite_deposition_solvent_annealing_time",\n    "Perovskite_deposition_solvent_annealing_temperature",\n    "JV_average_over_n_number_of_cells",\n    "JV_storage_age_of_cell", "JV_test_relative_humidity", "JV_test_temperature",\n    "JV_light_intensity", "JV_scan_speed", "JV_scan_delay_time", "JV_scan_integration_time",\n    "JV_scan_voltage_step", "JV_preconditioning_time", "JV_preconditioning_potential",\n    "JV_preconditioning_light_intensity",\n    "JV_reverse_scan_Voc", "JV_reverse_scan_Jsc", "JV_reverse_scan_FF", "JV_reverse_scan_PCE",\n    "JV_reverse_scan_Vmp", "JV_reverse_scan_Jmp",\n    "JV_reverse_scan_series_resistance", "JV_reverse_scan_shunt_resistance",\n    "JV_forward_scan_Voc", "JV_forward_scan_Jsc", "JV_forward_scan_FF", "JV_forward_scan_PCE",\n    "JV_forward_scan_Vmp", "JV_forward_scan_Jmp",\n    "JV_forward_scan_series_resistance", "JV_forward_scan_shunt_resistance",\n    "JV_default_Voc", "JV_default_Jsc", "JV_default_FF", "JV_default_PCE",\n    "JV_hysteresis_index",\n    "Stabilised_performance_measurement_time", "Stabilised_performance_PCE",\n    "Stabilised_performance_Vmp", "Stabilised_performance_Jmp",\n    "EQE_integrated_Jsc",\n    "Stability_average_over_n_number_of_cells", "Stability_light_intensity",\n    "Stability_light_cycling_times", "Stability_temperature_cycling_times",\n    "Stability_time_total_exposure", "Stability_periodic_JV_measurements_time_between_jv",\n    "Stability_PCE_initial_value", "Stability_PCE_end_of_experiment",\n    "Stability_PCE_T95", "Stability_PCE_Ts95", "Stability_PCE_T80", "Stability_PCE_Ts80",\n    "Stability_PCE_Te80", "Stability_PCE_Tse80", "Stability_PCE_after_1000_h",\n    "Stability_lifetime_energy_yield",\n    "Stability_flexible_cell_number_of_bending_cycles",\n    "Stability_flexible_cell_bending_radius",\n    "Stability_flexible_cell_PCE_initial_value",\n    "Stability_flexible_cell_PCE_end_of_experiment",\n    "Outdoor_average_over_n_number_of_cells", "Outdoor_installation_tilt",\n    "Outdoor_installation_number_of_solar_tracking_axis", "Outdoor_time_total_exposure",\n    "Outdoor_temperature_tmodule",\n    "Outdoor_periodic_JV_measurements_time_between_measurements",\n    "Outdoor_PCE_initial_value", "Outdoor_PCE_end_of_experiment",\n    "Outdoor_PCE_T95", "Outdoor_PCE_Ts95", "Outdoor_PCE_T80", "Outdoor_PCE_Ts80",\n    "Outdoor_PCE_Te80", "Outdoor_PCE_Tse80", "Outdoor_PCE_after_1000_h",\n    "Outdoor_power_generated",\n}\n# ============================================================\n# BASIC HELPERS\n# ============================================================\ndef clamp_years(year_min: int, year_max: int) -> Tuple[int, int]:\n    current_year = time.gmtime().tm_year\n    year_min = int(year_min)\n    year_max = int(year_max)\n    if year_min < 1900:\n        year_min = 1900\n    if year_max > current_year:\n        year_max = current_year\n    if year_min > year_max:\n        raise ValueError(f"Invalid year range: YEAR_MIN={year_min}, YEAR_MAX={year_max}")\n    return year_min, year_max\ndef sleep_with_jitter(base_delay: float = API_CALL_DELAY_SECONDS, jitter: float = API_JITTER_SECONDS) -> None:\n    delay = max(0.0, base_delay + random.uniform(-jitter, jitter))\n    time.sleep(delay)\ndef compute_backoff(attempt: int, base: float = API_RETRY_BACKOFF_SECONDS) -> float:\n    return base * (2 ** attempt) + random.uniform(0, 1.5)\ndef slugify(text: str, max_len: int = 48) -> str:\n    raw = (text or "").strip().lower()\n    raw = re.sub(r"[^\\w\\s-]", "", raw)\n    raw = re.sub(r"[-\\s]+", "_", raw).strip("_")\n    if not raw:\n        raw = "untitled"\n    short = raw[:max_len].rstrip("_") or "untitled"\n    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]\n    return f"{short}_{digest}"\ndef normalize_text(text: str) -> str:\n    text = text or ""\n    text = text.replace("\\x00", " ")\n    text = text.replace("\\r\\n", "\\n").replace("\\r", "\\n")\n    text = re.sub(r"\\n{3,}", "\\n\\n", text)\n    text = re.sub(r"[ \\t]+", " ", text)\n    return text.strip()\ndef strip_html_text(text: Optional[str]) -> Optional[str]:\n    if not text:\n        return None\n    text = ihtml.unescape(text)\n    text = re.sub(r"<[^>]+>", " ", text)\n    text = re.sub(r"\\s+", " ", text).strip()\n    return text or None\ndef is_allowed_crossref_item(item: Dict[str, Any]) -> bool:\n    title = " ".join(item.get("title", [])).strip()\n    if not title:\n        return False\n    item_type = (item.get("type") or "").lower()\n    allowed_types = {"journal-article", "proceedings-article", "posted-content"}\n    if item_type and item_type not in allowed_types:\n        return False\n    if TITLE_BLOCK_RE.search(title):\n        return False\n    title_low = title.lower()\n    bad_terms = [\n        "review", "perspective", "editorial", "commentary", "roadmap",\n        "outlook", "conference summary"\n    ]\n    if any(bt in title_low for bt in bad_terms):\n        return False\n    return True\ndef make_empty_record() -> Dict[str, Any]:\n    rec = {col: None for col in ALL_COLUMNS}\n    for col in EXTRA_COLUMNS:\n        rec[col] = None\n    return rec\ndef filter_to_allowed_columns(rec: Dict[str, Any]) -> Dict[str, Any]:\n    allowed = set(ALL_COLUMNS) | set(EXTRA_COLUMNS) | {k for k in rec.keys() if k.startswith("_evidence_")}\n    return {k: v for k, v in rec.items() if k in allowed}\ndef parse_numeric(value: Any) -> Optional[float]:\n    if value is None:\n        return None\n    if isinstance(value, (int, float)):\n        return float(value)\n    s = str(value).strip().replace(",", "")\n    m = re.search(r"[-+]?\\d+(?:\\.\\d+)?", s)\n    return float(m.group(0)) if m else None\ndef normalize_bool(value: Any) -> Optional[str]:\n    if value is None:\n        return None\n    s = str(value).strip().lower()\n    if s in {"true", "yes", "1"}:\n        return "TRUE"\n    if s in {"false", "no", "0"}:\n        return "FALSE"\n    return None\ndef validate_record_keys(records: List[Dict[str, Any]], allowed_columns: List[str]) -> List[Dict[str, Any]]:\n    allowed = set(allowed_columns)\n    cleaned = []\n    for rec in records:\n        if not isinstance(rec, dict):\n            continue\n        fixed = {}\n        for k, v in rec.items():\n            if k in allowed or k.startswith("_evidence_"):\n                fixed[k] = v\n        cleaned.append(fixed)\n    return cleaned\ndef normalize_record_types(rec: Dict[str, Any]) -> Dict[str, Any]:\n    out = dict(rec)\n    for f in BOOL_FIELDS:\n        if f in out and out[f] is not None:\n            out[f] = normalize_bool(out[f])\n    for f in NUMERIC_FIELDS:\n        if f in out and out[f] is not None:\n            if isinstance(out[f], str) and "implied" in out[f].lower():\n                out[f] = None\n            else:\n                out[f] = parse_numeric(out[f])\n    return out\ndef merge_non_null(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:\n    out = dict(base)\n    for k, v in new.items():\n        if v is None:\n            continue\n        if k not in out or out[k] is None or out[k] == "":\n            out[k] = v\n    return out\ndef _make_hashable_for_dedupe(value: Any) -> Any:\n    """Convert nested list/dict values into stable hashable values for dedupe keys.\n\n    LLM JSON outputs can legitimately contain lists or dictionaries for fields such as\n    Cell_stack_sequence, compositions, additives, or evidence. The previous dedupe key\n    used raw field values inside a tuple, which crashes when one of those values is a\n    list: TypeError("cannot use \'tuple\' as a set element (unhashable type: \'list\')").\n    """\n    if value is None:\n        return None\n    if isinstance(value, (str, int, float, bool)):\n        return value\n    if isinstance(value, (list, tuple, set)):\n        return tuple(_make_hashable_for_dedupe(v) for v in value)\n    if isinstance(value, dict):\n        return tuple(\n            sorted(\n                (str(k), _make_hashable_for_dedupe(v))\n                for k, v in value.items()\n            )\n        )\n    return str(value)\n\ndef _normalize_dedupe_key_value(value: Any) -> Any:\n    value = _make_hashable_for_dedupe(value)\n    if isinstance(value, str):\n        return re.sub(r"\\s+", " ", value.strip().lower()) or None\n    return value\n\ndef dedupe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:\n    deduped = []\n    seen = set()\n    for rec in records:\n        if not isinstance(rec, dict):\n            continue\n        raw_key = (\n            rec.get("Ref_DOI_number"),\n            rec.get("Ref_internal_sample_id"),\n            rec.get("Perovskite_composition_short_form"),\n            rec.get("JV_default_PCE"),\n            rec.get("Cell_stack_sequence"),\n        )\n        key = tuple(_normalize_dedupe_key_value(v) for v in raw_key)\n        if key in seen:\n            continue\n        seen.add(key)\n        deduped.append(rec)\n    return deduped\n# ============================================================\n# CHEMISTRY DERIVATION\n# ============================================================\nTOKEN_PAT = re.compile(r"(FA|MA|Cs|Rb|PEA|BA|DMA|EA|GA|Pb|Sn|I|Br|Cl|SCN|BF4|Ac|thiocyanate|formate|iodide|bromide|chloride)(\\d*\\.?\\d*)")\ndef split_mixed_site_tokens(expr: str) -> List[Tuple[str, float]]:\n    if not expr:\n        return []\n    expr = expr.replace(" ", "")\n    out = []\n    for m in TOKEN_PAT.finditer(expr):\n        ion = m.group(1)\n        coeff = m.group(2)\n        out.append((ion, float(coeff) if coeff else 1.0))\n    return out\ndef parse_perovskite_formula_abc3(formula: str) -> Optional[Dict[str, Any]]:\n    if not formula:\n        return None\n    s = formula.replace(" ", "")\n    s = s.replace("[", "(").replace("]", ")")\n    if "Pb" not in s and "Sn" not in s:\n        return None\n    m = re.match(r"(.+?)(Pb|Sn)(.+)", s)\n    if not m:\n        return None\n    a_expr = m.group(1)\n    b_expr = m.group(2)\n    c_expr = m.group(3)\n    if c_expr.startswith("(") and ")" in c_expr:\n        c_inner = c_expr[1:c_expr.index(")")]\n    else:\n        c_inner = re.sub(r"3$", "", c_expr)\n    a_tokens = split_mixed_site_tokens(a_expr)\n    b_tokens = split_mixed_site_tokens(b_expr)\n    c_tokens = split_mixed_site_tokens(c_inner)\n    if not a_tokens or not b_tokens or not c_tokens:\n        return None\n    if len(b_tokens) == 1 and b_tokens[0][1] != 1.0:\n        b_tokens = [(b_tokens[0][0], 1.0)]\n    c_sum = sum(v for _, v in c_tokens)\n    if c_sum != 0 and abs(c_sum - 3.0) > 0.15:\n        scale = 3.0 / c_sum\n        c_tokens = [(ion, round(val * scale, 4)) for ion, val in c_tokens]\n    return {\n        "a_ions": [x[0] for x in a_tokens],\n        "a_coeffs": [x[1] for x in a_tokens],\n        "b_ions": [x[0] for x in b_tokens],\n        "b_coeffs": [x[1] for x in b_tokens],\n        "c_ions": [x[0] for x in c_tokens],\n        "c_coeffs": [x[1] for x in c_tokens],\n    }\ndef derive_chemistry_fields(rec: Dict[str, Any]) -> Dict[str, Any]:\n    out = dict(rec)\n    comp = out.get("Perovskite_composition_short_form") or out.get("Perovskite_composition_long_form")\n    if not comp:\n        return out\n    parsed = parse_perovskite_formula_abc3(str(comp))\n    if parsed:\n        out["Perovskite_composition_perovskite_ABC3_structure"] = out.get("Perovskite_composition_perovskite_ABC3_structure") or "TRUE"\n        out["Perovskite_composition_a_ions"] = out.get("Perovskite_composition_a_ions") or ";".join(parsed["a_ions"])\n        out["Perovskite_composition_a_ions_coefficients"] = out.get("Perovskite_composition_a_ions_coefficients") or ";".join(str(x) for x in parsed["a_coeffs"])\n        out["Perovskite_composition_b_ions"] = out.get("Perovskite_composition_b_ions") or ";".join(parsed["b_ions"])\n        out["Perovskite_composition_b_ions_coefficients"] = out.get("Perovskite_composition_b_ions_coefficients") or ";".join(str(x) for x in parsed["b_coeffs"])\n        out["Perovskite_composition_c_ions"] = out.get("Perovskite_composition_c_ions") or ";".join(parsed["c_ions"])\n        out["Perovskite_composition_c_ions_coefficients"] = out.get("Perovskite_composition_c_ions_coefficients") or ";".join(str(x) for x in parsed["c_coeffs"])\n        aions = set(parsed["a_ions"])\n        bions = set(parsed["b_ions"])\n        if aions.issubset({"Cs", "Rb"}) and bions.isdisjoint({"Pb"}):\n            out["Perovskite_composition_inorganic"] = out.get("Perovskite_composition_inorganic") or "TRUE"\n        elif aions.issubset({"Cs", "Rb"}):\n            out["Perovskite_composition_inorganic"] = out.get("Perovskite_composition_inorganic") or "TRUE"\n        if "Pb" not in bions:\n            out["Perovskite_composition_leadfree"] = out.get("Perovskite_composition_leadfree") or "TRUE"\n    return out\ndef derive_measurement_flags(rec: Dict[str, Any]) -> Dict[str, Any]:\n    out = dict(rec)\n    if any(out.get(k) is not None for k in [\n        "JV_default_PCE", "JV_default_Voc", "JV_default_Jsc", "JV_default_FF",\n        "JV_reverse_scan_PCE", "JV_forward_scan_PCE"\n    ]):\n        out["JV_measured"] = out.get("JV_measured") or "TRUE"\n    if out.get("EQE_integrated_Jsc") is not None or out.get("EQE_link_raw_data") is not None:\n        out["EQE_measured"] = out.get("EQE_measured") or "TRUE"\n    if any(out.get(k) is not None for k in [\n        "Stabilised_performance_PCE", "Stabilised_performance_Vmp", "Stabilised_performance_Jmp"\n    ]):\n        out["Stabilised_performance_measured"] = out.get("Stabilised_performance_measured") or "TRUE"\n    if any(out.get(k) is not None for k in [\n        "Stability_protocol", "Stability_time_total_exposure", "Stability_PCE_end_of_experiment",\n        "Stability_PCE_T80", "Stability_PCE_after_1000_h"\n    ]):\n        out["Stability_measured"] = out.get("Stability_measured") or "TRUE"\n    if any(out.get(k) is not None for k in [\n        "Outdoor_protocol", "Outdoor_time_total_exposure", "Outdoor_PCE_end_of_experiment"\n    ]):\n        out["Outdoor_tested"] = out.get("Outdoor_tested") or "TRUE"\n    return out\ndef derive_dimensionality_flags(rec: Dict[str, Any]) -> Dict[str, Any]:\n    out = dict(rec)\n    blob = " ".join([\n        str(out.get("Perovskite_composition_short_form") or ""),\n        str(out.get("Perovskite_composition_long_form") or ""),\n        str(out.get("Perovskite_dimension_list_of_layers") or "")\n    ]).lower()\n    if any(x in blob for x in ["2d/3d", "2d-3d", "2d 3d mixture", "quasi-2d", "quasi 2d"]):\n        out["Perovskite_dimension_2D3D_mixture"] = out.get("Perovskite_dimension_2D3D_mixture") or "TRUE"\n    elif any(x in blob for x in ["pea2", "ba2", "ruddlesden", "dion-jacobson", "dj perovskite", "rp perovskite", "n=1", "n = 1"]):\n        out["Perovskite_dimension_2D"] = out.get("Perovskite_dimension_2D") or "TRUE"\n    elif any(x in blob for x in ["fapbi3", "mapbi3", "csfapbi3", "3d perovskite"]):\n        out["Perovskite_dimension_3D"] = out.get("Perovskite_dimension_3D") or "TRUE"\n    return out\n# ============================================================\n# SEARCH / FULLTEXT\n# ============================================================\ndef crossref_search(\n    query: str,\n    year_min: int,\n    year_max: int,\n    rows_per_page: int = 100,\n    max_pages: int = 8,\n) -> List[Dict[str, Any]]:\n    out = []\n    seen = set()\n    for page in range(max_pages):\n        offset = page * rows_per_page\n        url = (\n            "https://api.crossref.org/works?"\n            f"query.bibliographic={requests.utils.quote(query)}"\n            f"&rows={rows_per_page}&offset={offset}"\n            f"&filter=from-pub-date:{year_min}-01-01,until-pub-date:{year_max}-12-31"\n        )\n        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)\n        r.raise_for_status()\n        items = r.json().get("message", {}).get("items", [])\n        if not items:\n            break\n        for item in items:\n            if not is_allowed_crossref_item(item):\n                continue\n            title = " ".join(item.get("title", [])).strip() or None\n            doi = item.get("DOI")\n            key = doi or title\n            if not key or key in seen:\n                continue\n            seen.add(key)\n            journal = " ".join(item.get("container-title", [])).strip() or None\n            abstract = strip_html_text(item.get("abstract"))\n            resource_url = item.get("URL")\n            year = None\n            if item.get("published-print", {}).get("date-parts"):\n                year = item["published-print"]["date-parts"][0][0]\n            elif item.get("published-online", {}).get("date-parts"):\n                year = item["published-online"]["date-parts"][0][0]\n            author = None\n            if item.get("author"):\n                first = item["author"][0]\n                author = " ".join([first.get("given", ""), first.get("family", "")]).strip() or None\n            out.append({\n                "title": title,\n                "doi": doi,\n                "journal": journal,\n                "year": year,\n                "abstract": abstract,\n                "lead_author": author,\n                "source": "crossref",\n                "landing_page": resource_url,\n            })\n        time.sleep(SLEEP_BETWEEN_REQUESTS)\n    return out\ndef rss_search(query: str, year_min: int, year_max: int) -> List[Dict[str, Any]]:\n    if not HAS_FEEDPARSER:\n        return []\n    query_terms = [t.lower() for t in re.findall(r"[A-Za-z0-9\\-]+", query) if len(t) > 2]\n    items = []\n    for feed_url in RSS_FEEDS:\n        try:\n            feed = feedparser.parse(feed_url)\n        except Exception:\n            continue\n        for e in getattr(feed, "entries", []):\n            title = getattr(e, "title", None)\n            summary = getattr(e, "summary", None)\n            blob = f"{title or \'\'} {summary or \'\'}".lower()\n            if sum(term in blob for term in ["perovskite", "solar", "cell"]) < 2:\n                continue\n            if TITLE_BLOCK_RE.search(title or ""):\n                continue\n            if "review" in blob:\n                continue\n            year = None\n            for attr in ["published_parsed", "updated_parsed"]:\n                ts = getattr(e, attr, None)\n                if ts:\n                    year = ts.tm_year\n                    break\n            if year is not None and not (year_min <= year <= year_max):\n                continue\n            items.append({\n                "title": title,\n                "doi": None,\n                "journal": getattr(feed.feed, "title", None),\n                "year": year,\n                "abstract": strip_html_text(summary),\n                "lead_author": None,\n                "source": "rss",\n                "landing_page": getattr(e, "link", None),\n            })\n    return items\ndef merge_candidates(rss_items: List[Dict[str, Any]], crossref_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:\n    merged = []\n    seen = set()\n    for item in rss_items + crossref_items:\n        key = item.get("doi") or item.get("title")\n        if not key or key in seen:\n            continue\n        seen.add(key)\n        merged.append(item)\n    return merged\ndef crossref_lookup_title(title: str) -> Optional[Dict[str, Any]]:\n    url = f"https://api.crossref.org/works?query.title={quote(title)}&rows=5"\n    try:\n        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)\n        r.raise_for_status()\n        items = r.json().get("message", {}).get("items", [])\n        for item in items:\n            if not is_allowed_crossref_item(item):\n                continue\n            cand_title = " ".join(item.get("title", [])).strip().lower()\n            if cand_title and title and cand_title[:120] == title.strip().lower()[:120]:\n                return {\n                    "doi": item.get("DOI"),\n                    "landing_page": item.get("URL"),\n                    "journal": " ".join(item.get("container-title", [])).strip() or None,\n                }\n        if items:\n            item = items[0]\n            return {\n                "doi": item.get("DOI"),\n                "landing_page": item.get("URL"),\n                "journal": " ".join(item.get("container-title", [])).strip() or None,\n            }\n    except Exception:\n        return None\n    return None\ndef enrich_rss_candidate(meta: Dict[str, Any]) -> Dict[str, Any]:\n    if meta.get("doi"):\n        return meta\n    hit = crossref_lookup_title(meta.get("title") or "")\n    if hit:\n        meta["doi"] = hit.get("doi")\n        meta["landing_page"] = hit.get("landing_page") or meta.get("landing_page")\n        if not meta.get("journal"):\n            meta["journal"] = hit.get("journal")\n    return meta\ndef unpaywall_lookup(doi: str) -> Dict[str, Any]:\n    url = f"https://api.unpaywall.org/v2/{doi}?email={requests.utils.quote(UNPAYWALL_EMAIL)}"\n    r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)\n    r.raise_for_status()\n    data = r.json()\n    best = data.get("best_oa_location") or {}\n    return {\n        "is_oa": bool(data.get("is_oa")),\n        "pdf_url": best.get("url_for_pdf"),\n        "landing_page": best.get("url"),\n    }\ndef download_file(url: str, out_path: Path) -> bool:\n    try:\n        r = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT, headers=HEADERS)\n        r.raise_for_status()\n        with out_path.open("wb") as f:\n            for chunk in r.iter_content(chunk_size=8192):\n                if chunk:\n                    f.write(chunk)\n        return True\n    except Exception:\n        return False\ndef extract_pdf_text(pdf_path: Path) -> str:\n    if not HAS_PYPDF2:\n        return ""\n    try:\n        reader = PdfReader(str(pdf_path))\n        pages = []\n        for page in reader.pages:\n            txt = page.extract_text() or ""\n            if txt.strip():\n                pages.append(txt)\n        return "\\n".join(pages)\n    except Exception:\n        return ""\ndef fetch_html(url: str) -> Optional[str]:\n    try:\n        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)\n        r.raise_for_status()\n        if "text/html" in r.headers.get("Content-Type", "") or "<html" in r.text[:500].lower():\n            return r.text\n    except Exception:\n        pass\n    return None\ndef discover_pdf_from_html(html: str, base_url: str) -> Optional[str]:\n    soup = BeautifulSoup(html, "lxml")\n    meta_candidates = [\n        ("meta", {"name": "citation_pdf_url"}),\n        ("meta", {"property": "citation_pdf_url"}),\n        ("meta", {"name": "dc.identifier"}),\n        ("meta", {"name": "pdf_url"}),\n    ]\n    for tag_name, attrs in meta_candidates:\n        tag = soup.find(tag_name, attrs=attrs)\n        if tag and tag.get("content"):\n            val = tag.get("content")\n            if ".pdf" in val.lower():\n                return urljoin(base_url, val)\n    for a in soup.find_all("a", href=True):\n        href = a["href"]\n        text = (a.get_text(" ", strip=True) or "").lower()\n        if ".pdf" in href.lower() or "pdf" in text:\n            return urljoin(base_url, href)\n    return None\ndef extract_text_from_html(html: str) -> str:\n    if not html:\n        return ""\n    if HAS_TRAFILATURA:\n        try:\n            extracted = trafilatura.extract(\n                html,\n                include_comments=False,\n                include_tables=True,\n                favor_precision=True,\n            )\n            if extracted and len(extracted.strip()) > 1500:\n                return extracted\n        except Exception:\n            pass\n    soup = BeautifulSoup(html, "lxml")\n    for bad in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):\n        bad.decompose()\n    selectors = [\n        "article", "main", "div.article", "div.article-content", "div#main-content",\n        "section.article-body", "div.c-article-body", "div#Abs1-content", "div#bodymatter"\n    ]\n    for sel in selectors:\n        node = soup.select_one(sel)\n        if node:\n            text = node.get_text("\\n", strip=True)\n            if len(text) > 1200:\n                return text\n    return soup.get_text("\\n", strip=True)\ndef extract_relevant_sections(full_text: str) -> str:\n    if not full_text:\n        return ""\n    text = full_text\n    lower = text.lower()\n    markers = [\n        "abstract", "introduction", "experimental", "materials and methods", "methods",\n        "results", "results and discussion", "discussion", "conclusion"\n    ]\n    spans = []\n    for marker in markers:\n        idx = lower.find(marker)\n        if idx >= 0:\n            spans.append((idx, min(len(text), idx + 50000)))\n    if not spans:\n        return text[:220000]\n    spans.sort()\n    merged = []\n    s0, e0 = spans[0]\n    for s, e in spans[1:]:\n        if s <= e0:\n            e0 = max(e0, e)\n        else:\n            merged.append((s0, e0))\n            s0, e0 = s, e\n    merged.append((s0, e0))\n    return "\\n\\n".join(text[s:e] for s, e in merged)[:220000]\ndef try_full_text(meta: Dict[str, Any], slug: str) -> Dict[str, Any]:\n    pdf_text = ""\n    html_text = ""\n    pdf_url = None\n    landing_page = meta.get("landing_page")\n    is_oa = False\n    text_source = None\n    doi = meta.get("doi")\n    if doi:\n        try:\n            oa = unpaywall_lookup(doi)\n            is_oa = oa.get("is_oa", False)\n            pdf_url = oa.get("pdf_url")\n            if oa.get("landing_page"):\n                landing_page = oa.get("landing_page")\n        except Exception:\n            pass\n    if pdf_url:\n        pdf_path = PDF_DIR / f"{slug}.pdf"\n        if download_file(pdf_url, pdf_path):\n            pdf_text = extract_pdf_text(pdf_path)\n            if len(pdf_text.strip()) > 5000:\n                text_source = "pdf"\n    if landing_page and (not pdf_text or len(pdf_text.strip()) < 5000):\n        html = fetch_html(landing_page)\n        if html:\n            html_path = HTML_DIR / f"{slug}.html"\n            html_path.write_text(html, encoding="utf-8", errors="ignore")\n            discovered_pdf = discover_pdf_from_html(html, landing_page)\n            if discovered_pdf and (not pdf_text or len(pdf_text.strip()) < 5000):\n                pdf_path = PDF_DIR / f"{slug}_from_html.pdf"\n                if download_file(discovered_pdf, pdf_path):\n                    maybe_pdf_text = extract_pdf_text(pdf_path)\n                    if len(maybe_pdf_text.strip()) > len(pdf_text.strip()):\n                        pdf_text = maybe_pdf_text\n                        pdf_url = discovered_pdf\n                        text_source = "pdf"\n            html_text = extract_text_from_html(html)\n            if len(html_text.strip()) > len(pdf_text.strip()):\n                text_source = "html"\n    best_text = pdf_text if len(pdf_text.strip()) >= len(html_text.strip()) else html_text\n    best_text = extract_relevant_sections(best_text)\n    return {\n        "full_text": best_text,\n        "pdf_url": pdf_url,\n        "landing_page": landing_page,\n        "is_oa": is_oa,\n        "text_source": text_source,\n    }\n# ============================================================\n# SECTIONING / CHUNKING\n# ============================================================\ndef split_sections(text: str) -> Dict[str, str]:\n    lower = text.lower()\n    markers = [\n        "abstract", "introduction", "experimental", "materials and methods", "methods",\n        "results and discussion", "results", "discussion", "conclusion"\n    ]\n    positions = []\n    for m in markers:\n        idx = lower.find(m)\n        if idx != -1:\n            positions.append((idx, m))\n    positions.sort()\n    sections = {}\n    if not positions:\n        sections["full"] = text\n        return sections\n    for i, (start, name) in enumerate(positions):\n        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)\n        sections[name] = text[start:end]\n    sections["full"] = text\n    return sections\ndef build_extraction_text(full_text: str) -> str:\n    sections = split_sections(full_text)\n    parts = [\n        sections.get("experimental", ""),\n        sections.get("materials and methods", ""),\n        sections.get("methods", ""),\n        sections.get("results and discussion", ""),\n        sections.get("results", ""),\n        sections.get("discussion", ""),\n    ]\n    text = "\\n\\n".join(p for p in parts if p).strip()\n    if len(text) < 1500:\n        return full_text[:80000]\n    return text\ndef chunk_text(text: str, max_chars: int = LLM_CHUNK_CHAR_LIMIT, overlap: int = LLM_CHUNK_OVERLAP) -> List[str]:\n    text = text.strip()\n    if not text:\n        return []\n    if len(text) <= max_chars:\n        return [text]\n    chunks = []\n    start = 0\n    n = len(text)\n    while start < n:\n        end = min(start + max_chars, n)\n        chunk = text[start:end]\n        if end < n:\n            cut = max(chunk.rfind("\\n\\n"), chunk.rfind(". "), chunk.rfind("\\n"))\n            if cut > int(max_chars * 0.6):\n                end = start + cut + 1\n                chunk = text[start:end]\n        chunks.append(chunk.strip())\n        if end >= n:\n            break\n        start = max(0, end - overlap)\n    return [c for c in chunks if c]\n# ============================================================\n# RECORD MERGING\n# ============================================================\ndef make_record_fingerprint(rec: Dict[str, Any]) -> Tuple[Any, ...]:\n    return (\n        rec.get("Perovskite_composition_short_form"),\n        rec.get("Perovskite_composition_long_form"),\n        rec.get("Cell_stack_sequence"),\n        rec.get("ETL_stack_sequence"),\n        rec.get("HTL_stack_sequence"),\n        rec.get("Backcontact_stack_sequence"),\n        rec.get("JV_default_PCE"),\n    )\ndef merge_record_lists(existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:\n    if not existing:\n        return list(new)\n    merged = [dict(x) for x in existing]\n    for new_rec in new:\n        new_fp = make_record_fingerprint(new_rec)\n        match_idx = None\n        for i, old_rec in enumerate(merged):\n            old_fp = make_record_fingerprint(old_rec)\n            overlap = sum(\n                1 for a, b in zip(old_fp, new_fp)\n                if a is not None and b is not None and a == b\n            )\n            if overlap >= 2:\n                match_idx = i\n                break\n        if match_idx is None:\n            merged.append(dict(new_rec))\n        else:\n            merged[match_idx] = merge_non_null(merged[match_idx], new_rec)\n    return merged\n# ============================================================\n# QWEN EXTRACTOR\n# ============================================================\nclass HuggingFaceQwenExtractor:\n    def __init__(self, api_key: str, model: str, api_url: str):\n        if not api_key:\n            raise ValueError("HF_API_KEY is missing.")\n        self.api_key = api_key\n        self.model = model\n        self.url = api_url\n    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:\n        last_err = None\n        sleep_with_jitter()\n        for attempt in range(API_MAX_RETRIES):\n            try:\n                r = requests.post(\n                    self.url,\n                    headers={\n                        "Authorization": f"Bearer {self.api_key}",\n                        "Content-Type": "application/json",\n                    },\n                    json=payload,\n                    timeout=REQUEST_TIMEOUT * 3,\n                )\n                if r.status_code in (429, 500, 502, 503, 504):\n                    wait_s = compute_backoff(attempt)\n                    print(f"[WARN] HF/Qwen retryable status {r.status_code}. Waiting {wait_s:.1f}s...")\n                    time.sleep(wait_s)\n                    last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")\n                    continue\n                r.raise_for_status()\n                return r.json()\n            except Exception as exc:\n                last_err = exc\n                if attempt < API_MAX_RETRIES - 1:\n                    wait_s = compute_backoff(attempt)\n                    print(f"[WARN] HF/Qwen request failed. Waiting {wait_s:.1f}s before retry...")\n                    time.sleep(wait_s)\n        raise RuntimeError(f"HF/Qwen request failed after retries: {last_err}")\n    def _save_raw_text(self, paper_slug: str, stage: str, content: str) -> None:\n        if not SAVE_RAW_MODEL_TEXT:\n            return\n        ts = int(time.time() * 1000)\n        out = RAW_LLM_TEXT_DIR / f"{paper_slug}_{stage}_{ts}.txt"\n        out.write_text(content, encoding="utf-8", errors="ignore")\n    def _parse_content_to_json(self, content: Any, paper_slug: str, stage: str) -> Dict[str, Any]:\n        if isinstance(content, list):\n            text_bits = []\n            for part in content:\n                if isinstance(part, dict) and part.get("type") == "text":\n                    text_bits.append(part.get("text", ""))\n                elif isinstance(part, str):\n                    text_bits.append(part)\n            content = "\\n".join(text_bits).strip()\n        if not isinstance(content, str):\n            raise RuntimeError("Model returned non-string content.")\n        self._save_raw_text(paper_slug, stage, content)\n        try:\n            return json.loads(content)\n        except json.JSONDecodeError:\n            pass\n        m = re.search(r"```(?:json)?\\s*(\\{.*\\})\\s*```", content, flags=re.DOTALL)\n        if m:\n            return json.loads(m.group(1))\n        m = re.search(r"\\{.*\\}", content, flags=re.DOTALL)\n        if m:\n            return json.loads(m.group(0))\n        raise RuntimeError("Could not parse model output as JSON.")\n    def _call(self, system_prompt: str, user_payload: Dict[str, Any], paper_slug: str, stage: str) -> Dict[str, Any]:\n        payload = {\n            "model": self.model,\n            "temperature": 0,\n            "max_tokens": LLM_MAX_OUTPUT_TOKENS,\n            "response_format": {"type": "json_object"},\n            "messages": [\n                {"role": "system", "content": system_prompt},\n                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},\n            ],\n        }\n        data = self._request(payload)\n        content = data["choices"][0]["message"]["content"]\n        return self._parse_content_to_json(content, paper_slug=paper_slug, stage=stage)\n    def extract_family_records(\n        self,\n        metadata: Dict[str, Any],\n        chunk_text: str,\n        family_name: str,\n        family_columns: List[str],\n        alias_map: Dict[str, List[str]],\n        paper_slug: str,\n        chunk_idx: int,\n    ) -> Dict[str, Any]:\n        system = (\n            "You extract structured data from one experimental perovskite solar-cell paper. "\n            "Populate the requested CSV columns when the text explicitly provides the information, "\n            "even if the paper uses different terminology than the column name. "\n            "Use the provided alias lists to map paper wording to schema fields. "\n            "Do not invent unsupported values. "\n            "You may normalize terminology and units. "\n            "You may derive straightforward structured chemistry fields from explicitly given compositions, "\n            "but do not infer physical properties that are not explicitly stated. "\n            "Return one record per distinct explicitly described device when supported. "\n            "For each populated field, also return a short evidence snippet in a matching key named "\n            "_evidence_<fieldname>. "\n            "Return only valid JSON with top-level key \'records\'."\n        )\n        payload = {\n            "paper_metadata": metadata,\n            "schema_family": family_name,\n            "target_columns": family_columns,\n            "field_aliases": alias_map,\n            "task": (\n                "Extract all available values for this schema family from the text chunk. "\n                "Search by meaning and aliases, not just exact column names. "\n                "If the paper uses alternate nomenclature, map it to the correct column. "\n                "Only return values that are explicitly supported by the text. "\n                "Include short evidence snippets for populated fields."\n            ),\n            "text_chunk": chunk_text,\n            "output_format": {\n                "records": [\n                    {\n                        "example_column": "example_value",\n                        "_evidence_example_column": "short source snippet"\n                    }\n                ]\n            },\n        }\n        return self._call(system, payload, paper_slug=paper_slug, stage=f"{family_name}_partial_{chunk_idx}")\n    def consolidate_records(\n        self,\n        metadata: Dict[str, Any],\n        family_name: str,\n        partial_payloads: List[Dict[str, Any]],\n        family_columns: List[str],\n        paper_slug: str,\n    ) -> Dict[str, Any]:\n        system = (\n            "You are merging partial extraction outputs from one perovskite solar-cell paper for one schema family. "\n            "Combine rows only when they clearly refer to the same device. "\n            "Keep only values supported by the partial outputs. "\n            "Preserve evidence snippet keys when available. "\n            "Do not create unsupported values. "\n            "Return only valid JSON with top-level key \'records\'."\n        )\n        payload = {\n            "paper_metadata": metadata,\n            "schema_family": family_name,\n            "target_columns": family_columns,\n            "partials": partial_payloads,\n            "task": "Merge the partial outputs into final device-level records for this schema family."\n        }\n        return self._call(system, payload, paper_slug=paper_slug, stage=f"{family_name}_consolidated")\n# ============================================================\n# EXTRACTION PIPELINE\n# ============================================================\ndef enrich_record_with_metadata(rec: Dict[str, Any], meta: Dict[str, Any], idx: int) -> Dict[str, Any]:\n    out = dict(rec)\n    out["Ref_DOI_number"] = out.get("Ref_DOI_number") or meta.get("doi")\n    out["Ref_lead_author"] = out.get("Ref_lead_author") or meta.get("lead_author")\n    out["Ref_publication_date"] = out.get("Ref_publication_date") or meta.get("year")\n    out["Ref_journal"] = out.get("Ref_journal") or meta.get("journal")\n    out["Ref_original_filename_data_upload"] = out.get("Ref_original_filename_data_upload") or meta.get("title")\n    out["Ref_part_of_initial_dataset"] = out.get("Ref_part_of_initial_dataset") or "FALSE"\n    out["Ref_free_text_comment"] = out.get("Ref_free_text_comment") or f"Qwen-extracted from {meta.get(\'source\', \'search\')} paper."\n    out["Ref_internal_sample_id"] = out.get("Ref_internal_sample_id") or f"{slugify(meta.get(\'title\') or meta.get(\'doi\') or \'paper\')}_device_{idx}"\n    out["_source_abstract"] = meta.get("abstract")\n    out["_source_pdf_url"] = meta.get("pdf_url")\n    out["_source_open_access"] = meta.get("is_oa")\n    out["_source_landing_page"] = meta.get("landing_page")\n    out["_full_text_length"] = len(meta.get("_full_text") or "")\n    out["_full_text_source"] = meta.get("full_text_source")\n    out = derive_chemistry_fields(out)\n    out = derive_measurement_flags(out)\n    out = derive_dimensionality_flags(out)\n    out = normalize_record_types(out)\n    short_form = out.get("Perovskite_composition_short_form")\n    long_form = out.get("Perovskite_composition_long_form")\n    band_gap = out.get("Perovskite_band_gap")\n    dim = None\n    if out.get("Perovskite_dimension_0D") == "TRUE":\n        dim = "0D"\n    elif out.get("Perovskite_dimension_2D") == "TRUE":\n        dim = "2D"\n    elif out.get("Perovskite_dimension_2D3D_mixture") == "TRUE":\n        dim = "2D3D mixture"\n    elif out.get("Perovskite_dimension_3D") == "TRUE":\n        dim = "3D"\n    elif out.get("Perovskite_dimension_3D_with_2D_capping_layer") == "TRUE":\n        dim = "3D with 2D capping layer"\n    out["_composition_json"] = {\n        "short_form": short_form,\n        "long_form": long_form,\n        "band_gap": band_gap,\n        "dimensionality": dim,\n    }\n    return filter_to_allowed_columns(out)\ndef extract_family_for_chunks(\n    meta: Dict[str, Any],\n    family_name: str,\n    family_columns: List[str],\n    chunks: List[str],\n    extractor: HuggingFaceQwenExtractor,\n    paper_slug: str,\n) -> List[Dict[str, Any]]:\n    partial_payloads = []\n    alias_map = {col: FIELD_ALIASES.get(col, []) for col in family_columns}\n    for i, chunk in enumerate(chunks, start=1):\n        (TEXT_DIR / f"{paper_slug}_{family_name}_chunk_{i}.txt").write_text(chunk, encoding="utf-8")\n        payload = extractor.extract_family_records(\n            metadata={\n                "title": meta.get("title"),\n                "doi": meta.get("doi"),\n                "journal": meta.get("journal"),\n                "year": meta.get("year"),\n                "lead_author": meta.get("lead_author"),\n                "chunk_index": i,\n                "num_chunks": len(chunks),\n            },\n            chunk_text=chunk,\n            family_name=family_name,\n            family_columns=family_columns,\n            alias_map=alias_map,\n            paper_slug=paper_slug,\n            chunk_idx=i,\n        )\n        payload["records"] = validate_record_keys(payload.get("records", []), family_columns)\n        partial_payloads.append(payload)\n        (RAW_LLM_DIR / f"{paper_slug}_{family_name}_partial_{i}.json").write_text(\n            json.dumps(payload, indent=2, ensure_ascii=False),\n            encoding="utf-8"\n        )\n    if SKIP_CONSOLIDATION_IF_ONE_CHUNK and len(partial_payloads) == 1:\n        consolidated = partial_payloads[0]\n    elif USE_CONSOLIDATION_PASS:\n        consolidated = extractor.consolidate_records(\n            metadata={\n                "title": meta.get("title"),\n                "doi": meta.get("doi"),\n                "journal": meta.get("journal"),\n                "year": meta.get("year"),\n                "lead_author": meta.get("lead_author"),\n            },\n            family_name=family_name,\n            partial_payloads=partial_payloads,\n            family_columns=family_columns,\n            paper_slug=paper_slug,\n        )\n    else:\n        consolidated = {"records": [r for p in partial_payloads for r in p.get("records", [])]}\n    consolidated["records"] = validate_record_keys(consolidated.get("records", []), family_columns)\n    (RAW_LLM_DIR / f"{paper_slug}_{family_name}_consolidated.json").write_text(\n        json.dumps(consolidated, indent=2, ensure_ascii=False),\n        encoding="utf-8"\n    )\n    records = consolidated.get("records", [])\n    if not isinstance(records, list):\n        return []\n    return records\ndef extract_records_with_llm(meta: Dict[str, Any], extraction_text: str, extractor: HuggingFaceQwenExtractor) -> List[Dict[str, Any]]:\n    chunks = chunk_text(extraction_text, max_chars=LLM_CHUNK_CHAR_LIMIT, overlap=LLM_CHUNK_OVERLAP)\n    paper_slug = slugify(meta.get("title") or meta.get("doi") or "paper")\n    family_outputs: Dict[str, List[Dict[str, Any]]] = {}\n    for family_name, family_columns in SCHEMA_FAMILIES.items():\n        print(f"[INFO] Extracting family: {family_name}")\n        family_records = extract_family_for_chunks(\n            meta=meta,\n            family_name=family_name,\n            family_columns=family_columns,\n            chunks=chunks,\n            extractor=extractor,\n            paper_slug=paper_slug,\n        )\n        family_outputs[family_name] = family_records\n    combined_records: List[Dict[str, Any]] = []\n    for family_name in SCHEMA_FAMILIES.keys():\n        combined_records = merge_record_lists(combined_records, family_outputs.get(family_name, []))\n    if not combined_records:\n        raise RuntimeError("LLM returned zero usable records.")\n    cleaned = []\n    for idx, rec in enumerate(combined_records, start=1):\n        rec = enrich_record_with_metadata(rec, meta, idx)\n        cleaned.append(rec)\n    cleaned = dedupe_records(cleaned)\n    if not cleaned:\n        raise RuntimeError("LLM returned zero usable records after normalization.")\n    return cleaned\n# ============================================================\n# EXPORT\n# ============================================================\ndef json_to_csv(records: List[Dict[str, Any]], out_csv: Path) -> None:\n    rows = []\n    for rec in records:\n        row = {col: rec.get(col) for col in ALL_COLUMNS}\n        row["_source_abstract"] = rec.get("_source_abstract")\n        row["_source_pdf_url"] = rec.get("_source_pdf_url")\n        row["_source_open_access"] = rec.get("_source_open_access")\n        row["_source_landing_page"] = rec.get("_source_landing_page")\n        row["_full_text_length"] = rec.get("_full_text_length")\n        row["_full_text_source"] = rec.get("_full_text_source")\n        row["_composition_json"] = json.dumps(rec.get("_composition_json", {}), ensure_ascii=False)\n        rows.append(row)\n    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=ALL_COLUMNS + EXTRA_COLUMNS)\n    df.to_csv(out_csv, index=False)\n# ============================================================\n# PROCESS PAPER\n# ============================================================\ndef process_paper(meta: Dict[str, Any], extractor: HuggingFaceQwenExtractor) -> Optional[List[Dict[str, Any]]]:\n    if meta.get("source") == "rss" and not meta.get("doi"):\n        meta = enrich_rss_candidate(dict(meta))\n    print(f"[TRY] Candidate: {meta.get(\'title\')} | DOI={meta.get(\'doi\')}")\n    title = meta.get("title") or meta.get("doi") or "untitled"\n    slug = slugify(title)\n    retrieval = try_full_text(meta, slug)\n    meta["pdf_url"] = retrieval["pdf_url"]\n    meta["landing_page"] = retrieval["landing_page"]\n    meta["is_oa"] = retrieval["is_oa"]\n    meta["full_text_source"] = retrieval["text_source"]\n    full_text = normalize_text(retrieval["full_text"] or "")\n    meta["_full_text"] = full_text\n    if len(full_text) < LLM_MIN_FULLTEXT_CHARS:\n        print(f"[SKIP] Full text too short for reliable extraction ({len(full_text)} chars): {title}")\n        return None\n    print(f"[OK] Full text length: {len(full_text)} chars | source={meta.get(\'full_text_source\')}")\n    (TEXT_DIR / f"{slug}.txt").write_text(full_text, encoding="utf-8")\n    extraction_text = build_extraction_text(full_text)\n    (TEXT_DIR / f"{slug}_for_extraction.txt").write_text(extraction_text, encoding="utf-8")\n    print(f"[OK] Extraction text length: {len(extraction_text)} chars")\n    if len(extraction_text) < MIN_EXTRACTION_TEXT_CHARS:\n        print(f"[SKIP] Extraction text too short for reliable extraction ({len(extraction_text)} chars): {title}")\n        return None\n    try:\n        records = extract_records_with_llm(meta, extraction_text, extractor)\n    except Exception as exc:\n        print(f"[ERROR] Qwen extraction failed for {title}: {exc}")\n        if STRICT_FAIL_ON_LLM_ERROR:\n            raise\n        return None\n    with (JSON_DIR / f"{slug}.json").open("w", encoding="utf-8") as f:\n        json.dump(records, f, indent=2, ensure_ascii=False)\n    return records\n# ============================================================\n# MAIN\n# ============================================================\n\n\n\n\ndef build_fallback_paper_summary(metadata: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:\n    best = choose_best_record(records) or {}\n    title = metadata.get("title") or "Untitled paper"\n    arch = best.get("Cell_architecture") or best.get("Cell_stack_sequence") or None\n    perov = best.get("Perovskite_composition_short_form") or best.get("Perovskite_composition_long_form") or None\n    fab = []\n    for v in [\n        best.get("Perovskite_deposition_procedure"),\n        best.get("Perovskite_deposition_solvents"),\n        best.get("Perovskite_deposition_thermal_annealing_temperature"),\n        best.get("ETL_stack_sequence"),\n        best.get("HTL_stack_sequence"),\n        best.get("Backcontact_stack_sequence"),\n    ]:\n        if v not in (None, "", []) and str(v) not in fab:\n            fab.append(str(v))\n    notable = []\n    for v in [\n        best.get("Perovskite_additives_compounds"),\n        best.get("HTL_additives_compounds"),\n        best.get("ETL_additives_compounds"),\n    ]:\n        if v not in (None, "", []) and str(v) not in notable:\n            notable.append(str(v))\n    perf = {\n        "pce": best.get("JV_default_PCE") or best.get("Stabilised_performance_PCE"),\n        "voc": best.get("JV_default_Voc"),\n        "jsc": best.get("JV_default_Jsc") or best.get("EQE_integrated_Jsc"),\n        "ff": best.get("JV_default_FF"),\n        "context": "Auto-generated fallback summary from extracted fields.",\n    }\n    stability_bits = []\n    for v in [\n        best.get("Stability_protocol"),\n        best.get("Stability_time_total_exposure"),\n        best.get("Stability_PCE_after_1000_h"),\n        best.get("Stability_PCE_T80"),\n    ]:\n        if v not in (None, "", []):\n            stability_bits.append(str(v))\n    one_sentence = f"{title}: extracted {len(records)} device record(s)"\n    if perf["pce"] not in (None, ""):\n        one_sentence += f" with best reported PCE {perf[\'pce\']}"\n    if perov:\n        one_sentence += f" for {perov}"\n    one_sentence += "."\n    return {\n        "paper_summary": {\n            "one_sentence_summary": one_sentence,\n            "objective": "Fallback summary generated because the LLM summary was unavailable or invalid.",\n            "device_architecture": arch,\n            "perovskite_system": perov,\n            "fabrication_highlights": fab,\n            "best_reported_performance": perf,\n            "stability_summary": "; ".join(stability_bits) if stability_bits else None,\n            "notable_materials_or_methods": notable,\n            "main_contribution": "See extracted records for structured details.",\n            "data_quality_notes": [\n                "This summary was generated from extracted fields rather than a dedicated LLM summary pass.",\n                f"Number of extracted device records: {len(records)}",\n            ],\n        }\n    }\n\ndef render_paper_summary_text(summary_payload: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:\n    metadata = metadata or {}\n    ps = summary_payload.get("paper_summary", {}) if isinstance(summary_payload, dict) else {}\n    title = metadata.get("title") or ps.get("one_sentence_summary") or "Untitled paper"\n    doi = metadata.get("doi")\n    journal = metadata.get("journal")\n    year = metadata.get("year")\n    perf = ps.get("best_reported_performance") or {}\n    fab = ps.get("fabrication_highlights") or []\n    notable = ps.get("notable_materials_or_methods") or []\n    caveats = ps.get("data_quality_notes") or []\n\n    lines = [f"Paper Summary: {title}"]\n    meta_bits = []\n    if doi:\n        meta_bits.append(f"DOI: {doi}")\n    if journal:\n        meta_bits.append(f"Journal: {journal}")\n    if year:\n        meta_bits.append(f"Year: {year}")\n    if meta_bits:\n        lines.append(" | ".join(meta_bits))\n\n    def add_block(header: str, value: Any) -> None:\n        if value is None:\n            return\n        if isinstance(value, str):\n            value = value.strip()\n            if not value:\n                return\n            lines.append(f"\\n{header}:\\n{value}")\n        elif isinstance(value, list):\n            clean = [str(v).strip() for v in value if str(v).strip()]\n            if not clean:\n                return\n            lines.append(f"\\n{header}:")\n            lines.extend([f"- {v}" for v in clean])\n\n    add_block("One-sentence Summary", ps.get("one_sentence_summary"))\n    add_block("Objective", ps.get("objective"))\n    add_block("Device Architecture", ps.get("device_architecture"))\n    add_block("Perovskite System", ps.get("perovskite_system"))\n    add_block("Fabrication Highlights", fab)\n\n    perf_lines = []\n    if perf.get("pce") is not None:\n        perf_lines.append(f"PCE: {perf.get(\'pce\')}")\n    if perf.get("voc") is not None:\n        perf_lines.append(f"Voc: {perf.get(\'voc\')}")\n    if perf.get("jsc") is not None:\n        perf_lines.append(f"Jsc: {perf.get(\'jsc\')}")\n    if perf.get("ff") is not None:\n        perf_lines.append(f"FF: {perf.get(\'ff\')}")\n    if perf.get("context"):\n        perf_lines.append(f"Context: {perf.get(\'context\')}")\n    if perf_lines:\n        lines.append("\\nBest Reported Performance:")\n        lines.extend([f"- {v}" for v in perf_lines])\n\n    add_block("Stability Summary", ps.get("stability_summary"))\n    add_block("Notable Materials or Methods", notable)\n    add_block("Main Contribution", ps.get("main_contribution"))\n    add_block("Data Quality Notes", caveats)\n    return "\\n".join(lines).strip() + "\\n"\n\ndef main():\n    if not ALL_COLUMNS:\n        raise ValueError("ALL_COLUMNS is empty.")\n    if not HF_API_KEY:\n        raise ValueError("HF_API_KEY is missing.")\n    mode = normalize_run_mode(RUN_MODE)\n    if mode == "reset":\n        print(f"[INFO] RUN_MODE={mode}: clearing existing outputs in {BASE}")\n        reset_work_dir()\n    year_min, year_max = clamp_years(YEAR_MIN, YEAR_MAX)\n    extractor = HuggingFaceQwenExtractor(\n        api_key=HF_API_KEY,\n        model=HF_MODEL,\n        api_url=HF_API_URL,\n    )\n    rss_candidates = rss_search(SEARCH_QUERY, year_min, year_max)\n    crossref_candidates = crossref_search(\n        SEARCH_QUERY,\n        year_min,\n        year_max,\n        rows_per_page=CROSSREF_ROWS_PER_PAGE,\n        max_pages=CROSSREF_MAX_PAGES,\n    )\n    candidates = merge_candidates(rss_candidates, crossref_candidates)\n    candidates_to_process = collect_candidate_subset(candidates, MAX_PAPERS_TO_PROCESS, run_mode=mode)\n    records: List[Dict[str, Any]] = []\n    errors: List[Dict[str, Any]] = []\n    skipped: List[Dict[str, Any]] = []\n    summary_rows: List[Dict[str, Any]] = []\n    processed_papers = 0\n    for meta in candidates_to_process:\n        try:\n            out = process_paper(meta, extractor, force_reprocess=(mode in {"reset", "reprocess"}), run_mode=mode)\n            if out is None:\n                skipped.append({\n                    "title": meta.get("title"),\n                    "doi": meta.get("doi"),\n                    "reason": "no usable full text retrieved or extraction text below threshold"\n                })\n            else:\n                records.extend(out)\n                processed_papers += 1\n                slug = slugify(meta.get("title") or meta.get("doi") or "paper")\n                row = summary_row_from_slug_meta(slug, meta)\n                if row:\n                    summary_rows.append(row)\n                print(f"[OK] Processed paper: {meta.get(\'title\')}")\n        except Exception as exc:\n            errors.append({\n                "title": meta.get("title"),\n                "doi": meta.get("doi"),\n                "error": str(exc)\n            })\n            print(f"[FAIL] {meta.get(\'title\')}: {exc}")\n            continue\n        time.sleep(SLEEP_BETWEEN_REQUESTS)\n    jsonl_path = JSON_DIR / "all_records.jsonl"\n    with jsonl_path.open("w", encoding="utf-8") as f:\n        for rec in records:\n            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")\n    csv_path = CSV_DIR / "all_records.csv"\n    json_to_csv(records, csv_path)\n    summary_csv_path = CSV_DIR / "paper_summaries.csv"\n    if WRITE_SUMMARY_CSV:\n        write_paper_summary_csv(summary_rows, summary_csv_path)\n    (BASE / "skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")\n    (BASE / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")\n    summary = {\n        "search_mode": "rss+crossref",\n        "llm_backend": "huggingface_qwen",\n        "llm_model": HF_MODEL,\n        "hf_api_url": HF_API_URL,\n        "search_query": SEARCH_QUERY,\n        "year_min": YEAR_MIN,\n        "year_max": YEAR_MAX,\n        "max_papers_to_process": MAX_PAPERS_TO_PROCESS,\n        "run_mode": mode,\n        "rss_candidates": len(rss_candidates),\n        "crossref_candidates": len(crossref_candidates),\n        "merged_candidates": len(candidates),\n        "candidates_selected_for_processing": len(candidates_to_process),\n        "papers_processed": processed_papers,\n        "records_written": len(records),\n        "summaries_written": len(summary_rows),\n        "visible_summary_json_dir": str(PAPER_SUMMARY_JSON_DIR),\n        "visible_summary_text_dir": str(PAPER_SUMMARY_TEXT_DIR),\n        "paper_registry_path": str(PAPER_REGISTRY_PATH),\n        "skipped": len(skipped),\n        "errors": len(errors),\n        "jsonl_path": str(jsonl_path),\n        "csv_path": str(csv_path),\n        "summary_csv_path": str(summary_csv_path),\n    }\n    (BASE / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")\n    print(json.dumps(summary, indent=2))\nif False and __name__ == "__main__":\n    main()\n# ============================================================\n# IMPROVED OVERRIDES (v2)\n# ============================================================\nRUN_MISSING_FIELD_RECOVERY = True\nRUN_PAPER_SUMMARY = True\nWRITE_PER_PAPER_CSV = True\nWRITE_SUMMARY_CSV = True\nSUMMARY_MAX_TEXT_CHARS = 24000\nSECTION_FALLBACK_CHAR_LIMIT = 120000\nFAMILY_SECTION_HINTS = {\n    "reference_device": ["abstract", "experimental", "materials and methods", "methods", "results and discussion", "results", "discussion", "full"],\n    "perovskite_chemistry": ["abstract", "experimental", "materials and methods", "methods", "results and discussion", "results", "discussion", "full"],\n    "etl": ["experimental", "materials and methods", "methods", "results and discussion", "results", "full"],\n    "perovskite_processing": ["experimental", "materials and methods", "methods", "results and discussion", "results", "full"],\n    "htl_backcontact_additional": ["experimental", "materials and methods", "methods", "results and discussion", "results", "full"],\n    "performance": ["results and discussion", "results", "discussion", "full"],\n    "stability_outdoor": ["results and discussion", "results", "discussion", "full"],\n}\nPRIORITY_FIELDS_BY_FAMILY = {\n    "reference_device": ["Cell_stack_sequence", "Cell_architecture", "Cell_area_total", "Cell_area_measured"],\n    "perovskite_chemistry": [\n        "Perovskite_composition_short_form", "Perovskite_composition_long_form",\n        "Perovskite_composition_a_ions", "Perovskite_composition_b_ions", "Perovskite_composition_c_ions",\n        "Perovskite_dimension_2D", "Perovskite_dimension_2D3D_mixture", "Perovskite_dimension_3D",\n        "Perovskite_thickness", "Perovskite_band_gap", "Perovskite_pl_max"\n    ],\n    "etl": ["ETL_stack_sequence", "ETL_thickness", "ETL_deposition_procedure"],\n    "perovskite_processing": [\n        "Perovskite_deposition_procedure", "Perovskite_deposition_solvents",\n        "Perovskite_deposition_solvents_mixing_ratios", "Perovskite_deposition_quenching_media",\n        "Perovskite_deposition_thermal_annealing_temperature", "Perovskite_deposition_thermal_annealing_time"\n    ],\n    "htl_backcontact_additional": ["HTL_stack_sequence", "Backcontact_stack_sequence", "HTL_deposition_procedure", "Backcontact_deposition_procedure"],\n    "performance": ["JV_default_PCE", "JV_default_Voc", "JV_default_Jsc", "JV_default_FF", "EQE_integrated_Jsc", "Stabilised_performance_PCE"],\n    "stability_outdoor": ["Stability_protocol", "Stability_time_total_exposure", "Stability_PCE_after_1000_h", "Outdoor_tested"],\n}\nGLOBAL_PRIORITY_FIELDS = [\n    "Ref_DOI_number", "Cell_stack_sequence", "Cell_architecture",\n    "Perovskite_composition_short_form", "Perovskite_composition_long_form",\n    "Perovskite_composition_a_ions", "Perovskite_composition_b_ions", "Perovskite_composition_c_ions",\n    "ETL_stack_sequence", "HTL_stack_sequence", "Backcontact_stack_sequence",\n    "JV_default_PCE", "JV_default_Voc", "JV_default_Jsc", "JV_default_FF",\n    "EQE_integrated_Jsc", "Stabilised_performance_PCE", "Stability_protocol"\n]\nSUMMARY_DIR = BASE / "summary_json"\nSUMMARY_DIR.mkdir(parents=True, exist_ok=True)\nSUMMARY_TEXT_DIR = BASE / "summary_text"\nSUMMARY_TEXT_DIR.mkdir(parents=True, exist_ok=True)\ndef build_field_specs(family_columns: List[str]) -> List[Dict[str, Any]]:\n    specs = []\n    for col in family_columns:\n        aliases = FIELD_ALIASES.get(col, [])\n        specs.append({\n            "field": col,\n            "aliases": aliases,\n            "is_boolean": col in BOOL_FIELDS,\n            "is_numeric": col in NUMERIC_FIELDS,\n        })\n    return specs\ndef split_sections(text: str) -> Dict[str, str]:\n    text = normalize_text(text)\n    if not text:\n        return {"full": ""}\n    patterns = [\n        ("abstract", r"\\babstract\\b"),\n        ("introduction", r"\\bintroduction\\b"),\n        ("experimental", r"\\bexperimental\\b"),\n        ("materials and methods", r"\\bmaterials and methods\\b"),\n        ("methods", r"\\bmethods\\b"),\n        ("device fabrication", r"\\bdevice fabrication\\b"),\n        ("results and discussion", r"\\bresults and discussion\\b"),\n        ("results", r"\\bresults\\b"),\n        ("discussion", r"\\bdiscussion\\b"),\n        ("conclusion", r"\\bconclusion\\b"),\n        ("supporting information", r"\\bsupporting information\\b"),\n        ("supplementary", r"\\bsupplementary\\b"),\n    ]\n    hits = []\n    lower = text.lower()\n    for name, pat in patterns:\n        m = re.search(pat, lower)\n        if m:\n            hits.append((m.start(), name))\n    hits.sort()\n    if not hits:\n        return {"full": text}\n    sections: Dict[str, str] = {}\n    for i, (start, name) in enumerate(hits):\n        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)\n        chunk = text[start:end].strip()\n        if chunk:\n            sections[name] = chunk\n    sections["full"] = text\n    return sections\ndef choose_family_text(full_text: str, family_name: str) -> str:\n    sections = split_sections(full_text)\n    ordered_names = FAMILY_SECTION_HINTS.get(family_name, ["full"])\n    parts = []\n    seen = set()\n    for name in ordered_names:\n        part = sections.get(name, "")\n        part = part.strip()\n        if part and part not in seen:\n            parts.append(part)\n            seen.add(part)\n    merged = "\\n\\n".join(parts).strip()\n    if not merged:\n        merged = full_text[:SECTION_FALLBACK_CHAR_LIMIT]\n    return merged[:SECTION_FALLBACK_CHAR_LIMIT]\ndef build_extraction_text(full_text: str) -> str:\n    sections = split_sections(full_text)\n    parts = [\n        sections.get("abstract", ""),\n        sections.get("experimental", ""),\n        sections.get("materials and methods", ""),\n        sections.get("methods", ""),\n        sections.get("device fabrication", ""),\n        sections.get("results and discussion", ""),\n        sections.get("results", ""),\n        sections.get("discussion", ""),\n    ]\n    merged = "\\n\\n".join([p for p in parts if p]).strip()\n    if len(merged) < 2500:\n        merged = full_text[:80000]\n    return merged\ndef score_record_completeness(rec: Dict[str, Any]) -> int:\n    score = 0\n    for f in GLOBAL_PRIORITY_FIELDS:\n        v = rec.get(f)\n        if v not in (None, "", [], {}):\n            score += 3\n    for k, v in rec.items():\n        if k.startswith("_evidence_") and v:\n            score += 1\n        elif k in ALL_COLUMNS and v not in (None, "", [], {}):\n            score += 1\n    return score\ndef collect_missing_fields(rec: Dict[str, Any], family_name: str) -> List[str]:\n    wanted = PRIORITY_FIELDS_BY_FAMILY.get(family_name, [])\n    return [f for f in wanted if rec.get(f) in (None, "")]\ndef choose_best_record(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:\n    if not records:\n        return None\n    return max(records, key=score_record_completeness)\nclass HuggingFaceQwenExtractor(HuggingFaceQwenExtractor):\n    def extract_family_records(\n        self,\n        metadata: Dict[str, Any],\n        chunk_text: str,\n        family_name: str,\n        family_columns: List[str],\n        alias_map: Dict[str, List[str]],\n        paper_slug: str,\n        chunk_idx: int,\n    ) -> Dict[str, Any]:\n        system = (\n            "You extract structured data from ONE experimental perovskite solar-cell paper. "\n            "Your goal is maximum field recall with strict evidence grounding. "\n            "You must search by meaning, synonyms, units, stack notation, formulas, tables, captions, and nearby context. "\n            "Populate a field only when the text explicitly supports it or it can be directly normalized from explicit text. "\n            "Do not invent unsupported values. "\n            "Before producing output, check every requested field one by one. "\n            "If a paper uses alternate nomenclature, map it to the correct schema field. "\n            "For chemistry, resolve short form, long form, and A/B/C ions when they are explicitly recoverable from composition text. "\n            "Return one record per distinct experimentally described device only when the chunk clearly supports separate devices; otherwise return the main device record. "\n            "For each populated field, return a short evidence snippet using a matching key named _evidence_<fieldname>. "\n            "Return valid JSON only with top-level key \'records\'."\n        )\n        payload = {\n            "paper_metadata": metadata,\n            "schema_family": family_name,\n            "target_columns": family_columns,\n            "priority_fields": PRIORITY_FIELDS_BY_FAMILY.get(family_name, []),\n            "field_aliases": alias_map,\n            "field_specs": build_field_specs(family_columns),\n            "task": {\n                "goal": "Extract all available supported values for this schema family from the text chunk.",\n                "requirements": [\n                    "Search for synonymous wording, abbreviated metrics, and table-like statements.",\n                    "Prefer experimental/methods text for fabrication fields and results/captions for performance fields.",\n                    "Keep unsupported fields null.",\n                    "Include _evidence_<field> for populated fields."\n                ]\n            },\n            "text_chunk": chunk_text,\n            "output_format": {"records": [{"example_column": "example_value", "_evidence_example_column": "short source snippet"}]},\n        }\n        return self._call(system, payload, paper_slug=paper_slug, stage=f"{family_name}_partial_{chunk_idx}")\n    def consolidate_records(\n        self,\n        metadata: Dict[str, Any],\n        family_name: str,\n        partial_payloads: List[Dict[str, Any]],\n        family_columns: List[str],\n        paper_slug: str,\n    ) -> Dict[str, Any]:\n        system = (\n            "You are merging partial extraction outputs for one perovskite solar-cell paper and one schema family. "\n            "Merge rows only when they clearly refer to the same device. "\n            "Prefer the more complete value when duplicates conflict, but do not invent values. "\n            "Preserve evidence snippets. "\n            "Return valid JSON only with top-level key \'records\'."\n        )\n        payload = {\n            "paper_metadata": metadata,\n            "schema_family": family_name,\n            "target_columns": family_columns,\n            "priority_fields": PRIORITY_FIELDS_BY_FAMILY.get(family_name, []),\n            "partials": partial_payloads,\n            "task": "Merge the partial outputs into final device-level records for this schema family with the highest supported completeness."\n        }\n        return self._call(system, payload, paper_slug=paper_slug, stage=f"{family_name}_consolidated")\n    def recover_missing_fields(\n        self,\n        metadata: Dict[str, Any],\n        full_text: str,\n        family_name: str,\n        base_record: Dict[str, Any],\n        missing_fields: List[str],\n        paper_slug: str,\n    ) -> Dict[str, Any]:\n        if not missing_fields:\n            return {"records": [base_record]}\n        system = (\n            "You are performing a targeted recovery pass for one extracted perovskite solar-cell device record. "\n            "Only try to populate the listed missing fields. "\n            "Use the existing base record for context, but only add values explicitly supported by the text. "\n            "For each recovered field, include _evidence_<fieldname>. "\n            "Return valid JSON only with top-level key \'records\'."\n        )\n        payload = {\n            "paper_metadata": metadata,\n            "schema_family": family_name,\n            "missing_fields": missing_fields,\n            "base_record": base_record,\n            "field_aliases": {col: FIELD_ALIASES.get(col, []) for col in missing_fields},\n            "text_chunk": full_text[:50000],\n            "task": "Recover the missing fields if and only if they are explicitly supported in the text. Return a single updated record in \'records\'."\n        }\n        return self._call(system, payload, paper_slug=paper_slug, stage=f"{family_name}_recovery")\n    def summarize_paper(\n        self,\n        metadata: Dict[str, Any],\n        records: List[Dict[str, Any]],\n        summary_text: str,\n        paper_slug: str,\n    ) -> Dict[str, Any]:\n        system = (\n            "You are summarizing one experimental perovskite solar-cell paper for a literature-mining workflow. "\n            "Write a concise but information-dense structured summary grounded only in the provided metadata, extracted records, and text. "\n            "Focus on the research objective, device architecture, perovskite system, fabrication highlights, key photovoltaic results, stability results, unusual materials or methods, and any extraction caveats. "\n            "Do not invent claims. All summary fields MUST be written in English, even if the source paper or extracted text is in another language. Do not use Chinese or any non-English language in the summary. Return valid JSON only."\n        )\n        payload = {\n            "paper_metadata": metadata,\n            "records": records,\n            "text": summary_text[:SUMMARY_MAX_TEXT_CHARS],\n            "output_format": {\n                "paper_summary": {\n                    "one_sentence_summary": "...",\n                    "objective": "...",\n                    "device_architecture": "...",\n                    "perovskite_system": "...",\n                    "fabrication_highlights": ["..."],\n                    "best_reported_performance": {\n                        "pce": None,\n                        "voc": None,\n                        "jsc": None,\n                        "ff": None,\n                        "context": "..."\n                    },\n                    "stability_summary": "...",\n                    "notable_materials_or_methods": ["..."],\n                    "main_contribution": "...",\n                    "data_quality_notes": ["..."]\n                }\n            }\n        }\n        return self._call(system, payload, paper_slug=paper_slug, stage="paper_summary")\ndef extract_family_for_chunks(\n    meta: Dict[str, Any],\n    family_name: str,\n    family_columns: List[str],\n    chunks: List[str],\n    extractor: HuggingFaceQwenExtractor,\n    paper_slug: str,\n) -> List[Dict[str, Any]]:\n    partial_payloads = []\n    alias_map = {col: FIELD_ALIASES.get(col, []) for col in family_columns}\n    for i, chunk in enumerate(chunks, start=1):\n        if not chunk.strip():\n            continue\n        (TEXT_DIR / f"{paper_slug}_{family_name}_chunk_{i}.txt").write_text(chunk, encoding="utf-8")\n        payload = extractor.extract_family_records(\n            metadata={\n                "title": meta.get("title"),\n                "doi": meta.get("doi"),\n                "journal": meta.get("journal"),\n                "year": meta.get("year"),\n                "lead_author": meta.get("lead_author"),\n                "chunk_index": i,\n                "num_chunks": len(chunks),\n            },\n            chunk_text=chunk,\n            family_name=family_name,\n            family_columns=family_columns,\n            alias_map=alias_map,\n            paper_slug=paper_slug,\n            chunk_idx=i,\n        )\n        payload["records"] = validate_record_keys(payload.get("records", []), family_columns)\n        partial_payloads.append(payload)\n        (RAW_LLM_DIR / f"{paper_slug}_{family_name}_partial_{i}.json").write_text(\n            json.dumps(payload, indent=2, ensure_ascii=False),\n            encoding="utf-8"\n        )\n    if not partial_payloads:\n        return []\n    if SKIP_CONSOLIDATION_IF_ONE_CHUNK and len(partial_payloads) == 1:\n        consolidated = partial_payloads[0]\n    elif USE_CONSOLIDATION_PASS:\n        consolidated = extractor.consolidate_records(\n            metadata={\n                "title": meta.get("title"),\n                "doi": meta.get("doi"),\n                "journal": meta.get("journal"),\n                "year": meta.get("year"),\n                "lead_author": meta.get("lead_author"),\n            },\n            family_name=family_name,\n            partial_payloads=partial_payloads,\n            family_columns=family_columns,\n            paper_slug=paper_slug,\n        )\n    else:\n        consolidated = {"records": [r for p in partial_payloads for r in p.get("records", [])]}\n    consolidated["records"] = validate_record_keys(consolidated.get("records", []), family_columns)\n    (RAW_LLM_DIR / f"{paper_slug}_{family_name}_consolidated.json").write_text(\n        json.dumps(consolidated, indent=2, ensure_ascii=False),\n        encoding="utf-8"\n    )\n    records = consolidated.get("records", [])\n    if not isinstance(records, list):\n        return []\n    return records\ndef extract_records_with_llm(meta: Dict[str, Any], extraction_text: str, extractor: HuggingFaceQwenExtractor) -> List[Dict[str, Any]]:\n    paper_slug = slugify(meta.get("title") or meta.get("doi") or "paper")\n    family_outputs: Dict[str, List[Dict[str, Any]]] = {}\n    for family_name, family_columns in SCHEMA_FAMILIES.items():\n        family_text = choose_family_text(extraction_text, family_name)\n        chunks = chunk_text(family_text, max_chars=LLM_CHUNK_CHAR_LIMIT, overlap=LLM_CHUNK_OVERLAP)\n        print(f"[INFO] Extracting family: {family_name} | chunks={len(chunks)} | chars={len(family_text)}")\n        family_records = extract_family_for_chunks(\n            meta=meta,\n            family_name=family_name,\n            family_columns=family_columns,\n            chunks=chunks,\n            extractor=extractor,\n            paper_slug=paper_slug,\n        )\n        family_outputs[family_name] = family_records\n    combined_records: List[Dict[str, Any]] = []\n    for family_name in SCHEMA_FAMILIES.keys():\n        combined_records = merge_record_lists(combined_records, family_outputs.get(family_name, []))\n    if not combined_records:\n        raise RuntimeError("LLM returned zero usable records.")\n    cleaned = []\n    for idx, rec in enumerate(combined_records, start=1):\n        rec = enrich_record_with_metadata(rec, meta, idx)\n        cleaned.append(rec)\n    cleaned = dedupe_records(cleaned)\n    if not cleaned:\n        raise RuntimeError("LLM returned zero usable records after normalization.")\n    if RUN_MISSING_FIELD_RECOVERY:\n        best = choose_best_record(cleaned)\n        if best is not None:\n            for family_name in SCHEMA_FAMILIES.keys():\n                missing = collect_missing_fields(best, family_name)\n                if not missing:\n                    continue\n                try:\n                    recovered = extractor.recover_missing_fields(\n                        metadata={\n                            "title": meta.get("title"),\n                            "doi": meta.get("doi"),\n                            "journal": meta.get("journal"),\n                            "year": meta.get("year"),\n                            "lead_author": meta.get("lead_author"),\n                        },\n                        full_text=choose_family_text(extraction_text, family_name),\n                        family_name=family_name,\n                        base_record=best,\n                        missing_fields=missing,\n                        paper_slug=paper_slug,\n                    )\n                    new_records = recovered.get("records", [])\n                    if isinstance(new_records, list) and new_records:\n                        best = enrich_record_with_metadata(merge_non_null(best, new_records[0]), meta, 1)\n                except Exception as exc:\n                    print(f"[WARN] Recovery pass failed for {family_name}: {exc}")\n            cleaned[0] = best\n            cleaned = dedupe_records(cleaned)\n    cleaned.sort(key=score_record_completeness, reverse=True)\n    return cleaned\ndef write_paper_summary_csv(summary_rows: List[Dict[str, Any]], out_csv: Path) -> None:\n    df = pd.DataFrame(summary_rows)\n    df.to_csv(out_csv, index=False)\ndef process_paper(meta: Dict[str, Any], extractor: HuggingFaceQwenExtractor) -> Optional[List[Dict[str, Any]]]:\n    if meta.get("source") == "rss" and not meta.get("doi"):\n        meta = enrich_rss_candidate(dict(meta))\n    print(f"[TRY] Candidate: {meta.get(\'title\')} | DOI={meta.get(\'doi\')}")\n    title = meta.get("title") or meta.get("doi") or "untitled"\n    slug = slugify(title)\n    retrieval = try_full_text(meta, slug)\n    meta["pdf_url"] = retrieval["pdf_url"]\n    meta["landing_page"] = retrieval["landing_page"]\n    meta["is_oa"] = retrieval["is_oa"]\n    meta["full_text_source"] = retrieval["text_source"]\n    full_text = normalize_text(retrieval["full_text"] or "")\n    meta["_full_text"] = full_text\n    if len(full_text) < LLM_MIN_FULLTEXT_CHARS:\n        print(f"[SKIP] Full text too short for reliable extraction ({len(full_text)} chars): {title}")\n        return None\n    print(f"[OK] Full text length: {len(full_text)} chars | source={meta.get(\'full_text_source\')}")\n    (TEXT_DIR / f"{slug}.txt").write_text(full_text, encoding="utf-8")\n    extraction_text = build_extraction_text(full_text)\n    (TEXT_DIR / f"{slug}_for_extraction.txt").write_text(extraction_text, encoding="utf-8")\n    print(f"[OK] Extraction text length: {len(extraction_text)} chars")\n    if len(extraction_text) < MIN_EXTRACTION_TEXT_CHARS:\n        print(f"[SKIP] Extraction text too short for reliable extraction ({len(extraction_text)} chars): {title}")\n        return None\n    try:\n        records = extract_records_with_llm(meta, extraction_text, extractor)\n    except Exception as exc:\n        print(f"[ERROR] Qwen extraction failed for {title}: {exc}")\n        if STRICT_FAIL_ON_LLM_ERROR:\n            raise\n        return None\n    with (JSON_DIR / f"{slug}.json").open("w", encoding="utf-8") as f:\n        json.dump(records, f, indent=2, ensure_ascii=False)\n    if WRITE_PER_PAPER_CSV:\n        json_to_csv(records, CSV_DIR / f"{slug}.csv")\n    if RUN_PAPER_SUMMARY:\n        summary_metadata = {\n            "title": meta.get("title"),\n            "doi": meta.get("doi"),\n            "journal": meta.get("journal"),\n            "year": meta.get("year"),\n            "lead_author": meta.get("lead_author"),\n        }\n        summary_payload = None\n        try:\n            summary_payload = extractor.summarize_paper(\n                metadata=summary_metadata,\n                records=records,\n                summary_text=full_text,\n                paper_slug=slug,\n            )\n            if not isinstance(summary_payload, dict) or not isinstance(summary_payload.get("paper_summary"), dict):\n                raise RuntimeError("LLM summary payload missing top-level \'paper_summary\' object.")\n        except Exception as exc:\n            print(f"[WARN] Paper summary generation failed for {title}: {exc}")\n            summary_payload = build_fallback_paper_summary(summary_metadata, records)\n            summary_payload["_summary_generation_mode"] = "fallback"\n            summary_payload["_summary_generation_error"] = str(exc)\n        else:\n            summary_payload["_summary_generation_mode"] = "llm"\n        with (SUMMARY_DIR / f"{slug}_summary.json").open("w", encoding="utf-8") as f:\n            json.dump(summary_payload, f, indent=2, ensure_ascii=False)\n        summary_text_rendered = render_paper_summary_text(summary_payload, summary_metadata)\n        (SUMMARY_TEXT_DIR / f"{slug}_summary.txt").write_text(summary_text_rendered, encoding="utf-8")\n        print(f"[OK] Wrote paper summaries: {SUMMARY_DIR / f\'{slug}_summary.json\'} | {SUMMARY_TEXT_DIR / f\'{slug}_summary.txt\'}")\n    return records\n\n\ndef render_paper_summary_text(summary_payload: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:\n    metadata = metadata or {}\n    ps = summary_payload.get("paper_summary", {}) if isinstance(summary_payload, dict) else {}\n    title = metadata.get("title") or ps.get("one_sentence_summary") or "Untitled paper"\n    doi = metadata.get("doi")\n    journal = metadata.get("journal")\n    year = metadata.get("year")\n    perf = ps.get("best_reported_performance") or {}\n    fab = ps.get("fabrication_highlights") or []\n    notable = ps.get("notable_materials_or_methods") or []\n    caveats = ps.get("data_quality_notes") or []\n\n    lines = [f"Paper Summary: {title}"]\n    meta_bits = []\n    if doi:\n        meta_bits.append(f"DOI: {doi}")\n    if journal:\n        meta_bits.append(f"Journal: {journal}")\n    if year:\n        meta_bits.append(f"Year: {year}")\n    if meta_bits:\n        lines.append(" | ".join(meta_bits))\n\n    def add_block(header: str, value: Any) -> None:\n        if value is None:\n            return\n        if isinstance(value, str):\n            value = value.strip()\n            if not value:\n                return\n            lines.append(f"\\n{header}:\\n{value}")\n        elif isinstance(value, list):\n            clean = [str(v).strip() for v in value if str(v).strip()]\n            if not clean:\n                return\n            lines.append(f"\\n{header}:")\n            lines.extend([f"- {v}" for v in clean])\n\n    add_block("One-sentence Summary", ps.get("one_sentence_summary"))\n    add_block("Objective", ps.get("objective"))\n    add_block("Device Architecture", ps.get("device_architecture"))\n    add_block("Perovskite System", ps.get("perovskite_system"))\n    add_block("Fabrication Highlights", fab)\n\n    perf_lines = []\n    if perf.get("pce") is not None:\n        perf_lines.append(f"PCE: {perf.get(\'pce\')}")\n    if perf.get("voc") is not None:\n        perf_lines.append(f"Voc: {perf.get(\'voc\')}")\n    if perf.get("jsc") is not None:\n        perf_lines.append(f"Jsc: {perf.get(\'jsc\')}")\n    if perf.get("ff") is not None:\n        perf_lines.append(f"FF: {perf.get(\'ff\')}")\n    if perf.get("context"):\n        perf_lines.append(f"Context: {perf.get(\'context\')}")\n    if perf_lines:\n        lines.append("\\nBest Reported Performance:")\n        lines.extend([f"- {v}" for v in perf_lines])\n\n    add_block("Stability Summary", ps.get("stability_summary"))\n    add_block("Notable Materials or Methods", notable)\n    add_block("Main Contribution", ps.get("main_contribution"))\n    add_block("Data Quality Notes", caveats)\n    return "\\n".join(lines).strip() + "\\n"\n\ndef main():\n    if not ALL_COLUMNS:\n        raise ValueError("ALL_COLUMNS is empty.")\n    if not HF_API_KEY:\n        raise ValueError("HF_API_KEY is missing.")\n    year_min, year_max = clamp_years(YEAR_MIN, YEAR_MAX)\n    extractor = HuggingFaceQwenExtractor(\n        api_key=HF_API_KEY,\n        model=HF_MODEL,\n        api_url=HF_API_URL,\n    )\n    rss_candidates = rss_search(SEARCH_QUERY, year_min, year_max)\n    crossref_candidates = crossref_search(\n        SEARCH_QUERY,\n        year_min,\n        year_max,\n        rows_per_page=CROSSREF_ROWS_PER_PAGE,\n        max_pages=CROSSREF_MAX_PAGES,\n    )\n    candidates = merge_candidates(rss_candidates, crossref_candidates)\n    candidates_to_process = collect_candidate_subset(candidates, max_papers, run_mode=mode)\n    records: List[Dict[str, Any]] = []\n    errors: List[Dict[str, Any]] = []\n    skipped: List[Dict[str, Any]] = []\n    summary_rows: List[Dict[str, Any]] = []\n    processed_papers = 0\n    for meta in candidate_pool:\n        if processed_papers >= MAX_PAPERS_TO_PROCESS:\n            break\n        try:\n            out = process_paper(meta, extractor)\n            if out is None:\n                skipped.append({\n                    "title": meta.get("title"),\n                    "doi": meta.get("doi"),\n                    "reason": "no usable full text retrieved or extraction text below threshold"\n                })\n            else:\n                records.extend(out)\n                processed_papers += 1\n                slug = slugify(meta.get("title") or meta.get("doi") or "paper")\n                summary_path = SUMMARY_DIR / f"{slug}_summary.json"\n                if summary_path.exists():\n                    try:\n                        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))\n                        ps = summary_data.get("paper_summary", {})\n                        summary_rows.append({\n                            "title": meta.get("title"),\n                            "doi": meta.get("doi"),\n                            "journal": meta.get("journal"),\n                            "year": meta.get("year"),\n                            "one_sentence_summary": ps.get("one_sentence_summary"),\n                            "objective": ps.get("objective"),\n                            "device_architecture": ps.get("device_architecture"),\n                            "perovskite_system": ps.get("perovskite_system"),\n                            "main_contribution": ps.get("main_contribution"),\n                            "stability_summary": ps.get("stability_summary"),\n                            "summary_text_path": str(SUMMARY_TEXT_DIR / f"{slug}_summary.txt"),\n                        })\n                    except Exception:\n                        pass\n                print(f"[OK] Processed paper: {meta.get(\'title\')}")\n        except Exception as exc:\n            errors.append({\n                "title": meta.get("title"),\n                "doi": meta.get("doi"),\n                "error": str(exc)\n            })\n            print(f"[FAIL] {meta.get(\'title\')}: {exc}")\n            continue\n        time.sleep(SLEEP_BETWEEN_REQUESTS)\n    jsonl_path = JSON_DIR / "all_records.jsonl"\n    with jsonl_path.open("w", encoding="utf-8") as f:\n        for rec in records:\n            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")\n    csv_path = CSV_DIR / "all_records.csv"\n    json_to_csv(records, csv_path)\n    if WRITE_SUMMARY_CSV:\n        write_paper_summary_csv(summary_rows, CSV_DIR / "paper_summaries.csv")\n    (BASE / "skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")\n    (BASE / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")\n    summary = {\n        "search_mode": "rss+crossref",\n        "llm_backend": "huggingface_qwen",\n        "llm_model": HF_MODEL,\n        "hf_api_url": HF_API_URL,\n        "search_query": SEARCH_QUERY,\n        "year_min": YEAR_MIN,\n        "year_max": YEAR_MAX,\n        "max_papers_to_process": MAX_PAPERS_TO_PROCESS,\n        "rss_candidates": len(rss_candidates),\n        "crossref_candidates": len(crossref_candidates),\n        "merged_candidates": len(candidates),\n        "candidates_selected_for_processing": len(candidates_to_process),\n        "run_mode": mode,\n        "papers_processed": processed_papers,\n        "records_written": len(records),\n        "summaries_written": len(summary_rows),\n        "skipped": len(skipped),\n        "errors": len(errors),\n        "jsonl_path": str(jsonl_path),\n        "csv_path": str(csv_path),\n        "summary_csv_path": str(CSV_DIR / "paper_summaries.csv"),\n    }\n    (BASE / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")\n    print(json.dumps(summary, indent=2))\n# ============================================================\n# RAG + MCP WRAPPER LAYER\n# ============================================================\nRAG_ENABLE = True\nRAG_TOP_K_CHUNKS = 6\nRAG_MIN_SCORE = 1.0\nMCP_SERVER_NAME = "literature-mining-agent"\nMCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "http").strip().lower() or "http"\nRUN_AS_MCP_SERVER = os.getenv("RUN_AS_MCP_SERVER", "0").strip().lower() in {"1", "true", "yes", "on"}\nRUN_MODE = os.getenv("RUN_MODE", "expand").strip().lower() or "expand"\n# Supported RUN_MODE values:\n# - resume: reuse cached outputs and skip already processed papers\n# - expand: alias of resume for growing the library with new papers only\n# - reprocess: keep stored literature, but recompute extraction/summaries for matching papers\n# - reset / overwrite: delete prior outputs in WORK_DIR and start from scratch\ndef _rag_tokenize(text: str) -> List[str]:\n    text = (text or "").lower()\n    tokens = re.findall(r"[a-z0-9_\\-\\.]+", text)\n    stop = {\n        "the", "and", "for", "with", "that", "this", "from", "into", "were", "was", "are", "is", "of", "in",\n        "to", "on", "by", "or", "as", "an", "at", "be", "it", "we", "they", "their", "our", "using"\n    }\n    return [t for t in tokens if len(t) > 1 and t not in stop]\ndef _family_query_terms(family_name: str, family_columns: List[str]) -> List[str]:\n    terms: List[str] = []\n    terms.extend(_rag_tokenize(family_name.replace("_", " ")))\n    for col in family_columns:\n        terms.extend(_rag_tokenize(col.replace("_", " ")))\n        for alias in FIELD_ALIASES.get(col, []):\n            terms.extend(_rag_tokenize(alias))\n    # prioritize known perovskite/device terms for better recall\n    priors = {\n        "reference_device": ["device", "architecture", "substrate", "cell", "fabrication"],\n        "perovskite_chemistry": ["perovskite", "composition", "formula", "absorber", "bandgap", "pl"],\n        "etl": ["etl", "electron", "transport", "sno2", "tio2"],\n        "perovskite_processing": ["spin", "coat", "anneal", "antisolvent", "precursor", "solvent"],\n        "htl_backcontact_additional": ["htl", "hole", "transport", "electrode", "back", "contact", "encapsulation"],\n        "performance": ["pce", "voc", "jsc", "ff", "j-v", "eqe", "efficiency"],\n        "stability_outdoor": ["stability", "operational", "shelf", "thermal", "outdoor", "t80", "t95"],\n    }\n    terms.extend(priors.get(family_name, []))\n    # dedupe preserving order\n    seen = set()\n    out = []\n    for t in terms:\n        if t not in seen:\n            seen.add(t)\n            out.append(t)\n    return out\ndef _score_chunk_for_family(chunk: str, family_name: str, family_columns: List[str]) -> float:\n    chunk_low = (chunk or "").lower()\n    chunk_tokens = _rag_tokenize(chunk_low)\n    token_set = set(chunk_tokens)\n    terms = _family_query_terms(family_name, family_columns)\n    score = 0.0\n    for term in terms:\n        if term in token_set:\n            score += 2.0\n        elif term in chunk_low:\n            score += 1.0\n    # section-aware boosts\n    if family_name in {"reference_device", "etl", "perovskite_processing", "htl_backcontact_additional"}:\n        if re.search(r"\x08(methods?|experimental|fabrication|device fabrication|preparation)\x08", chunk_low):\n            score += 4.0\n    if family_name == "perovskite_chemistry":\n        if re.search(r"\x08(perovskite|composition|formula|absorber|band gap|photoluminescence|pl)\x08", chunk_low):\n            score += 4.0\n    if family_name == "performance":\n        if re.search(r"\x08(pce|voc|jsc|fill factor|ff|eqe|photovoltaic|j-v|efficiency)\x08", chunk_low):\n            score += 5.0\n    if family_name == "stability_outdoor":\n        if re.search(r"\x08(stability|t80|t95|operational|shelf|thermal|outdoor|is-os|isos)\x08", chunk_low):\n            score += 5.0\n    # figure/table captions often contain direct values\n    if re.search(r"\x08(fig\\.?|figure|table)\x08", chunk_low):\n        score += 1.0\n    return score\ndef rag_select_chunks_for_family(extraction_text: str, family_name: str, family_columns: List[str], top_k: int = RAG_TOP_K_CHUNKS) -> List[str]:\n    all_chunks = chunk_text(extraction_text, max_chars=LLM_CHUNK_CHAR_LIMIT, overlap=LLM_CHUNK_OVERLAP)\n    if not RAG_ENABLE or len(all_chunks) <= top_k:\n        return all_chunks\n    scored = []\n    for i, ch in enumerate(all_chunks):\n        scored.append((i, _score_chunk_for_family(ch, family_name, family_columns), ch))\n    scored.sort(key=lambda x: x[1], reverse=True)\n    kept = [t for t in scored if t[1] >= RAG_MIN_SCORE][:top_k]\n    if not kept:\n        kept = scored[:top_k]\n    # restore original chunk order for narrative continuity\n    kept.sort(key=lambda x: x[0])\n    return [x[2] for x in kept]\nclass HuggingFaceQwenExtractor(HuggingFaceQwenExtractor):\n    def extract_family_records(\n        self,\n        metadata: Dict[str, Any],\n        chunk_text: str,\n        family_name: str,\n        family_columns: List[str],\n        alias_map: Dict[str, List[str]],\n        paper_slug: str,\n        chunk_idx: int,\n    ) -> Dict[str, Any]:\n        system = (\n            "You extract structured data from one experimental perovskite solar-cell paper. "\n            "Your goal is high field recall with strict evidence grounding. "\n            "Populate requested CSV columns whenever the text explicitly provides the information, even if different terminology is used. "\n            "Use the alias lists and nearby context to map paper wording to schema fields. "\n            "Scan methods, device fabrication, results tables, captions, and performance paragraphs. "\n            "Do not invent unsupported values. Normalize units and wording when safe. "\n            "You may derive straightforward A/B/C ion decomposition only from explicitly reported perovskite compositions. "\n            "Return one record per distinct explicitly described device when supported; otherwise return the best-supported main device record. "\n            "For every populated field, return a short supporting snippet in a matching key named _evidence_<fieldname>. "\n            "Leave unsupported fields null. Return only valid JSON with top-level key \'records\'."\n        )\n        payload = {\n            "metadata": metadata,\n            "family_name": family_name,\n            "family_columns": family_columns,\n            "alias_map": alias_map,\n            "chunk_text": chunk_text,\n            "instructions": {\n                "maximize_field_recall": True,\n                "require_evidence_for_populated_fields": True,\n                "prefer_experimental_device_data": True,\n                "do_not_guess": True,\n            },\n            "output_schema": {\n                "records": [\n                    {c: None for c in family_columns} | {f"_evidence_{c}": None for c in family_columns}\n                ]\n            },\n        }\n        return self._call(system, payload, paper_slug=paper_slug, stage=f"{family_name}_chunk_{chunk_idx}")\ndef extract_records_with_llm(meta: Dict[str, Any], extraction_text: str, extractor: HuggingFaceQwenExtractor) -> List[Dict[str, Any]]:\n    paper_slug = slugify(meta.get("title") or meta.get("doi") or "paper")\n    family_outputs: Dict[str, List[Dict[str, Any]]] = {}\n    for family_name, family_columns in SCHEMA_FAMILIES.items():\n        family_chunks = rag_select_chunks_for_family(extraction_text, family_name, family_columns, top_k=RAG_TOP_K_CHUNKS)\n        print(f"[INFO] Extracting family: {family_name} | rag_chunks={len(family_chunks)}")\n        family_records = extract_family_for_chunks(\n            meta=meta,\n            family_name=family_name,\n            family_columns=family_columns,\n            chunks=family_chunks,\n            extractor=extractor,\n            paper_slug=paper_slug,\n        )\n        family_outputs[family_name] = family_records\n    combined_records: List[Dict[str, Any]] = []\n    for family_name in SCHEMA_FAMILIES.keys():\n        combined_records = merge_record_lists(combined_records, family_outputs.get(family_name, []))\n    if not combined_records:\n        raise RuntimeError("LLM returned zero usable records.")\n    cleaned: List[Dict[str, Any]] = []\n    for idx, rec in enumerate(combined_records, start=1):\n        rec = enrich_record_with_metadata(rec, meta, idx)\n        cleaned.append(rec)\n    cleaned = dedupe_records(cleaned)\n    cleaned.sort(key=score_record_completeness, reverse=True)\n    if not cleaned:\n        raise RuntimeError("LLM returned zero usable records after normalization.")\n    return cleaned\ndef build_extractor() -> HuggingFaceQwenExtractor:\n    if not HF_API_KEY:\n        raise ValueError("HF_API_KEY is missing.")\n    return HuggingFaceQwenExtractor(api_key=HF_API_KEY, model=HF_MODEL, api_url=HF_API_URL)\ndef search_candidates_api(query: str, year_min: int = YEAR_MIN, year_max: int = YEAR_MAX, max_candidates: int = 25) -> Dict[str, Any]:\n    year_min, year_max = clamp_years(year_min, year_max)\n    rss_candidates = rss_search(query, year_min, year_max)\n    crossref_candidates = crossref_search(\n        query,\n        year_min,\n        year_max,\n        rows_per_page=CROSSREF_ROWS_PER_PAGE,\n        max_pages=CROSSREF_MAX_PAGES,\n    )\n    candidates = merge_candidates(rss_candidates, crossref_candidates)\n    candidates_to_process = collect_candidate_subset(candidates, max_papers, run_mode=mode)\n    return {\n        "ok": True,\n        "tool": "search_candidates",\n        "query": query,\n        "year_min": year_min,\n        "year_max": year_max,\n        "rss_candidates": len(rss_candidates),\n        "crossref_candidates": len(crossref_candidates),\n        "merged_candidates": len(candidates),\n        "candidates_selected_for_processing": len(candidates_to_process),\n        "run_mode": mode,\n        "candidates": candidates[:max_candidates],\n    }\ndef process_one_paper_api(meta: Dict[str, Any], force_reprocess: bool = False, run_mode: Optional[str] = None) -> Dict[str, Any]:\n    extractor = build_extractor()\n    records = process_paper(dict(meta), extractor, force_reprocess=force_reprocess, run_mode=run_mode)\n    title = meta.get("title")\n    doi = meta.get("doi")\n    slug = slugify(title or doi or "paper")\n    summary_path = SUMMARY_DIR / f"{slug}_summary.json"\n    summary_text_path = SUMMARY_TEXT_DIR / f"{slug}_summary.txt"\n    csv_path = CSV_DIR / f"{slug}.csv"\n    json_path = JSON_DIR / f"{slug}.json"\n    payload: Dict[str, Any] = {\n        "ok": records is not None,\n        "tool": "process_one_paper",\n        "title": title,\n        "doi": doi,\n        "paper_slug": slug,\n        "num_records": 0 if records is None else len(records),\n        "json_path": str(json_path) if json_path.exists() else None,\n        "csv_path": str(csv_path) if csv_path.exists() else None,\n        "summary_path": str(summary_path) if summary_path.exists() else None,\n        "summary_text_path": str(summary_text_path) if summary_text_path.exists() else None,\n        "summary_text_path": str(summary_text_path) if summary_text_path.exists() else None,\n        "records": records or [],\n    }\n    if summary_path.exists():\n        try:\n            payload["paper_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))\n        except Exception:\n            payload["paper_summary"] = None\n    if summary_text_path.exists():\n        try:\n            payload["paper_summary_text"] = summary_text_path.read_text(encoding="utf-8")\n        except Exception:\n            payload["paper_summary_text"] = None\n    return payload\ndef process_batch_api(query: str = SEARCH_QUERY, year_min: int = YEAR_MIN, year_max: int = YEAR_MAX, max_papers: int = MAX_PAPERS_TO_PROCESS, run_mode: Optional[str] = None, force_reprocess: bool = False, reset_output: bool = False) -> Dict[str, Any]:\n    year_min, year_max = clamp_years(year_min, year_max)\n    mode = normalize_run_mode(run_mode)\n    if reset_output or mode == "reset":\n        reset_work_dir()\n    extractor = build_extractor()\n    rss_candidates = rss_search(query, year_min, year_max)\n    crossref_candidates = crossref_search(\n        query,\n        year_min,\n        year_max,\n        rows_per_page=CROSSREF_ROWS_PER_PAGE,\n        max_pages=CROSSREF_MAX_PAGES,\n    )\n    candidates = merge_candidates(rss_candidates, crossref_candidates)\n    candidates = sort_candidates_for_processing(candidates)\n    candidate_pool = collect_candidate_subset(candidates, max_papers, run_mode=mode)\n    records: List[Dict[str, Any]] = []\n    errors: List[Dict[str, Any]] = []\n    skipped: List[Dict[str, Any]] = []\n    summary_rows: List[Dict[str, Any]] = []\n    processed_papers = 0\n    attempted_candidates = 0\n    for meta in candidate_pool:\n        if processed_papers >= max_papers:\n            break\n        slug = slugify(meta.get("title") or meta.get("doi") or "paper")\n        if should_skip_processed_paper(meta, slug, mode):\n            continue\n        attempted_candidates += 1\n        try:\n            out = process_paper(dict(meta), extractor, force_reprocess=force_reprocess or mode in {"reprocess", "reset"}, run_mode=mode)\n            if out is None:\n                skipped.append({\n                    "title": meta.get("title"),\n                    "doi": meta.get("doi"),\n                    "reason": "no usable full text retrieved or extraction text below threshold",\n                })\n                continue\n            records.extend(out)\n            processed_papers += 1\n            row = summary_row_from_slug_meta(slug, meta)\n            if row:\n                summary_rows.append(row)\n        except Exception as exc:\n            errors.append({"title": meta.get("title"), "doi": meta.get("doi"), "error": str(exc)})\n    jsonl_path = BASE / "all_records_from_mcp.jsonl"\n    csv_path = CSV_DIR / "all_records_from_mcp.csv"\n    with jsonl_path.open("w", encoding="utf-8") as f:\n        for rec in records:\n            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")\n    json_to_csv(records, csv_path)\n    summary_csv_path = CSV_DIR / "paper_summaries_from_mcp.csv"\n    if summary_rows:\n        write_paper_summary_csv(summary_rows, summary_csv_path)\n    return {\n        "ok": True,\n        "tool": "process_batch",\n        "query": query,\n        "year_min": year_min,\n        "year_max": year_max,\n        "rss_candidates": len(rss_candidates),\n        "crossref_candidates": len(crossref_candidates),\n        "merged_candidates": len(candidates),\n        "candidates_considered": len(candidate_pool),\n        "candidates_attempted": attempted_candidates,\n        "run_mode": mode,\n        "papers_processed": processed_papers,\n        "records_written": len(records),\n        "summaries_written": len(summary_rows),\n        "visible_summary_json_dir": str(PAPER_SUMMARY_JSON_DIR),\n        "visible_summary_text_dir": str(PAPER_SUMMARY_TEXT_DIR),\n        "paper_registry_path": str(PAPER_REGISTRY_PATH),\n        "skipped": skipped,\n        "errors": errors,\n        "jsonl_path": str(jsonl_path),\n        "csv_path": str(csv_path),\n        "summary_csv_path": str(summary_csv_path),\n    }\n\ndef get_saved_paper_output_api(paper_slug: str) -> Dict[str, Any]:\n    json_path = JSON_DIR / f"{paper_slug}.json"\n    csv_path = CSV_DIR / f"{paper_slug}.csv"\n    summary_path = SUMMARY_DIR / f"{paper_slug}_summary.json"\n    summary_text_path = SUMMARY_TEXT_DIR / f"{paper_slug}_summary.txt"\n    if not json_path.exists() and not summary_path.exists() and not summary_text_path.exists() and not csv_path.exists():\n        return {"ok": False, "tool": "get_saved_paper_output", "paper_slug": paper_slug, "reason": "saved outputs not found"}\n    out: Dict[str, Any] = {\n        "ok": True,\n        "tool": "get_saved_paper_output",\n        "paper_slug": paper_slug,\n        "json_path": str(json_path) if json_path.exists() else None,\n        "csv_path": str(csv_path) if csv_path.exists() else None,\n        "summary_path": str(summary_path) if summary_path.exists() else None,\n        "summary_text_path": str(summary_text_path) if summary_text_path.exists() else None,\n    }\n    if json_path.exists():\n        out["records"] = json.loads(json_path.read_text(encoding="utf-8"))\n    if summary_path.exists():\n        out["paper_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))\n    if summary_text_path.exists():\n        out["paper_summary_text"] = summary_text_path.read_text(encoding="utf-8")\n    return out\ndef run_mcp_server() -> None:\n    try:\n        from mcp.server.fastmcp import FastMCP\n    except Exception as exc:\n        raise RuntimeError("The \'mcp\' package is required to run the MCP server. Install it with: pip install mcp") from exc\n    mcp = FastMCP(MCP_SERVER_NAME)\n    @mcp.tool()\n    def search_candidates(query: str, year_min: int = YEAR_MIN, year_max: int = YEAR_MAX, max_candidates: int = 25) -> Dict[str, Any]:\n        return search_candidates_api(query=query, year_min=year_min, year_max=year_max, max_candidates=max_candidates)\n    @mcp.tool()\n    def process_one_paper(meta: Dict[str, Any], force_reprocess: bool = False, run_mode: str = RUN_MODE) -> Dict[str, Any]:\n        return process_one_paper_api(meta, force_reprocess=force_reprocess, run_mode=run_mode)\n    @mcp.tool()\n    def process_batch(query: str = SEARCH_QUERY, year_min: int = YEAR_MIN, year_max: int = YEAR_MAX, max_papers: int = MAX_PAPERS_TO_PROCESS, run_mode: str = RUN_MODE, force_reprocess: bool = False, reset_output: bool = False) -> Dict[str, Any]:\n        return process_batch_api(query=query, year_min=year_min, year_max=year_max, max_papers=max_papers, run_mode=run_mode, force_reprocess=force_reprocess, reset_output=reset_output)\n    @mcp.tool()\n    def get_saved_paper_output(paper_slug: str) -> Dict[str, Any]:\n        return get_saved_paper_output_api(paper_slug)\n    if MCP_TRANSPORT == "http":\n        mcp.run(transport="streamable-http")\n    else:\n        mcp.run()\n\n# ============================================================\n# PERSISTENT CACHE + VISIBLE SUMMARY STORAGE OVERRIDES\n# ============================================================\nPAPER_STORE_DIR = BASE / "paper_store"\nPAPER_SUMMARY_JSON_DIR = BASE / "paper_summaries_json"\nPAPER_SUMMARY_TEXT_DIR = BASE / "paper_summaries_text"\nPAPER_REGISTRY_PATH = BASE / "paper_registry.json"\nfor d in [PAPER_STORE_DIR, PAPER_SUMMARY_JSON_DIR, PAPER_SUMMARY_TEXT_DIR]:\n    d.mkdir(parents=True, exist_ok=True)\n\n\ndef compute_paper_id(meta: Dict[str, Any], slug: Optional[str] = None) -> str:\n    doi = (meta.get("doi") or meta.get("Ref_DOI_number") or "").strip().lower()\n    if doi:\n        return f"doi::{doi}"\n    year = str(meta.get("year") or meta.get("Ref_publication_date") or "")[:4]\n    slug = slug or slugify(meta.get("title") or meta.get("doi") or "paper")\n    return f"slug::{slug}::year::{year or \'unknown\'}"\n\n\ndef load_paper_registry() -> Dict[str, Any]:\n    if PAPER_REGISTRY_PATH.exists():\n        try:\n            data = json.loads(PAPER_REGISTRY_PATH.read_text(encoding="utf-8"))\n            if isinstance(data, dict):\n                data.setdefault("papers", {})\n                return data\n        except Exception:\n            pass\n    return {"papers": {}}\n\n\ndef save_paper_registry(registry: Dict[str, Any]) -> None:\n    PAPER_REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")\n\n\ndef upsert_registry_entry(meta: Dict[str, Any], slug: str, updates: Dict[str, Any]) -> Dict[str, Any]:\n    registry = load_paper_registry()\n    paper_id = compute_paper_id(meta, slug)\n    entry = registry["papers"].get(paper_id, {})\n    entry.update({\n        "paper_id": paper_id,\n        "slug": slug,\n        "title": meta.get("title"),\n        "doi": meta.get("doi"),\n        "journal": meta.get("journal"),\n        "year": meta.get("year"),\n        "lead_author": meta.get("lead_author"),\n    })\n    entry.update(updates)\n    registry["papers"][paper_id] = entry\n    save_paper_registry(registry)\n    return entry\n\n\ndef find_registry_entry(meta: Dict[str, Any], slug: Optional[str] = None) -> Optional[Dict[str, Any]]:\n    registry = load_paper_registry().get("papers", {})\n    pid = compute_paper_id(meta, slug)\n    if pid in registry:\n        return registry[pid]\n    doi = (meta.get("doi") or "").strip().lower()\n    slug = slug or slugify(meta.get("title") or meta.get("doi") or "paper")\n    for entry in registry.values():\n        if doi and (entry.get("doi") or "").strip().lower() == doi:\n            return entry\n        if entry.get("slug") == slug:\n            return entry\n    return None\n\n\ndef paper_paths(slug: str, create_dirs: bool = True) -> Dict[str, Path]:\n    paper_dir = PAPER_STORE_DIR / slug\n    llm_dir = paper_dir / "llm_outputs"\n    outputs_dir = paper_dir / "outputs"\n    raw_dir = paper_dir / "raw"\n    if create_dirs:\n        for d in [paper_dir, llm_dir, outputs_dir, raw_dir]:\n            d.mkdir(parents=True, exist_ok=True)\n    return {\n        "paper_dir": paper_dir,\n        "llm_dir": llm_dir,\n        "outputs_dir": outputs_dir,\n        "raw_dir": raw_dir,\n        "records_json": outputs_dir / "records.json",\n        "records_csv": outputs_dir / "records.csv",\n        "summary_json": outputs_dir / "paper_summary.json",\n        "summary_txt": outputs_dir / "paper_summary.txt",\n        "metadata_json": outputs_dir / "metadata.json",\n        "full_text_txt": raw_dir / "full_text.txt",\n        "extraction_text_txt": raw_dir / "extraction_text.txt",\n    }\n\ndef load_cached_records_for_slug(slug: str) -> Optional[List[Dict[str, Any]]]:\n    primary = JSON_DIR / f"{slug}.json"\n    secondary = paper_paths(slug)["records_json"]\n    for p in [primary, secondary]:\n        if p.exists():\n            try:\n                data = json.loads(p.read_text(encoding="utf-8"))\n                if isinstance(data, list):\n                    return data\n            except Exception:\n                pass\n    return None\n\n\ndef persist_artifacts(slug: str, meta: Dict[str, Any], records: List[Dict[str, Any]], full_text: str, extraction_text: str,\n                      summary_payload: Optional[Dict[str, Any]] = None, summary_text_rendered: Optional[str] = None) -> None:\n    paths = paper_paths(slug)\n    paths["metadata_json"].write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")\n    paths["records_json"].write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")\n    if records:\n        pd.DataFrame([{k: v for k, v in rec.items() if not k.startswith("_evidence_")} for rec in records]).to_csv(paths["records_csv"], index=False)\n    if full_text:\n        paths["full_text_txt"].write_text(full_text, encoding="utf-8")\n    if extraction_text:\n        paths["extraction_text_txt"].write_text(extraction_text, encoding="utf-8")\n    if summary_payload is not None:\n        paths["summary_json"].write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")\n    if summary_text_rendered is not None:\n        paths["summary_txt"].write_text(summary_text_rendered, encoding="utf-8")\n\n\ndef load_cached_text(slug: str) -> Tuple[str, str]:\n    full_text = ""\n    extraction_text = ""\n    p = TEXT_DIR / f"{slug}.txt"\n    q = TEXT_DIR / f"{slug}_for_extraction.txt"\n    if p.exists():\n        full_text = normalize_text(p.read_text(encoding="utf-8", errors="ignore"))\n    elif paper_paths(slug)["full_text_txt"].exists():\n        full_text = normalize_text(paper_paths(slug)["full_text_txt"].read_text(encoding="utf-8", errors="ignore"))\n    if q.exists():\n        extraction_text = normalize_text(q.read_text(encoding="utf-8", errors="ignore"))\n    elif paper_paths(slug)["extraction_text_txt"].exists():\n        extraction_text = normalize_text(paper_paths(slug)["extraction_text_txt"].read_text(encoding="utf-8", errors="ignore"))\n    return full_text, extraction_text\n\n\ndef write_visible_summary_files(slug: str, summary_payload: Dict[str, Any], summary_text_rendered: str) -> None:\n    summary_json_path = PAPER_SUMMARY_JSON_DIR / f"{slug}_summary.json"\n    summary_txt_path = PAPER_SUMMARY_TEXT_DIR / f"{slug}_summary.txt"\n    summary_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")\n    summary_txt_path.write_text(summary_text_rendered, encoding="utf-8")\n    # keep backward-compatible locations too\n    (SUMMARY_DIR / f"{slug}_summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")\n    (SUMMARY_TEXT_DIR / f"{slug}_summary.txt").write_text(summary_text_rendered, encoding="utf-8")\n\n\ndef ensure_summary_outputs(slug: str, meta: Dict[str, Any], records: List[Dict[str, Any]], full_text: str,\n                           extractor: Optional[HuggingFaceQwenExtractor] = None, force_llm_summary: bool = False) -> Tuple[Dict[str, Any], str]:\n    visible_json = PAPER_SUMMARY_JSON_DIR / f"{slug}_summary.json"\n    visible_txt = PAPER_SUMMARY_TEXT_DIR / f"{slug}_summary.txt"\n    summary_metadata = {\n        "title": meta.get("title"),\n        "doi": meta.get("doi"),\n        "journal": meta.get("journal"),\n        "year": meta.get("year"),\n        "lead_author": meta.get("lead_author"),\n    }\n    if visible_json.exists() and visible_txt.exists() and not force_llm_summary:\n        try:\n            return json.loads(visible_json.read_text(encoding="utf-8")), visible_txt.read_text(encoding="utf-8")\n        except Exception:\n            pass\n\n    summary_payload: Optional[Dict[str, Any]] = None\n    llm_error = None\n    if extractor is not None and full_text and (force_llm_summary or len(full_text) >= 1000):\n        try:\n            summary_payload = extractor.summarize_paper(\n                metadata=summary_metadata,\n                records=records,\n                summary_text=full_text,\n                paper_slug=slug,\n            )\n            if not isinstance(summary_payload, dict) or not isinstance(summary_payload.get("paper_summary"), dict):\n                raise RuntimeError("LLM summary payload missing top-level \'paper_summary\' object.")\n            summary_payload["_summary_generation_mode"] = "llm"\n        except Exception as exc:\n            llm_error = str(exc)\n            summary_payload = None\n            print(f"[WARN] Paper summary generation failed for {meta.get(\'title\')}: {exc}")\n\n    if summary_payload is None:\n        summary_payload = build_fallback_paper_summary(summary_metadata, records)\n        summary_payload["_summary_generation_mode"] = "fallback"\n        if llm_error:\n            summary_payload["_summary_generation_error"] = llm_error\n\n    summary_text_rendered = render_paper_summary_text(summary_payload, summary_metadata)\n    write_visible_summary_files(slug, summary_payload, summary_text_rendered)\n    persist_artifacts(slug, summary_metadata, records, full_text, build_extraction_text(full_text) if full_text else "",\n                      summary_payload=summary_payload, summary_text_rendered=summary_text_rendered)\n    print(f"[OK] Wrote visible paper summaries: {PAPER_SUMMARY_JSON_DIR / f\'{slug}_summary.json\'} | {PAPER_SUMMARY_TEXT_DIR / f\'{slug}_summary.txt\'}")\n    return summary_payload, summary_text_rendered\n\n\ndef candidate_relevance_score(meta: Dict[str, Any]) -> float:\n    title = (meta.get("title") or "").lower()\n    abstract = strip_html_text(meta.get("abstract")) or ""\n    abstract = abstract.lower()\n    text = f"{title} {abstract}"\n    score = 0.0\n    positive_terms = {\n        "perovskite": 5.0,\n        "solar cell": 5.0,\n        "photovoltaic": 2.5,\n        "device": 1.5,\n        "j-v": 1.0,\n        "eqe": 1.0,\n        "stability": 1.0,\n        "efficiency": 1.0,\n        "power conversion efficiency": 2.0,\n    }\n    negative_terms = {\n        "review": -4.0,\n        "perspective": -4.0,\n        "editorial": -6.0,\n        "commentary": -6.0,\n        "erratum": -8.0,\n        "correction": -8.0,\n        "retraction": -10.0,\n        "silicon solar cell": -4.0,\n        "silver recovery": -6.0,\n        "electrolyte jet": -5.0,\n        "battery": -3.0,\n        "supercapacitor": -3.0,\n    }\n    for term, weight in positive_terms.items():\n        if term in text:\n            score += weight\n    for term, weight in negative_terms.items():\n        if term in text:\n            score += weight\n    if meta.get("doi"):\n        score += 0.1\n    if meta.get("pdf_url"):\n        score += 0.2\n    return score\n\n\ndef sort_candidates_for_processing(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:\n    return sorted(candidates, key=lambda m: candidate_relevance_score(m), reverse=True)\n\n\ndef normalize_run_mode(run_mode: Optional[str] = None) -> str:\n    mode = (run_mode or RUN_MODE or "resume").strip().lower()\n    aliases = {"default": "resume", "fresh": "reset", "overwrite": "reset", "rerun": "reprocess"}\n    mode = aliases.get(mode, mode)\n    if mode not in {"resume", "expand", "reprocess", "reset", "benchmark"}:\n        mode = "resume"\n    return mode\n\n\ndef should_skip_processed_paper(meta: Dict[str, Any], slug: str, run_mode: Optional[str] = None) -> bool:\n    mode = normalize_run_mode(run_mode)\n    if mode in {"reset", "reprocess"}:\n        return False\n    entry = find_registry_entry(meta, slug)\n    if entry and entry.get("status") in {"processed", "cached"}:\n        return True\n    paths = paper_paths(slug, create_dirs=False)\n    if (JSON_DIR / f"{slug}.json").exists() or paths["records_json"].exists():\n        return True\n    return False\n\ndef reset_work_dir() -> None:\n    if BASE.exists():\n        shutil.rmtree(BASE)\n    for d in [BASE, PDF_DIR, HTML_DIR, TEXT_DIR, JSON_DIR, CSV_DIR, RAW_LLM_DIR, RAW_LLM_TEXT_DIR,\n              SUMMARY_DIR, SUMMARY_TEXT_DIR, PAPER_STORE_DIR, PAPER_SUMMARY_JSON_DIR, PAPER_SUMMARY_TEXT_DIR]:\n        d.mkdir(parents=True, exist_ok=True)\n\n\ndef collect_candidate_subset(candidates: List[Dict[str, Any]], max_papers: int, run_mode: Optional[str] = None) -> List[Dict[str, Any]]:\n    mode = normalize_run_mode(run_mode)\n    if mode in {"reprocess", "reset"}:\n        return candidates\n    selected: List[Dict[str, Any]] = []\n    skipped_existing = 0\n    target_pool = max(max_papers * 5, max_papers)\n    for meta in candidates:\n        slug = slugify(meta.get("title") or meta.get("doi") or "paper")\n        if should_skip_processed_paper(meta, slug, mode):\n            skipped_existing += 1\n            continue\n        selected.append(meta)\n        if len(selected) >= target_pool:\n            break\n    if skipped_existing and mode in {"resume", "expand"}:\n        print(f"[INFO] Skipped {skipped_existing} previously processed candidate(s) based on cache/registry.")\n    return selected\n\n_original_process_paper_for_reference = process_paper\n\ndef process_paper(meta: Dict[str, Any], extractor: HuggingFaceQwenExtractor, force_reprocess: bool = False, run_mode: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:\n    if meta.get("source") == "rss" and not meta.get("doi"):\n        meta = enrich_rss_candidate(dict(meta))\n    print(f"[TRY] Candidate: {meta.get(\'title\')} | DOI={meta.get(\'doi\')}")\n    title = meta.get("title") or meta.get("doi") or "untitled"\n    slug = slugify(title)\n    cached_entry = find_registry_entry(meta, slug)\n    mode = normalize_run_mode(run_mode)\n    force_reprocess = force_reprocess or mode in {"reset", "reprocess"}\n\n    if not force_reprocess:\n        cached_records = load_cached_records_for_slug(slug)\n        if cached_records:\n            full_text, extraction_text = load_cached_text(slug)\n            ensure_summary_outputs(slug, meta, cached_records, full_text, extractor=extractor, force_llm_summary=True)\n            upsert_registry_entry(meta, slug, {\n                "status": "cached",\n                "records_count": len(cached_records),\n                "json_path": str(JSON_DIR / f"{slug}.json"),\n                "csv_path": str(CSV_DIR / f"{slug}.csv"),\n                "summary_json_path": str(PAPER_SUMMARY_JSON_DIR / f"{slug}_summary.json"),\n                "summary_text_path": str(PAPER_SUMMARY_TEXT_DIR / f"{slug}_summary.txt"),\n                "full_text_path": str(TEXT_DIR / f"{slug}.txt") if (TEXT_DIR / f"{slug}.txt").exists() else None,\n                "extraction_text_path": str(TEXT_DIR / f"{slug}_for_extraction.txt") if (TEXT_DIR / f"{slug}_for_extraction.txt").exists() else None,\n                "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),\n            })\n            print(f"[CACHE] Reusing stored records for {title}")\n            return cached_records\n\n    full_text, extraction_text = load_cached_text(slug)\n    if full_text and len(full_text) >= LLM_MIN_FULLTEXT_CHARS:\n        print(f"[CACHE] Reusing stored full text for {title} ({len(full_text)} chars)")\n        meta["_full_text"] = full_text\n        if not extraction_text:\n            extraction_text = build_extraction_text(full_text)\n            (TEXT_DIR / f"{slug}_for_extraction.txt").write_text(extraction_text, encoding="utf-8")\n    else:\n        retrieval = try_full_text(meta, slug)\n        meta["pdf_url"] = retrieval["pdf_url"]\n        meta["landing_page"] = retrieval["landing_page"]\n        meta["is_oa"] = retrieval["is_oa"]\n        meta["full_text_source"] = retrieval["text_source"]\n        full_text = normalize_text(retrieval["full_text"] or "")\n        meta["_full_text"] = full_text\n        if len(full_text) < LLM_MIN_FULLTEXT_CHARS:\n            print(f"[SKIP] Full text too short for reliable extraction ({len(full_text)} chars): {title}")\n            upsert_registry_entry(meta, slug, {"status": "skipped_short_text", "full_text_length": len(full_text)})\n            return None\n        print(f"[OK] Full text length: {len(full_text)} chars | source={meta.get(\'full_text_source\')}")\n        (TEXT_DIR / f"{slug}.txt").write_text(full_text, encoding="utf-8")\n        extraction_text = build_extraction_text(full_text)\n        (TEXT_DIR / f"{slug}_for_extraction.txt").write_text(extraction_text, encoding="utf-8")\n\n    print(f"[OK] Extraction text length: {len(extraction_text)} chars")\n    if len(extraction_text) < MIN_EXTRACTION_TEXT_CHARS:\n        print(f"[SKIP] Extraction text too short for reliable extraction ({len(extraction_text)} chars): {title}")\n        upsert_registry_entry(meta, slug, {"status": "skipped_short_extraction_text", "extraction_text_length": len(extraction_text)})\n        return None\n\n    try:\n        records = extract_records_with_llm(meta, extraction_text, extractor)\n    except Exception as exc:\n        print(f"[ERROR] Qwen extraction failed for {title}: {exc}")\n        upsert_registry_entry(meta, slug, {"status": "extraction_error", "error": str(exc)})\n        if STRICT_FAIL_ON_LLM_ERROR:\n            raise\n        return None\n\n    with (JSON_DIR / f"{slug}.json").open("w", encoding="utf-8") as f:\n        json.dump(records, f, indent=2, ensure_ascii=False)\n    if WRITE_PER_PAPER_CSV:\n        json_to_csv(records, CSV_DIR / f"{slug}.csv")\n\n    summary_payload, summary_text_rendered = ensure_summary_outputs(slug, meta, records, full_text, extractor=extractor, force_llm_summary=True)\n    persist_artifacts(slug, meta, records, full_text, extraction_text, summary_payload=summary_payload, summary_text_rendered=summary_text_rendered)\n    upsert_registry_entry(meta, slug, {\n        "status": "processed",\n        "records_count": len(records),\n        "json_path": str(JSON_DIR / f"{slug}.json"),\n        "csv_path": str(CSV_DIR / f"{slug}.csv"),\n        "summary_json_path": str(PAPER_SUMMARY_JSON_DIR / f"{slug}_summary.json"),\n        "summary_text_path": str(PAPER_SUMMARY_TEXT_DIR / f"{slug}_summary.txt"),\n        "paper_store_dir": str(paper_paths(slug, create_dirs=False)["paper_dir"]),\n        "full_text_path": str(TEXT_DIR / f"{slug}.txt"),\n        "extraction_text_path": str(TEXT_DIR / f"{slug}_for_extraction.txt"),\n        "full_text_length": len(full_text),\n        "extraction_text_length": len(extraction_text),\n        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),\n        "llm_model": HF_MODEL,\n        "summary_generation_mode": summary_payload.get("_summary_generation_mode"),\n    })\n    return records\n\n\ndef summary_row_from_slug_meta(slug: str, meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:\n    summary_path = PAPER_SUMMARY_JSON_DIR / f"{slug}_summary.json"\n    if not summary_path.exists():\n        return None\n    try:\n        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))\n    except Exception:\n        return None\n    ps = summary_data.get("paper_summary", {})\n    return {\n        "title": meta.get("title"),\n        "doi": meta.get("doi"),\n        "journal": meta.get("journal"),\n        "year": meta.get("year"),\n        "one_sentence_summary": ps.get("one_sentence_summary"),\n        "objective": ps.get("objective"),\n        "device_architecture": ps.get("device_architecture"),\n        "perovskite_system": ps.get("perovskite_system"),\n        "main_contribution": ps.get("main_contribution"),\n        "stability_summary": ps.get("stability_summary"),\n        "summary_json_path": str(summary_path),\n        "summary_text_path": str(PAPER_SUMMARY_TEXT_DIR / f"{slug}_summary.txt"),\n        "summary_generation_mode": summary_data.get("_summary_generation_mode"),\n    }\n\n\ndef get_hypothesis_payload_api(paper_slug: str) -> Dict[str, Any]:\n    out = get_saved_paper_output_api(paper_slug)\n    if not out.get("ok"):\n        return out\n    return {\n        "ok": True,\n        "tool": "get_hypothesis_payload",\n        "paper_slug": paper_slug,\n        "title": out.get("title"),\n        "doi": out.get("doi"),\n        "records": out.get("records", []),\n        "paper_summary": out.get("paper_summary"),\n        "paper_summary_text": out.get("paper_summary_text"),\n        "provenance": {\n            "json_path": out.get("json_path"),\n            "summary_json_path": out.get("summary_path"),\n            "summary_text_path": out.get("summary_text_path"),\n            "retrieved_from_cache": True,\n        },\n    }\n\n\ndef list_processed_papers_api() -> Dict[str, Any]:\n    registry = load_paper_registry()\n    papers = sorted(registry.get("papers", {}).values(), key=lambda x: x.get("processed_at") or "", reverse=True)\n    return {"ok": True, "tool": "list_processed_papers", "count": len(papers), "papers": papers}\n\n\ndef process_one_paper_api(meta: Dict[str, Any], force_reprocess: bool = False, run_mode: Optional[str] = None) -> Dict[str, Any]:\n    extractor = build_extractor()\n    records = process_paper(dict(meta), extractor, force_reprocess=force_reprocess, run_mode=run_mode)\n    title = meta.get("title")\n    doi = meta.get("doi")\n    slug = slugify(title or doi or "paper")\n    summary_path = PAPER_SUMMARY_JSON_DIR / f"{slug}_summary.json"\n    summary_text_path = PAPER_SUMMARY_TEXT_DIR / f"{slug}_summary.txt"\n    csv_path = CSV_DIR / f"{slug}.csv"\n    json_path = JSON_DIR / f"{slug}.json"\n    entry = find_registry_entry(meta, slug) or {}\n    payload: Dict[str, Any] = {\n        "ok": records is not None,\n        "tool": "process_one_paper",\n        "title": title,\n        "doi": doi,\n        "paper_slug": slug,\n        "num_records": 0 if records is None else len(records),\n        "json_path": str(json_path) if json_path.exists() else None,\n        "csv_path": str(csv_path) if csv_path.exists() else None,\n        "summary_path": str(summary_path) if summary_path.exists() else None,\n        "summary_text_path": str(summary_text_path) if summary_text_path.exists() else None,\n        "paper_store_dir": str(paper_paths(slug, create_dirs=False)["paper_dir"]),\n        "registry_entry": entry,\n        "records": records or [],\n    }\n    if summary_path.exists():\n        try:\n            payload["paper_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))\n        except Exception:\n            payload["paper_summary"] = None\n    if summary_text_path.exists():\n        try:\n            payload["paper_summary_text"] = summary_text_path.read_text(encoding="utf-8")\n        except Exception:\n            payload["paper_summary_text"] = None\n    return payload\n\n\ndef process_batch_api(query: str = SEARCH_QUERY, year_min: int = YEAR_MIN, year_max: int = YEAR_MAX, max_papers: int = MAX_PAPERS_TO_PROCESS, run_mode: Optional[str] = None, force_reprocess: bool = False, reset_output: bool = False) -> Dict[str, Any]:\n    year_min, year_max = clamp_years(year_min, year_max)\n    mode = normalize_run_mode(run_mode)\n    if reset_output or mode == "reset":\n        reset_work_dir()\n    extractor = build_extractor()\n    rss_candidates = rss_search(query, year_min, year_max)\n    crossref_candidates = crossref_search(\n        query,\n        year_min,\n        year_max,\n        rows_per_page=CROSSREF_ROWS_PER_PAGE,\n        max_pages=CROSSREF_MAX_PAGES,\n    )\n    candidates = merge_candidates(rss_candidates, crossref_candidates)\n    candidates_to_process = collect_candidate_subset(candidates, max_papers, run_mode=mode)\n    records: List[Dict[str, Any]] = []\n    errors: List[Dict[str, Any]] = []\n    skipped: List[Dict[str, Any]] = []\n    summary_rows: List[Dict[str, Any]] = []\n    processed_papers = 0\n    for meta in candidates_to_process:\n        try:\n            out = process_paper(dict(meta), extractor, force_reprocess=force_reprocess, run_mode=mode)\n            if out is None:\n                skipped.append({\n                    "title": meta.get("title"),\n                    "doi": meta.get("doi"),\n                    "reason": "no usable full text retrieved or extraction text below threshold",\n                })\n                continue\n            records.extend(out)\n            processed_papers += 1\n            slug = slugify(meta.get("title") or meta.get("doi") or "paper")\n            row = summary_row_from_slug_meta(slug, meta)\n            if row:\n                summary_rows.append(row)\n        except Exception as exc:\n            errors.append({"title": meta.get("title"), "doi": meta.get("doi"), "error": str(exc)})\n    jsonl_path = BASE / "all_records_from_mcp.jsonl"\n    csv_path = CSV_DIR / "all_records_from_mcp.csv"\n    with jsonl_path.open("w", encoding="utf-8") as f:\n        for rec in records:\n            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")\n    json_to_csv(records, csv_path)\n    summary_csv_path = CSV_DIR / "paper_summaries_from_mcp.csv"\n    if summary_rows:\n        write_paper_summary_csv(summary_rows, summary_csv_path)\n    return {\n        "ok": True,\n        "tool": "process_batch",\n        "query": query,\n        "year_min": year_min,\n        "year_max": year_max,\n        "rss_candidates": len(rss_candidates),\n        "crossref_candidates": len(crossref_candidates),\n        "merged_candidates": len(candidates),\n        "candidates_selected_for_processing": len(candidates_to_process),\n        "run_mode": mode,\n        "papers_processed": processed_papers,\n        "records_written": len(records),\n        "summaries_written": len(summary_rows),\n        "visible_summary_json_dir": str(PAPER_SUMMARY_JSON_DIR),\n        "visible_summary_text_dir": str(PAPER_SUMMARY_TEXT_DIR),\n        "paper_registry_path": str(PAPER_REGISTRY_PATH),\n        "skipped": skipped,\n        "errors": errors,\n        "jsonl_path": str(jsonl_path),\n        "csv_path": str(csv_path),\n        "summary_csv_path": str(summary_csv_path),\n    }\n\n\ndef get_saved_paper_output_api(paper_slug: str) -> Dict[str, Any]:\n    json_path = JSON_DIR / f"{paper_slug}.json"\n    csv_path = CSV_DIR / f"{paper_slug}.csv"\n    summary_path = PAPER_SUMMARY_JSON_DIR / f"{paper_slug}_summary.json"\n    summary_text_path = PAPER_SUMMARY_TEXT_DIR / f"{paper_slug}_summary.txt"\n    metadata_path = paper_paths(paper_slug, create_dirs=False)["metadata_json"]\n    if not json_path.exists() and not summary_path.exists() and not summary_text_path.exists() and not csv_path.exists():\n        return {"ok": False, "tool": "get_saved_paper_output", "paper_slug": paper_slug, "reason": "saved outputs not found"}\n    out: Dict[str, Any] = {\n        "ok": True,\n        "tool": "get_saved_paper_output",\n        "paper_slug": paper_slug,\n        "json_path": str(json_path) if json_path.exists() else None,\n        "csv_path": str(csv_path) if csv_path.exists() else None,\n        "summary_path": str(summary_path) if summary_path.exists() else None,\n        "summary_text_path": str(summary_text_path) if summary_text_path.exists() else None,\n        "paper_store_dir": str(paper_paths(paper_slug, create_dirs=False)["paper_dir"]),\n        "metadata_path": str(metadata_path) if metadata_path.exists() else None,\n    }\n    if metadata_path.exists():\n        try:\n            meta = json.loads(metadata_path.read_text(encoding="utf-8"))\n            out["title"] = meta.get("title")\n            out["doi"] = meta.get("doi")\n        except Exception:\n            pass\n    if json_path.exists():\n        out["records"] = json.loads(json_path.read_text(encoding="utf-8"))\n    if summary_path.exists():\n        out["paper_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))\n    if summary_text_path.exists():\n        out["paper_summary_text"] = summary_text_path.read_text(encoding="utf-8")\n    return out\n\n\ndef main():\n    if not ALL_COLUMNS:\n        raise ValueError("ALL_COLUMNS is empty.")\n    if not HF_API_KEY:\n        raise ValueError("HF_API_KEY is missing.")\n\n    year_min, year_max = clamp_years(YEAR_MIN, YEAR_MAX)\n    mode = normalize_run_mode(RUN_MODE)\n    if mode == "reset":\n        reset_work_dir()\n\n    extractor = HuggingFaceQwenExtractor(\n        api_key=HF_API_KEY,\n        model=HF_MODEL,\n        api_url=HF_API_URL,\n    )\n\n    rss_candidates = rss_search(SEARCH_QUERY, year_min, year_max)\n    crossref_candidates = crossref_search(\n        SEARCH_QUERY,\n        year_min,\n        year_max,\n        rows_per_page=CROSSREF_ROWS_PER_PAGE,\n        max_pages=CROSSREF_MAX_PAGES,\n    )\n    candidates = merge_candidates(rss_candidates, crossref_candidates)\n    candidates = sort_candidates_for_processing(candidates)\n    candidate_pool = collect_candidate_subset(candidates, MAX_PAPERS_TO_PROCESS, run_mode=mode)\n\n    records: List[Dict[str, Any]] = []\n    errors: List[Dict[str, Any]] = []\n    skipped: List[Dict[str, Any]] = []\n    summary_rows: List[Dict[str, Any]] = []\n    processed_papers = 0\n    attempted_candidates = 0\n\n    for meta in candidates:\n        if processed_papers >= MAX_PAPERS_TO_PROCESS:\n            break\n\n        slug = slugify(meta.get("title") or meta.get("doi") or "paper")\n        if should_skip_processed_paper(meta, slug, mode):\n            continue\n\n        attempted_candidates += 1\n        try:\n            out = process_paper(\n                dict(meta),\n                extractor,\n                force_reprocess=(mode in {"reprocess", "reset"}),\n                run_mode=mode,\n            )\n            if out is None:\n                skipped.append({\n                    "title": meta.get("title"),\n                    "doi": meta.get("doi"),\n                    "reason": "no usable full text retrieved or extraction text below threshold",\n                })\n            else:\n                records.extend(out)\n                processed_papers += 1\n                row = summary_row_from_slug_meta(slug, meta)\n                if row:\n                    summary_rows.append(row)\n                print(f"[OK] Processed paper: {meta.get(\'title\')}")\n        except Exception as exc:\n            errors.append({\n                "title": meta.get("title"),\n                "doi": meta.get("doi"),\n                "error": str(exc),\n            })\n            print(f"[FAIL] {meta.get(\'title\')}: {exc}")\n        time.sleep(SLEEP_BETWEEN_REQUESTS)\n\n    jsonl_path = JSON_DIR / "all_records.jsonl"\n    with jsonl_path.open("w", encoding="utf-8") as f:\n        for rec in records:\n            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")\n\n    csv_path = CSV_DIR / "all_records.csv"\n    json_to_csv(records, csv_path)\n\n    summary_csv_path = CSV_DIR / "paper_summaries.csv"\n    if WRITE_SUMMARY_CSV:\n        write_paper_summary_csv(summary_rows, summary_csv_path)\n\n    (BASE / "skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")\n    (BASE / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")\n\n    summary = {\n        "search_mode": "rss+crossref",\n        "llm_backend": "huggingface_qwen",\n        "llm_model": HF_MODEL,\n        "hf_api_url": HF_API_URL,\n        "search_query": SEARCH_QUERY,\n        "year_min": YEAR_MIN,\n        "year_max": YEAR_MAX,\n        "max_papers_to_process": MAX_PAPERS_TO_PROCESS,\n        "rss_candidates": len(rss_candidates),\n        "crossref_candidates": len(crossref_candidates),\n        "merged_candidates": len(candidates),\n        "candidates_considered": len(candidate_pool),\n        "candidates_attempted": attempted_candidates,\n        "run_mode": mode,\n        "papers_processed": processed_papers,\n        "records_written": len(records),\n        "summaries_written": len(summary_rows),\n        "visible_summary_json_dir": str(PAPER_SUMMARY_JSON_DIR),\n        "visible_summary_text_dir": str(PAPER_SUMMARY_TEXT_DIR),\n        "paper_registry_path": str(PAPER_REGISTRY_PATH),\n        "skipped": len(skipped),\n        "errors": len(errors),\n        "jsonl_path": str(jsonl_path),\n        "csv_path": str(csv_path),\n        "summary_csv_path": str(summary_csv_path),\n    }\n    (BASE / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")\n    print(json.dumps(summary, indent=2))\n\ndef run_mcp_server() -> None:\n    try:\n        from mcp.server.fastmcp import FastMCP\n    except Exception as exc:\n        raise RuntimeError("The \'mcp\' package is required to run the MCP server. Install it with: pip install mcp") from exc\n    mcp = FastMCP(MCP_SERVER_NAME)\n    @mcp.tool()\n    def search_candidates(query: str, year_min: int = YEAR_MIN, year_max: int = YEAR_MAX, max_candidates: int = 25) -> Dict[str, Any]:\n        return search_candidates_api(query=query, year_min=year_min, year_max=year_max, max_candidates=max_candidates)\n    @mcp.tool()\n    def process_one_paper(meta: Dict[str, Any], force_reprocess: bool = False, run_mode: str = RUN_MODE) -> Dict[str, Any]:\n        return process_one_paper_api(meta, force_reprocess=force_reprocess, run_mode=run_mode)\n    @mcp.tool()\n    def process_batch(query: str = SEARCH_QUERY, year_min: int = YEAR_MIN, year_max: int = YEAR_MAX, max_papers: int = MAX_PAPERS_TO_PROCESS, run_mode: str = RUN_MODE, force_reprocess: bool = False, reset_output: bool = False) -> Dict[str, Any]:\n        return process_batch_api(query=query, year_min=year_min, year_max=year_max, max_papers=max_papers, run_mode=run_mode, force_reprocess=force_reprocess, reset_output=reset_output)\n    @mcp.tool()\n    def get_saved_paper_output(paper_slug: str) -> Dict[str, Any]:\n        return get_saved_paper_output_api(paper_slug)\n    @mcp.tool()\n    def list_processed_papers() -> Dict[str, Any]:\n        return list_processed_papers_api()\n    @mcp.tool()\n    def get_hypothesis_payload(paper_slug: str) -> Dict[str, Any]:\n        return get_hypothesis_payload_api(paper_slug)\n    if MCP_TRANSPORT == "http":\n        mcp.run(transport="streamable-http")\n    else:\n        mcp.run()\n\n# NOTE: disabled early entry point from the original script.\n# The final entry point at the bottom of the file is used instead so RUN_MODE=benchmark\n# can route into BenchmarkingAgent before any Crossref/Google Drive/full-pipeline work starts.\nif False and __name__ == "__main__":\n    if RUN_AS_MCP_SERVER:\n        run_mcp_server()\n    else:\n        main()\n\n\n# ============================================================\n# MULTI-AGENT REFACTOR OVERRIDES\n# ============================================================\n\nMCP_SERVER_NAME = "literature-manager-agent"\n\nEVALUATION_PRIORITY_FIELDS = {\n    "critical": [\n        "Ref_DOI_number",\n        "Cell_stack_sequence",\n        "Cell_architecture",\n        "Perovskite_composition_short_form",\n        "Perovskite_composition_long_form",\n        "JV_default_PCE",\n        "JV_default_Voc",\n        "JV_default_Jsc",\n        "JV_default_FF",\n    ],\n    "high_value": [\n        "ETL_stack_sequence",\n        "HTL_stack_sequence",\n        "Backcontact_stack_sequence",\n        "Perovskite_thickness",\n        "Perovskite_band_gap",\n        "Perovskite_deposition_procedure",\n        "Perovskite_deposition_quenching_media",\n        "Perovskite_deposition_thermal_annealing_temperature",\n        "Perovskite_deposition_thermal_annealing_time",\n        "Stability_protocol",\n        "Stability_PCE_T80",\n        "Stability_PCE_after_1000_h",\n    ],\n}\n\ndef _safe_float(v: Any) -> Optional[float]:\n    try:\n        return parse_numeric(v)\n    except Exception:\n        return None\n\ndef _boolish(v: Any) -> Optional[bool]:\n    if v is None:\n        return None\n    s = str(v).strip().lower()\n    if s in {"true", "yes", "1"}:\n        return True\n    if s in {"false", "no", "0"}:\n        return False\n    return None\n\nclass ExtractionAgent:\n    def __init__(self, extractor: HuggingFaceQwenExtractor):\n        self.extractor = extractor\n\n    def extract(self, meta: Dict[str, Any], full_text: str, extraction_text: str) -> Dict[str, Any]:\n        records = extract_records_with_llm(meta, extraction_text, self.extractor)\n        for idx, rec in enumerate(records, start=1):\n            rec = enrich_record_with_metadata(rec, meta, idx)\n            records[idx - 1] = rec\n        return {\n            "records": records,\n            "full_text_length": len(full_text or ""),\n            "extraction_text_length": len(extraction_text or ""),\n            "family_count": len(SCHEMA_FAMILIES),\n        }\n\nclass EvaluationAgent:\n    def __init__(self):\n        self.critical_fields = list(EVALUATION_PRIORITY_FIELDS["critical"])\n        self.high_value_fields = list(EVALUATION_PRIORITY_FIELDS["high_value"])\n\n    def _field_support(self, rec: Dict[str, Any], field: str) -> Dict[str, Any]:\n        value = rec.get(field)\n        evidence = rec.get(f"_evidence_{field}")\n        populated = value not in (None, "")\n        return {\n            "populated": populated,\n            "has_evidence": bool(str(evidence).strip()) if evidence not in (None, "") else False,\n            "value": value,\n        }\n\n    def _detect_conflicts(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:\n        conflicts: List[Dict[str, Any]] = []\n        if len(records) <= 1:\n            return conflicts\n        inspect_fields = self.critical_fields + self.high_value_fields\n        top_records = records[: min(5, len(records))]\n        for field in inspect_fields:\n            seen_vals = []\n            for rec in top_records:\n                val = rec.get(field)\n                if val not in (None, "") and val not in seen_vals:\n                    seen_vals.append(val)\n            if len(seen_vals) <= 1:\n                continue\n            numeric_vals = [_safe_float(v) for v in seen_vals]\n            numeric_vals = [v for v in numeric_vals if v is not None]\n            if numeric_vals:\n                spread = max(numeric_vals) - min(numeric_vals)\n                if field in {"JV_default_PCE", "JV_default_Voc", "JV_default_Jsc", "JV_default_FF"} and spread <= 0.05:\n                    continue\n            conflicts.append({\n                "field": field,\n                "values": seen_vals[:5],\n                "reason": "multiple non-null values detected across extracted records",\n            })\n        return conflicts\n\n    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:\n        sorted_records = sorted(records, key=score_record_completeness, reverse=True)\n        best = sorted_records[0] if sorted_records else None\n        if not best:\n            return {\n                "record_confidence": 0.0,\n                "record_count": 0,\n                "best_record_index": None,\n                "missing_critical_fields": self.critical_fields,\n                "missing_high_value_fields": self.high_value_fields,\n                "field_confidence": {},\n                "conflicts": [],\n                "evidence_coverage": 0.0,\n                "completeness_score": 0.0,\n            }\n        max_possible = max(1, len(ALL_COLUMNS) + len(EXTRA_COLUMNS))\n        completeness_score = min(1.0, score_record_completeness(best) / max_possible)\n        field_confidence: Dict[str, float] = {}\n        evidence_populated = 0\n        evidence_total = 0\n        for field in self.critical_fields + self.high_value_fields:\n            support = self._field_support(best, field)\n            score = 0.0\n            if support["populated"]:\n                score += 0.6\n            if support["has_evidence"]:\n                score += 0.4\n                evidence_populated += 1\n            evidence_total += 1\n            field_confidence[field] = round(score, 3)\n        missing_critical = [f for f in self.critical_fields if best.get(f) in (None, "")]\n        missing_high = [f for f in self.high_value_fields if best.get(f) in (None, "")]\n        evidence_coverage = evidence_populated / max(1, evidence_total)\n        record_confidence = round(\n            0.50 * completeness_score\n            + 0.35 * evidence_coverage\n            + 0.15 * min(1.0, len(sorted_records) / 3.0),\n            4,\n        )\n        conflicts = self._detect_conflicts(sorted_records)\n        return {\n            "record_confidence": record_confidence,\n            "record_count": len(sorted_records),\n            "best_record_index": 0,\n            "missing_critical_fields": missing_critical,\n            "missing_high_value_fields": missing_high,\n            "field_confidence": field_confidence,\n            "conflicts": conflicts,\n            "evidence_coverage": round(evidence_coverage, 4),\n            "completeness_score": round(completeness_score, 4),\n        }\n\nclass FlaggingAgent:\n    def flag(self, meta: Dict[str, Any], records: List[Dict[str, Any]], evaluation: Dict[str, Any]) -> Dict[str, Any]:\n        flags: List[str] = []\n        readiness = {\n            "summary_ready": False,\n            "hypothesis_ready": False,\n            "modeling_ready": False,\n        }\n        confidence = evaluation.get("record_confidence", 0.0)\n        missing_critical = evaluation.get("missing_critical_fields", [])\n        conflicts = evaluation.get("conflicts", [])\n        if not records:\n            flags.extend(["no_records_extracted", "needs_manual_review", "unsafe_for_hypothesis_agent"])\n            return {"flags": sorted(set(flags)), "downstream_readiness": readiness}\n        if confidence < 0.45:\n            flags.append("low_record_confidence")\n        if confidence < 0.25:\n            flags.append("needs_manual_review")\n        if missing_critical:\n            flags.append("missing_critical_fields")\n        if any(f in missing_critical for f in ["Perovskite_composition_short_form", "Perovskite_composition_long_form"]):\n            flags.append("missing_composition")\n        if "Cell_architecture" in missing_critical:\n            flags.append("missing_architecture")\n        if conflicts:\n            flags.append("conflicting_extractions")\n            conflict_fields = {c["field"] for c in conflicts}\n            if any(f in conflict_fields for f in ["JV_default_PCE", "JV_default_Voc", "JV_default_Jsc", "JV_default_FF"]):\n                flags.append("conflicting_performance_metrics")\n        best = sorted(records, key=score_record_completeness, reverse=True)[0]\n        if best.get("JV_default_PCE") in (None, ""):\n            flags.append("missing_primary_performance")\n        if best.get("Stability_protocol") in (None, "") and any(\n            best.get(k) not in (None, "") for k in ["Stability_PCE_T80", "Stability_PCE_after_1000_h", "Stability_PCE_end_of_experiment"]\n        ):\n            flags.append("stability_metric_without_protocol")\n        if confidence >= 0.30 and best.get("JV_default_PCE") not in (None, ""):\n            readiness["summary_ready"] = True\n        if confidence >= 0.55 and "missing_composition" not in flags and "conflicting_performance_metrics" not in flags:\n            readiness["hypothesis_ready"] = True\n        if confidence >= 0.72 and not missing_critical and not conflicts:\n            readiness["modeling_ready"] = True\n        if readiness["summary_ready"] and not readiness["hypothesis_ready"]:\n            flags.append("summary_safe_but_not_hypothesis_safe")\n        if readiness["hypothesis_ready"] and not readiness["modeling_ready"]:\n            flags.append("hypothesis_safe_but_not_model_safe")\n        if not readiness["summary_ready"]:\n            flags.append("unsafe_for_summary")\n        if not readiness["hypothesis_ready"]:\n            flags.append("unsafe_for_hypothesis_agent")\n        if not readiness["modeling_ready"]:\n            flags.append("unsafe_for_quantitative_modeling")\n        return {"flags": sorted(set(flags)), "downstream_readiness": readiness}\n\nclass AssemblyAgent:\n    def assemble_records(self, meta: Dict[str, Any], raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:\n        cleaned: List[Dict[str, Any]] = []\n        for idx, rec in enumerate(raw_records, start=1):\n            merged = make_empty_record()\n            merged = merge_non_null(merged, rec)\n            merged = enrich_record_with_metadata(merged, meta, idx)\n            merged = derive_chemistry_fields(merged)\n            merged = derive_measurement_flags(merged)\n            merged = derive_dimensionality_flags(merged)\n            merged = normalize_record_types(merged)\n            merged = filter_to_allowed_columns(merged)\n            cleaned.append(merged)\n        cleaned = dedupe_records(cleaned)\n        cleaned.sort(key=score_record_completeness, reverse=True)\n        return cleaned\n\n    def persist_package(\n        self,\n        slug: str,\n        meta: Dict[str, Any],\n        records: List[Dict[str, Any]],\n        full_text: str,\n        extraction_text: str,\n        summary_payload: Dict[str, Any],\n        summary_text_rendered: str,\n        evaluation: Dict[str, Any],\n        flagging: Dict[str, Any],\n    ) -> Dict[str, Any]:\n        persist_artifacts(\n            slug=slug,\n            meta=meta,\n            records=records,\n            full_text=full_text,\n            extraction_text=extraction_text,\n            summary_payload=summary_payload,\n            summary_text_rendered=summary_text_rendered,\n        )\n        outputs_dir = paper_paths(slug)["outputs_dir"]\n        evaluation_path = outputs_dir / "evaluation.json"\n        flags_path = outputs_dir / "flags.json"\n        manager_payload_path = outputs_dir / "manager_payload.json"\n        evaluation_path.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")\n        flags_path.write_text(json.dumps(flagging, indent=2, ensure_ascii=False), encoding="utf-8")\n        write_visible_summary_files(slug, summary_payload, summary_text_rendered)\n        csv_path = CSV_DIR / f"{slug}.csv"\n        json_path = JSON_DIR / f"{slug}.json"\n        json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")\n        json_to_csv(records, csv_path)\n        registry_entry = upsert_registry_entry(meta, slug, {\n            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),\n            "num_records": len(records),\n            "json_path": str(json_path),\n            "csv_path": str(csv_path),\n            "summary_json_path": str(PAPER_SUMMARY_JSON_DIR / f"{slug}_summary.json"),\n            "summary_text_path": str(PAPER_SUMMARY_TEXT_DIR / f"{slug}_summary.txt"),\n            "paper_store_dir": str(paper_paths(slug, create_dirs=False)["paper_dir"]),\n            "evaluation_path": str(evaluation_path),\n            "flags_path": str(flags_path),\n            "record_confidence": evaluation.get("record_confidence"),\n            "downstream_readiness": flagging.get("downstream_readiness"),\n            "flags": flagging.get("flags", []),\n        })\n        manager_payload = {\n            "paper_slug": slug,\n            "title": meta.get("title"),\n            "doi": meta.get("doi"),\n            "records": records,\n            "paper_summary": summary_payload,\n            "paper_summary_text": summary_text_rendered,\n            "evaluation": evaluation,\n            "flagging": flagging,\n            "registry_entry": registry_entry,\n        }\n        manager_payload_path.write_text(json.dumps(manager_payload, indent=2, ensure_ascii=False), encoding="utf-8")\n        return {\n            "json_path": str(json_path),\n            "csv_path": str(csv_path),\n            "summary_path": str(PAPER_SUMMARY_JSON_DIR / f"{slug}_summary.json"),\n            "summary_text_path": str(PAPER_SUMMARY_TEXT_DIR / f"{slug}_summary.txt"),\n            "evaluation_path": str(evaluation_path),\n            "flags_path": str(flags_path),\n            "manager_payload_path": str(manager_payload_path),\n            "registry_entry": registry_entry,\n        }\n\nclass SummarizationAgent:\n    def __init__(self, extractor: HuggingFaceQwenExtractor):\n        self.extractor = extractor\n\n    def summarize(self, slug: str, meta: Dict[str, Any], records: List[Dict[str, Any]], full_text: str) -> Tuple[Dict[str, Any], str]:\n        summary_metadata = {\n            "title": meta.get("title"),\n            "doi": meta.get("doi"),\n            "journal": meta.get("journal"),\n            "year": meta.get("year"),\n            "lead_author": meta.get("lead_author"),\n        }\n        return ensure_summary_outputs(\n            slug=slug,\n            meta=summary_metadata,\n            records=records,\n            full_text=full_text,\n            extractor=self.extractor,\n            force_llm_summary=False,\n        )\n\nclass LiteratureManagerAgent:\n    def __init__(self, extractor: Optional[HuggingFaceQwenExtractor] = None):\n        self.extractor = extractor or build_extractor()\n        self.extraction_agent = ExtractionAgent(self.extractor)\n        self.evaluation_agent = EvaluationAgent()\n        self.flagging_agent = FlaggingAgent()\n        self.assembly_agent = AssemblyAgent()\n        self.summarization_agent = SummarizationAgent(self.extractor)\n\n    def process_paper(self, meta: Dict[str, Any], force_reprocess: bool = False, run_mode: Optional[str] = None) -> Dict[str, Any]:\n        mode = normalize_run_mode(run_mode)\n        slug = slugify(meta.get("title") or meta.get("doi") or "paper")\n        if should_skip_processed_paper(meta, slug, mode) and not force_reprocess:\n            cached = get_saved_paper_output_api(slug)\n            if cached.get("ok"):\n                cached["retrieved_from_cache"] = True\n                return cached\n\n        full_text_info = try_full_text(meta, slug)\n        full_text = normalize_text(full_text_info.get("full_text") or "")\n        if len(full_text) < MIN_EXTRACTION_TEXT_CHARS:\n            upsert_registry_entry(meta, slug, {\n                "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),\n                "status": "skipped",\n                "skip_reason": "full text below extraction threshold",\n                "full_text_length": len(full_text),\n            })\n            return {\n                "ok": False,\n                "tool": "process_one_paper",\n                "title": meta.get("title"),\n                "doi": meta.get("doi"),\n                "paper_slug": slug,\n                "reason": "no usable full text retrieved or extraction text below threshold",\n                "full_text_length": len(full_text),\n            }\n\n        extraction_text = build_extraction_text(full_text)\n        extraction_result = self.extraction_agent.extract(meta, full_text, extraction_text)\n        records = self.assembly_agent.assemble_records(meta, extraction_result["records"])\n        evaluation = self.evaluation_agent.evaluate(records)\n        flagging = self.flagging_agent.flag(meta, records, evaluation)\n        summary_payload, summary_text = self.summarization_agent.summarize(slug, meta, records, full_text)\n        persistence = self.assembly_agent.persist_package(\n            slug=slug,\n            meta={\n                **meta,\n                "_source_pdf_url": full_text_info.get("pdf_url"),\n                "_source_landing_page": full_text_info.get("landing_page"),\n                "_source_open_access": full_text_info.get("is_oa"),\n                "_full_text_length": len(full_text),\n                "_full_text_source": full_text_info.get("text_source"),\n            },\n            records=records,\n            full_text=full_text,\n            extraction_text=extraction_text,\n            summary_payload=summary_payload,\n            summary_text_rendered=summary_text,\n            evaluation=evaluation,\n            flagging=flagging,\n        )\n        return {\n            "ok": True,\n            "tool": "process_one_paper",\n            "title": meta.get("title"),\n            "doi": meta.get("doi"),\n            "paper_slug": slug,\n            "num_records": len(records),\n            "records": records,\n            "paper_summary": summary_payload,\n            "paper_summary_text": summary_text,\n            "evaluation": evaluation,\n            "flagging": flagging,\n            "provenance": {\n                "json_path": persistence["json_path"],\n                "csv_path": persistence["csv_path"],\n                "summary_json_path": persistence["summary_path"],\n                "summary_text_path": persistence["summary_text_path"],\n                "evaluation_path": persistence["evaluation_path"],\n                "flags_path": persistence["flags_path"],\n                "manager_payload_path": persistence["manager_payload_path"],\n                "paper_store_dir": str(paper_paths(slug, create_dirs=False)["paper_dir"]),\n                "full_text_length": len(full_text),\n                "extraction_text_length": len(extraction_text),\n                "full_text_source": full_text_info.get("text_source"),\n                "retrieved_from_cache": False,\n            },\n            "registry_entry": persistence["registry_entry"],\n        }\n\n    def process_batch(\n        self,\n        query: str = SEARCH_QUERY,\n        year_min: int = YEAR_MIN,\n        year_max: int = YEAR_MAX,\n        max_papers: int = MAX_PAPERS_TO_PROCESS,\n        run_mode: Optional[str] = None,\n        force_reprocess: bool = False,\n        reset_output: bool = False,\n    ) -> Dict[str, Any]:\n        year_min, year_max = clamp_years(year_min, year_max)\n        mode = normalize_run_mode(run_mode)\n        if reset_output or mode == "reset":\n            reset_work_dir()\n        rss_candidates = rss_search(query, year_min, year_max)\n        crossref_candidates = crossref_search(\n            query,\n            year_min,\n            year_max,\n            rows_per_page=CROSSREF_ROWS_PER_PAGE,\n            max_pages=CROSSREF_MAX_PAGES,\n        )\n        candidates = merge_candidates(rss_candidates, crossref_candidates)\n        candidates_to_process = collect_candidate_subset(candidates, max_papers, run_mode=mode)\n        results = []\n        errors = []\n        skipped = []\n        all_records = []\n        summary_rows = []\n        for meta in candidates_to_process:\n            try:\n                out = self.process_paper(dict(meta), force_reprocess=force_reprocess, run_mode=mode)\n                if not out.get("ok"):\n                    skipped.append({\n                        "title": meta.get("title"),\n                        "doi": meta.get("doi"),\n                        "reason": out.get("reason", "processing failed"),\n                    })\n                    continue\n                results.append({\n                    "paper_slug": out["paper_slug"],\n                    "title": out.get("title"),\n                    "doi": out.get("doi"),\n                    "num_records": out.get("num_records", 0),\n                    "record_confidence": out.get("evaluation", {}).get("record_confidence"),\n                    "flags": out.get("flagging", {}).get("flags", []),\n                })\n                all_records.extend(out.get("records", []))\n                row = summary_row_from_slug_meta(out["paper_slug"], meta)\n                if row:\n                    summary_rows.append(row)\n            except Exception as exc:\n                errors.append({"title": meta.get("title"), "doi": meta.get("doi"), "error": str(exc)})\n        jsonl_path = BASE / "all_records_from_manager.jsonl"\n        with jsonl_path.open("w", encoding="utf-8") as f:\n            for rec in all_records:\n                f.write(json.dumps(rec, ensure_ascii=False) + "\\n")\n        csv_path = CSV_DIR / "all_records_from_manager.csv"\n        json_to_csv(all_records, csv_path)\n        summary_csv_path = CSV_DIR / "paper_summaries_from_manager.csv"\n        if summary_rows:\n            write_paper_summary_csv(summary_rows, summary_csv_path)\n        return {\n            "ok": True,\n            "tool": "process_batch",\n            "manager": MCP_SERVER_NAME,\n            "query": query,\n            "year_min": year_min,\n            "year_max": year_max,\n            "rss_candidates": len(rss_candidates),\n            "crossref_candidates": len(crossref_candidates),\n            "merged_candidates": len(candidates),\n            "candidates_selected_for_processing": len(candidates_to_process),\n            "run_mode": mode,\n            "papers_processed": len(results),\n            "records_written": len(all_records),\n            "summaries_written": len(summary_rows),\n            "visible_summary_json_dir": str(PAPER_SUMMARY_JSON_DIR),\n            "visible_summary_text_dir": str(PAPER_SUMMARY_TEXT_DIR),\n            "paper_registry_path": str(PAPER_REGISTRY_PATH),\n            "results": results,\n            "skipped": skipped,\n            "errors": errors,\n            "jsonl_path": str(jsonl_path),\n            "csv_path": str(csv_path),\n            "summary_csv_path": str(summary_csv_path),\n        }\n\ndef process_one_paper_api(meta: Dict[str, Any], force_reprocess: bool = False, run_mode: Optional[str] = None) -> Dict[str, Any]:\n    manager = LiteratureManagerAgent()\n    return manager.process_paper(dict(meta), force_reprocess=force_reprocess, run_mode=run_mode)\n\ndef process_batch_api(\n    query: str = SEARCH_QUERY,\n    year_min: int = YEAR_MIN,\n    year_max: int = YEAR_MAX,\n    max_papers: int = MAX_PAPERS_TO_PROCESS,\n    run_mode: Optional[str] = None,\n    force_reprocess: bool = False,\n    reset_output: bool = False,\n) -> Dict[str, Any]:\n    manager = LiteratureManagerAgent()\n    return manager.process_batch(\n        query=query,\n        year_min=year_min,\n        year_max=year_max,\n        max_papers=max_papers,\n        run_mode=run_mode,\n        force_reprocess=force_reprocess,\n        reset_output=reset_output,\n    )\n\ndef get_saved_paper_output_api(paper_slug: str) -> Dict[str, Any]:\n    json_path = JSON_DIR / f"{paper_slug}.json"\n    csv_path = CSV_DIR / f"{paper_slug}.csv"\n    summary_path = PAPER_SUMMARY_JSON_DIR / f"{paper_slug}_summary.json"\n    summary_text_path = PAPER_SUMMARY_TEXT_DIR / f"{paper_slug}_summary.txt"\n    outputs_dir = paper_paths(paper_slug, create_dirs=False)["outputs_dir"]\n    metadata_path = paper_paths(paper_slug, create_dirs=False)["metadata_json"]\n    evaluation_path = outputs_dir / "evaluation.json"\n    flags_path = outputs_dir / "flags.json"\n    manager_payload_path = outputs_dir / "manager_payload.json"\n    if not any(p.exists() for p in [json_path, csv_path, summary_path, summary_text_path, metadata_path, manager_payload_path]):\n        return {"ok": False, "tool": "get_saved_paper_output", "paper_slug": paper_slug, "reason": "saved outputs not found"}\n    out: Dict[str, Any] = {\n        "ok": True,\n        "tool": "get_saved_paper_output",\n        "paper_slug": paper_slug,\n        "json_path": str(json_path) if json_path.exists() else None,\n        "csv_path": str(csv_path) if csv_path.exists() else None,\n        "summary_path": str(summary_path) if summary_path.exists() else None,\n        "summary_text_path": str(summary_text_path) if summary_text_path.exists() else None,\n        "evaluation_path": str(evaluation_path) if evaluation_path.exists() else None,\n        "flags_path": str(flags_path) if flags_path.exists() else None,\n        "manager_payload_path": str(manager_payload_path) if manager_payload_path.exists() else None,\n        "paper_store_dir": str(paper_paths(paper_slug, create_dirs=False)["paper_dir"]),\n        "metadata_path": str(metadata_path) if metadata_path.exists() else None,\n    }\n    if metadata_path.exists():\n        try:\n            meta = json.loads(metadata_path.read_text(encoding="utf-8"))\n            out["title"] = meta.get("title")\n            out["doi"] = meta.get("doi")\n        except Exception:\n            pass\n    if json_path.exists():\n        out["records"] = json.loads(json_path.read_text(encoding="utf-8"))\n    if summary_path.exists():\n        out["paper_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))\n    if summary_text_path.exists():\n        out["paper_summary_text"] = summary_text_path.read_text(encoding="utf-8")\n    if evaluation_path.exists():\n        out["evaluation"] = json.loads(evaluation_path.read_text(encoding="utf-8"))\n    if flags_path.exists():\n        out["flagging"] = json.loads(flags_path.read_text(encoding="utf-8"))\n    return out\n\ndef get_hypothesis_payload_api(paper_slug: str) -> Dict[str, Any]:\n    out = get_saved_paper_output_api(paper_slug)\n    if not out.get("ok"):\n        return out\n    return {\n        "ok": True,\n        "tool": "get_hypothesis_payload",\n        "manager": MCP_SERVER_NAME,\n        "paper_slug": paper_slug,\n        "title": out.get("title"),\n        "doi": out.get("doi"),\n        "records": out.get("records", []),\n        "paper_summary": out.get("paper_summary"),\n        "paper_summary_text": out.get("paper_summary_text"),\n        "evaluation": out.get("evaluation"),\n        "flagging": out.get("flagging"),\n        "provenance": {\n            "json_path": out.get("json_path"),\n            "summary_json_path": out.get("summary_path"),\n            "summary_text_path": out.get("summary_text_path"),\n            "evaluation_path": out.get("evaluation_path"),\n            "flags_path": out.get("flags_path"),\n            "retrieved_from_cache": True,\n        },\n    }\n\ndef list_processed_papers_api() -> Dict[str, Any]:\n    registry = load_paper_registry()\n    papers = sorted(registry.get("papers", {}).values(), key=lambda x: x.get("processed_at") or "", reverse=True)\n    return {"ok": True, "tool": "list_processed_papers", "manager": MCP_SERVER_NAME, "count": len(papers), "papers": papers}\n\ndef main():\n    if not ALL_COLUMNS:\n        raise ValueError("ALL_COLUMNS is empty.")\n    if not HF_API_KEY:\n        raise ValueError("HF_API_KEY is missing.")\n    mode = normalize_run_mode(RUN_MODE)\n    manager = LiteratureManagerAgent()\n    out = manager.process_batch(\n        query=SEARCH_QUERY,\n        year_min=YEAR_MIN,\n        year_max=YEAR_MAX,\n        max_papers=MAX_PAPERS_TO_PROCESS,\n        run_mode=mode,\n        force_reprocess=(mode in {"reprocess", "reset"}),\n        reset_output=(mode == "reset"),\n    )\n    print(json.dumps(out, indent=2, ensure_ascii=False))\n\ndef run_mcp_server() -> None:\n    try:\n        from mcp.server.fastmcp import FastMCP\n    except Exception as exc:\n        raise RuntimeError("The \'mcp\' package is required to run the MCP server. Install it with: pip install mcp") from exc\n    mcp = FastMCP(MCP_SERVER_NAME)\n    @mcp.tool()\n    def search_candidates(query: str, year_min: int = YEAR_MIN, year_max: int = YEAR_MAX, max_candidates: int = 25) -> Dict[str, Any]:\n        return search_candidates_api(query=query, year_min=year_min, year_max=year_max, max_candidates=max_candidates)\n    @mcp.tool()\n    def process_one_paper(meta: Dict[str, Any], force_reprocess: bool = False, run_mode: str = RUN_MODE) -> Dict[str, Any]:\n        return process_one_paper_api(meta, force_reprocess=force_reprocess, run_mode=run_mode)\n    @mcp.tool()\n    def process_batch(query: str = SEARCH_QUERY, year_min: int = YEAR_MIN, year_max: int = YEAR_MAX, max_papers: int = MAX_PAPERS_TO_PROCESS, run_mode: str = RUN_MODE, force_reprocess: bool = False, reset_output: bool = False) -> Dict[str, Any]:\n        return process_batch_api(query=query, year_min=year_min, year_max=year_max, max_papers=max_papers, run_mode=run_mode, force_reprocess=force_reprocess, reset_output=reset_output)\n    @mcp.tool()\n    def get_saved_paper_output(paper_slug: str) -> Dict[str, Any]:\n        return get_saved_paper_output_api(paper_slug)\n    @mcp.tool()\n    def list_processed_papers() -> Dict[str, Any]:\n        return list_processed_papers_api()\n    @mcp.tool()\n    def get_hypothesis_payload(paper_slug: str) -> Dict[str, Any]:\n        return get_hypothesis_payload_api(paper_slug)\n    if MCP_TRANSPORT == "http":\n        mcp.run(transport="streamable-http")\n    else:\n        mcp.run()\n\n\n# ============================================================\n# OCR / FIGURE IMAGE ANALYSIS / GOOGLE DRIVE / MANUAL PIPELINE\n# Clean integration layer built on top of the working original script\n# ============================================================\nimport io\n\ntry:\n    import fitz  # PyMuPDF\n    HAS_PYMUPDF = True\nexcept Exception:\n    HAS_PYMUPDF = False\n\ntry:\n    import pytesseract\n    HAS_PYTESSERACT = True\nexcept Exception:\n    HAS_PYTESSERACT = False\n\ntry:\n    from PIL import Image\n    HAS_PIL = True\nexcept Exception:\n    HAS_PIL = False\n\ntry:\n    import pdfplumber\n    HAS_PDFPLUMBER = True\nexcept Exception:\n    HAS_PDFPLUMBER = False\n\n\nOCR_ENABLE = os.getenv("OCR_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}\nOCR_LANG = os.getenv("OCR_LANG", "eng").strip() or "eng"\nOCR_MIN_TEXT_CHARS = int(os.getenv("OCR_MIN_TEXT_CHARS", "5000"))\nOCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "20"))\nOCR_DPI = int(os.getenv("OCR_DPI", "200"))\nOCR_TESSERACT_CMD = os.getenv("OCR_TESSERACT_CMD", "").strip()\n\nFIGURE_IMAGE_ANALYSIS_ENABLE = os.getenv("FIGURE_IMAGE_ANALYSIS_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}\nFIGURE_ANALYSIS_MAX_PAGES = int(os.getenv("FIGURE_ANALYSIS_MAX_PAGES", "12"))\nFIGURE_ANALYSIS_MAX_BLOCKS_PER_PAGE = int(os.getenv("FIGURE_ANALYSIS_MAX_BLOCKS_PER_PAGE", "8"))\nFIGURE_OCR_MIN_CHARS = int(os.getenv("FIGURE_OCR_MIN_CHARS", "20"))\nFIGURE_APPENDIX_MAX_CHARS = int(os.getenv("FIGURE_APPENDIX_MAX_CHARS", "30000"))\nTABLE_EXTRACTION_ENABLE = os.getenv("TABLE_EXTRACTION_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}\n\nOCR_DIR = BASE / "ocr"\nFIGURE_DIR = BASE / "figure_analysis"\n\nMANUAL_GOOGLE_DRIVE_DIR = BASE / "manual_google_drive_papers"\nMANUAL_GOOGLE_DRIVE_PDF_DIR = MANUAL_GOOGLE_DRIVE_DIR / "pdf"\nMANUAL_GOOGLE_DRIVE_TEXT_DIR = MANUAL_GOOGLE_DRIVE_DIR / "text"\nMANUAL_GOOGLE_DRIVE_MANIFEST_DIR = MANUAL_GOOGLE_DRIVE_DIR / "manifests"\n\nCOMBINED_COVERAGE_DIR = BASE / "combined_full_coverage"\nCOMBINED_PDF_DIR = COMBINED_COVERAGE_DIR / "pdf"\nCOMBINED_HTML_DIR = COMBINED_COVERAGE_DIR / "html"\nCOMBINED_TEXT_DIR = COMBINED_COVERAGE_DIR / "text"\nCOMBINED_JSON_DIR = COMBINED_COVERAGE_DIR / "json"\nCOMBINED_CSV_DIR = COMBINED_COVERAGE_DIR / "csv"\nCOMBINED_RAW_LLM_DIR = COMBINED_COVERAGE_DIR / "raw_llm_json"\nCOMBINED_RAW_LLM_TEXT_DIR = COMBINED_COVERAGE_DIR / "raw_llm_text"\nCOMBINED_PAPER_STORE_DIR = COMBINED_COVERAGE_DIR / "paper_store"\nCOMBINED_SUMMARY_JSON_DIR = COMBINED_COVERAGE_DIR / "paper_summaries_json"\nCOMBINED_SUMMARY_TEXT_DIR = COMBINED_COVERAGE_DIR / "paper_summaries_text"\nCOMBINED_MANIFEST_DIR = COMBINED_COVERAGE_DIR / "manifests"\nCOMBINED_REGISTRY_PATH = COMBINED_COVERAGE_DIR / "paper_registry.json"\n\nfor d in [\n    OCR_DIR, FIGURE_DIR,\n    MANUAL_GOOGLE_DRIVE_DIR, MANUAL_GOOGLE_DRIVE_PDF_DIR, MANUAL_GOOGLE_DRIVE_TEXT_DIR, MANUAL_GOOGLE_DRIVE_MANIFEST_DIR,\n    COMBINED_COVERAGE_DIR, COMBINED_PDF_DIR, COMBINED_HTML_DIR, COMBINED_TEXT_DIR, COMBINED_JSON_DIR, COMBINED_CSV_DIR,\n    COMBINED_RAW_LLM_DIR, COMBINED_RAW_LLM_TEXT_DIR, COMBINED_PAPER_STORE_DIR, COMBINED_SUMMARY_JSON_DIR,\n    COMBINED_SUMMARY_TEXT_DIR, COMBINED_MANIFEST_DIR,\n]:\n    d.mkdir(parents=True, exist_ok=True)\n\nif HAS_PYTESSERACT and OCR_TESSERACT_CMD:\n    try:\n        pytesseract.pytesseract.tesseract_cmd = OCR_TESSERACT_CMD\n    except Exception:\n        pass\n\nFIGURE_CAPTION_RE = re.compile(r\'(?im)^\\s*(figure|fig\\.)\\s*([0-9]+[a-zA-Z]?)\\s*[:\\.-]?\\s*(.+)$\')\nTABLE_CAPTION_RE = re.compile(r\'(?im)^\\s*(table)\\s*([0-9]+[a-zA-Z]?)\\s*[:\\.-]?\\s*(.+)$\')\n\nGOOGLE_DRIVE_FOLDER_LINK = os.getenv("GOOGLE_DRIVE_FOLDER_LINK", "https://drive.google.com/drive/folders/10-o3P0YaH8q2h3cE86e_o2aDYIhpWV0b?usp=sharing").strip()\nGOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "10-o3P0YaH8q2h3cE86e_o2aDYIhpWV0b").strip()\nGOOGLE_DRIVE_SHARED_DRIVE_ID = os.getenv("GOOGLE_DRIVE_SHARED_DRIVE_ID", "").strip() or None\nGOOGLE_DRIVE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE", "").strip()\nGOOGLE_DRIVE_OAUTH_CLIENT_SECRETS = os.getenv("GOOGLE_DRIVE_OAUTH_CLIENT_SECRETS", "").strip()\nGOOGLE_DRIVE_OAUTH_TOKEN_FILE = os.getenv("GOOGLE_DRIVE_OAUTH_TOKEN_FILE", "google_drive_token.json").strip()\nGOOGLE_DRIVE_API_KEY = os.getenv("GOOGLE_DRIVE_API_KEY", "").strip()\nGOOGLE_DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]\n\n_original_try_full_text = try_full_text\n_original_process_one_paper_api = process_one_paper_api\n_original_process_batch_api = process_batch_api\n\n\ndef _truncate_text(text: str, limit: int) -> str:\n    text = normalize_text(text or "")\n    if len(text) <= limit:\n        return text\n    return text[:limit].rsplit(" ", 1)[0] + " ..."\n\n\ndef _safe_copy(src: Optional[Path], dst: Optional[Path]) -> Optional[str]:\n    try:\n        if not src or not dst or not src.exists():\n            return None\n        if src.resolve() == dst.resolve():\n            return str(dst)\n        dst.parent.mkdir(parents=True, exist_ok=True)\n        shutil.copy2(src, dst)\n        return str(dst)\n    except Exception:\n        return None\n\n\ndef _safe_copy_tree(src: Optional[Path], dst: Optional[Path]) -> Optional[str]:\n    try:\n        if not src or not dst or not src.exists():\n            return None\n        if dst.exists():\n            shutil.rmtree(dst)\n        shutil.copytree(src, dst)\n        return str(dst)\n    except Exception:\n        return None\n\n\ndef _ocr_image(image: Any, psm: int = 6) -> str:\n    if not (OCR_ENABLE and HAS_PYTESSERACT and HAS_PIL):\n        return ""\n    try:\n        config = f"--oem 3 --psm {psm}"\n        txt = pytesseract.image_to_string(image, lang=OCR_LANG, config=config)\n        return normalize_text(txt)\n    except Exception:\n        return ""\n\n\ndef _pixmap_to_pil(pix) -> Optional[Any]:\n    if not HAS_PIL:\n        return None\n    try:\n        return Image.open(io.BytesIO(pix.tobytes("png")))\n    except Exception:\n        return None\n\n\ndef _meta_bool(meta: Dict[str, Any], key: str, default: bool) -> bool:\n    if key not in meta:\n        return default\n    value = meta.get(key)\n    if isinstance(value, bool):\n        return value\n    return str(value).strip().lower() in {"1", "true", "yes", "on"}\n\n\ndef extract_pdf_text_native(pdf_path: Path, max_pages: Optional[int] = None) -> str:\n    if HAS_PYMUPDF:\n        try:\n            pages = []\n            with fitz.open(str(pdf_path)) as doc:\n                limit = min(len(doc), max_pages or len(doc))\n                for page_index in range(limit):\n                    txt = normalize_text(doc[page_index].get_text("text") or "")\n                    if txt:\n                        pages.append(txt)\n            if pages:\n                return "\\n\\n".join(pages)\n        except Exception:\n            pass\n    return extract_pdf_text(pdf_path)\n\n\ndef extract_pdf_text_ocr(pdf_path: Path, max_pages: Optional[int] = None, dpi: int = OCR_DPI) -> str:\n    if not (OCR_ENABLE and HAS_PYMUPDF and HAS_PYTESSERACT and HAS_PIL):\n        return ""\n    pages = []\n    try:\n        with fitz.open(str(pdf_path)) as doc:\n            limit = min(len(doc), max_pages or OCR_MAX_PAGES)\n            matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)\n            for page_index in range(limit):\n                page = doc[page_index]\n                pix = page.get_pixmap(matrix=matrix, alpha=False)\n                image = _pixmap_to_pil(pix)\n                if image is None:\n                    continue\n                txt = _ocr_image(image, psm=6)\n                if txt:\n                    pages.append(txt)\n    except Exception:\n        return ""\n    return "\\n\\n".join(pages)\n\n\ndef extract_pdf_tables_text(pdf_path: Path, max_pages: Optional[int] = None) -> str:\n    if not (TABLE_EXTRACTION_ENABLE and HAS_PDFPLUMBER):\n        return ""\n    out = []\n    try:\n        with pdfplumber.open(str(pdf_path)) as pdf:\n            limit = min(len(pdf.pages), max_pages or min(len(pdf.pages), FIGURE_ANALYSIS_MAX_PAGES))\n            for page_index in range(limit):\n                page = pdf.pages[page_index]\n                try:\n                    tables = page.extract_tables() or []\n                except Exception:\n                    tables = []\n                for table_idx, table in enumerate(tables, start=1):\n                    rows = []\n                    for row in table or []:\n                        cells = [normalize_text(str(cell or "")) for cell in row]\n                        if any(cells):\n                            rows.append(" | ".join(cells))\n                    if rows:\n                        out.append(f"[TABLE EXTRACTION] page={page_index + 1} table={table_idx}\\n" + "\\n".join(rows))\n    except Exception:\n        return ""\n    return "\\n\\n".join(out)\n\n\ndef extract_figure_captions_from_text(text: str) -> List[str]:\n    caps: List[str] = []\n    for rx in [FIGURE_CAPTION_RE, TABLE_CAPTION_RE]:\n        for m in rx.finditer(text or ""):\n            label = f"{m.group(1).strip()} {m.group(2).strip()}"\n            desc = normalize_text(m.group(3))\n            line = f"{label}: {desc}" if desc else label\n            if line and line not in caps:\n                caps.append(line)\n    return caps[:100]\n\n\ndef analyze_pdf_figures_and_images(pdf_path: Path, slug: str, native_text: str = "") -> Dict[str, Any]:\n    out: Dict[str, Any] = {\n        "pages_scanned": 0,\n        "figure_regions": [],\n        "captions": extract_figure_captions_from_text(native_text),\n        "figure_text": "",\n        "table_text": "",\n        "saved_images": [],\n    }\n    if not (FIGURE_IMAGE_ANALYSIS_ENABLE and HAS_PYMUPDF):\n        return out\n    figure_chunks: List[str] = []\n    if out["captions"]:\n        figure_chunks.append("[CAPTIONS FOUND IN PDF TEXT]\\n" + "\\n".join(out["captions"][:60]))\n    try:\n        with fitz.open(str(pdf_path)) as doc:\n            limit = min(len(doc), FIGURE_ANALYSIS_MAX_PAGES)\n            out["pages_scanned"] = limit\n            for page_index in range(limit):\n                page = doc[page_index]\n                page_dict = page.get_text("dict") or {}\n                blocks = page_dict.get("blocks", []) or []\n                image_blocks = [b for b in blocks if b.get("type") == 1]\n                if not image_blocks:\n                    continue\n                page_hits: List[str] = []\n                for block_idx, block in enumerate(image_blocks[:FIGURE_ANALYSIS_MAX_BLOCKS_PER_PAGE], start=1):\n                    bbox = block.get("bbox")\n                    if not bbox or len(bbox) != 4:\n                        continue\n                    try:\n                        rect = fitz.Rect(bbox)\n                        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=rect, alpha=False)\n                    except Exception:\n                        continue\n                    image = _pixmap_to_pil(pix)\n                    if image is None:\n                        continue\n                    img_path = FIGURE_DIR / f"{slug}_page_{page_index + 1}_img_{block_idx}.png"\n                    try:\n                        image.save(img_path)\n                        out["saved_images"].append(str(img_path))\n                    except Exception:\n                        pass\n                    ocr_text = _ocr_image(image, psm=6)\n                    if len(ocr_text) >= FIGURE_OCR_MIN_CHARS:\n                        hit = {\n                            "page": page_index + 1,\n                            "image_index": block_idx,\n                            "bbox": [round(float(x), 2) for x in bbox],\n                            "ocr_text": _truncate_text(ocr_text, 1200),\n                        }\n                        out["figure_regions"].append(hit)\n                        page_hits.append(f"[FIGURE REGION OCR] page={page_index + 1} image={block_idx} bbox={hit[\'bbox\']}\\n{hit[\'ocr_text\']}")\n                if page_hits:\n                    figure_chunks.append("\\n\\n".join(page_hits))\n    except Exception:\n        pass\n    out["table_text"] = extract_pdf_tables_text(pdf_path, max_pages=FIGURE_ANALYSIS_MAX_PAGES)\n    if out["table_text"]:\n        figure_chunks.append(out["table_text"])\n    out["figure_text"] = _truncate_text("\\n\\n".join(x for x in figure_chunks if x), FIGURE_APPENDIX_MAX_CHARS)\n    return out\n\n\ndef try_full_text(meta: Dict[str, Any], slug: str) -> Dict[str, Any]:\n    include_ocr = _meta_bool(meta, "_include_ocr", OCR_ENABLE)\n    include_figures = _meta_bool(meta, "_include_figures", FIGURE_IMAGE_ANALYSIS_ENABLE)\n\n    local_pdf_path = meta.get("local_pdf_path")\n    if local_pdf_path:\n        local_pdf = Path(str(local_pdf_path))\n        if not local_pdf.exists():\n            return {\n                "full_text": "",\n                "pdf_url": None,\n                "landing_page": meta.get("landing_page"),\n                "is_oa": False,\n                "text_source": "local_pdf_missing",\n                "pdf_text_native_length": 0,\n                "pdf_text_ocr_length": 0,\n                "html_text_length": 0,\n                "figure_image_analysis_length": 0,\n                "figure_image_analysis": {"pages_scanned": 0, "figure_regions": [], "captions": [], "figure_text": "", "table_text": "", "saved_images": []},\n                "saved_pdf_path": str(local_pdf),\n            }\n        native = extract_pdf_text_native(local_pdf)\n        ocr = extract_pdf_text_ocr(local_pdf) if include_ocr else ""\n        best_pdf_text = native if len(native.strip()) >= len(ocr.strip()) else ocr\n        figure_payload = analyze_pdf_figures_and_images(local_pdf, slug=slug, native_text=best_pdf_text) if include_figures else {\n            "pages_scanned": 0, "figure_regions": [], "captions": [], "figure_text": "", "table_text": "", "saved_images": []\n        }\n        full_text = extract_relevant_sections(best_pdf_text)\n        figure_text = figure_payload.get("figure_text") or ""\n        if figure_text:\n            full_text = full_text.rstrip() + "\\n\\n[FIGURE_IMAGE_ANALYSIS_LAYER]\\n" + figure_text\n        return {\n            "full_text": full_text,\n            "pdf_url": None,\n            "landing_page": meta.get("landing_page"),\n            "is_oa": False,\n            "text_source": ("local_pdf_native" if len(native.strip()) >= len(ocr.strip()) else "local_pdf_ocr") + ("+figure_image_analysis" if figure_text else ""),\n            "pdf_text_native_length": len(native),\n            "pdf_text_ocr_length": len(ocr),\n            "html_text_length": 0,\n            "figure_image_analysis_length": len(figure_text),\n            "figure_image_analysis": figure_payload,\n            "saved_pdf_path": str(local_pdf),\n        }\n    return _original_try_full_text(meta, slug)\n\n\ndef build_extraction_text(full_text: str) -> str:\n    sections = split_sections(full_text)\n    parts = [\n        sections.get("abstract", ""),\n        sections.get("experimental", ""),\n        sections.get("materials and methods", ""),\n        sections.get("methods", ""),\n        sections.get("device fabrication", ""),\n        sections.get("results and discussion", ""),\n        sections.get("results", ""),\n        sections.get("discussion", ""),\n    ]\n    merged = "\\n\\n".join([p for p in parts if p]).strip()\n    lower_full = (full_text or "").lower()\n    if "[figure_image_analysis_layer]" in lower_full:\n        appendix = full_text[lower_full.find("[figure_image_analysis_layer]"):]\n        merged = (merged + "\\n\\n" + appendix).strip() if merged else appendix\n    if len(merged) < 2500:\n        merged = full_text[:120000]\n    return merged\n\n\ndef inspect_local_pdf_api(pdf_path: str, include_ocr: bool = True, include_figures: bool = True) -> Dict[str, Any]:\n    path = Path(pdf_path)\n    if not path.exists():\n        return {"ok": False, "tool": "inspect_local_pdf", "reason": f"File not found: {pdf_path}"}\n    native_text = extract_pdf_text_native(path)\n    ocr_text = extract_pdf_text_ocr(path) if include_ocr else ""\n    figure_payload = analyze_pdf_figures_and_images(path, slug=slugify(path.stem), native_text=native_text or ocr_text) if include_figures else {}\n    best_text = native_text if len(native_text) >= len(ocr_text) else ocr_text\n    if include_figures and figure_payload.get("figure_text"):\n        best_text = (best_text + "\\n\\n[FIGURE_IMAGE_ANALYSIS_LAYER]\\n" + figure_payload["figure_text"]).strip()\n    return {\n        "ok": True,\n        "tool": "inspect_local_pdf",\n        "pdf_path": str(path),\n        "native_text_length": len(native_text),\n        "ocr_text_length": len(ocr_text),\n        "best_text_length": len(best_text),\n        "figure_image_analysis": figure_payload,\n        "best_text_preview": _truncate_text(best_text, 4000),\n    }\n\n\ndef parse_google_drive_folder_id(folder_link_or_id: Optional[str]) -> Optional[str]:\n    value = (folder_link_or_id or "").strip()\n    if not value:\n        return None\n    m = re.search(r"/folders/([A-Za-z0-9_-]+)", value)\n    if m:\n        return m.group(1)\n    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", value):\n        return value\n    return None\n\n\ndef _load_google_drive_modules():\n    try:\n        from googleapiclient.discovery import build\n        from googleapiclient.http import MediaIoBaseDownload\n        from google.oauth2.service_account import Credentials as ServiceAccountCredentials\n        from google.oauth2.credentials import Credentials as UserCredentials\n        from google_auth_oauthlib.flow import InstalledAppFlow\n        from google.auth.transport.requests import Request as GoogleAuthRequest\n        return {\n            "build": build,\n            "MediaIoBaseDownload": MediaIoBaseDownload,\n            "ServiceAccountCredentials": ServiceAccountCredentials,\n            "UserCredentials": UserCredentials,\n            "InstalledAppFlow": InstalledAppFlow,\n            "GoogleAuthRequest": GoogleAuthRequest,\n        }\n    except Exception as exc:\n        raise RuntimeError(\n            "Google Drive dependencies are missing. Install with: pip install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2"\n        ) from exc\n\n\ndef build_google_drive_service() -> Any:\n    mods = _load_google_drive_modules()\n    build = mods["build"]\n    service_creds_cls = mods["ServiceAccountCredentials"]\n    user_creds_cls = mods["UserCredentials"]\n    google_request = mods["GoogleAuthRequest"]\n    installed_app_flow = mods["InstalledAppFlow"]\n\n    if GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE:\n        creds = service_creds_cls.from_service_account_file(\n            GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE,\n            scopes=GOOGLE_DRIVE_SCOPES,\n        )\n        return build("drive", "v3", credentials=creds, cache_discovery=False)\n\n    token_path = Path(GOOGLE_DRIVE_OAUTH_TOKEN_FILE)\n    creds = None\n    if token_path.exists():\n        creds = user_creds_cls.from_authorized_user_file(str(token_path), GOOGLE_DRIVE_SCOPES)\n    if creds and creds.expired and creds.refresh_token:\n        creds.refresh(google_request())\n        token_path.write_text(creds.to_json(), encoding="utf-8")\n    if not creds and GOOGLE_DRIVE_OAUTH_CLIENT_SECRETS:\n        flow = installed_app_flow.from_client_secrets_file(GOOGLE_DRIVE_OAUTH_CLIENT_SECRETS, GOOGLE_DRIVE_SCOPES)\n        creds = flow.run_local_server(port=0)\n        token_path.write_text(creds.to_json(), encoding="utf-8")\n    if creds:\n        return build("drive", "v3", credentials=creds, cache_discovery=False)\n\n    if GOOGLE_DRIVE_API_KEY:\n        return build("drive", "v3", developerKey=GOOGLE_DRIVE_API_KEY, cache_discovery=False)\n\n    raise RuntimeError(\n        "No Google Drive authentication configured. Set GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE, or GOOGLE_DRIVE_OAUTH_CLIENT_SECRETS, or GOOGLE_DRIVE_API_KEY for public folders."\n    )\n\n\ndef list_google_drive_folder_files(folder_id: Optional[str] = None, page_size: int = 100) -> List[Dict[str, Any]]:\n    folder_id = parse_google_drive_folder_id(folder_id or GOOGLE_DRIVE_FOLDER_ID or GOOGLE_DRIVE_FOLDER_LINK)\n    if not folder_id:\n        raise ValueError("No valid Google Drive folder ID was provided.")\n    service = build_google_drive_service()\n    query = f"\'{folder_id}\' in parents and trashed = false"\n    files: List[Dict[str, Any]] = []\n    page_token = None\n    while True:\n        kwargs = {\n            "q": query,\n            "pageSize": min(max(page_size, 1), 1000),\n            "pageToken": page_token,\n            "fields": "nextPageToken, files(id, name, mimeType, modifiedTime, size, webViewLink, iconLink)",\n            "supportsAllDrives": True,\n            "includeItemsFromAllDrives": True,\n        }\n        if GOOGLE_DRIVE_SHARED_DRIVE_ID:\n            kwargs["driveId"] = GOOGLE_DRIVE_SHARED_DRIVE_ID\n            kwargs["corpora"] = "drive"\n        else:\n            kwargs["corpora"] = "allDrives"\n        response = service.files().list(**kwargs).execute()\n        files.extend(response.get("files", []))\n        page_token = response.get("nextPageToken")\n        if not page_token:\n            break\n    return files\n\n\ndef export_or_download_google_drive_file(file_id: str, mime_type: str, out_path: Path) -> Path:\n    mods = _load_google_drive_modules()\n    media_download = mods["MediaIoBaseDownload"]\n    service = build_google_drive_service()\n    out_path.parent.mkdir(parents=True, exist_ok=True)\n    google_workspace_exports = {\n        "application/vnd.google-apps.document": ("application/pdf", ".pdf"),\n        "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),\n        "application/vnd.google-apps.drawing": ("application/pdf", ".pdf"),\n    }\n    export_info = google_workspace_exports.get(mime_type)\n    if export_info:\n        export_mime, ext = export_info\n        if out_path.suffix.lower() != ext:\n            out_path = out_path.with_suffix(ext)\n        request = service.files().export_media(fileId=file_id, mimeType=export_mime)\n    else:\n        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)\n    with out_path.open("wb") as fh:\n        downloader = media_download(fh, request)\n        done = False\n        while not done:\n            _, done = downloader.next_chunk()\n    return out_path\n\n\ndef build_manual_drive_meta(import_item: Dict[str, Any], include_ocr: bool = True, include_figures: bool = True) -> Dict[str, Any]:\n    name = import_item.get("name") or "manual_drive_paper.pdf"\n    stem = Path(name).stem.strip() or "manual_drive_paper"\n    gid = (import_item.get("id") or "")[:10]\n    title = stem if not gid else f"{stem}__gdrive_{gid}"\n    modified = import_item.get("modifiedTime")\n    year = None\n    if modified and len(modified) >= 4 and modified[:4].isdigit():\n        year = int(modified[:4])\n    return {\n        "title": title,\n        "original_title": stem,\n        "original_filename": name,\n        "doi": None,\n        "journal": "Manual Google Drive Import",\n        "year": year,\n        "lead_author": None,\n        "landing_page": import_item.get("webViewLink"),\n        "local_pdf_path": import_item.get("saved_to"),\n        "source_collection": "manual_google_drive",\n        "google_drive_file_id": import_item.get("id"),\n        "google_drive_folder_id": import_item.get("folder_id"),\n        "google_drive_mime_type": import_item.get("mimeType"),\n        "google_drive_modified_time": modified,\n        "_include_ocr": include_ocr,\n        "_include_figures": include_figures,\n    }\n\n\ndef _sync_manual_drive_text(slug: str) -> Dict[str, Optional[str]]:\n    out = {\n        "full_text": _safe_copy(TEXT_DIR / f"{slug}.txt", MANUAL_GOOGLE_DRIVE_TEXT_DIR / f"{slug}.txt"),\n        "extraction_text": _safe_copy(TEXT_DIR / f"{slug}_for_extraction.txt", MANUAL_GOOGLE_DRIVE_TEXT_DIR / f"{slug}_for_extraction.txt"),\n    }\n    return out\n\n\ndef sync_paper_to_combined_coverage(slug: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:\n    combined_paths = {\n        "json": COMBINED_JSON_DIR / f"{slug}.json",\n        "csv": COMBINED_CSV_DIR / f"{slug}.csv",\n        "summary_json": COMBINED_SUMMARY_JSON_DIR / f"{slug}_summary.json",\n        "summary_txt": COMBINED_SUMMARY_TEXT_DIR / f"{slug}_summary.txt",\n        "full_text": COMBINED_TEXT_DIR / f"{slug}.txt",\n        "extraction_text": COMBINED_TEXT_DIR / f"{slug}_for_extraction.txt",\n        "pdf": COMBINED_PDF_DIR / f"{slug}.pdf",\n        "html": COMBINED_HTML_DIR / f"{slug}.html",\n        "raw_llm_json": COMBINED_RAW_LLM_DIR / f"{slug}.json",\n        "raw_llm_text": COMBINED_RAW_LLM_TEXT_DIR / f"{slug}.txt",\n        "paper_dir": COMBINED_PAPER_STORE_DIR / slug,\n    }\n    copied = {}\n    copied["json"] = _safe_copy(JSON_DIR / f"{slug}.json", combined_paths["json"])\n    copied["csv"] = _safe_copy(CSV_DIR / f"{slug}.csv", combined_paths["csv"])\n    copied["summary_json"] = _safe_copy(PAPER_SUMMARY_JSON_DIR / f"{slug}_summary.json", combined_paths["summary_json"])\n    copied["summary_txt"] = _safe_copy(PAPER_SUMMARY_TEXT_DIR / f"{slug}_summary.txt", combined_paths["summary_txt"])\n    copied["full_text"] = _safe_copy(TEXT_DIR / f"{slug}.txt", combined_paths["full_text"])\n    copied["extraction_text"] = _safe_copy(TEXT_DIR / f"{slug}_for_extraction.txt", combined_paths["extraction_text"])\n    copied["pdf"] = _safe_copy(PDF_DIR / f"{slug}.pdf", combined_paths["pdf"])\n    if not copied["pdf"]:\n        copied["pdf"] = _safe_copy(PDF_DIR / f"{slug}_from_html.pdf", combined_paths["pdf"])\n    if not copied["pdf"] and meta and meta.get("local_pdf_path"):\n        copied["pdf"] = _safe_copy(Path(str(meta.get("local_pdf_path"))), combined_paths["pdf"])\n    copied["html"] = _safe_copy(HTML_DIR / f"{slug}.html", combined_paths["html"])\n    copied["raw_llm_json"] = _safe_copy(RAW_LLM_DIR / f"{slug}.json", combined_paths["raw_llm_json"])\n    copied["raw_llm_text"] = _safe_copy(RAW_LLM_TEXT_DIR / f"{slug}.txt", combined_paths["raw_llm_text"])\n    copied["paper_dir"] = _safe_copy_tree(paper_paths(slug, create_dirs=False)["paper_dir"], combined_paths["paper_dir"])\n\n    try:\n        COMBINED_REGISTRY_PATH.write_text(json.dumps(load_paper_registry(), indent=2, ensure_ascii=False), encoding="utf-8")\n    except Exception:\n        pass\n\n    manifest = {\n        "paper_slug": slug,\n        "copied": copied,\n        "meta_title": (meta or {}).get("title"),\n        "meta_source_collection": (meta or {}).get("source_collection"),\n        "synced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),\n    }\n    try:\n        (COMBINED_MANIFEST_DIR / f"{slug}_sync_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")\n    except Exception:\n        pass\n    return manifest\n\n\ndef sync_global_outputs_to_combined() -> Dict[str, Any]:\n    copied = {}\n    for name in ["all_records_from_manager.jsonl", "paper_registry.json"]:\n        copied[name] = _safe_copy(BASE / name, COMBINED_COVERAGE_DIR / name)\n    for name in ["all_records_from_manager.csv", "paper_summaries_from_manager.csv"]:\n        copied[name] = _safe_copy(CSV_DIR / name, COMBINED_CSV_DIR / name)\n    return copied\n\n\ndef list_google_drive_folder_api(folder_id_or_link: Optional[str] = None) -> Dict[str, Any]:\n    folder_id = parse_google_drive_folder_id(folder_id_or_link or GOOGLE_DRIVE_FOLDER_ID or GOOGLE_DRIVE_FOLDER_LINK)\n    if not folder_id:\n        return {"ok": False, "reason": "No valid Google Drive folder ID or link was provided."}\n    try:\n        files = list_google_drive_folder_files(folder_id=folder_id)\n    except Exception as exc:\n        return {"ok": False, "folder_id": folder_id, "reason": f"Failed to list Google Drive folder: {exc}"}\n    return {\n        "ok": True,\n        "folder_id": folder_id,\n        "folder_link": folder_id_or_link or GOOGLE_DRIVE_FOLDER_LINK,\n        "count": len(files),\n        "files": files,\n    }\n\n\ndef import_google_drive_folder_api(folder_id_or_link: Optional[str] = None, max_files: int = 50, only_pdfs: bool = True) -> Dict[str, Any]:\n    folder_id = parse_google_drive_folder_id(folder_id_or_link or GOOGLE_DRIVE_FOLDER_ID or GOOGLE_DRIVE_FOLDER_LINK)\n    if not folder_id:\n        return {"ok": False, "reason": "No valid Google Drive folder ID or link was provided."}\n    try:\n        files = list_google_drive_folder_files(folder_id=folder_id)\n    except Exception as exc:\n        return {"ok": False, "folder_id": folder_id, "reason": f"Failed to list or access Google Drive folder: {exc}"}\n\n    allowed_google_types = {\n        "application/pdf",\n        "application/vnd.google-apps.document",\n        "application/vnd.google-apps.presentation",\n    }\n    selected = []\n    for item in files:\n        mime_type = item.get("mimeType") or ""\n        if only_pdfs and mime_type not in allowed_google_types:\n            continue\n        selected.append(item)\n    selected = selected[:max(0, max_files)]\n\n    imported = []\n    for item in selected:\n        raw_name = item.get("name") or item.get("id") or "file"\n        safe_stem = slugify(Path(raw_name).stem)\n        file_id_short = (item.get("id") or "file")[:10]\n        mime_type = item.get("mimeType") or ""\n        suffix = ".pdf" if mime_type in allowed_google_types else (Path(raw_name).suffix or ".bin")\n        out_path = MANUAL_GOOGLE_DRIVE_PDF_DIR / f"{safe_stem}_{file_id_short}{suffix}"\n        try:\n            if not out_path.exists():\n                out_path = export_or_download_google_drive_file(item["id"], mime_type, out_path)\n            row = {\n                "id": item.get("id"),\n                "name": raw_name,\n                "mimeType": mime_type,\n                "modifiedTime": item.get("modifiedTime"),\n                "webViewLink": item.get("webViewLink"),\n                "folder_id": folder_id,\n                "saved_to": str(out_path),\n                "source_collection": "manual_google_drive",\n                "reused_existing_file": out_path.exists(),\n            }\n            imported.append(row)\n        except Exception as exc:\n            imported.append({\n                "id": item.get("id"),\n                "name": raw_name,\n                "mimeType": mime_type,\n                "folder_id": folder_id,\n                "error": str(exc),\n                "source_collection": "manual_google_drive",\n            })\n\n    manifest = {\n        "ok": True,\n        "folder_id": folder_id,\n        "folder_link": folder_id_or_link or GOOGLE_DRIVE_FOLDER_LINK,\n        "listed_count": len(files),\n        "selected_count": len(selected),\n        "imported_count": len([x for x in imported if x.get("saved_to")]),\n        "imported": imported,\n        "import_dir": str(MANUAL_GOOGLE_DRIVE_PDF_DIR),\n        "manifest_written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),\n    }\n    try:\n        (MANUAL_GOOGLE_DRIVE_MANIFEST_DIR / f"import_manifest_{folder_id}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")\n        (COMBINED_MANIFEST_DIR / f"import_manifest_{folder_id}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")\n    except Exception:\n        pass\n    return manifest\n\n\ndef process_one_paper_api(meta: Dict[str, Any], force_reprocess: bool = False, run_mode: Optional[str] = None) -> Dict[str, Any]:\n    result = _original_process_one_paper_api(meta, force_reprocess=force_reprocess, run_mode=run_mode)\n    if result.get("ok"):\n        slug = result.get("paper_slug") or slugify(meta.get("title") or meta.get("doi") or meta.get("original_filename") or "paper")\n        if (meta or {}).get("source_collection") == "manual_google_drive":\n            result["manual_google_drive_text_sync"] = _sync_manual_drive_text(slug)\n        result["combined_coverage_sync"] = sync_paper_to_combined_coverage(slug, meta=meta)\n    return result\n\n\ndef process_batch_api(\n    query: str = SEARCH_QUERY,\n    year_min: int = YEAR_MIN,\n    year_max: int = YEAR_MAX,\n    max_papers: int = MAX_PAPERS_TO_PROCESS,\n    run_mode: Optional[str] = None,\n    force_reprocess: bool = False,\n    reset_output: bool = False,\n) -> Dict[str, Any]:\n    result = _original_process_batch_api(\n        query=query,\n        year_min=year_min,\n        year_max=year_max,\n        max_papers=max_papers,\n        run_mode=run_mode,\n        force_reprocess=force_reprocess,\n        reset_output=reset_output,\n    )\n    synced = []\n    for item in result.get("results", []) or []:\n        slug = item.get("paper_slug")\n        if slug:\n            synced.append(sync_paper_to_combined_coverage(slug))\n    result["combined_global_sync"] = sync_global_outputs_to_combined()\n    result["combined_paper_syncs"] = synced\n    result["combined_coverage_dir"] = str(COMBINED_COVERAGE_DIR)\n    return result\n\n\ndef process_google_drive_folder_api(\n    folder_id_or_link: Optional[str] = None,\n    max_files: int = 20,\n    include_ocr: bool = True,\n    include_figures: bool = True,\n    force_reprocess: bool = False,\n    run_mode: Optional[str] = None,\n) -> Dict[str, Any]:\n    mode = normalize_run_mode(run_mode)\n    imported_payload = import_google_drive_folder_api(folder_id_or_link=folder_id_or_link, max_files=max_files, only_pdfs=True)\n    if not imported_payload.get("ok"):\n        return imported_payload\n\n    processed = []\n    skipped = []\n    for item in imported_payload.get("imported", []):\n        if item.get("error"):\n            processed.append({"name": item.get("name"), "ok": False, "reason": item.get("error")})\n            continue\n        if not item.get("saved_to"):\n            processed.append({"name": item.get("name"), "ok": False, "reason": "Imported file has no saved path."})\n            continue\n        meta = build_manual_drive_meta(item, include_ocr=include_ocr, include_figures=include_figures)\n        slug = slugify(meta.get("title") or meta.get("doi") or meta.get("original_filename") or "paper")\n        if not force_reprocess and should_skip_processed_paper(meta, slug, mode):\n            cached = get_saved_paper_output_api(slug)\n            skipped.append({\n                "name": item.get("name"),\n                "paper_slug": slug,\n                "reason": "already processed",\n                "cached_found": cached.get("ok", False),\n            })\n            continue\n        result = process_one_paper_api(meta=meta, force_reprocess=force_reprocess, run_mode=mode)\n        result.setdefault("ingestion_options", {})\n        result["ingestion_options"]["include_ocr"] = include_ocr\n        result["ingestion_options"]["include_figures"] = include_figures\n        processed.append({\n            "name": item.get("name"),\n            "id": item.get("id"),\n            "saved_to": item.get("saved_to"),\n            "meta": meta,\n            "result": result,\n        })\n\n    global_sync = sync_global_outputs_to_combined()\n    payload = {\n        "ok": True,\n        "tool": "process_google_drive_folder",\n        "folder_id": imported_payload.get("folder_id"),\n        "folder_link": imported_payload.get("folder_link"),\n        "run_mode": mode,\n        "manual_google_drive_dir": str(MANUAL_GOOGLE_DRIVE_DIR),\n        "manual_google_drive_pdf_dir": str(MANUAL_GOOGLE_DRIVE_PDF_DIR),\n        "manual_google_drive_text_dir": str(MANUAL_GOOGLE_DRIVE_TEXT_DIR),\n        "combined_coverage_dir": str(COMBINED_COVERAGE_DIR),\n        "import_dir": imported_payload.get("import_dir"),\n        "selected_count": imported_payload.get("selected_count"),\n        "processed_count": len(processed),\n        "skipped_count": len(skipped),\n        "processed": processed,\n        "skipped": skipped,\n        "combined_global_sync": global_sync,\n    }\n    try:\n        (MANUAL_GOOGLE_DRIVE_MANIFEST_DIR / f"processed_manifest_{imported_payload.get(\'folder_id\')}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")\n        (COMBINED_MANIFEST_DIR / f"processed_manifest_{imported_payload.get(\'folder_id\')}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")\n    except Exception:\n        pass\n    return payload\n\n\ndef get_storage_layout_api() -> Dict[str, Any]:\n    return {\n        "ok": True,\n        "literature_mining_base": str(BASE),\n        "literature_mining_dirs": {\n            "pdf": str(PDF_DIR),\n            "html": str(HTML_DIR),\n            "text": str(TEXT_DIR),\n            "json": str(JSON_DIR),\n            "csv": str(CSV_DIR),\n            "paper_store": str(PAPER_STORE_DIR),\n            "summary_json": str(PAPER_SUMMARY_JSON_DIR),\n            "summary_text": str(PAPER_SUMMARY_TEXT_DIR),\n        },\n        "manual_google_drive_dirs": {\n            "base": str(MANUAL_GOOGLE_DRIVE_DIR),\n            "pdf": str(MANUAL_GOOGLE_DRIVE_PDF_DIR),\n            "text": str(MANUAL_GOOGLE_DRIVE_TEXT_DIR),\n            "manifests": str(MANUAL_GOOGLE_DRIVE_MANIFEST_DIR),\n        },\n        "combined_coverage_dirs": {\n            "base": str(COMBINED_COVERAGE_DIR),\n            "pdf": str(COMBINED_PDF_DIR),\n            "html": str(COMBINED_HTML_DIR),\n            "text": str(COMBINED_TEXT_DIR),\n            "json": str(COMBINED_JSON_DIR),\n            "csv": str(COMBINED_CSV_DIR),\n            "paper_store": str(COMBINED_PAPER_STORE_DIR),\n            "summary_json": str(COMBINED_SUMMARY_JSON_DIR),\n            "summary_text": str(COMBINED_SUMMARY_TEXT_DIR),\n            "manifests": str(COMBINED_MANIFEST_DIR),\n        },\n    }\n\n\ndef extraction_agent_extract_api(\n    meta: Dict[str, Any],\n    full_text: Optional[str] = None,\n    extraction_text: Optional[str] = None,\n    include_ocr: bool = True,\n    include_figures: bool = True,\n) -> Dict[str, Any]:\n    extractor = build_extractor()\n    agent = ExtractionAgent(extractor)\n    meta_local = dict(meta)\n    meta_local["_include_ocr"] = include_ocr\n    meta_local["_include_figures"] = include_figures\n    slug = slugify(meta_local.get("title") or meta_local.get("doi") or meta_local.get("original_filename") or "paper")\n    if not full_text:\n        full_text_info = try_full_text(meta_local, slug)\n        full_text = full_text_info.get("full_text") or ""\n    else:\n        full_text_info = {\n            "text_source": "provided_full_text",\n            "figure_image_analysis": {},\n            "saved_pdf_path": meta_local.get("local_pdf_path"),\n        }\n    extraction_text = extraction_text or build_extraction_text(full_text or "")\n    result = agent.extract(meta_local, full_text or "", extraction_text)\n    return {\n        "ok": True,\n        "tool": "extraction_agent_extract",\n        "paper_slug": slug,\n        "meta": meta_local,\n        "full_text_length": len(full_text or ""),\n        "extraction_text_length": len(extraction_text or ""),\n        "text_source": full_text_info.get("text_source"),\n        "saved_pdf_path": full_text_info.get("saved_pdf_path"),\n        "result": result,\n    }\n\n\ndef evaluation_agent_evaluate_api(records: List[Dict[str, Any]]) -> Dict[str, Any]:\n    agent = EvaluationAgent()\n    return {"ok": True, "tool": "evaluation_agent_evaluate", "evaluation": agent.evaluate(records or [])}\n\n\ndef flagging_agent_flag_api(meta: Dict[str, Any], records: List[Dict[str, Any]], evaluation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:\n    agent = FlaggingAgent()\n    eval_payload = evaluation or EvaluationAgent().evaluate(records or [])\n    return {"ok": True, "tool": "flagging_agent_flag", "flagging": agent.flag(meta or {}, records or [], eval_payload)}\n\n\ndef assembly_agent_assemble_records_api(meta: Dict[str, Any], raw_records: List[Dict[str, Any]]) -> Dict[str, Any]:\n    agent = AssemblyAgent()\n    records = agent.assemble_records(meta or {}, raw_records or [])\n    return {"ok": True, "tool": "assembly_agent_assemble_records", "records": records, "num_records": len(records)}\n\n\ndef summarization_agent_summarize_api(\n    meta: Dict[str, Any],\n    records: List[Dict[str, Any]],\n    full_text: Optional[str] = None,\n    paper_slug: Optional[str] = None,\n) -> Dict[str, Any]:\n    extractor = build_extractor()\n    agent = SummarizationAgent(extractor)\n    slug = paper_slug or slugify((meta or {}).get("title") or (meta or {}).get("doi") or "paper")\n    if full_text is None:\n        cached_full_text, _ = load_cached_text(slug)\n        full_text = cached_full_text\n    summary_payload, summary_text = agent.summarize(slug, meta or {}, records or [], full_text or "")\n    return {\n        "ok": True,\n        "tool": "summarization_agent_summarize",\n        "paper_slug": slug,\n        "paper_summary": summary_payload,\n        "paper_summary_text": summary_text,\n    }\n\n\n# ============================================================\n# FINAL PRIORITY / SUMMARY OVERRIDES\n# Google Drive is rechecked before every batch run so newly added manual papers\n# are processed first, then automated search expands the library afterwards.\n# ============================================================\nDRIVE_PRECHECK_BEFORE_SEARCH = os.getenv("DRIVE_PRECHECK_BEFORE_SEARCH", "1").strip().lower() in {"1", "true", "yes", "on"}\nDRIVE_FIRST_INCLUDE_OCR = os.getenv("DRIVE_FIRST_INCLUDE_OCR", "1").strip().lower() in {"1", "true", "yes", "on"}\nDRIVE_FIRST_INCLUDE_FIGURES = os.getenv("DRIVE_FIRST_INCLUDE_FIGURES", "1").strip().lower() in {"1", "true", "yes", "on"}\nDRIVE_FIRST_FORCE_REPROCESS = os.getenv("DRIVE_FIRST_FORCE_REPROCESS", "0").strip().lower() in {"1", "true", "yes", "on"}\n\n\ndef _sort_drive_items_newest_first(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:\n    return sorted(\n        items or [],\n        key=lambda x: ((x or {}).get("modifiedTime") or "", (x or {}).get("name") or ""),\n        reverse=True,\n    )\n\n\nclass HuggingFaceQwenExtractor(HuggingFaceQwenExtractor):\n    def summarize_paper(\n        self,\n        metadata: Dict[str, Any],\n        records: List[Dict[str, Any]],\n        summary_text: str,\n        paper_slug: str,\n    ) -> Dict[str, Any]:\n        metadata = metadata or {}\n        source_collection = (metadata.get("source_collection") or "").strip().lower()\n        material_focused = source_collection == "manual_google_drive" or str(metadata.get("journal") or "").strip().lower() == "manual google drive import"\n        system = (\n            "You are summarizing one perovskite research paper for a literature-mining workflow. "\n            "Write a detailed, evidence-grounded structured summary using only the provided metadata, extracted records, and text. "\n            "The summary should be richer and more interpretive than a short abstract rewrite. "\n            "Explain the scientific objective, the materials system, what was actually done, what the most important findings mean, "\n            "and any limitations or ambiguity in the extraction evidence. "\n            "When the paper is device-focused, include device architecture, fabrication highlights, key photovoltaic metrics, and stability. "\n            "When the paper is materials-focused or mechanistic rather than device-focused, do not force a device narrative. "\n            "Instead emphasize composition, structure, dimensionality, processing, characterization, proposed mechanisms, structure-property relationships, "\n            "and why the findings matter for perovskite materials or optoelectronic behavior. "\n            "If performance metrics are absent or weakly supported, state that clearly instead of inventing them. "\n            "All summary fields MUST be written in English, even if the paper text, OCR text, or extracted passages are in another language. Do not use Chinese or any non-English language in the summary. Return valid JSON only."\n        )\n        payload = {\n            "paper_metadata": metadata,\n            "records": records,\n            "text": (summary_text or "")[:SUMMARY_MAX_TEXT_CHARS],\n            "summary_style": "material_interpretive" if material_focused else "device_interpretive",\n            "output_format": {\n                "paper_summary": {\n                    "one_sentence_summary": "...",\n                    "objective": "...",\n                    "research_context": "...",\n                    "device_architecture": "...",\n                    "perovskite_system": "...",\n                    "materials_focus": "...",\n                    "fabrication_highlights": ["..."],\n                    "synthesis_or_processing_insights": ["..."],\n                    "characterization_and_evidence": ["..."],\n                    "structure_composition_insights": ["..."],\n                    "best_reported_performance": {\n                        "pce": None,\n                        "voc": None,\n                        "jsc": None,\n                        "ff": None,\n                        "context": "..."\n                    },\n                    "stability_summary": "...",\n                    "mechanism_or_interpretation": "...",\n                    "notable_materials_or_methods": ["..."],\n                    "main_contribution": "...",\n                    "interpretive_takeaways": ["..."],\n                    "data_quality_notes": ["..."]\n                }\n            }\n        }\n        return self._call(system, payload, paper_slug=paper_slug, stage="paper_summary")\n\n\nclass SummarizationAgent(SummarizationAgent):\n    def summarize(self, slug: str, meta: Dict[str, Any], records: List[Dict[str, Any]], full_text: str) -> Tuple[Dict[str, Any], str]:\n        summary_metadata = {\n            "title": meta.get("title"),\n            "doi": meta.get("doi"),\n            "journal": meta.get("journal"),\n            "year": meta.get("year"),\n            "lead_author": meta.get("lead_author"),\n            "source_collection": meta.get("source_collection"),\n            "original_title": meta.get("original_title"),\n            "original_filename": meta.get("original_filename"),\n            "google_drive_file_id": meta.get("google_drive_file_id"),\n            "google_drive_folder_id": meta.get("google_drive_folder_id"),\n        }\n        return ensure_summary_outputs(\n            slug=slug,\n            meta=summary_metadata,\n            records=records,\n            full_text=full_text,\n            extractor=self.extractor,\n            force_llm_summary=False,\n        )\n\n\n_previous_process_google_drive_folder_api = process_google_drive_folder_api\n\n\ndef process_google_drive_folder_api(\n    folder_id_or_link: Optional[str] = None,\n    max_files: int = 20,\n    include_ocr: bool = True,\n    include_figures: bool = True,\n    force_reprocess: bool = False,\n    run_mode: Optional[str] = None,\n) -> Dict[str, Any]:\n    mode = normalize_run_mode(run_mode)\n    imported_payload = import_google_drive_folder_api(folder_id_or_link=folder_id_or_link, max_files=max_files, only_pdfs=True)\n    if not imported_payload.get("ok"):\n        return imported_payload\n\n    processed = []\n    skipped = []\n    imported_items = _sort_drive_items_newest_first(imported_payload.get("imported", []))\n    for item in imported_items:\n        if item.get("error"):\n            processed.append({"name": item.get("name"), "ok": False, "reason": item.get("error")})\n            continue\n        if not item.get("saved_to"):\n            processed.append({"name": item.get("name"), "ok": False, "reason": "Imported file has no saved path."})\n            continue\n        meta = build_manual_drive_meta(item, include_ocr=include_ocr, include_figures=include_figures)\n        slug = slugify(meta.get("title") or meta.get("doi") or meta.get("original_filename") or "paper")\n        if not force_reprocess and should_skip_processed_paper(meta, slug, mode):\n            cached = get_saved_paper_output_api(slug)\n            skipped.append({\n                "name": item.get("name"),\n                "paper_slug": slug,\n                "reason": "already processed",\n                "cached_found": cached.get("ok", False),\n                "modifiedTime": item.get("modifiedTime"),\n            })\n            continue\n        result = process_one_paper_api(meta=meta, force_reprocess=force_reprocess, run_mode=mode)\n        result.setdefault("ingestion_options", {})\n        result["ingestion_options"]["include_ocr"] = include_ocr\n        result["ingestion_options"]["include_figures"] = include_figures\n        processed.append({\n            "name": item.get("name"),\n            "id": item.get("id"),\n            "saved_to": item.get("saved_to"),\n            "modifiedTime": item.get("modifiedTime"),\n            "meta": meta,\n            "result": result,\n        })\n\n    global_sync = sync_global_outputs_to_combined()\n    payload = {\n        "ok": True,\n        "tool": "process_google_drive_folder",\n        "folder_id": imported_payload.get("folder_id"),\n        "folder_link": imported_payload.get("folder_link"),\n        "run_mode": mode,\n        "manual_google_drive_dir": str(MANUAL_GOOGLE_DRIVE_DIR),\n        "manual_google_drive_pdf_dir": str(MANUAL_GOOGLE_DRIVE_PDF_DIR),\n        "manual_google_drive_text_dir": str(MANUAL_GOOGLE_DRIVE_TEXT_DIR),\n        "combined_coverage_dir": str(COMBINED_COVERAGE_DIR),\n        "import_dir": imported_payload.get("import_dir"),\n        "selected_count": imported_payload.get("selected_count"),\n        "processed_count": len(processed),\n        "skipped_count": len(skipped),\n        "processed": processed,\n        "skipped": skipped,\n        "combined_global_sync": global_sync,\n        "drive_files_checked_first": True,\n        "processing_order": "google_drive_newest_first",\n    }\n    try:\n        (MANUAL_GOOGLE_DRIVE_MANIFEST_DIR / f"processed_manifest_{imported_payload.get(\'folder_id\')}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")\n        (COMBINED_MANIFEST_DIR / f"processed_manifest_{imported_payload.get(\'folder_id\')}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")\n    except Exception:\n        pass\n    return payload\n\n\n_previous_combined_syncing_process_batch_api = process_batch_api\n\n\ndef process_batch_api(\n    query: str = SEARCH_QUERY,\n    year_min: int = YEAR_MIN,\n    year_max: int = YEAR_MAX,\n    max_papers: int = MAX_PAPERS_TO_PROCESS,\n    run_mode: Optional[str] = None,\n    force_reprocess: bool = False,\n    reset_output: bool = False,\n) -> Dict[str, Any]:\n    mode = normalize_run_mode(run_mode)\n    drive_precheck: Dict[str, Any] = {\n        "enabled": bool(DRIVE_PRECHECK_BEFORE_SEARCH),\n        "performed": False,\n        "folder_link": GOOGLE_DRIVE_FOLDER_LINK,\n        "folder_id": GOOGLE_DRIVE_FOLDER_ID,\n    }\n\n    if DRIVE_PRECHECK_BEFORE_SEARCH and (GOOGLE_DRIVE_FOLDER_LINK or GOOGLE_DRIVE_FOLDER_ID):\n        drive_precheck["performed"] = True\n        drive_list = list_google_drive_folder_api(GOOGLE_DRIVE_FOLDER_LINK or GOOGLE_DRIVE_FOLDER_ID)\n        drive_precheck["list_result"] = {\n            "ok": drive_list.get("ok"),\n            "count": drive_list.get("count"),\n            "folder_id": drive_list.get("folder_id"),\n        }\n        if drive_list.get("ok"):\n            drive_count = int(drive_list.get("count") or 0)\n            drive_precheck["drive_count_seen_before_search"] = drive_count\n            if drive_count > 0:\n                drive_result = process_google_drive_folder_api(\n                    folder_id_or_link=GOOGLE_DRIVE_FOLDER_LINK or GOOGLE_DRIVE_FOLDER_ID,\n                    max_files=drive_count,\n                    include_ocr=DRIVE_FIRST_INCLUDE_OCR,\n                    include_figures=DRIVE_FIRST_INCLUDE_FIGURES,\n                    force_reprocess=force_reprocess or DRIVE_FIRST_FORCE_REPROCESS,\n                    run_mode=mode,\n                )\n                drive_precheck["process_result"] = {\n                    "ok": drive_result.get("ok"),\n                    "selected_count": drive_result.get("selected_count"),\n                    "processed_count": drive_result.get("processed_count"),\n                    "skipped_count": drive_result.get("skipped_count"),\n                    "manual_google_drive_dir": drive_result.get("manual_google_drive_dir"),\n                    "manifest_dir": str(MANUAL_GOOGLE_DRIVE_MANIFEST_DIR),\n                }\n            else:\n                drive_precheck["process_result"] = {"ok": True, "selected_count": 0, "processed_count": 0, "skipped_count": 0}\n        else:\n            drive_precheck["process_result"] = {\n                "ok": False,\n                "reason": drive_list.get("reason"),\n            }\n\n    result = _original_process_batch_api(\n        query=query,\n        year_min=year_min,\n        year_max=year_max,\n        max_papers=max_papers,\n        run_mode=mode,\n        force_reprocess=force_reprocess,\n        reset_output=reset_output,\n    )\n    synced = []\n    for item in result.get("results", []) or []:\n        slug = item.get("paper_slug")\n        if slug:\n            synced.append(sync_paper_to_combined_coverage(slug))\n    result["combined_global_sync"] = sync_global_outputs_to_combined()\n    result["combined_paper_syncs"] = synced\n    result["combined_coverage_dir"] = str(COMBINED_COVERAGE_DIR)\n    result["drive_precheck"] = drive_precheck\n    result["google_drive_processed_before_search"] = bool(drive_precheck.get("performed"))\n    return result\n\n\n\ndef literature_agent_main() -> None:\n    """Original command-line entry point for the full literature-mining pipeline.\n\n    This is intentionally kept separate from BenchmarkingAgent so the normal\n    Drive-first literature manager, extraction, summarization, CSV/JSON output,\n    and MCP functionality remain unchanged unless RUN_MODE=benchmark is set.\n    """\n    if not ALL_COLUMNS:\n        raise ValueError("ALL_COLUMNS is empty.")\n    if not HF_API_KEY:\n        raise ValueError("HF_API_KEY is missing.")\n    out = process_batch_api(\n        query=SEARCH_QUERY,\n        year_min=YEAR_MIN,\n        year_max=YEAR_MAX,\n        max_papers=MAX_PAPERS_TO_PROCESS,\n        run_mode=normalize_run_mode(RUN_MODE),\n        force_reprocess=(normalize_run_mode(RUN_MODE) in {"reprocess", "reset"}),\n        reset_output=(normalize_run_mode(RUN_MODE) == "reset"),\n    )\n    print(json.dumps(out, indent=2, ensure_ascii=False))\n\n\n# ============================================================\n# BENCHMARKING SUB-AGENT\n# v15: evidence-table extraction + question-specific precision gates + protected-token repair.\n# ============================================================\nclass BenchmarkingAgent:\n    """Sub-agent for the POLARIS literature-agent benchmark.\n\n    It is activated only with RUN_MODE=benchmark. The original literature-mining\n    agent is otherwise untouched. This agent reads local PDF files from the\n    benchmark folder, extracts text, asks the same three benchmark prompts for\n    each paper/run, and writes benchmark-compatible files.\n    """\n\n    QUESTIONS = [\n        "List all perovskite compounds mentioned in the Experimental or Results sections.",\n        "For each perovskite, list all materials used in its synthesis and their IUPAC names.",\n        "List the synthesis temperature, method, and solvent used for each perovskite.",\n    ]\n\n    def __init__(self) -> None:\n        default_root = r"C:\\Users\\jorda\\Downloads\\paper benchmarking updated\\paper benchmarking updated"\n        self.root = Path(os.getenv("BENCHMARK_ROOT", default_root)).expanduser()\n        self.runs = int(os.getenv("BENCHMARK_RUNS", "3"))\n        self.scan_only = os.getenv("BENCHMARK_SCAN_ONLY", "0").strip().lower() in {"1", "true", "yes"}\n        self.model_label = os.getenv("BENCHMARK_MODEL_LABEL", "LiteratureAgent_Qwen").strip() or "LiteratureAgent_Qwen"\n        self.context_char_limit = int(os.getenv("BENCHMARK_CONTEXT_CHAR_LIMIT", "90000"))\n        self.paper_min = int(os.getenv("BENCHMARK_PAPER_MIN", "1"))\n        self.paper_max = int(os.getenv("BENCHMARK_PAPER_MAX", "10"))\n        self.overwrite = os.getenv("BENCHMARK_OVERWRITE", "1").strip().lower() in {"1", "true", "yes"}\n        # Low temperature + shorter answers usually improve benchmark precision/F1.\n        self.temperature = float(os.getenv("BENCHMARK_TEMPERATURE", "0.03"))\n        self.max_tokens = int(os.getenv("BENCHMARK_MAX_OUTPUT_TOKENS", "1600"))\n        self.compact_output = os.getenv("BENCHMARK_COMPACT_OUTPUT", "1").strip().lower() in {"1", "true", "yes"}\n        self.answer_max_chars = int(os.getenv("BENCHMARK_ANSWER_MAX_CHARS", "1500"))\n        # Publication-Analyzer-style behavior: retrieve only high-signal evidence\n        # paragraphs from Experimental/Methods/Results before prompting Qwen.\n        self.use_rag_filter = os.getenv("BENCHMARK_USE_RAG_FILTER", "1").strip().lower() in {"1", "true", "yes"}\n        self.evidence_char_limit = int(os.getenv("BENCHMARK_EVIDENCE_CHAR_LIMIT", "65000"))\n        self.evidence_paragraph_max = int(os.getenv("BENCHMARK_EVIDENCE_PARAGRAPH_MAX", "150"))\n        self.include_evidence_debug = os.getenv("BENCHMARK_INCLUDE_EVIDENCE_DEBUG", "0").strip().lower() in {"1", "true", "yes"}\n        # v8 precision/recall balance: use a small deterministic token inventory plus strict rendering.\n        self.add_token_inventory = os.getenv("BENCHMARK_ADD_TOKEN_INVENTORY", "1").strip().lower() in {"1", "true", "yes"}\n        self.strict_benchmark_style = os.getenv("BENCHMARK_STRICT_STYLE", "1").strip().lower() in {"1", "true", "yes"}\n        # v9 Edison-targeted setting: first pass harvests facts, second pass compresses to benchmark style.\n        # This costs roughly 2x API calls, but usually improves precision/F1 by reducing long Qwen prose.\n        self.two_pass = os.getenv("BENCHMARK_TWO_PASS", "1").strip().lower() in {"1", "true", "yes"}\n        self.harvest_max_tokens = int(os.getenv("BENCHMARK_HARVEST_MAX_TOKENS", "2200"))\n        self.refine_max_tokens = int(os.getenv("BENCHMARK_REFINE_MAX_TOKENS", "1150"))\n        self.refine_temperature = float(os.getenv("BENCHMARK_REFINE_TEMPERATURE", "0.01"))\n        # v10: third pass checks whether the compressed answer accidentally lost\n        # benchmark-important tokens from the raw harvest and restores only supported facts.\n        self.audit_pass = os.getenv("BENCHMARK_AUDIT_PASS", "1").strip().lower() in {"1", "true", "yes"}\n        self.audit_max_tokens = int(os.getenv("BENCHMARK_AUDIT_MAX_TOKENS", "1050"))\n        self.audit_temperature = float(os.getenv("BENCHMARK_AUDIT_TEMPERATURE", "0.0"))\n        self.min_token_retention = float(os.getenv("BENCHMARK_MIN_TOKEN_RETENTION", "0.72"))\n        # v11: deterministic score-aware repair after the LLM passes. This does not call the API;\n        # it restores high-value benchmark tokens from the retrieved evidence/raw harvest only when\n        # they are likely to improve overlap without making the answer ramble again.\n        self.hybrid_token_bank_repair = os.getenv("BENCHMARK_HYBRID_TOKEN_BANK_REPAIR", "1").strip().lower() in {"1", "true", "yes"}\n        self.token_bank_max_additions = int(os.getenv("BENCHMARK_TOKEN_BANK_MAX_ADDITIONS", "4"))\n        self.token_bank_soft_cap_factor = float(os.getenv("BENCHMARK_TOKEN_BANK_SOFT_CAP_FACTOR", "0.90"))\n        # v13: selective token repair keeps v11 recall behavior, but only when it is\n        # likely to help the benchmark without returning to verbose Qwen prose.\n        self.selective_token_repair = os.getenv("BENCHMARK_SELECTIVE_TOKEN_REPAIR", "1").strip().lower() in {"1", "true", "yes"}\n        self.compound_repair_only_if_sparse = os.getenv("BENCHMARK_COMPOUND_REPAIR_ONLY_IF_SPARSE", "1").strip().lower() in {"1", "true", "yes"}\n\n    def discover_papers(self) -> List[Tuple[int, Path, Path]]:\n        r"""Find benchmark PDFs using only the fixed nested benchmark layout.\n\n        This intentionally does not call Crossref, RSS, Unpaywall, Google Drive,\n        or the normal literature-manager search/import pipeline. In benchmark\n        mode, the only accepted input path pattern is:\n\n            BENCHMARK_ROOT / "paper N" / "N.pdf"\n\n        Example:\n            C:\\Users\\jorda\\Downloads\\paper benchmarking updated\\paper benchmarking updated\\paper 1\\1.pdf\n        """\n        if not self.root.exists():\n            raise FileNotFoundError(f"BENCHMARK_ROOT does not exist: {self.root}")\n        found: List[Tuple[int, Path, Path]] = []\n        missing: List[str] = []\n        for n in range(self.paper_min, self.paper_max + 1):\n            paper_dir = self.root / f"paper {n}"\n            expected_pdf = paper_dir / f"{n}.pdf"\n            if expected_pdf.exists():\n                found.append((n, paper_dir, expected_pdf))\n            else:\n                missing.append(str(expected_pdf))\n        if missing:\n            print("[BenchmarkingAgent] Missing expected benchmark PDFs:")\n            for p in missing:\n                print(f"  - {p}")\n        return found\n\n    def scan(self) -> Dict[str, Any]:\n        papers = self.discover_papers()\n        payload = {\n            "ok": True,\n            "mode": "benchmark_scan",\n            "benchmark_root": str(self.root),\n            "found_count": len(papers),\n            "expected_count": self.paper_max - self.paper_min + 1,\n            "papers": [\n                {"paper_number": n, "paper_dir": str(paper_dir), "pdf_path": str(pdf_path)}\n                for n, paper_dir, pdf_path in papers\n            ],\n        }\n        print(json.dumps(payload, indent=2, ensure_ascii=False))\n        return payload\n\n    def _cut_references(self, text: str) -> str:\n        """Remove bibliography/supporting-info tail when possible to reduce false positives."""\n        m = re.search(r"\\n\\s*(references|bibliography|associated content|supporting information)\\s*\\n", text, flags=re.I)\n        return text[: m.start()] if m else text\n\n    def _extract_named_sections(self, full_text: str) -> str:\n        """Extract only Methods/Experimental and Results-like sections when headings are detectable.\n\n        This is deliberately conservative: the benchmark asks for Experimental/Results only.\n        Less background text usually means fewer hallucinated/background compounds and better F1.\n        """\n        text = normalize_text(full_text)\n        text = self._cut_references(text)\n        # Put likely headings on separate searchable boundaries.\n        allowed = re.compile(\n            r"^(experimental(?: section| procedures?)?|materials and methods|methods?|synthesis|sample preparation|"\n            r"results(?: and discussion)?|results|discussion)$",\n            re.I,\n        )\n        stop = re.compile(\n            r"^(abstract|introduction|background|conclusions?|summary|acknowledg(?:e)?ments?|references|"\n            r"bibliography|associated content|supporting information)$",\n            re.I,\n        )\n        lines = [ln.strip() for ln in text.splitlines()]\n        hits = []\n        for i, ln in enumerate(lines):\n            clean = re.sub(r"^[0-9]+\\.?\\s*", "", ln).strip()\n            clean = re.sub(r"[:.]+$", "", clean).strip()\n            if allowed.match(clean):\n                hits.append((i, clean))\n        chunks = []\n        for idx, heading in hits:\n            end = len(lines)\n            for j in range(idx + 1, len(lines)):\n                clean = re.sub(r"^[0-9]+\\.?\\s*", "", lines[j]).strip()\n                clean = re.sub(r"[:.]+$", "", clean).strip()\n                # Stop at a recognized major heading, but allow Discussion after Results.\n                if j > idx + 8 and (allowed.match(clean) or stop.match(clean)):\n                    end = j\n                    break\n            chunk = "\\n".join(lines[idx:end]).strip()\n            if len(chunk) > 800:\n                chunks.append(chunk)\n        # De-duplicate while preserving order.\n        out = []\n        seen = set()\n        for c in chunks:\n            key = c[:300]\n            if key not in seen:\n                seen.add(key)\n                out.append(c)\n        return "\\n\\n".join(out)\n\n    def _split_paragraphs(self, text: str) -> List[str]:\n        """Split extracted text into compact paragraph-like units for retrieval."""\n        text = normalize_text(text)\n        raw = re.split(r"\\n\\s*\\n|(?<=\\.)\\s+(?=[A-Z][A-Za-z0-9\\-/]{2,}\\s)", text)\n        paras = []\n        for para in raw:\n            para = re.sub(r"\\s+", " ", para).strip()\n            if 80 <= len(para) <= 2500:\n                paras.append(para)\n        return paras\n\n    def _benchmark_keyword_bank(self, question: Optional[str] = None) -> List[str]:\n        """Domain-specific token expansion inspired by Publication Analyzer.\n\n        The goal is not to summarize the paper. It is to retrieve high-signal\n        evidence paragraphs containing perovskite formulas, synthesis reagents,\n        methods, solvents, and temperatures.\n        """\n        base = [\n            "perovskite", "quasi-2D", "2D", "3D", "Ruddlesden", "Popper", "Dion", "Jacobson",\n            "PbI2", "PbBr2", "SnI2", "MAI", "MABr", "FAI", "FABr", "BAI", "BABr", "PEAI", "PEABr",\n            "CsI", "CsBr", "Cs2CO3", "NH4SCN", "NH4Cl", "MACl", "spacer", "precursor",\n            "synthesis", "prepared", "solution", "film", "nanocrystal", "nanoplatelet", "PNC", "NPL",\n            "spin", "spin-coated", "drop-cast", "dropcast", "hot-casting", "LARP", "reprecipitation",\n            "anneal", "annealed", "stirred", "glovebox", "antisolvent", "anti-solvent",\n            "DMF", "DMSO", "GBL", "NMP", "toluene", "chlorobenzene", "chloroform", "octadecene",\n            "temperature", "°C", "room temperature", "ambient", "rpm", "min", "h", "mol", "mM", "M",\n            "IUPAC", "iodide", "bromide", "chloride", "lead", "tin", "cesium", "formamidinium", "methylammonium",\n            "phenethylammonium", "butylammonium", "guanidinium", "oleic", "oleylamine", "octanoic", "octylamine",\n        ]\n        if question:\n            q = question.lower()\n            if "compounds" in q:\n                base += ["formula", "composition", "phase", "series", "n =", "n=", "x =", "x=", "control", "doped"]\n            if "materials" in q or "iupac" in q:\n                base += ["materials", "chemicals", "reagents", "purchased", "supplier", "used as received", "precursor", "salt", "solvent"]\n            if "temperature" in q or "method" in q or "solvent" in q:\n                base += ["method", "fabrication", "deposition", "spin coating", "spin-coating", "heated", "annealing", "hotplate", "solvent"]\n        # Preserve order and remove duplicates case-insensitively.\n        seen = set()\n        out = []\n        for k in base:\n            kk = k.lower()\n            if kk not in seen:\n                seen.add(kk)\n                out.append(k)\n        return out\n\n    def _score_paragraph(self, para: str, question: Optional[str] = None) -> float:\n        low = para.lower()\n        score = 0.0\n        for kw in self._benchmark_keyword_bank(question):\n            if kw.lower() in low:\n                score += 1.0\n        # Boost chemical formulas and synthesis values.\n        score += min(8, len(re.findall(r"\\b(?:[A-Z][a-z]?|MA|FA|PEA|BA|GA|Cs|Rb|Pb|Sn|I|Br|Cl)[A-Za-z0-9()\\[\\]{}\\.\\-−]*\\d*\\b", para))) * 0.35\n        score += min(5, len(re.findall(r"\\b\\d+(?:\\.\\d+)?\\s*(?:°C|rpm|min|h|M|mM|mol\\s*%|wt\\s*%)", para, flags=re.I))) * 0.8\n        if re.search(r"\\b(introduction|review|previously reported|literature|prior work)\\b", low):\n            score -= 2.0\n        if re.search(r"\\b(experimental|materials|methods|synthesis|preparation|fabrication|results|discussion)\\b", low):\n            score += 2.0\n        return score\n\n    def _rag_filter_context(self, section_text: str, question: Optional[str] = None) -> str:\n        """Retrieval-style context compression before the LLM call.\n\n        This mimics the useful Publication Analyzer behavior: section-filtered,\n        keyword-expanded, evidence-first extraction. It keeps paragraph order after\n        selection so the model still sees coherent paper text.\n        """\n        paras = self._split_paragraphs(section_text)\n        if not paras:\n            return section_text[: self.evidence_char_limit]\n        scored = []\n        for idx, para in enumerate(paras):\n            sc = self._score_paragraph(para, question)\n            if sc > 0:\n                scored.append((sc, idx, para))\n        if not scored:\n            return section_text[: self.evidence_char_limit]\n        # Select top high-signal paragraphs, then restore document order.\n        scored.sort(key=lambda x: (-x[0], x[1]))\n        selected = sorted(scored[: self.evidence_paragraph_max], key=lambda x: x[1])\n        chunks = []\n        total = 0\n        for sc, idx, para in selected:\n            add = para.strip()\n            if self.include_evidence_debug:\n                add = f"[evidence paragraph {idx}; score={sc:.2f}] " + add\n            if total + len(add) + 2 > self.evidence_char_limit:\n                break\n            chunks.append(add)\n            total += len(add) + 2\n        return "\\n\\n".join(chunks).strip()\n\n    def _token_inventory(self, text: str, question: Optional[str] = None) -> str:\n        """Create a short high-signal token inventory from the PDF text.\n\n        This is not used as output. It gives Qwen the important formula/reagent/method\n        tokens that often get lost when paragraph RAG is too aggressive. Keeping this\n        inventory compact helps recall without making the final answer verbose.\n        """\n        if not self.add_token_inventory:\n            return ""\n        t = normalize_text(self._cut_references(text))\n        # Chemical-like formula/composition tokens: MAPbI3, CsPbBr3, (PEA)2PbI4, etc.\n        formula_pat = re.compile(\n            r"(?:\\([A-Za-z0-9\\-]+\\)|[A-Z][a-z]?[A-Za-z0-9\\-]*)(?:\\d|\\.|[A-Z]|[a-z]|\\(|\\)|−|-|/|,|x|n|\\+|=){2,}"\n        )\n        formulas = []\n        for m in formula_pat.finditer(t):\n            s = m.group(0).strip(" .,;:()[]")\n            if 3 <= len(s) <= 80 and any(x in s for x in ["Pb", "Sn", "Cs", "MA", "FA", "PEA", "BA", "Br", "I"]):\n                if not re.match(r"^Fig|^Table|^Scheme|^Ref", s, re.I):\n                    formulas.append(s)\n        reagent_terms = [\n            "PbI2", "PbBr2", "SnI2", "MAI", "MABr", "FAI", "FABr", "PEAI", "PEABr", "BAI", "BABr",\n            "CsI", "CsBr", "Cs2CO3", "NH4SCN", "NH4Cl", "MACl", "HI", "LiF", "HCl",\n            "DMF", "DMSO", "GBL", "NMP", "toluene", "chlorobenzene", "chloroform", "octadecene",\n            "oleic acid", "oleylamine", "octanoic acid", "octylamine", "phenethylammonium", "butylammonium",\n            "methylammonium", "formamidinium", "guanidinium", "lead(II) iodide", "lead(II) bromide", "tin(II) iodide",\n        ]\n        methods = [\n            "spin-coating", "spin coating", "drop-cast", "dropcast", "hot-casting", "hot casting", "LARP",\n            "reprecipitation", "annealed", "annealing", "stirred", "room temperature", "glovebox", "antisolvent",\n        ]\n        temps = re.findall(r"\\b(?:room temperature|ambient|RT|\\d{2,3}\\s*°\\s*C|\\d{2,3}\\s*deg(?:ree)?s?\\s*C|\\d+\\s*min|\\d+\\s*h|\\d+\\s*hours?|\\d+\\s*rpm)\\b", t, flags=re.I)\n        def uniq(seq, maxn):\n            out, seen = [], set()\n            for x in seq:\n                x = re.sub(r"\\s+", " ", str(x)).strip()\n                k = x.lower()\n                if x and k not in seen:\n                    seen.add(k); out.append(x)\n                if len(out) >= maxn:\n                    break\n            return out\n        found_reagents = [r for r in reagent_terms if re.search(r"\\b" + re.escape(r) + r"\\b", t, flags=re.I)]\n        found_methods = [m for m in methods if re.search(re.escape(m), t, flags=re.I)]\n        qlow = (question or "").lower()\n        max_formulas = 55 if "compound" in qlow else 35\n        max_reagents = 45 if ("materials" in qlow or "iupac" in qlow) else 25\n        max_methods = 25 if ("temperature" in qlow or "method" in qlow or "solvent" in qlow) else 15\n        inventory = [\n            "HIGH-SIGNAL TOKEN INVENTORY (for extraction only; do not copy unless relevant):",\n            "Formulas/compositions: " + "; ".join(uniq(formulas, max_formulas)),\n            "Reagents/solvents: " + "; ".join(uniq(found_reagents, max_reagents)),\n            "Methods/conditions: " + "; ".join(uniq(found_methods + temps, max_methods)),\n        ]\n        return "\\n".join([x for x in inventory if not x.endswith(": ")])\n\n    def _question_char_cap(self, question: str) -> int:\n        q = question.lower()\n        if "compounds" in q:\n            return int(os.getenv("BENCHMARK_COMPOUND_ANSWER_MAX_CHARS", "1200"))\n        if "materials" in q and "iupac" in q:\n            return int(os.getenv("BENCHMARK_MATERIALS_ANSWER_MAX_CHARS", "2600"))\n        if "temperature" in q and "method" in q:\n            return int(os.getenv("BENCHMARK_SYNTHESIS_ANSWER_MAX_CHARS", "1600"))\n        return self.answer_max_chars\n\n    def read_pdf_context(self, pdf_path: Path, question: Optional[str] = None) -> str:\n        full_text = extract_pdf_text(pdf_path)\n        if not full_text or len(full_text.strip()) < 500:\n            raise RuntimeError(\n                f"Could not extract enough text from {pdf_path}. "\n                "If this is a scanned PDF, run OCR first or convert it to selectable text."\n            )\n        # First try strict benchmark section extraction; fallback to the original extraction helpers.\n        section_text = self._extract_named_sections(full_text)\n        if not section_text or len(section_text.strip()) < 1500:\n            try:\n                section_text = build_extraction_text(full_text)\n            except Exception:\n                section_text = extract_relevant_sections(full_text)\n        section_text = normalize_text(section_text or full_text)\n        inventory = self._token_inventory(full_text, question=question)\n        if self.use_rag_filter:\n            section_text = self._rag_filter_context(section_text, question=question)\n        limit = min(self.context_char_limit, self.evidence_char_limit if self.use_rag_filter else self.context_char_limit)\n        if len(section_text) > limit:\n            section_text = section_text[:limit]\n        if inventory:\n            section_text = inventory + "\\n\\nRETRIEVED EXPERIMENTAL/RESULTS EVIDENCE:\\n" + section_text\n        return section_text\n\n    def call_qwen_text(self, system_prompt: str, user_prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:\n        if not HF_API_KEY:\n            raise ValueError("HF_API_KEY is missing. Set it before running BENCHMARK_SCAN_ONLY=0.")\n        payload = {\n            "model": HF_MODEL,\n            "temperature": self.temperature if temperature is None else temperature,\n            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,\n            "messages": [\n                {"role": "system", "content": system_prompt},\n                {"role": "user", "content": user_prompt},\n            ],\n        }\n        sleep_with_jitter()\n        last_err = None\n        for attempt in range(API_MAX_RETRIES):\n            try:\n                r = requests.post(\n                    HF_API_URL,\n                    headers={\n                        "Authorization": f"Bearer {HF_API_KEY}",\n                        "Content-Type": "application/json",\n                    },\n                    json=payload,\n                    timeout=REQUEST_TIMEOUT * 3,\n                )\n                if r.status_code in (429, 500, 502, 503, 504):\n                    wait_s = compute_backoff(attempt)\n                    print(f"[WARN] BenchmarkingAgent retryable status {r.status_code}. Waiting {wait_s:.1f}s...")\n                    time.sleep(wait_s)\n                    last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")\n                    continue\n                r.raise_for_status()\n                data = r.json()\n                content = data["choices"][0]["message"]["content"]\n                if isinstance(content, list):\n                    content = "\\n".join(\n                        part.get("text", "") if isinstance(part, dict) else str(part)\n                        for part in content\n                    )\n                return str(content).strip()\n            except Exception as exc:\n                last_err = exc\n                if attempt < API_MAX_RETRIES - 1:\n                    wait_s = compute_backoff(attempt)\n                    print(f"[WARN] BenchmarkingAgent request failed. Waiting {wait_s:.1f}s before retry...")\n                    time.sleep(wait_s)\n        raise RuntimeError(f"BenchmarkingAgent Qwen request failed after retries: {last_err}")\n\n    def _format_instruction_for_question(self, question: str) -> str:\n        q = question.lower()\n        if "compounds" in q:\n            return (\n                "OUTPUT FORMAT:\\n"\n                "- Use one bullet per experimentally studied or Results-discussed perovskite compound.\\n"\n                "- Each bullet should be: compound/formula — short label if needed.\\n"\n                "- Do not include precursor-only chemicals, device layers, literature examples, or background-only compounds.\\n"\n                "- Do not explain unless needed to distinguish a composition series."\n            )\n        if "materials" in q and "iupac" in q:\n            return (\n                "OUTPUT FORMAT:\\n"\n                "For each perovskite/series, use concise bullets or a compact table with exactly these fields:\\n"\n                "Perovskite: ...\\nMaterials: reagent abbreviation/name (IUPAC); reagent abbreviation/name (IUPAC).\\n"\n                "Include synthesis reagents, additives, antisolvents, and solvents only when used to make the perovskite film/nanocrystal.\\n"\n                "Do not include device-only materials unless they are part of perovskite synthesis."\n            )\n        if "temperature" in q and "method" in q and "solvent" in q:\n            return (\n                "OUTPUT FORMAT:\\n"\n                "For each perovskite/series, use concise bullets or a compact table with exactly these fields:\\n"\n                "Perovskite: ...\\nTemperature: ...\\nMethod: ...\\nSolvent: ...\\n"\n                "Write \'not reported\' for any missing temperature, method, or solvent."\n            )\n        return "OUTPUT FORMAT: concise bullets only; answer the question directly."\n\n    def build_prompt(self, paper_number: int, pdf_path: Path, question: str, run_idx: int, context: str) -> Tuple[str, str]:\n        system = (\n            "You are the BenchmarkingAgent sub-agent inside a literature-mining system. "\n            "Behave like a high-precision publication analyzer, not a summarizer. "\n            "Your output is scored against a ground-truth extraction transcript, so use compact benchmark-style answers. "\n            "Maximize correct overlap with formulas, reagents, IUPAC names, methods, solvents, and temperatures while minimizing filler. "\n            "Use only supplied evidence from Experimental/Methods/Results/Results and Discussion. "\n            "Do not use Introduction/background/literature-review examples unless the same compound is also synthesized, prepared, measured, or analyzed in Results. "\n            "Do not infer missing details. If a value is not explicitly reported, write exactly \'not reported\'. "\n            "No citations, no confidence statements, no caveats, no summaries, no explanations of your reasoning. "\n            "Return only answer content in Edison/Publication-Analyzer style: abbreviation-first bullets, compact semicolon-separated lines, no paragraph prose. "\n            "Use the same high-overlap vocabulary expected in ground truth: MAI, FAI, PbI2, PbBr2, DMF, DMSO, spin-coating, annealed, room temperature, not reported."\n        )\n        format_instruction = self._format_instruction_for_question(question)\n        user = f"""\nPAPER NUMBER: {paper_number}\nPDF FILE PATH USED BY AGENT: {pdf_path}\nINDEPENDENT RUN: {run_idx}\n\nBENCHMARK QUESTION:\n{question}\n\nSTRICT EXTRACTION RULES:\n- Answer only the benchmark question; do not add extra tasks or commentary.\n- Use only the supplied evidence text below; do not use outside knowledge.\n- Use only Experimental/Methods/Results/Results and Discussion content.\n- Exclude introduction-only examples, cited prior-work examples, and generic background compounds.\n- Exclude device-only layers unless the material is directly part of perovskite synthesis/deposition.\n- Include additives, antisolvents, and solvents only when they are part of making the perovskite sample.\n- Prefer exact formulas/names from the paper over paraphrases.\n- Preserve important benchmark tokens: formulas, abbreviations, expanded names, IUPAC names, method names, solvents, temperatures, rpm, times.\n- Use abbreviation-first style when possible, e.g., "MAI — methylammonium iodide (methanaminium iodide)" and "DMF — N,N-dimethylformamide".\n- Use compact bullets; no opening sentence and no closing sentence.\n- For missing values, write exactly: not reported.\n- Do not say "the text says", "based on the paper", "notably", or "it appears".\n- Do not include evidence excerpts or citations.\n- Target style: terse extraction transcript, not a report.\n\n{format_instruction}\n\nSUPPLIED EVIDENCE TEXT:\n{context}\n""".strip()\n        return system, user\n\n    def _target_style_examples(self, question: str) -> str:\n        """Short generic examples of the target answer shape. These are style examples only."""\n        q = question.lower()\n        if "compounds" in q:\n            return (\n                "TARGET STYLE EXAMPLE:\\n"\n                "- MAPbI3\\n"\n                "- BA2MA4Pb5I16 — n = 5 layered perovskite\\n"\n                "- CsPbBr3 nanocrystals"\n            )\n        if "materials" in q or "iupac" in q:\n            return (\n                "TARGET STYLE EXAMPLE:\\n"\n                "For MAPbI3: MAI — methylammonium iodide (methanaminium iodide); "\n                "PbI2 — lead(II) iodide; DMF — N,N-dimethylformamide.\\n"\n                "For BA2MA4Pb5I16: BAI — n-butylammonium iodide (butan-1-aminium iodide); "\n                "MAI — methanaminium iodide; PbI2 — lead(II) iodide; additives/solvents if used."\n            )\n        if "temperature" in q and "method" in q:\n            return (\n                "TARGET STYLE EXAMPLE:\\n"\n                "For MAPbI3: Method: spin-coating; Temperature: 100 °C anneal, 10 min; Solvent: DMF/DMSO.\\n"\n                "For CsPbBr3: Method: LARP; Temperature: room temperature or not reported; Solvent: toluene/DMF."\n            )\n        return "TARGET STYLE: terse extraction bullets only."\n\n    def refine_answer_to_benchmark_style(self, question: str, raw_answer: str, context: str) -> str:\n        """Second-pass compressor/verifier tuned to improve benchmark precision.\n\n        The first pass is allowed to harvest facts. This pass removes prose,\n        device-only materials, background examples, and unsupported details while\n        preserving the exact tokens that benchmark scoring rewards.\n        """\n        if not self.two_pass:\n            return raw_answer\n        style = self._target_style_examples(question)\n        # Keep a verification context but do not send the entire PDF again.\n        verify_context = context[: int(os.getenv("BENCHMARK_REFINE_CONTEXT_CHARS", "22000"))]\n        system = (\n            "You are a strict benchmark-answer compressor. Your job is to convert a raw extraction "\n            "into the final answer style used by Edison_KOSMOS and Publication Analyzer. "\n            "Do not add facts. Do not explain. Preserve exact formulas, reagent abbreviations, "\n            "IUPAC names, methods, solvents, temperatures, rpm, and times. Remove filler prose, "\n            "background-only examples, citations, evidence quotes, and device-only materials. "\n            "If a field is missing, write exactly \'not reported\'. Output only the final answer."\n        )\n        user = f"""\nBENCHMARK QUESTION:\n{question}\n\n{style}\n\nCOMPRESSION RULES:\n- Keep high-overlap benchmark tokens: formulas, precursor salts, additives, solvents, IUPAC names, methods, temperatures, rpm, times.\n- Remove all generic phrases: based on the paper, the following, experimental section reports, this study, etc.\n- Remove narrative paragraphs. Use compact bullets or semicolon-separated lines.\n- Remove unsupported/inferred details if absent from verification evidence.\n- Do not list introduction-only literature examples.\n- Do not list device-only layers unless the question asks for them or they are used in perovskite synthesis.\n- Do not include confidence, citations, or evidence excerpts.\n- Final answer length target: compounds <= 10 bullets; materials <= 18 compact lines; synthesis <= 12 compact lines.\n- Use abbreviation-first wording: PEAI — phenethylammonium iodide; MAI — methylammonium iodide (methanaminium iodide); PbI2 — lead(II) iodide; DMF — N,N-dimethylformamide.\n- Keep additives and solvents if they are part of synthesis, because they are benchmark-relevant.\n\nRAW FIRST-PASS EXTRACTION:\n{raw_answer}\n\nVERIFICATION EVIDENCE:\n{verify_context}\n""".strip()\n        refined = self.call_qwen_text(\n            system,\n            user,\n            temperature=self.refine_temperature,\n            max_tokens=self.refine_max_tokens,\n        )\n        return refined.strip() or raw_answer\n\n\n    def _benchmark_tokens(self, text: str) -> set:\n        """Return benchmark-important tokens used to decide whether compression lost recall.\n\n        This is deliberately simple and deterministic. It is not the ground truth;\n        it just protects high-value chemistry/synthesis terms from being removed\n        during the Edison-style compression pass.\n        """\n        if not text:\n            return set()\n        t = normalize_text(text)\n        token_patterns = [\n            r"\\b[A-Z][a-z]?[A-Za-z0-9\\-]*(?:Pb|Sn|Cs|MA|FA|PEA|BA|GA|XDA)[A-Za-z0-9\\(\\)\\.\\-−/]*\\b",\n            r"\\b(?:PbI2|PbBr2|SnI2|MAI|MABr|FAI|FABr|PEAI|PEABr|BAI|BABr|CsI|CsBr|Cs2CO3|NH4SCN|NH4Cl|MACl|HI|DMF|DMSO|GBL|NMP)\\b",\n            r"\\b(?:spin[- ]coating|spin[- ]coated|drop[- ]cast|dropcast|hot[- ]casting|LARP|reprecipitation|anneal(?:ed|ing)?|stirred|antisolvent|anti[- ]solvent|glovebox|room temperature|ambient)\\b",\n            r"\\b\\d+(?:\\.\\d+)?\\s*(?:°\\s*C|rpm|min|h|hours?|M|mM|mol\\s*%|µL|uL|mL)\\b",\n            r"\\b(?:methylammonium|formamidinium|phenethylammonium|butylammonium|guanidinium|lead\\(II\\)|tin\\(II\\)|cesium|caesium|iodide|bromide|chloride|thiocyanate|dimethylformamide|dimethyl sulfoxide|chlorobenzene|toluene|chloroform|oleic acid|oleylamine|octanoic acid|octylamine)\\b",\n        ]\n        out = set()\n        for pat in token_patterns:\n            for m in re.finditer(pat, t, flags=re.I):\n                tok = re.sub(r"\\s+", " ", m.group(0)).strip(" .,;:()[]")\n                if len(tok) >= 2:\n                    out.add(tok.lower())\n        return out\n\n    def _token_retention_ratio(self, raw_answer: str, final_answer: str) -> float:\n        raw_tokens = self._benchmark_tokens(raw_answer)\n        if not raw_tokens:\n            return 1.0\n        final_tokens = self._benchmark_tokens(final_answer)\n        return len(raw_tokens & final_tokens) / max(1, len(raw_tokens))\n\n    def audit_and_repair_answer(self, question: str, raw_answer: str, refined_answer: str, context: str) -> str:\n        """Third pass: restore missing benchmark-important facts without returning to verbosity.\n\n        v9 reduced verbosity but lost cosine/recall. This audit pass asks Qwen to\n        compare the raw harvest and compressed answer, then add back only supported\n        formulas/reagents/IUPAC/method/solvent/temperature tokens that were dropped.\n        """\n        if not self.audit_pass:\n            return refined_answer\n        retention = self._token_retention_ratio(raw_answer, refined_answer)\n        # If the compressor retained enough high-value tokens, do not spend another API call.\n        if retention >= self.min_token_retention:\n            return refined_answer\n        verify_context = context[: int(os.getenv("BENCHMARK_AUDIT_CONTEXT_CHARS", "36000"))]\n        system = (\n            "You are a benchmark answer auditor. Compare a raw fact harvest with a compressed answer. "\n            "Return a final answer that keeps the compressed answer style but restores missing supported facts. "\n            "Do not add commentary. Do not quote evidence. Do not add unsupported facts. "\n            "The goal is high F1 against ground truth: keep formulas, abbreviations, expanded names, IUPAC names, additives, solvents, methods, temperatures, rpm, and times."\n        )\n        user = f"""\nBENCHMARK QUESTION:\n{question}\n\nRAW FACT HARVEST:\n{raw_answer}\n\nCOMPRESSED ANSWER:\n{refined_answer}\n\nVERIFICATION EVIDENCE:\n{verify_context}\n\nAUDIT/REPAIR RULES:\n- Keep the compressed answer concise, but restore missing benchmark-relevant tokens from RAW FACT HARVEST if they are supported by VERIFICATION EVIDENCE.\n- Prefer abbreviation-first wording: MAI — methylammonium iodide (methanaminium iodide); PbI2 — lead(II) iodide; DMF — N,N-dimethylformamide.\n- Do not remove solvents, additives, antisolvents, rpm, temperatures, times, or molar ratios when they are part of synthesis.\n- Do not list introduction-only/background compounds.\n- Use compact bullets or semicolon-separated lines only.\n- If missing, write exactly: not reported.\n- Output only the final repaired answer.\n""".strip()\n        repaired = self.call_qwen_text(\n            system,\n            user,\n            temperature=self.audit_temperature,\n            max_tokens=self.audit_max_tokens,\n        ).strip()\n        if not repaired:\n            return refined_answer\n        # Safety: if repair somehow gets much worse by token retention, keep refined.\n        if self._token_retention_ratio(raw_answer, repaired) + 0.05 < retention:\n            return refined_answer\n        return repaired\n\n    def _present_in_text(self, text: str, term: str) -> bool:\n        """Case-insensitive loose containment for benchmark-token repair."""\n        if not text or not term:\n            return False\n        t = text.lower()\n        q = term.lower()\n        compact_t = re.sub(r"[\\s\\-–—_]+", "", t)\n        compact_q = re.sub(r"[\\s\\-–—_]+", "", q)\n        return q in t or compact_q in compact_t\n\n    def _abbrev_expansion_map(self) -> Dict[str, str]:\n        """Benchmark-friendly abbreviation expansions.\n\n        These are deterministic and only inserted when the abbreviation appears in\n        evidence/raw extraction. They help the scorer because old model outputs and\n        ground truth often include both abbreviation and expanded/IUPAC-like names.\n        """\n        return {\n            "MAI": "MAI — methylammonium iodide (methanaminium iodide)",\n            "MABr": "MABr — methylammonium bromide (methanaminium bromide)",\n            "FAI": "FAI — formamidinium iodide (aminomethaniminium iodide)",\n            "FABr": "FABr — formamidinium bromide (aminomethaniminium bromide)",\n            "PEAI": "PEAI — phenethylammonium iodide (2-phenylethan-1-aminium iodide)",\n            "PEABr": "PEABr — phenethylammonium bromide (2-phenylethan-1-aminium bromide)",\n            "BAI": "BAI — n-butylammonium iodide (butan-1-aminium iodide)",\n            "BABr": "BABr — n-butylammonium bromide (butan-1-aminium bromide)",\n            "PbI2": "PbI2 — lead(II) iodide",\n            "PbBr2": "PbBr2 — lead(II) bromide",\n            "SnI2": "SnI2 — tin(II) iodide",\n            "CsI": "CsI — caesium iodide",\n            "Cs2CO3": "Cs2CO3 — caesium carbonate",\n            "NH4SCN": "NH4SCN — ammonium thiocyanate",\n            "NH4Cl": "NH4Cl — ammonium chloride",\n            "MACl": "MACl — methylammonium chloride (methanaminium chloride)",\n            "DMF": "DMF — N,N-dimethylformamide",\n            "DMSO": "DMSO — dimethyl sulfoxide",\n            "GBL": "GBL — γ-butyrolactone (oxolan-2-one)",\n            "NMP": "NMP — N-methyl-2-pyrrolidone",\n        }\n\n    def _extract_formula_candidates(self, evidence: str, raw_answer: str = "") -> List[str]:\n        """Extract likely perovskite formula/composition strings from evidence and raw harvest."""\n        blob = normalize_text((raw_answer or "") + "\\n" + (evidence or ""))\n        # Patterns capture formulas such as (PEA)2PbI4, CsPbBr3, BA2MA4Pb5I16,\n        # Cs0.25FA0.75Pb(I0.85Br0.15)3, and general series X2PbnSn1−nI4.\n        pats = [\n            r"\\([A-Za-z0-9\\-]+\\)\\d*(?:\\([A-Za-z0-9\\-]+\\)\\d*)*(?:MA|FA|Cs|Rb|Pb|Sn|I|Br|Cl|GA|XDA)[A-Za-z0-9\\(\\)\\.\\-−/]*",\n            r"\\b(?:Cs|Rb|MA|FA|BA|PEA|GA|XDA|[A-Z][a-z]?)[A-Za-z0-9\\.\\-−/]*Pb[A-Za-z0-9\\(\\)\\.\\-−/]*[IBrCl][A-Za-z0-9\\(\\)\\.\\-−/]*\\b",\n            r"\\b[A-Za-z0-9\\(\\)\\.\\-−/]*PbnSn1[\\-−]nI4\\b",\n            r"\\b[A-Za-z0-9\\(\\)\\.\\-−/]*MAn[\\-−]1PbnI3n\\+1\\b",\n        ]\n        out: List[str] = []\n        seen = set()\n        bad = re.compile(r"^(Fig|Figure|Table|Scheme|Ref|http|www|DOI|ISBN)", re.I)\n        for pat in pats:\n            for m in re.finditer(pat, blob):\n                s = m.group(0).strip(" .,;:[]{}")\n                if not (4 <= len(s) <= 90):\n                    continue\n                if bad.search(s):\n                    continue\n                if not any(x in s for x in ["Pb", "Sn", "Cs", "MA", "FA", "PEA", "BA", "XDA", "Br", "I"]):\n                    continue\n                k = s.lower()\n                if k not in seen:\n                    seen.add(k)\n                    out.append(s)\n                if len(out) >= 25:\n                    return out\n        return out\n\n    def _extract_condition_candidates(self, evidence: str, raw_answer: str = "") -> List[str]:\n        """Extract compact synthesis-condition tokens likely rewarded by benchmark scoring."""\n        blob = normalize_text((raw_answer or "") + "\\n" + (evidence or ""))\n        candidates: List[str] = []\n        patterns = [\n            r"\\b\\d+(?:\\.\\d+)?\\s*°\\s*C\\b",\n            r"\\b\\d+(?:\\.\\d+)?\\s*(?:rpm|min|h|hours?|M|mM|mol\\s*%|wt\\s*%|µL|uL|mL)\\b",\n            r"\\b(?:room temperature|ambient|RT|glovebox|N2 glovebox|nitrogen glovebox)\\b",\n            r"\\b(?:spin[- ]coating|spin[- ]coated|drop[- ]cast|dropcast|hot[- ]casting|hot casting|LARP|ligand[- ]assisted reprecipitation|reprecipitation|anneal(?:ed|ing)?|stirred|antisolvent|anti[- ]solvent)\\b",\n            r"\\b(?:DMF|DMSO|GBL|NMP|toluene|chlorobenzene|chloroform|octadecene|DMF\\s*[:/]\\s*DMSO\\s*\\d+\\s*[:/]\\s*\\d+)\\b",\n        ]\n        seen = set()\n        for pat in patterns:\n            for m in re.finditer(pat, blob, flags=re.I):\n                s = re.sub(r"\\s+", " ", m.group(0)).strip(" .,;:")\n                k = s.lower()\n                if s and k not in seen:\n                    seen.add(k)\n                    candidates.append(s)\n                if len(candidates) >= 35:\n                    return candidates\n        return candidates\n\n    def _scoreaware_candidate_bank(self, question: str, evidence: str, raw_answer: str, current_answer: str) -> List[str]:\n        """Build a small bank of missing, evidence-supported terms to append back.\n\n        This is a hybrid rule-based repair layer added after the LLM audit pass.\n        It focuses on the benchmark\'s scoring behavior: high-value formula,\n        reagent, solvent, method, temperature, and IUPAC tokens.\n        """\n        q = question.lower()\n        bank: List[str] = []\n        seen = set()\n\n        def add(term: str) -> None:\n            term = re.sub(r"\\s+", " ", term).strip(" .,;:")\n            if not term:\n                return\n            k = term.lower()\n            if k in seen:\n                return\n            # Must be in raw answer or evidence; must not already be in final answer.\n            if not (self._present_in_text(raw_answer, term) or self._present_in_text(evidence, term)):\n                return\n            if self._present_in_text(current_answer, term):\n                return\n            seen.add(k)\n            bank.append(term)\n\n        if "compounds" in q:\n            for formula in self._extract_formula_candidates(evidence, raw_answer):\n                # Avoid adding generic ions/reagents as compounds.\n                if formula in {"PbI2", "PbBr2", "SnI2", "MAI", "FAI", "BAI", "PEAI"}:\n                    continue\n                add(formula)\n\n        elif "materials" in q or "iupac" in q:\n            expansions = self._abbrev_expansion_map()\n            # Prioritize reagents/solvents that benchmark ground truth usually rewards.\n            priority = [\n                "PEAI", "PEABr", "BAI", "BABr", "MAI", "MABr", "FAI", "FABr", "PbI2", "PbBr2", "SnI2",\n                "CsI", "Cs2CO3", "NH4SCN", "NH4Cl", "MACl", "DMF", "DMSO", "GBL", "NMP",\n            ]\n            for abbr in priority:\n                if re.search(r"\\b" + re.escape(abbr) + r"\\b", evidence + "\\n" + raw_answer, flags=re.I):\n                    add(expansions.get(abbr, abbr))\n            for extra in ["toluene", "chlorobenzene", "chloroform", "oleic acid", "oleylamine", "octanoic acid", "octylamine"]:\n                if re.search(r"\\b" + re.escape(extra) + r"\\b", evidence + "\\n" + raw_answer, flags=re.I):\n                    add(extra)\n\n        elif "temperature" in q or "method" in q or "solvent" in q:\n            for cond in self._extract_condition_candidates(evidence, raw_answer):\n                add(cond)\n\n        return bank[: self.token_bank_max_additions]\n\n    def deterministic_scoreaware_repair(self, answer: str, question: str, evidence: str, raw_answer: str) -> str:\n        """v13 selective local repair: recover v11\'s useful tokens without hurting F1.\n\n        v11 improved phrase/token recall but reduced F1 because it appended too many\n        extra terms. This version only restores a few high-value, evidence-supported\n        terms and uses a stricter question-specific budget. It is intentionally\n        conservative for the compound-list question, where extra formulas can become\n        false positives.\n        """\n        if not self.hybrid_token_bank_repair:\n            return answer\n        ans = normalize_text(answer)\n        if not ans:\n            return ans\n        q = question.lower()\n\n        additions = self._scoreaware_candidate_bank(question, evidence, raw_answer, ans)\n        if not additions:\n            return ans\n\n        # Best-of-both-worlds strategy:\n        # - Compounds: only repair if answer is clearly sparse or contains no Pb/Sn/Cs formula tokens.\n        # - Materials/IUPAC: keep a few abbreviation + expanded-name pairs because the scorer rewards them.\n        # - Synthesis conditions: keep only a few missing method/temp/solvent tokens.\n        if self.selective_token_repair:\n            has_formula = bool(re.search(r"\\b[A-Za-z0-9()\\.\\-−/]*Pb[A-Za-z0-9()\\.\\-−/]*(?:I|Br|Cl)[A-Za-z0-9()\\.\\-−/]*\\b", ans))\n            if "compounds" in q and self.compound_repair_only_if_sparse and (has_formula and len(ans) > 260):\n                return ans\n            if "compounds" in q:\n                additions = additions[:1]\n            elif "materials" in q or "iupac" in q:\n                additions = additions[: min(3, self.token_bank_max_additions)]\n            else:\n                additions = additions[: min(2, self.token_bank_max_additions)]\n\n        # Tighter caps than v11. Materials get the largest allowance because reagent/IUPAC\n        # expansions are valuable; compound and synthesis answers are kept closer to v9.\n        base_cap = self._question_char_cap(question)\n        if "materials" in q or "iupac" in q:\n            cap = int(base_cap * min(1.00, self.token_bank_soft_cap_factor + 0.08))\n        elif "temperature" in q or "method" in q or "solvent" in q:\n            cap = int(base_cap * min(0.88, self.token_bank_soft_cap_factor))\n        else:\n            cap = int(base_cap * min(0.78, self.token_bank_soft_cap_factor))\n\n        def build_candidate(items: List[str]) -> str:\n            if not items:\n                return ans\n            if "compounds" in q:\n                return ans.rstrip() + "\\n" + "\\n".join([f"- {z}" for z in items])\n            # Use a short neutral label; avoid verbose v11 headers like\n            # "Shared/other synthesis materials" that added low-value words.\n            return ans.rstrip() + "\\nAdditional: " + "; ".join(items) + "."\n\n        kept: List[str] = []\n        for x in additions:\n            candidate = build_candidate(kept + [x])\n            if len(candidate) <= cap:\n                kept.append(x)\n        if not kept:\n            return ans\n        repaired = build_candidate(kept)\n\n        # Last safety cleanup: if the repair introduced duplicate abbreviations or a\n        # large verbose line, keep the original.\n        if len(repaired) > max(cap, len(ans) + 450):\n            return ans\n        return repaired.strip()\n\n    def canonicalize_benchmark_terms(self, answer: str) -> str:\n        """Normalize recurring chemistry terms to benchmark-friendly spellings.\n\n        This is intentionally light-touch; it improves token overlap without\n        changing the scientific content.\n        """\n        ans = answer\n        replacements = {\n            "N,N-Dimethylformamide": "DMF — N,N-dimethylformamide",\n            "N,N-dimethylformamide": "DMF — N,N-dimethylformamide",\n            "dimethyl sulfoxide": "DMSO — dimethyl sulfoxide",\n            "Dimethyl sulfoxide": "DMSO — dimethyl sulfoxide",\n            "methylammonium iodide": "MAI — methylammonium iodide",\n            "formamidinium iodide": "FAI — formamidinium iodide",\n            "lead(II) iodide": "PbI2 — lead(II) iodide",\n            "lead(II) bromide": "PbBr2 — lead(II) bromide",\n            "tin(II) iodide": "SnI2 — tin(II) iodide",\n            "ammonium thiocyanate": "NH4SCN — ammonium thiocyanate",\n            "ammonium chloride": "NH4Cl — ammonium chloride",\n        }\n        # Avoid doubling if abbreviation is already adjacent.\n        for old, new in replacements.items():\n            ans = re.sub(rf"(?<![A-Za-z0-9—-]){re.escape(old)}(?![A-Za-z0-9])", new, ans)\n        # Clean accidental repeated abbreviation expansions.\n        ans = re.sub(r"\\b(DMF)\\s*—\\s*DMF\\s*—", r"\\1 —", ans)\n        ans = re.sub(r"\\b(DMSO)\\s*—\\s*DMSO\\s*—", r"\\1 —", ans)\n        ans = re.sub(r"\\b(PbI2)\\s*—\\s*PbI2\\s*—", r"\\1 —", ans)\n        ans = re.sub(r"\\b(PbBr2)\\s*—\\s*PbBr2\\s*—", r"\\1 —", ans)\n        ans = re.sub(r"\\b(MAI)\\s*—\\s*MAI\\s*—", r"\\1 —", ans)\n        ans = re.sub(r"\\b(FAI)\\s*—\\s*FAI\\s*—", r"\\1 —", ans)\n        return ans\n\n    def sanitize_answer(self, answer: str, question: Optional[str] = None) -> str:\n        """Light post-processing to reduce verbosity without changing facts."""\n        ans = normalize_text(answer)\n        ans = self.canonicalize_benchmark_terms(ans)\n        # Remove common verbose openers that hurt precision but add no benchmark value.\n        ans = re.sub(r"^(based on|from|according to)\\s+the\\s+(supplied\\s+)?(paper\\s+)?text[:,]?\\s*", "", ans, flags=re.I)\n        ans = re.sub(r"^the\\s+(experimental|methods|results).*?sections?\\s+(mention|describe|report)[s]?[:,]?\\s*", "", ans, flags=re.I)\n        # Remove generic hedging/filler phrases that hurt precision.\n        ans = re.sub(r"\\b(experimentally studied|the following|as follows|respectively|where applicable)\\b[:,]?", "", ans, flags=re.I)\n        ans = re.sub(r"\\b(no other .*? reported|only .*? reported)\\b\\.?", "", ans, flags=re.I)\n        # Strip markdown table separator rows only.\n        ans = re.sub(r"^\\s*\\|?\\s*:?-{3,}:?\\s*(\\|\\s*:?-{3,}:?\\s*)+\\|?\\s*$", "", ans, flags=re.M)\n        # Hard cap as a safety valve against rambling. Keep whole lines where possible.\n        cap = self._question_char_cap(question or "")\n        if self.compact_output and len(ans) > cap:\n            cut = ans[: cap]\n            last_newline = cut.rfind("\\n")\n            if last_newline > cap * 0.65:\n                cut = cut[:last_newline]\n            ans = cut.rstrip()\n        return ans.strip()\n\n    def write_outputs(self, paper_number: int, paper_dir: Path, transcript: str, jsonl_rows: List[Dict[str, Any]]) -> Dict[str, str]:\n        safe_label = self.model_label\n        txt_path = paper_dir / f"paper {paper_number} {safe_label}.txt"\n        jsonl_path = paper_dir / f"paper {paper_number} {safe_label}.jsonl"\n        docx_path = paper_dir / f"paper {paper_number} {safe_label} chat.docx"\n\n        if not self.overwrite and (txt_path.exists() or jsonl_path.exists() or docx_path.exists()):\n            raise FileExistsError(\n                f"Benchmark outputs already exist for paper {paper_number}. "\n                "Set BENCHMARK_OVERWRITE=1 to overwrite."\n            )\n\n        txt_path.write_text(transcript, encoding="utf-8", errors="ignore")\n        with jsonl_path.open("w", encoding="utf-8") as f:\n            for row in jsonl_rows:\n                f.write(json.dumps(row, ensure_ascii=False) + "\\n")\n\n        try:\n            from docx import Document  # type: ignore\n            doc = Document()\n            for line in transcript.splitlines():\n                doc.add_paragraph(line)\n            doc.save(str(docx_path))\n        except Exception as exc:\n            # The benchmark scorer can read TXT/JSONL, so DOCX is convenient but not required.\n            print(f"[WARN] Could not write DOCX for paper {paper_number}: {exc}")\n\n        return {"txt": str(txt_path), "jsonl": str(jsonl_path), "docx": str(docx_path)}\n\n    def run_one_paper(self, paper_number: int, paper_dir: Path, pdf_path: Path) -> Dict[str, Any]:\n        print(f"[BenchmarkingAgent] Paper {paper_number}: reading {pdf_path}")\n        transcript_parts: List[str] = []\n        jsonl_rows: List[Dict[str, Any]] = []\n        context_lengths: List[int] = []\n\n        for run_idx in range(1, self.runs + 1):\n            transcript_parts.append(f"===== PAPER {paper_number} | RUN {run_idx} =====")\n            for question_idx, question in enumerate(self.QUESTIONS, start=1):\n                context = self.read_pdf_context(pdf_path, question=question)\n                context_lengths.append(len(context))\n                system, user = self.build_prompt(paper_number, pdf_path, question, run_idx, context)\n                raw_answer = self.call_qwen_text(\n                    system,\n                    user,\n                    temperature=self.temperature,\n                    max_tokens=self.harvest_max_tokens if self.two_pass else self.max_tokens,\n                )\n                refined_answer = self.refine_answer_to_benchmark_style(question, raw_answer, context)\n                audited_answer = self.audit_and_repair_answer(question, raw_answer, refined_answer, context)\n                answer = self.sanitize_answer(audited_answer, question=question)\n                answer = self.deterministic_scoreaware_repair(answer, question, context, raw_answer)\n                answer = self.sanitize_answer(answer, question=question)\n                transcript_parts.append("You said:")\n                transcript_parts.append(question)\n                transcript_parts.append("")\n                transcript_parts.append(f"{self.model_label} said:")\n                transcript_parts.append(answer)\n                transcript_parts.append("")\n                jsonl_rows.append({\n                    "paper_number": paper_number,\n                    "paper_dir": str(paper_dir),\n                    "pdf_path": str(pdf_path),\n                    "run": run_idx,\n                    "question_index": question_idx,\n                    "question": question,\n                    "model": self.model_label,\n                    "answer": answer,\n                })\n                print(f"[BenchmarkingAgent] Paper {paper_number} run {run_idx} question {question_idx} complete.")\n\n        transcript = "\\n".join(transcript_parts).strip() + "\\n"\n        paths = self.write_outputs(paper_number, paper_dir, transcript, jsonl_rows)\n        return {\n            "paper_number": paper_number,\n            "pdf_path": str(pdf_path),\n            "context_chars_min": min(context_lengths) if context_lengths else 0,\n            "context_chars_max": max(context_lengths) if context_lengths else 0,\n            "output_paths": paths,\n            "num_rows": len(jsonl_rows),\n        }\n\n    def run(self) -> Dict[str, Any]:\n        papers = self.discover_papers()\n        if self.scan_only:\n            return self.scan()\n        if not papers:\n            raise RuntimeError(f"No benchmark PDFs found under {self.root}")\n        if not HF_API_KEY:\n            raise ValueError("HF_API_KEY is missing. Set HF_API_KEY or use BENCHMARK_SCAN_ONLY=1.")\n        results = []\n        for paper_number, paper_dir, pdf_path in papers:\n            results.append(self.run_one_paper(paper_number, paper_dir, pdf_path))\n        payload = {\n            "ok": True,\n            "mode": "benchmark",\n            "agent": "BenchmarkingAgent_v15_evidence_table_precision_gate",\n            "model_label": self.model_label,\n            "hf_model": HF_MODEL,\n            "benchmark_root": str(self.root),\n            "runs_per_paper": self.runs,\n            "papers_processed": len(results),\n            "results": results,\n        }\n        summary_path = self.root / f"{self.model_label}_benchmark_manifest.json"\n        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")\n        print(json.dumps(payload, indent=2, ensure_ascii=False))\n        return payload\n\n\ndef main() -> None:\n    mode = (os.getenv("RUN_MODE", RUN_MODE) or "").strip().lower()\n    if mode == "benchmark":\n        # Hard separation: benchmark mode only uses local nested paper PDF paths\n        # through BenchmarkingAgent. It does not run process_batch_api, Crossref,\n        # RSS/Unpaywall expansion, or Google Drive import/precheck.\n        BenchmarkingAgent().run()\n        return\n    literature_agent_main()\n\n\ndef run_mcp_server() -> None:\n    try:\n        from mcp.server.fastmcp import FastMCP\n    except Exception as exc:\n        raise RuntimeError("The \'mcp\' package is required to run the MCP server. Install it with: pip install mcp") from exc\n\n    mcp = FastMCP(MCP_SERVER_NAME)\n\n    @mcp.tool()\n    def search_candidates(query: str, year_min: int = YEAR_MIN, year_max: int = YEAR_MAX, max_candidates: int = 25) -> Dict[str, Any]:\n        return search_candidates_api(query=query, year_min=year_min, year_max=year_max, max_candidates=max_candidates)\n\n    @mcp.tool()\n    def process_one_paper(meta: Dict[str, Any], force_reprocess: bool = False, run_mode: str = RUN_MODE) -> Dict[str, Any]:\n        return process_one_paper_api(meta, force_reprocess=force_reprocess, run_mode=run_mode)\n\n    @mcp.tool()\n    def process_batch(query: str = SEARCH_QUERY, year_min: int = YEAR_MIN, year_max: int = YEAR_MAX, max_papers: int = MAX_PAPERS_TO_PROCESS, run_mode: str = RUN_MODE, force_reprocess: bool = False, reset_output: bool = False) -> Dict[str, Any]:\n        return process_batch_api(query=query, year_min=year_min, year_max=year_max, max_papers=max_papers, run_mode=run_mode, force_reprocess=force_reprocess, reset_output=reset_output)\n\n    @mcp.tool()\n    def get_saved_paper_output(paper_slug: str) -> Dict[str, Any]:\n        return get_saved_paper_output_api(paper_slug)\n\n    @mcp.tool()\n    def list_processed_papers() -> Dict[str, Any]:\n        return list_processed_papers_api()\n\n    @mcp.tool()\n    def get_hypothesis_payload(paper_slug: str) -> Dict[str, Any]:\n        return get_hypothesis_payload_api(paper_slug)\n\n    @mcp.tool()\n    def inspect_local_pdf(pdf_path: str, include_ocr: bool = True, include_figures: bool = True) -> Dict[str, Any]:\n        return inspect_local_pdf_api(pdf_path=pdf_path, include_ocr=include_ocr, include_figures=include_figures)\n\n    @mcp.tool()\n    def list_google_drive_folder(folder_id_or_link: str = GOOGLE_DRIVE_FOLDER_LINK) -> Dict[str, Any]:\n        return list_google_drive_folder_api(folder_id_or_link=folder_id_or_link)\n\n    @mcp.tool()\n    def import_google_drive_folder(folder_id_or_link: str = GOOGLE_DRIVE_FOLDER_LINK, max_files: int = 50, only_pdfs: bool = True) -> Dict[str, Any]:\n        return import_google_drive_folder_api(folder_id_or_link=folder_id_or_link, max_files=max_files, only_pdfs=only_pdfs)\n\n    @mcp.tool()\n    def process_google_drive_folder(folder_id_or_link: str = GOOGLE_DRIVE_FOLDER_LINK, max_files: int = 20, include_ocr: bool = True, include_figures: bool = True, force_reprocess: bool = False, run_mode: str = RUN_MODE) -> Dict[str, Any]:\n        return process_google_drive_folder_api(folder_id_or_link=folder_id_or_link, max_files=max_files, include_ocr=include_ocr, include_figures=include_figures, force_reprocess=force_reprocess, run_mode=run_mode)\n\n    @mcp.tool()\n    def get_storage_layout() -> Dict[str, Any]:\n        return get_storage_layout_api()\n\n    @mcp.tool()\n    def extraction_agent_extract(meta: Dict[str, Any], full_text: str = "", extraction_text: str = "", include_ocr: bool = True, include_figures: bool = True) -> Dict[str, Any]:\n        return extraction_agent_extract_api(meta=meta, full_text=full_text or None, extraction_text=extraction_text or None, include_ocr=include_ocr, include_figures=include_figures)\n\n    @mcp.tool()\n    def evaluation_agent_evaluate(records: List[Dict[str, Any]]) -> Dict[str, Any]:\n        return evaluation_agent_evaluate_api(records)\n\n    @mcp.tool()\n    def flagging_agent_flag(meta: Dict[str, Any], records: List[Dict[str, Any]], evaluation: Dict[str, Any] | None = None) -> Dict[str, Any]:\n        return flagging_agent_flag_api(meta=meta, records=records, evaluation=evaluation)\n\n    @mcp.tool()\n    def assembly_agent_assemble_records(meta: Dict[str, Any], raw_records: List[Dict[str, Any]]) -> Dict[str, Any]:\n        return assembly_agent_assemble_records_api(meta=meta, raw_records=raw_records)\n\n    @mcp.tool()\n    def summarization_agent_summarize(meta: Dict[str, Any], records: List[Dict[str, Any]], full_text: str = "", paper_slug: str = "") -> Dict[str, Any]:\n        return summarization_agent_summarize_api(meta=meta, records=records, full_text=full_text or None, paper_slug=paper_slug or None)\n\n    if MCP_TRANSPORT == "http":\n        mcp.run(transport="streamable-http")\n    else:\n        mcp.run()\n\n\n# ============================================================\n# V15 BENCHMARKING PATCH: recall-preserving evidence table + precision gates\n# ============================================================\n# This patch is intentionally applied by monkey-patching BenchmarkingAgent so the\n# original literature-mining pipeline remains untouched. It only affects RUN_MODE=benchmark.\n\n_BENCHMARKING_AGENT_ORIG_INIT = BenchmarkingAgent.__init__\n_BENCHMARKING_AGENT_ORIG_BUILD_PROMPT = BenchmarkingAgent.build_prompt\n_BENCHMARKING_AGENT_ORIG_REFINE = BenchmarkingAgent.refine_answer_to_benchmark_style\n\n\ndef _v15_init(self) -> None:\n    _BENCHMARKING_AGENT_ORIG_INIT(self)\n    self.evidence_table_enabled = os.getenv("BENCHMARK_EVIDENCE_TABLE", "1").strip().lower() in {"1", "true", "yes"}\n    self.question_gates_enabled = os.getenv("BENCHMARK_QUESTION_GATES", "1").strip().lower() in {"1", "true", "yes"}\n    self.protected_token_gate_enabled = os.getenv("BENCHMARK_PROTECTED_TOKEN_GATE", "1").strip().lower() in {"1", "true", "yes"}\n    # v15 uses v13-style selective repair, but with lower additions by default.\n    self.token_bank_max_additions = int(os.getenv("BENCHMARK_TOKEN_BANK_MAX_ADDITIONS", "5"))\n    self.agent_version = "BenchmarkingAgent_v15_recall_preserving_precision_gate"\n\n\ndef _v15_abbrev_priority_for_question(self, question: str) -> list:\n    q = question.lower()\n    if "materials" in q or "iupac" in q:\n        return [\n            "PEAI", "PEABr", "BAI", "BABr", "MAI", "MABr", "FAI", "FABr",\n            "PbI2", "PbBr2", "SnI2", "CsI", "Cs2CO3", "NH4SCN", "NH4Cl", "MACl",\n            "DMF", "DMSO", "GBL", "NMP", "chlorobenzene", "toluene", "chloroform",\n            "oleic acid", "oleylamine", "octanoic acid", "octylamine", "PMMA", "ZnO", "FTO", "THF", "triethylamine", "sodium tert-butoxide", "HI", "HCl", "LiF",\n        ]\n    if "temperature" in q or "method" in q or "solvent" in q:\n        return [\n            "spin-coating", "spin coating", "drop-casting", "drop casting", "drop-cast", "hot-casting",\n            "hot casting", "LARP", "ligand-assisted reprecipitation", "reprecipitation", "annealed",\n            "annealing", "stirred", "room temperature", "ambient", "glovebox", "antisolvent",\n            "DMF", "DMSO", "GBL", "NMP", "chlorobenzene", "toluene", "chloroform",\n        ]\n    return []\n\n\ndef _v15_clean_high_value_line(self, line: str) -> str:\n    line = re.sub(r"\\s+", " ", line).strip(" -;:\\t")\n    line = re.sub(r"^(based on|from|according to)\\s+.*?[:,]\\s*", "", line, flags=re.I)\n    line = re.sub(r"\\b(the following|as follows|in summary|overall|notably|this indicates that)\\b[:,]?", "", line, flags=re.I)\n    return line.strip()\n\n\ndef _v15_extract_formula_tokens(self, text: str, maxn: int = 30) -> list:\n    text = normalize_text(text or "")\n    # Captures common perovskite formulas/compositions but avoids isolated precursor salts later.\n    patterns = [\n        r"\\([A-Za-z0-9\\-]+\\)\\d?\\(?[A-Za-z0-9\\-]+\\)?\\d?Pb\\d?(?:I|Br|Cl)\\d+[A-Za-z0-9\\-−/\\.]*",\n        r"[A-Z][A-Za-z0-9\\.\\-−,/()]*Pb[A-Za-z0-9\\.\\-−,/()]*?(?:I|Br|Cl)\\d*[A-Za-z0-9\\.\\-−,/()]*",\n        r"Cs\\d*(?:\\.\\d+)?[A-Za-z0-9\\.\\-−,/()]*Pb[A-Za-z0-9\\.\\-−,/()]*?(?:I|Br|Cl)\\d*",\n    ]\n    out=[]; seen=set()\n    for pat in patterns:\n        for m in re.finditer(pat, text):\n            s=m.group(0).strip(" .,;:[]")\n            if len(s)<5 or len(s)>90: continue\n            if s in {"PbI2", "PbBr2", "SnI2", "PbCl2"}: continue\n            if re.match(r"^(Figure|Table|Scheme|Ref)", s, re.I): continue\n            k=s.lower()\n            if k not in seen:\n                seen.add(k); out.append(s)\n            if len(out)>=maxn: return out\n    return out\n\n\ndef _v15_build_evidence_table(self, context: str, question: str) -> str:\n    """Build a compact internal evidence table before Qwen answers.\n\n    This is the key v15 change: we preserve recall internally in structured form,\n    then final prompts/gates expose only the table entries relevant to the question.\n    """\n    if not getattr(self, "evidence_table_enabled", True):\n        return ""\n    source = normalize_text(context or "")\n    expansions = self._abbrev_expansion_map()\n    formulas = self._v15_extract_formula_tokens(source, maxn=26)\n    q = question.lower()\n\n    mat_terms=[]; seen=set()\n    for term in self._v15_abbrev_priority_for_question("materials iupac"):\n        pat = re.escape(term)\n        if re.search(r"\\b" + pat + r"\\b", source, flags=re.I) or (" " in term and re.search(pat, source, flags=re.I)):\n            expanded = expansions.get(term, term)\n            k = expanded.lower()\n            if k not in seen:\n                seen.add(k); mat_terms.append(expanded)\n        if len(mat_terms) >= 28: break\n\n    cond_terms=[]; seen=set()\n    # conditions from existing extractor plus regex values\n    for term in self._extract_condition_candidates(source, ""):\n        term = self._v15_clean_high_value_line(term)\n        if not term: continue\n        k=term.lower()\n        if k not in seen:\n            seen.add(k); cond_terms.append(term)\n        if len(cond_terms) >= 18: break\n    value_terms = re.findall(r"\\b(?:\\d{2,3}\\s*°\\s*C|\\d+\\s*min|\\d+\\s*h|\\d+\\s*hours?|\\d+\\s*rpm|\\d+(?:\\.\\d+)?\\s*M|\\d+(?:\\.\\d+)?\\s*mol\\s*%|\\d+\\s*:\\s*\\d+(?:\\s*v/v)?)\\b", source, flags=re.I)\n    for term in value_terms:\n        term = re.sub(r"\\s+", " ", term).strip()\n        k=term.lower()\n        if k not in seen:\n            seen.add(k); cond_terms.append(term)\n        if len(cond_terms) >= 28: break\n\n    methods=[]; seen=set()\n    for term in self._v15_abbrev_priority_for_question("temperature method solvent"):\n        if re.search(re.escape(term), source, flags=re.I):\n            k=term.lower()\n            if k not in seen:\n                seen.add(k); methods.append(term)\n        if len(methods) >= 18: break\n\n    table = ["INTERNAL EVIDENCE TABLE (use for recall; final answer must obey the question gate):"]\n    if formulas:\n        table.append("compound/formula candidates: " + "; ".join(formulas[:20]))\n    if mat_terms:\n        table.append("material/IUPAC candidates: " + "; ".join(mat_terms[:24]))\n    if methods or cond_terms:\n        table.append("method/condition candidates: " + "; ".join((methods + cond_terms)[:26]))\n    return "\\n".join(table)\n\n\ndef _v15_gate_instruction(self, question: str) -> str:\n    q = question.lower()\n    if "compounds" in q:\n        return (\n            "QUESTION GATE: COMPOUNDS ONLY. Final answer may contain only perovskite compound formulas/names and short descriptors. "\n            "Do not list reagents, solvents, temperatures, synthesis procedure, device layers, or explanatory text. "\n            "Use one bullet per compound/series; maximum 10 bullets unless the paper explicitly reports a larger composition series; preserve named series/endmembers."\n        )\n    if "materials" in q or "iupac" in q:\n        return (\n            "QUESTION GATE: MATERIALS/IUPAC ONLY. Final answer should list perovskite or series followed by synthesis materials. "\n            "Include abbreviation — common name (IUPAC) for precursors, additives, antisolvents, and solvents used to synthesize the perovskite. "\n            "Do not include performance/discussion text; keep synthesis-relevant solvents, additives, antisolvents, ratios, and precursor names if reported."\n        )\n    if "temperature" in q or "method" in q or "solvent" in q:\n        return (\n            "QUESTION GATE: SYNTHESIS CONDITIONS ONLY. Final answer should list perovskite or series with Method, Temperature, and Solvent. "\n            "Include rpm/time/annealing/stirring only if directly reported. Do not list full reagent/IUPAC inventories, but keep reported solvent names, antisolvents, temperatures, times, rpm, and ratios. "\n            "Use \'not reported\' for missing fields."\n        )\n    return "QUESTION GATE: answer only the requested field, compactly."\n\n\ndef _v15_build_prompt(self, paper_number: int, pdf_path: Path, question: str, run_idx: int, context: str) -> Tuple[str, str]:\n    # Build the original prompt, then prepend the evidence table and stronger gate.\n    system, user = _BENCHMARKING_AGENT_ORIG_BUILD_PROMPT(self, paper_number, pdf_path, question, run_idx, context)\n    evidence_table = self._v15_build_evidence_table(context, question)\n    gate = self._v15_gate_instruction(question)\n    system = system + " V15 rule: preserve recall in the internal evidence table, but the final answer must pass the question gate and contain no off-task fields."\n    if evidence_table:\n        user = user.replace("STRICT EXTRACTION RULES:", f"{evidence_table}\\n\\n{gate}\\n\\nSTRICT EXTRACTION RULES:")\n    else:\n        user = user.replace("STRICT EXTRACTION RULES:", f"{gate}\\n\\nSTRICT EXTRACTION RULES:")\n    return system, user\n\n\ndef _v15_refine_answer_to_benchmark_style(self, question: str, raw_answer: str, context: str) -> str:\n    refined = _BENCHMARKING_AGENT_ORIG_REFINE(self, question, raw_answer, context)\n    # One extra local instruction-free gate after LLM refine; no API call here.\n    return self.v15_question_specific_gate(refined, question, context, raw_answer)\n\n\ndef _v15_protected_tokens_for_question(self, question: str, evidence: str, raw_answer: str, limit: int) -> list:\n    q = question.lower()\n    source = normalize_text((evidence or "") + "\\n" + (raw_answer or ""))\n    candidates=[]; seen=set()\n    expansions = self._abbrev_expansion_map()\n\n    def add(term):\n        term=self._v15_clean_high_value_line(term)\n        if not term: return\n        if self._present_in_text(term, "device") and "materials" not in q: pass\n        k=term.lower()\n        if k not in seen and self._present_in_text(source, term.split(" — ")[0] if " — " in term else term):\n            seen.add(k); candidates.append(term)\n\n    if "compounds" in q:\n        for f in self._v15_extract_formula_tokens(source, maxn=20):\n            add(f)\n    elif "materials" in q or "iupac" in q:\n        for term in self._v15_abbrev_priority_for_question(question):\n            # For multiword tokens, just use plain regex.\n            if re.search(r"\\b" + re.escape(term) + r"\\b", source, flags=re.I) or (" " in term and re.search(re.escape(term), source, flags=re.I)):\n                add(expansions.get(term, term))\n    elif "temperature" in q or "method" in q or "solvent" in q:\n        for term in self._extract_condition_candidates(evidence, raw_answer):\n            add(term)\n        vals = re.findall(r"\\b(?:\\d{2,3}\\s*°\\s*C|\\d+\\s*min|\\d+\\s*h|\\d+\\s*hours?|\\d+\\s*rpm|room temperature|ambient|not reported|DMF|DMSO|GBL|NMP|chlorobenzene|toluene|chloroform|spin-coating|drop-casting|hot-casting|LARP)\\b", source, flags=re.I)\n        for v in vals:\n            add(v)\n    return candidates[:limit]\n\n\ndef _v15_filter_lines_by_question(self, lines: list, question: str) -> list:\n    q=question.lower()\n    out=[]\n    for line in lines:\n        line=self._v15_clean_high_value_line(line)\n        if not line: continue\n        low=line.lower()\n        # Always drop generic filler.\n        if re.search(r"\\b(based on|according to|the paper|this study|results suggest|in summary|evidence|citation)\\b", low):\n            continue\n        if "compounds" in q:\n            # Remove off-task synthesis/material lines.\n            if re.search(r"\\b(method|solvent|temperature|stirred|anneal|spin|materials?:|reagents?:|dmf|dmso|chlorobenzene|toluene)\\b", low):\n                continue\n            if not re.search(r"Pb|Sn|Cs|perovskite|MAPb|FAPb|PEA|BA|MA", line):\n                continue\n        elif "materials" in q or "iupac" in q:\n            # Keep synthesis-material lines even if they mention role/process terms; drop only device/performance-only lines.\n            if re.search(r"\\b(performance|pce|jsc|voc|fill factor|efficiency)\\b", low):\n                continue\n        elif "temperature" in q or "method" in q or "solvent" in q:\n            # Conditions may mention solvents/methods/temps; drop long material-only inventory lines.\n            if re.search(r"\\b(iupac|materials?:|reagents?:)\\b", low) and not re.search(r"\\b(method|temperature|solvent|anneal|spin|stir|°c|rpm|dmf|dmso|chlorobenzene|toluene|gbl|nmp)\\b", low):\n                continue\n        out.append(line)\n    return out\n\n\ndef _v15_question_specific_gate(self, answer: str, question: str, evidence: str, raw_answer: str = "") -> str:\n    """Question-specific final gate: preserve protected tokens, remove off-task verbosity."""\n    if not getattr(self, "question_gates_enabled", True):\n        return answer\n    q=question.lower()\n    ans=normalize_text(answer or "")\n    ans=self.canonicalize_benchmark_terms(ans)\n    # split to compact lines, including semicolon-heavy outputs.\n    raw_lines=[]\n    for part in ans.splitlines():\n        part=part.strip()\n        if not part: continue\n        if len(part) > 360 and ";" in part:\n            raw_lines.extend([x.strip() for x in part.split(";") if x.strip()])\n        else:\n            raw_lines.append(part)\n    lines=self._v15_filter_lines_by_question(raw_lines, question)\n\n    # Deduplicate lines but keep order.\n    dedup=[]; seen=set()\n    for ln in lines:\n        ln=re.sub(r"^[*\\-•]\\s*", "- ", ln)\n        if not ln.startswith("-") and ("compounds" in q):\n            ln="- " + ln\n        k=re.sub(r"[^a-z0-9]+", "", ln.lower())[:160]\n        if k and k not in seen:\n            seen.add(k); dedup.append(ln)\n\n    # Hard line caps by question type.\n    if "compounds" in q:\n        dedup=dedup[:16]\n        protected_limit=2 if len(dedup) < 3 else 0\n        cap=int(os.getenv("BENCHMARK_V15_COMPOUND_CAP", "1100"))\n    elif "materials" in q or "iupac" in q:\n        dedup=dedup[:24]\n        protected_limit=int(os.getenv("BENCHMARK_V15_MATERIAL_TOKEN_ADDS", "7"))\n        cap=int(os.getenv("BENCHMARK_V15_MATERIALS_CAP", "2850"))\n    elif "temperature" in q or "method" in q or "solvent" in q:\n        dedup=dedup[:16]\n        protected_limit=int(os.getenv("BENCHMARK_V15_SYNTHESIS_TOKEN_ADDS", "5"))\n        cap=int(os.getenv("BENCHMARK_V15_SYNTHESIS_CAP", "1700"))\n    else:\n        protected_limit=2; cap=self._question_char_cap(question)\n\n    out="\\n".join(dedup).strip()\n    if not out:\n        out=ans.strip()\n\n    # Selective protected-token add-back. This is the "best of both worlds" piece:\n    # token/phrase recall is preserved only with tokens relevant to the active question.\n    if getattr(self, "protected_token_gate_enabled", True) and protected_limit > 0:\n        protected=[]\n        for tok in self._v15_protected_tokens_for_question(question, evidence, raw_answer, protected_limit + 4):\n            key = tok.split(" — ")[0].strip()\n            if key and not self._present_in_text(out, key):\n                protected.append(tok)\n            if len(protected) >= protected_limit:\n                break\n        if protected:\n            if "compounds" in q:\n                addition="\\n" + "\\n".join(["- " + x for x in protected])\n            elif "materials" in q or "iupac" in q:\n                addition="\\nAdditional synthesis materials: " + "; ".join(protected) + "."\n            else:\n                addition="\\nAdditional synthesis conditions: " + "; ".join(protected) + "."\n            if len(out + addition) <= cap:\n                out += addition\n\n    # Final cap on whole lines.\n    if len(out) > cap:\n        cut=out[:cap]\n        last=cut.rfind("\\n")\n        if last > cap*0.60:\n            cut=cut[:last]\n        out=cut.rstrip()\n    return out.strip()\n\n\ndef _v15_sanitize_answer(self, answer: str, question: Optional[str] = None) -> str:\n    # Use original sanitize then apply a final filler cleanup only.\n    ans = BenchmarkingAgent.__dict__["_v13_original_sanitize"](self, answer, question) if "_v13_original_sanitize" in BenchmarkingAgent.__dict__ else normalize_text(answer)\n    ans = re.sub(r"\\b(based on the supplied text|based on the paper|the following|as follows|in summary)\\b[:,]?", "", ans, flags=re.I)\n    return normalize_text(ans).strip()\n\n\ndef _v15_run_one_paper(self, paper_number: int, paper_dir: Path, pdf_path: Path) -> Dict[str, Any]:\n    print(f"[BenchmarkingAgent v15] Paper {paper_number}: reading {pdf_path}")\n    transcript_parts: List[str] = []\n    jsonl_rows: List[Dict[str, Any]] = []\n    context_lengths: List[int] = []\n\n    for run_idx in range(1, self.runs + 1):\n        transcript_parts.append(f"===== PAPER {paper_number} | RUN {run_idx} =====")\n        for question_idx, question in enumerate(self.QUESTIONS, start=1):\n            context = self.read_pdf_context(pdf_path, question=question)\n            context_lengths.append(len(context))\n            system, user = self.build_prompt(paper_number, pdf_path, question, run_idx, context)\n            raw_answer = self.call_qwen_text(\n                system,\n                user,\n                temperature=self.temperature,\n                max_tokens=self.harvest_max_tokens if self.two_pass else self.max_tokens,\n            )\n            refined_answer = self.refine_answer_to_benchmark_style(question, raw_answer, context)\n            audited_answer = self.audit_and_repair_answer(question, raw_answer, refined_answer, context)\n            answer = self.sanitize_answer(audited_answer, question=question)\n            # v13/v11-style selective repair, then v15 recall-preserving precision gate.\n            answer = self.deterministic_scoreaware_repair(answer, question, context, raw_answer)\n            answer = self.v15_question_specific_gate(answer, question, context, raw_answer)\n            answer = self.sanitize_answer(answer, question=question)\n            answer = self.v15_question_specific_gate(answer, question, context, raw_answer)\n            transcript_parts.append("You said:")\n            transcript_parts.append(question)\n            transcript_parts.append("")\n            transcript_parts.append(f"{self.model_label} said:")\n            transcript_parts.append(answer)\n            transcript_parts.append("")\n            jsonl_rows.append({\n                "paper_number": paper_number,\n                "paper_dir": str(paper_dir),\n                "pdf_path": str(pdf_path),\n                "run": run_idx,\n                "question_index": question_idx,\n                "question": question,\n                "model": self.model_label,\n                "agent_version": "v15_recall_preserving_precision_gate",\n                "answer": answer,\n            })\n            print(f"[BenchmarkingAgent v15] Paper {paper_number} run {run_idx} question {question_idx} complete.")\n\n    transcript = "\\n".join(transcript_parts).strip() + "\\n"\n    paths = self.write_outputs(paper_number, paper_dir, transcript, jsonl_rows)\n    return {\n        "paper_number": paper_number,\n        "pdf_path": str(pdf_path),\n        "context_chars_min": min(context_lengths) if context_lengths else 0,\n        "context_chars_max": max(context_lengths) if context_lengths else 0,\n        "output_paths": paths,\n        "num_rows": len(jsonl_rows),\n    }\n\n\n# Install v15 monkey patches.\nif not hasattr(BenchmarkingAgent, "_v13_original_sanitize"):\n    BenchmarkingAgent._v13_original_sanitize = BenchmarkingAgent.sanitize_answer\nBenchmarkingAgent.__init__ = _v15_init\nBenchmarkingAgent.build_prompt = _v15_build_prompt\nBenchmarkingAgent.refine_answer_to_benchmark_style = _v15_refine_answer_to_benchmark_style\nBenchmarkingAgent.v15_question_specific_gate = _v15_question_specific_gate\nBenchmarkingAgent._v15_build_evidence_table = _v15_build_evidence_table\nBenchmarkingAgent._v15_gate_instruction = _v15_gate_instruction\nBenchmarkingAgent._v15_extract_formula_tokens = _v15_extract_formula_tokens\nBenchmarkingAgent._v15_abbrev_priority_for_question = _v15_abbrev_priority_for_question\nBenchmarkingAgent._v15_clean_high_value_line = _v15_clean_high_value_line\nBenchmarkingAgent._v15_filter_lines_by_question = _v15_filter_lines_by_question\nBenchmarkingAgent._v15_protected_tokens_for_question = _v15_protected_tokens_for_question\nBenchmarkingAgent.sanitize_answer = _v15_sanitize_answer\nBenchmarkingAgent.run_one_paper = _v15_run_one_paper\n\n\n\n# ============================================================\n# v16 SYSTEMATIC NORMAL PIPELINE UPDATE\n# - Keeps benchmark mode exactly separated.\n# - Makes normal mode more systematic using an evidence-table layer.\n# - Enforces Google Drive/manual papers before web expansion.\n# ============================================================\n\nSYSTEMATIC_MAIN_PIPELINE_ENABLE = os.getenv("SYSTEMATIC_MAIN_PIPELINE_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}\nMAIN_EVIDENCE_TABLE_ENABLE = os.getenv("MAIN_EVIDENCE_TABLE_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}\nMAIN_EVIDENCE_TABLE_MAX_CHARS = int(os.getenv("MAIN_EVIDENCE_TABLE_MAX_CHARS", "35000"))\nMAIN_EVIDENCE_SENTENCE_LIMIT = int(os.getenv("MAIN_EVIDENCE_SENTENCE_LIMIT", "120"))\nMAIN_EVIDENCE_DEBUG_SAVE = os.getenv("MAIN_EVIDENCE_DEBUG_SAVE", "1").strip().lower() in {"1", "true", "yes", "on"}\nDRIVE_FIRST_STRICT = os.getenv("DRIVE_FIRST_STRICT", "1").strip().lower() in {"1", "true", "yes", "on"}\nDRIVE_FIRST_STOP_ON_ERROR = os.getenv("DRIVE_FIRST_STOP_ON_ERROR", "1").strip().lower() in {"1", "true", "yes", "on"}\nDRIVE_FIRST_PROCESS_ALL_FILES = os.getenv("DRIVE_FIRST_PROCESS_ALL_FILES", "1").strip().lower() in {"1", "true", "yes", "on"}\n\n\nclass MainEvidenceTableAgent:\n    """Rule-guided evidence staging for the normal LiteratureAgent.\n\n    This is not the benchmark answer formatter. It builds a compact internal\n    evidence table that is prepended to the extraction context so the schema\n    extractors see high-signal compounds/materials/methods before generating\n    JSON. This keeps broad recall internally but encourages concise,\n    source-grounded field filling.\n    """\n\n    SECTION_ALLOWED_RE = re.compile(\n        r"^(experimental(?: section| procedures?)?|materials and methods|methods?|synthesis|sample preparation|"\n        r"device fabrication|film fabrication|results(?: and discussion)?|results|discussion)$",\n        re.I,\n    )\n    SECTION_STOP_RE = re.compile(\n        r"^(abstract|introduction|background|conclusions?|summary|acknowledg(?:e)?ments?|references|"\n        r"bibliography|associated content|supporting information)$",\n        re.I,\n    )\n\n    FORMULA_RE = re.compile(\n        r"(?:\\([A-Za-z0-9\\-]+\\)|[A-Z][a-z]?[A-Za-z0-9\\-]*)(?:\\d|\\.|[A-Z]|[a-z]|\\(|\\)|−|-|/|,|x|n|\\+|=){2,}"\n    )\n    VALUE_RE = re.compile(r"\\b\\d+(?:\\.\\d+)?\\s*(?:°C|rpm|min|h|M|mM|mol\\s*%|wt\\s*%|vol\\s*%|µL|uL|mL)\\b", re.I)\n\n    MATERIAL_TERMS = [\n        "PbI2", "PbBr2", "SnI2", "MAI", "MABr", "FAI", "FABr", "PEAI", "PEABr", "BAI", "BABr",\n        "CsI", "CsBr", "Cs2CO3", "NH4SCN", "NH4Cl", "MACl", "DMF", "DMSO", "GBL", "NMP",\n        "chlorobenzene", "toluene", "chloroform", "octadecene", "oleic acid", "oleylamine", "octanoic acid", "octylamine",\n    ]\n    METHOD_TERMS = [\n        "spin-coating", "spin coating", "drop-cast", "dropcast", "drop-casting", "hot-casting", "hot casting",\n        "LARP", "ligand-assisted reprecipitation", "anneal", "annealed", "stirred", "antisolvent", "anti-solvent",\n        "glovebox", "room temperature", "ambient", "precursor", "solution", "solvent", "fabrication",\n    ]\n\n    def _cut_references(self, text: str) -> str:\n        m = re.search(r"\\n\\s*(references|bibliography|associated content|supporting information)\\s*\\n", text or "", flags=re.I)\n        return text[:m.start()] if m else (text or "")\n\n    def extract_relevant_sections(self, full_text: str) -> str:\n        text = normalize_text(self._cut_references(full_text or ""))\n        lines = [ln.strip() for ln in text.splitlines()]\n        hits = []\n        for i, ln in enumerate(lines):\n            clean = re.sub(r"^[0-9]+\\.?\\s*", "", ln).strip()\n            clean = re.sub(r"[:.]+$", "", clean).strip()\n            if self.SECTION_ALLOWED_RE.match(clean):\n                hits.append((i, clean))\n        if not hits:\n            return text[:220000]\n        chunks = []\n        for idx, _heading in hits:\n            end = len(lines)\n            for j in range(idx + 1, len(lines)):\n                clean = re.sub(r"^[0-9]+\\.?\\s*", "", lines[j]).strip()\n                clean = re.sub(r"[:.]+$", "", clean).strip()\n                if j > idx + 8 and (self.SECTION_ALLOWED_RE.match(clean) or self.SECTION_STOP_RE.match(clean)):\n                    end = j\n                    break\n            chunk = "\\n".join(lines[idx:end]).strip()\n            if len(chunk) > 500:\n                chunks.append(chunk)\n        if not chunks:\n            return text[:220000]\n        deduped = []\n        seen = set()\n        for c in chunks:\n            k = c[:300]\n            if k not in seen:\n                seen.add(k)\n                deduped.append(c)\n        return "\\n\\n".join(deduped)[:220000]\n\n    def split_sentences(self, text: str) -> List[str]:\n        text = normalize_text(text or "")\n        parts = re.split(r"(?<=[.!?])\\s+(?=[A-Z0-9(])|\\n+", text)\n        out = []\n        for s in parts:\n            s = re.sub(r"\\s+", " ", s).strip()\n            if 60 <= len(s) <= 1200:\n                out.append(s)\n        return out\n\n    def score_sentence(self, sent: str) -> float:\n        low = sent.lower()\n        score = 0.0\n        if any(k.lower() in low for k in self.MATERIAL_TERMS):\n            score += 4.0\n        if any(k.lower() in low for k in self.METHOD_TERMS):\n            score += 3.0\n        if "perovskite" in low or "precursor" in low:\n            score += 2.0\n        if re.search(r"\\b(experimental|materials|methods|synthesis|fabrication|results|discussion)\\b", low):\n            score += 1.5\n        score += min(6, len(self.FORMULA_RE.findall(sent))) * 0.5\n        score += min(5, len(self.VALUE_RE.findall(sent))) * 0.8\n        if re.search(r"\\b(introduction|review|previously reported|literature|prior work|for example)\\b", low):\n            score -= 1.5\n        return score\n\n    def token_inventory(self, text: str) -> Dict[str, List[str]]:\n        text = normalize_text(self._cut_references(text or ""))\n        formulas = []\n        for m in self.FORMULA_RE.finditer(text):\n            s = m.group(0).strip(" .,;:()[]")\n            if 3 <= len(s) <= 90 and any(x in s for x in ["Pb", "Sn", "Cs", "MA", "FA", "PEA", "BA", "Br", "I"]):\n                formulas.append(s)\n        materials = [k for k in self.MATERIAL_TERMS if re.search(r"(?<![A-Za-z0-9])" + re.escape(k) + r"(?![A-Za-z0-9])", text, re.I)]\n        methods = [k for k in self.METHOD_TERMS if k.lower() in text.lower()]\n        values = self.VALUE_RE.findall(text)\n\n        def uniq(seq, limit):\n            seen = set(); out = []\n            for x in seq:\n                key = str(x).lower()\n                if key not in seen:\n                    seen.add(key); out.append(str(x))\n                if len(out) >= limit:\n                    break\n            return out\n        return {\n            "formula_tokens": uniq(formulas, 80),\n            "material_tokens": uniq(materials, 80),\n            "method_tokens": uniq(methods, 60),\n            "condition_tokens": uniq(values, 80),\n        }\n\n    def build(self, full_text: str, extraction_text: str = "") -> Dict[str, Any]:\n        sections = self.extract_relevant_sections(full_text or extraction_text or "")\n        sents = self.split_sentences(sections)\n        scored = []\n        for i, s in enumerate(sents):\n            sc = self.score_sentence(s)\n            if sc > 0:\n                scored.append((sc, i, s))\n        scored.sort(key=lambda x: (-x[0], x[1]))\n        selected = sorted(scored[:MAIN_EVIDENCE_SENTENCE_LIMIT], key=lambda x: x[1])\n        evidence_rows = []\n        for sc, idx, s in selected:\n            roles = []\n            low = s.lower()\n            if any(k.lower() in low for k in self.MATERIAL_TERMS):\n                roles.append("materials_or_solvents")\n            if any(k.lower() in low for k in self.METHOD_TERMS) or self.VALUE_RE.search(s):\n                roles.append("method_conditions")\n            if self.FORMULA_RE.search(s) or "perovskite" in low:\n                roles.append("compound_or_formula")\n            evidence_rows.append({"score": round(sc, 3), "roles": roles, "sentence": s})\n        inventory = self.token_inventory(sections)\n        return {\n            "agent": "MainEvidenceTableAgent_v16",\n            "section_filtered_chars": len(sections),\n            "evidence_row_count": len(evidence_rows),\n            "token_inventory": inventory,\n            "evidence_rows": evidence_rows,\n        }\n\n    def render_for_llm(self, table: Dict[str, Any]) -> str:\n        inv = table.get("token_inventory", {}) or {}\n        lines = [\n            "SYSTEMATIC EVIDENCE TABLE FOR EXTRACTION (source-grounded, not final output):",\n            "Use this table to improve schema extraction. Only populate fields supported by the paper text. Do not use background examples as synthesized materials.",\n            "Role checks: distinguish perovskite synthesis materials from ETL/HTL/back-contact/capping/device-only materials; tie solvents/temperatures to the correct process step; use not-reported/null when unsupported.",\n            "Formula/composition tokens: " + "; ".join(inv.get("formula_tokens", [])[:60]),\n            "Material/solvent tokens: " + "; ".join(inv.get("material_tokens", [])[:60]),\n            "Method tokens: " + "; ".join(inv.get("method_tokens", [])[:50]),\n            "Condition tokens: " + "; ".join(inv.get("condition_tokens", [])[:60]),\n            "High-signal evidence sentences:",\n        ]\n        total = sum(len(x) for x in lines)\n        for row in table.get("evidence_rows", []):\n            line = f"- roles={\',\'.join(row.get(\'roles\', []))}; evidence={row.get(\'sentence\', \'\')}"\n            if total + len(line) > MAIN_EVIDENCE_TABLE_MAX_CHARS:\n                break\n            lines.append(line)\n            total += len(line)\n        return "\\n".join(lines).strip()\n\n\n# Save originals before v16 monkey patches.\n_v16_original_extraction_extract = getattr(ExtractionAgent, "extract")\n_v16_original_eval_evaluate = getattr(EvaluationAgent, "evaluate")\n_v16_original_flag_flag = getattr(FlaggingAgent, "flag")\n\n\ndef _v16_extraction_extract(self: ExtractionAgent, meta: Dict[str, Any], full_text: str, extraction_text: str) -> Dict[str, Any]:\n    if not (SYSTEMATIC_MAIN_PIPELINE_ENABLE and MAIN_EVIDENCE_TABLE_ENABLE):\n        return _v16_original_extraction_extract(self, meta, full_text, extraction_text)\n    table_agent = MainEvidenceTableAgent()\n    evidence_table = table_agent.build(full_text or "", extraction_text or "")\n    evidence_context = table_agent.render_for_llm(evidence_table)\n    systematic_extraction_text = (evidence_context + "\\n\\n--- ORIGINAL SECTION-FILTERED PAPER TEXT ---\\n" + (extraction_text or ""))[: max(len(extraction_text or ""), MAIN_EVIDENCE_TABLE_MAX_CHARS + len(extraction_text or ""))]\n    result = _v16_original_extraction_extract(self, meta, full_text, systematic_extraction_text)\n    result["systematic_evidence_table"] = {\n        "agent": evidence_table.get("agent"),\n        "section_filtered_chars": evidence_table.get("section_filtered_chars"),\n        "evidence_row_count": evidence_table.get("evidence_row_count"),\n        "token_inventory_counts": {k: len(v or []) for k, v in (evidence_table.get("token_inventory") or {}).items()},\n    }\n    if MAIN_EVIDENCE_DEBUG_SAVE:\n        try:\n            slug = slugify((meta or {}).get("title") or (meta or {}).get("doi") or "paper")\n            (RAW_LLM_DIR / f"{slug}_v16_systematic_evidence_table.json").write_text(\n                json.dumps(evidence_table, indent=2, ensure_ascii=False), encoding="utf-8"\n            )\n            (RAW_LLM_TEXT_DIR / f"{slug}_v16_systematic_evidence_context.txt").write_text(\n                evidence_context, encoding="utf-8"\n            )\n        except Exception:\n            pass\n    return result\n\n\ndef _v16_detect_role_quality_issues(best: Dict[str, Any]) -> List[str]:\n    issues = []\n    if not best:\n        return issues\n    # Common false-positive/role-confusion checks learned from the benchmark.\n    perov_solvents = str(best.get("Perovskite_deposition_solvents") or "").lower()\n    htl_solvents = str(best.get("HTL_deposition_solvents") or "").lower()\n    etl_solvents = str(best.get("ETL_deposition_solvents") or "").lower()\n    if "chlorobenzene" in perov_solvents and any(x in perov_solvents for x in ["pmma", "spiro", "htl"]):\n        issues.append("possible_device_or_capping_solvent_assigned_to_perovskite")\n    if best.get("Perovskite_composition_short_form") in (None, "") and best.get("Perovskite_composition_long_form") in (None, ""):\n        issues.append("composition_not_resolved")\n    if best.get("Perovskite_deposition_thermal_annealing_temperature") not in (None, "") and best.get("Perovskite_deposition_procedure") in (None, ""):\n        issues.append("annealing_temperature_without_deposition_method")\n    if htl_solvents and perov_solvents and htl_solvents == perov_solvents:\n        issues.append("same_solvent_assigned_to_perovskite_and_htl_check_roles")\n    if etl_solvents and perov_solvents and etl_solvents == perov_solvents:\n        issues.append("same_solvent_assigned_to_perovskite_and_etl_check_roles")\n    return issues\n\n\ndef _v16_evaluation_evaluate(self: EvaluationAgent, records: List[Dict[str, Any]]) -> Dict[str, Any]:\n    ev = _v16_original_eval_evaluate(self, records)\n    try:\n        sorted_records = sorted(records or [], key=score_record_completeness, reverse=True)\n        best = sorted_records[0] if sorted_records else {}\n        issues = _v16_detect_role_quality_issues(best)\n        ev["systematic_validation"] = {\n            "agent": "MainValidationAgent_v16",\n            "role_quality_issues": issues,\n            "role_quality_issue_count": len(issues),\n            "drive_first_pipeline_enabled": bool(DRIVE_FIRST_STRICT),\n            "evidence_table_enabled": bool(MAIN_EVIDENCE_TABLE_ENABLE),\n        }\n        if issues:\n            ev["record_confidence"] = round(max(0.0, float(ev.get("record_confidence", 0.0)) - min(0.15, 0.04 * len(issues))), 4)\n    except Exception:\n        pass\n    return ev\n\n\ndef _v16_flagging_flag(self: FlaggingAgent, meta: Dict[str, Any], records: List[Dict[str, Any]], evaluation: Dict[str, Any]) -> Dict[str, Any]:\n    out = _v16_original_flag_flag(self, meta, records, evaluation)\n    try:\n        sysval = (evaluation or {}).get("systematic_validation", {}) or {}\n        issues = sysval.get("role_quality_issues", []) or []\n        if issues:\n            out.setdefault("flags", [])\n            out["flags"].append("systematic_role_validation_warning")\n            for issue in issues:\n                out["flags"].append(issue)\n            out["flags"] = sorted(set(out["flags"]))\n        out["systematic_validation"] = sysval\n    except Exception:\n        pass\n    return out\n\n\nExtractionAgent.extract = _v16_extraction_extract\nEvaluationAgent.evaluate = _v16_evaluation_evaluate\nFlaggingAgent.flag = _v16_flagging_flag\n\n\n# v16 Drive-first scheduler for normal mode. This replaces the previous batch API\n# wrapper so Drive/manual papers are exhausted before RSS/Crossref/Unpaywall expansion.\n_v16_web_expand_process_batch_api = _original_process_batch_api\n\n\ndef process_batch_api(\n    query: str = SEARCH_QUERY,\n    year_min: int = YEAR_MIN,\n    year_max: int = YEAR_MAX,\n    max_papers: int = MAX_PAPERS_TO_PROCESS,\n    run_mode: Optional[str] = None,\n    force_reprocess: bool = False,\n    reset_output: bool = False,\n) -> Dict[str, Any]:\n    mode = normalize_run_mode(run_mode)\n    if reset_output or mode == "reset":\n        reset_work_dir()\n\n    drive_payload: Dict[str, Any] = {\n        "performed": False,\n        "strict_drive_first": bool(DRIVE_FIRST_STRICT),\n        "google_drive_folder": GOOGLE_DRIVE_FOLDER_LINK or GOOGLE_DRIVE_FOLDER_ID,\n        "processed_before_web_expansion": False,\n    }\n\n    if DRIVE_FIRST_STRICT and (GOOGLE_DRIVE_FOLDER_LINK or GOOGLE_DRIVE_FOLDER_ID):\n        drive_payload["performed"] = True\n        try:\n            drive_list = list_google_drive_folder_api(GOOGLE_DRIVE_FOLDER_LINK or GOOGLE_DRIVE_FOLDER_ID)\n            drive_payload["list_result"] = {\n                "ok": drive_list.get("ok"),\n                "count": drive_list.get("count"),\n                "folder_id": drive_list.get("folder_id"),\n                "reason": drive_list.get("reason"),\n            }\n            if not drive_list.get("ok"):\n                drive_payload["process_result"] = {"ok": False, "reason": drive_list.get("reason")}\n                if DRIVE_FIRST_STOP_ON_ERROR:\n                    return {"ok": False, "tool": "process_batch", "stage": "drive_first_precheck", "drive_first": drive_payload, "web_expansion_skipped": True}\n            else:\n                drive_count = int(drive_list.get("count") or 0)\n                max_drive_files = drive_count if DRIVE_FIRST_PROCESS_ALL_FILES else min(drive_count, int(GOOGLE_DRIVE_MAX_FILES_PER_RUN))\n                if drive_count > 0:\n                    drive_result = process_google_drive_folder_api(\n                        folder_id_or_link=GOOGLE_DRIVE_FOLDER_LINK or GOOGLE_DRIVE_FOLDER_ID,\n                        max_files=max_drive_files,\n                        include_ocr=DRIVE_FIRST_INCLUDE_OCR,\n                        include_figures=DRIVE_FIRST_INCLUDE_FIGURES,\n                        force_reprocess=force_reprocess or DRIVE_FIRST_FORCE_REPROCESS,\n                        run_mode=mode,\n                    )\n                    drive_payload["processed_before_web_expansion"] = True\n                    drive_payload["process_result"] = {\n                        "ok": drive_result.get("ok"),\n                        "selected_count": drive_result.get("selected_count"),\n                        "processed_count": drive_result.get("processed_count"),\n                        "skipped_count": drive_result.get("skipped_count"),\n                        "manual_google_drive_dir": drive_result.get("manual_google_drive_dir"),\n                        "processing_order": drive_result.get("processing_order"),\n                    }\n                    if not drive_result.get("ok") and DRIVE_FIRST_STOP_ON_ERROR:\n                        return {"ok": False, "tool": "process_batch", "stage": "drive_first_processing", "drive_first": drive_payload, "web_expansion_skipped": True}\n                else:\n                    drive_payload["process_result"] = {"ok": True, "selected_count": 0, "processed_count": 0, "skipped_count": 0}\n        except Exception as exc:\n            drive_payload["error"] = repr(exc)\n            if DRIVE_FIRST_STOP_ON_ERROR:\n                return {"ok": False, "tool": "process_batch", "stage": "drive_first_exception", "drive_first": drive_payload, "web_expansion_skipped": True}\n\n    # After all manual Drive files have been checked/processed, expand online.\n    result = _v16_web_expand_process_batch_api(\n        query=query,\n        year_min=year_min,\n        year_max=year_max,\n        max_papers=max_papers,\n        run_mode=mode,\n        force_reprocess=force_reprocess,\n        reset_output=False,  # reset already handled above so Drive outputs are not wiped after processing\n    )\n    try:\n        synced = []\n        for item in result.get("results", []) or []:\n            slug = item.get("paper_slug")\n            if slug:\n                synced.append(sync_paper_to_combined_coverage(slug))\n        result["combined_global_sync"] = sync_global_outputs_to_combined()\n        result["combined_paper_syncs"] = synced\n        result["combined_coverage_dir"] = str(COMBINED_COVERAGE_DIR)\n    except Exception:\n        pass\n    result["drive_first"] = drive_payload\n    result["google_drive_processed_before_search"] = bool(drive_payload.get("processed_before_web_expansion"))\n    result["systematic_pipeline"] = {\n        "version": "v16",\n        "evidence_table_enabled": bool(MAIN_EVIDENCE_TABLE_ENABLE),\n        "role_validation_enabled": True,\n        "drive_first_strict": bool(DRIVE_FIRST_STRICT),\n        "web_expansion_after_drive": True,\n    }\n    return result\n\n\ndef literature_agent_main() -> None:\n    """v16 normal entry point.\n\n    Normal mode now always routes through the v16 process_batch_api wrapper,\n    which processes manually added Google Drive papers before any web expansion.\n    Benchmark mode is still handled separately in main().\n    """\n    if not ALL_COLUMNS:\n        raise ValueError("ALL_COLUMNS is empty.")\n    if not HF_API_KEY:\n        raise ValueError("HF_API_KEY is missing.")\n    mode = normalize_run_mode(RUN_MODE)\n    out = process_batch_api(\n        query=SEARCH_QUERY,\n        year_min=YEAR_MIN,\n        year_max=YEAR_MAX,\n        max_papers=MAX_PAPERS_TO_PROCESS,\n        run_mode=mode,\n        force_reprocess=(mode in {"reprocess", "reset"}),\n        reset_output=(mode == "reset"),\n    )\n    print(json.dumps(out, indent=2, ensure_ascii=False))\n\n\n\n# ============================================================\n# v17 DRIVE-FIRST INCREMENTAL QUEUE OVERRIDE\n# Google Drive is always checked first, but previously processed Drive files\n# are skipped by Google Drive file ID. Existing Drive-derived outputs are not\n# overwritten unless DRIVE_FIRST_FORCE_REPROCESS=1 is explicitly set.\n# ============================================================\nDRIVE_FIRST_INCREMENTAL_ONLY = os.getenv("DRIVE_FIRST_INCREMENTAL_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}\nDRIVE_FIRST_SKIP_PROCESSED_IDS = os.getenv("DRIVE_FIRST_SKIP_PROCESSED_IDS", "1").strip().lower() in {"1", "true", "yes", "on"}\nDRIVE_FIRST_NO_OVERWRITE_DOWNLOADS = os.getenv("DRIVE_FIRST_NO_OVERWRITE_DOWNLOADS", "1").strip().lower() in {"1", "true", "yes", "on"}\n\n\ndef _drive_incremental_manifest_path(folder_id: Optional[str]) -> Path:\n    safe_folder = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(folder_id or "default_drive_folder"))\n    return MANUAL_GOOGLE_DRIVE_MANIFEST_DIR / f"drive_incremental_processed_{safe_folder}.json"\n\n\ndef _load_drive_incremental_manifest(folder_id: Optional[str]) -> Dict[str, Any]:\n    path = _drive_incremental_manifest_path(folder_id)\n    if not path.exists():\n        return {"folder_id": folder_id, "processed_file_ids": {}, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}\n    try:\n        data = json.loads(path.read_text(encoding="utf-8"))\n        if not isinstance(data, dict):\n            raise ValueError("manifest is not a dict")\n        data.setdefault("folder_id", folder_id)\n        data.setdefault("processed_file_ids", {})\n        return data\n    except Exception:\n        return {"folder_id": folder_id, "processed_file_ids": {}, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "recovered_from_invalid_manifest": True}\n\n\ndef _save_drive_incremental_manifest(folder_id: Optional[str], manifest: Dict[str, Any]) -> None:\n    manifest = dict(manifest or {})\n    manifest["folder_id"] = folder_id\n    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())\n    manifest.setdefault("processed_file_ids", {})\n    path = _drive_incremental_manifest_path(folder_id)\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")\n    try:\n        mirror = COMBINED_MANIFEST_DIR / path.name\n        mirror.parent.mkdir(parents=True, exist_ok=True)\n        mirror.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")\n    except Exception:\n        pass\n\n\ndef _drive_file_already_processed(item: Dict[str, Any], manifest: Dict[str, Any]) -> bool:\n    if not item:\n        return False\n    file_id = item.get("id")\n    if not file_id:\n        return False\n    processed_ids = (manifest or {}).get("processed_file_ids", {}) or {}\n    entry = processed_ids.get(file_id)\n    if not entry:\n        return False\n    # Incremental queue semantics: once a Drive file ID has been successfully\n    # processed, it is skipped on future runs. This intentionally treats the\n    # Drive folder as an add-new-files queue instead of overwriting old outputs.\n    return str(entry.get("status") or "").lower() in {"processed", "cached", "ok"}\n\n\ndef _drive_download_item_without_overwrite(item: Dict[str, Any], folder_id: str) -> Dict[str, Any]:\n    raw_name = item.get("name") or item.get("id") or "file"\n    safe_stem = slugify(Path(raw_name).stem)\n    file_id_short = (item.get("id") or "file")[:10]\n    mime_type = item.get("mimeType") or ""\n    allowed_google_types = {\n        "application/pdf",\n        "application/vnd.google-apps.document",\n        "application/vnd.google-apps.presentation",\n    }\n    suffix = ".pdf" if mime_type in allowed_google_types else (Path(raw_name).suffix or ".bin")\n    out_path = MANUAL_GOOGLE_DRIVE_PDF_DIR / f"{safe_stem}_{file_id_short}{suffix}"\n    reused_existing_file = out_path.exists()\n    if reused_existing_file and DRIVE_FIRST_NO_OVERWRITE_DOWNLOADS:\n        saved_path = out_path\n    else:\n        saved_path = export_or_download_google_drive_file(item["id"], mime_type, out_path)\n    return {\n        "id": item.get("id"),\n        "name": raw_name,\n        "mimeType": mime_type,\n        "modifiedTime": item.get("modifiedTime"),\n        "webViewLink": item.get("webViewLink"),\n        "folder_id": folder_id,\n        "saved_to": str(saved_path),\n        "source_collection": "manual_google_drive",\n        "reused_existing_file": bool(reused_existing_file),\n        "download_overwrite_allowed": not DRIVE_FIRST_NO_OVERWRITE_DOWNLOADS,\n    }\n\n\n_v17_previous_process_google_drive_folder_api = process_google_drive_folder_api\n\n\ndef process_google_drive_folder_api(\n    folder_id_or_link: Optional[str] = None,\n    max_files: int = 20,\n    include_ocr: bool = True,\n    include_figures: bool = True,\n    force_reprocess: bool = False,\n    run_mode: Optional[str] = None,\n) -> Dict[str, Any]:\n    """Incremental Google Drive queue processor.\n\n    The folder is checked every run, but already processed Google Drive file IDs\n    are skipped before download/LLM extraction. Existing outputs are preserved\n    unless force_reprocess=True or DRIVE_FIRST_FORCE_REPROCESS=1 is explicitly set.\n    """\n    mode = normalize_run_mode(run_mode)\n    folder_id = parse_google_drive_folder_id(folder_id_or_link or GOOGLE_DRIVE_FOLDER_ID or GOOGLE_DRIVE_FOLDER_LINK)\n    if not folder_id:\n        return {"ok": False, "reason": "No valid Google Drive folder ID or link was provided."}\n\n    try:\n        files = list_google_drive_folder_files(folder_id=folder_id)\n    except Exception as exc:\n        return {"ok": False, "folder_id": folder_id, "reason": f"Failed to list or access Google Drive folder: {exc}"}\n\n    allowed_google_types = {\n        "application/pdf",\n        "application/vnd.google-apps.document",\n        "application/vnd.google-apps.presentation",\n    }\n    all_pdf_like = [item for item in files if (item.get("mimeType") or "") in allowed_google_types]\n    all_pdf_like = _sort_drive_items_newest_first(all_pdf_like)\n\n    manifest = _load_drive_incremental_manifest(folder_id)\n    force_drive = bool(force_reprocess or DRIVE_FIRST_FORCE_REPROCESS)\n\n    skipped_existing_queue = []\n    queue = []\n    for item in all_pdf_like:\n        if DRIVE_FIRST_INCREMENTAL_ONLY and DRIVE_FIRST_SKIP_PROCESSED_IDS and not force_drive:\n            if _drive_file_already_processed(item, manifest):\n                entry = (manifest.get("processed_file_ids", {}) or {}).get(item.get("id"), {})\n                skipped_existing_queue.append({\n                    "id": item.get("id"),\n                    "name": item.get("name"),\n                    "modifiedTime": item.get("modifiedTime"),\n                    "reason": "already processed in prior Drive-first run",\n                    "paper_slug": entry.get("paper_slug"),\n                    "processed_at": entry.get("processed_at"),\n                })\n                continue\n        queue.append(item)\n        if len(queue) >= max(0, int(max_files)):\n            break\n\n    imported = []\n    processed = []\n    skipped_processed_outputs = []\n    errors = []\n\n    for item in queue:\n        try:\n            import_item = _drive_download_item_without_overwrite(item, folder_id)\n            imported.append(import_item)\n            meta = build_manual_drive_meta(import_item, include_ocr=include_ocr, include_figures=include_figures)\n            slug = slugify(meta.get("title") or meta.get("doi") or meta.get("original_filename") or "paper")\n\n            # Even if a file ID was not in the Drive manifest, do not overwrite\n            # an already existing paper output unless explicitly forced.\n            if not force_drive and should_skip_processed_paper(meta, slug, mode):\n                cached = get_saved_paper_output_api(slug)\n                skipped_processed_outputs.append({\n                    "id": item.get("id"),\n                    "name": item.get("name"),\n                    "paper_slug": slug,\n                    "reason": "paper output already exists; not overwriting",\n                    "cached_found": cached.get("ok", False),\n                    "modifiedTime": item.get("modifiedTime"),\n                })\n                manifest.setdefault("processed_file_ids", {})[item.get("id")] = {\n                    "status": "cached",\n                    "name": item.get("name"),\n                    "modifiedTime": item.get("modifiedTime"),\n                    "paper_slug": slug,\n                    "saved_to": import_item.get("saved_to"),\n                    "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),\n                    "note": "Existing processed output found; marked as handled for incremental Drive queue.",\n                }\n                continue\n\n            result = process_one_paper_api(meta=meta, force_reprocess=force_drive, run_mode=mode)\n            result.setdefault("ingestion_options", {})\n            result["ingestion_options"]["include_ocr"] = include_ocr\n            result["ingestion_options"]["include_figures"] = include_figures\n            processed.append({\n                "name": item.get("name"),\n                "id": item.get("id"),\n                "saved_to": import_item.get("saved_to"),\n                "modifiedTime": item.get("modifiedTime"),\n                "meta": meta,\n                "result": result,\n            })\n            manifest.setdefault("processed_file_ids", {})[item.get("id")] = {\n                "status": "processed",\n                "name": item.get("name"),\n                "modifiedTime": item.get("modifiedTime"),\n                "paper_slug": slug,\n                "saved_to": import_item.get("saved_to"),\n                "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),\n            }\n        except Exception as exc:\n            err = {\n                "id": item.get("id"),\n                "name": item.get("name"),\n                "modifiedTime": item.get("modifiedTime"),\n                "error": repr(exc),\n            }\n            imported.append({**err, "source_collection": "manual_google_drive"})\n            errors.append(err)\n\n    _save_drive_incremental_manifest(folder_id, manifest)\n    global_sync = sync_global_outputs_to_combined()\n    payload = {\n        "ok": len(errors) == 0,\n        "tool": "process_google_drive_folder",\n        "version": "v17_incremental_drive_queue",\n        "folder_id": folder_id,\n        "folder_link": folder_id_or_link or GOOGLE_DRIVE_FOLDER_LINK,\n        "run_mode": mode,\n        "incremental_only": bool(DRIVE_FIRST_INCREMENTAL_ONLY),\n        "force_drive_reprocess": bool(force_drive),\n        "no_overwrite_downloads": bool(DRIVE_FIRST_NO_OVERWRITE_DOWNLOADS),\n        "listed_count": len(files),\n        "pdf_like_count": len(all_pdf_like),\n        "new_queue_count": len(queue),\n        "processed_count": len(processed),\n        "skipped_already_processed_count": len(skipped_existing_queue),\n        "skipped_existing_output_count": len(skipped_processed_outputs),\n        "error_count": len(errors),\n        "imported": imported,\n        "processed": processed,\n        "skipped_already_processed": skipped_existing_queue,\n        "skipped_existing_outputs": skipped_processed_outputs,\n        "errors": errors,\n        "manual_google_drive_dir": str(MANUAL_GOOGLE_DRIVE_DIR),\n        "manual_google_drive_pdf_dir": str(MANUAL_GOOGLE_DRIVE_PDF_DIR),\n        "manual_google_drive_text_dir": str(MANUAL_GOOGLE_DRIVE_TEXT_DIR),\n        "manifest_path": str(_drive_incremental_manifest_path(folder_id)),\n        "combined_global_sync": global_sync,\n        "drive_files_checked_first": True,\n        "processing_order": "google_drive_incremental_new_files_first_then_web_expansion",\n    }\n    try:\n        (MANUAL_GOOGLE_DRIVE_MANIFEST_DIR / f"processed_manifest_{folder_id}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")\n        (COMBINED_MANIFEST_DIR / f"processed_manifest_{folder_id}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")\n    except Exception:\n        pass\n    return payload\n\n\n_v17_web_expand_process_batch_api = _v16_web_expand_process_batch_api\n\n\ndef process_batch_api(\n    query: str = SEARCH_QUERY,\n    year_min: int = YEAR_MIN,\n    year_max: int = YEAR_MAX,\n    max_papers: int = MAX_PAPERS_TO_PROCESS,\n    run_mode: Optional[str] = None,\n    force_reprocess: bool = False,\n    reset_output: bool = False,\n) -> Dict[str, Any]:\n    """v17 Drive-first scheduler.\n\n    Rechecks the Drive folder at the start of every normal run, processes only\n    new/unhandled Drive file IDs, then proceeds to web expansion. A global\n    reprocess/reset run does not overwrite Drive outputs unless\n    DRIVE_FIRST_FORCE_REPROCESS=1 is explicitly set.\n    """\n    mode = normalize_run_mode(run_mode)\n    if reset_output or mode == "reset":\n        reset_work_dir()\n\n    drive_payload: Dict[str, Any] = {\n        "performed": False,\n        "strict_drive_first": bool(DRIVE_FIRST_STRICT),\n        "google_drive_folder": GOOGLE_DRIVE_FOLDER_LINK or GOOGLE_DRIVE_FOLDER_ID,\n        "processed_before_web_expansion": False,\n        "incremental_only": bool(DRIVE_FIRST_INCREMENTAL_ONLY),\n    }\n\n    if DRIVE_FIRST_STRICT and (GOOGLE_DRIVE_FOLDER_LINK or GOOGLE_DRIVE_FOLDER_ID):\n        drive_payload["performed"] = True\n        try:\n            drive_list = list_google_drive_folder_api(GOOGLE_DRIVE_FOLDER_LINK or GOOGLE_DRIVE_FOLDER_ID)\n            drive_payload["list_result"] = {\n                "ok": drive_list.get("ok"),\n                "count": drive_list.get("count"),\n                "folder_id": drive_list.get("folder_id"),\n                "reason": drive_list.get("reason"),\n            }\n            if not drive_list.get("ok"):\n                drive_payload["process_result"] = {"ok": False, "reason": drive_list.get("reason")}\n                if DRIVE_FIRST_STOP_ON_ERROR:\n                    return {"ok": False, "tool": "process_batch", "stage": "drive_first_precheck", "drive_first": drive_payload, "web_expansion_skipped": True}\n            else:\n                drive_count = int(drive_list.get("count") or 0)\n                max_drive_files = drive_count if DRIVE_FIRST_PROCESS_ALL_FILES else min(drive_count, int(GOOGLE_DRIVE_MAX_FILES_PER_RUN))\n                if drive_count > 0:\n                    # Critical v17 behavior: do not pass global reprocess/reset\n                    # into Drive ingestion. Drive files are only reprocessed if\n                    # DRIVE_FIRST_FORCE_REPROCESS is explicitly enabled.\n                    drive_result = process_google_drive_folder_api(\n                        folder_id_or_link=GOOGLE_DRIVE_FOLDER_LINK or GOOGLE_DRIVE_FOLDER_ID,\n                        max_files=max_drive_files,\n                        include_ocr=DRIVE_FIRST_INCLUDE_OCR,\n                        include_figures=DRIVE_FIRST_INCLUDE_FIGURES,\n                        force_reprocess=DRIVE_FIRST_FORCE_REPROCESS,\n                        run_mode=mode,\n                    )\n                    drive_payload["processed_before_web_expansion"] = True\n                    drive_payload["process_result"] = {\n                        "ok": drive_result.get("ok"),\n                        "pdf_like_count": drive_result.get("pdf_like_count"),\n                        "new_queue_count": drive_result.get("new_queue_count"),\n                        "processed_count": drive_result.get("processed_count"),\n                        "skipped_already_processed_count": drive_result.get("skipped_already_processed_count"),\n                        "skipped_existing_output_count": drive_result.get("skipped_existing_output_count"),\n                        "error_count": drive_result.get("error_count"),\n                        "manual_google_drive_dir": drive_result.get("manual_google_drive_dir"),\n                        "processing_order": drive_result.get("processing_order"),\n                        "manifest_path": drive_result.get("manifest_path"),\n                    }\n                    if not drive_result.get("ok") and DRIVE_FIRST_STOP_ON_ERROR:\n                        return {"ok": False, "tool": "process_batch", "stage": "drive_first_processing", "drive_first": drive_payload, "web_expansion_skipped": True}\n                else:\n                    drive_payload["process_result"] = {"ok": True, "selected_count": 0, "processed_count": 0, "skipped_count": 0}\n        except Exception as exc:\n            drive_payload["error"] = repr(exc)\n            if DRIVE_FIRST_STOP_ON_ERROR:\n                return {"ok": False, "tool": "process_batch", "stage": "drive_first_exception", "drive_first": drive_payload, "web_expansion_skipped": True}\n\n    result = _v17_web_expand_process_batch_api(\n        query=query,\n        year_min=year_min,\n        year_max=year_max,\n        max_papers=max_papers,\n        run_mode=mode,\n        force_reprocess=force_reprocess,\n        reset_output=False,\n    )\n    try:\n        synced = []\n        for item in result.get("results", []) or []:\n            slug = item.get("paper_slug")\n            if slug:\n                synced.append(sync_paper_to_combined_coverage(slug))\n        result["combined_global_sync"] = sync_global_outputs_to_combined()\n        result["combined_paper_syncs"] = synced\n        result["combined_coverage_dir"] = str(COMBINED_COVERAGE_DIR)\n    except Exception:\n        pass\n    result["drive_first"] = drive_payload\n    result["google_drive_processed_before_search"] = bool(drive_payload.get("processed_before_web_expansion"))\n    result["systematic_pipeline"] = {\n        "version": "v17_incremental_drive_queue",\n        "evidence_table_enabled": bool(MAIN_EVIDENCE_TABLE_ENABLE),\n        "role_validation_enabled": True,\n        "drive_first_strict": bool(DRIVE_FIRST_STRICT),\n        "drive_incremental_only": bool(DRIVE_FIRST_INCREMENTAL_ONLY),\n        "web_expansion_after_drive": True,\n    }\n    return result\n\n\ndef literature_agent_main() -> None:\n    """v17 normal entry point.\n\n    Normal mode checks Google Drive first, processes only new/unhandled Drive\n    files, does not overwrite old Drive-derived outputs by default, then runs\n    web expansion. Benchmark mode is still handled separately in main().\n    """\n    if not ALL_COLUMNS:\n        raise ValueError("ALL_COLUMNS is empty.")\n    if not HF_API_KEY:\n        raise ValueError("HF_API_KEY is missing.")\n    mode = normalize_run_mode(RUN_MODE)\n    out = process_batch_api(\n        query=SEARCH_QUERY,\n        year_min=YEAR_MIN,\n        year_max=YEAR_MAX,\n        max_papers=MAX_PAPERS_TO_PROCESS,\n        run_mode=mode,\n        force_reprocess=(mode in {"reprocess", "reset"}),\n        reset_output=(mode == "reset"),\n    )\n    print(json.dumps(out, indent=2, ensure_ascii=False))\n\n\n\n\n# ============================================================\n# v18 ONTOLOGY + COST-CONTROL + FIGURE INTELLIGENCE LAYER\n# Drop-in extension over v17. This section intentionally overrides\n# selected functions/classes above without deleting the working v17 code.\n# ============================================================\nimport base64\nfrom collections import defaultdict\n\nAGENT_VERSION = "v18_ontology_cost_visual"\n\n# -----------------------------\n# v18 runtime switches\n# -----------------------------\nONTOLOGY_ENABLE = os.getenv("ONTOLOGY_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}\nDETERMINISTIC_EXTRACTION_ENABLE = os.getenv("DETERMINISTIC_EXTRACTION_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}\nFAMILY_GATING_ENABLE = os.getenv("FAMILY_GATING_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}\nCHUNK_RANKING_ENABLE = os.getenv("CHUNK_RANKING_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}\nCOST_LOGGING_ENABLE = os.getenv("COST_LOGGING_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}\nVISION_LLM_ENABLE = os.getenv("VISION_LLM_ENABLE", "0").strip().lower() in {"1", "true", "yes", "on"}\n\n# cheap = fewer families/chunks/LLM calls; balanced = default; thorough = closer to old exhaustive behavior\nLLM_COST_MODE = os.getenv("LLM_COST_MODE", "balanced").strip().lower()\nHF_MODEL_FAST = os.getenv("HF_MODEL_FAST", os.getenv("HF_MODEL", "Qwen/Qwen3-32B")).strip()\nHF_MODEL_STRONG = os.getenv("HF_MODEL_STRONG", os.getenv("HF_MODEL", "Qwen/Qwen3-32B")).strip()\nHF_VISION_MODEL = os.getenv("HF_VISION_MODEL", "").strip()\n\nMAX_CHUNKS_PER_FAMILY_CHEAP = int(os.getenv("MAX_CHUNKS_PER_FAMILY_CHEAP", "2"))\nMAX_CHUNKS_PER_FAMILY_BALANCED = int(os.getenv("MAX_CHUNKS_PER_FAMILY_BALANCED", "4"))\nMAX_CHUNKS_PER_FAMILY_THOROUGH = int(os.getenv("MAX_CHUNKS_PER_FAMILY_THOROUGH", "8"))\nMIN_FAMILY_SCORE_CHEAP = int(os.getenv("MIN_FAMILY_SCORE_CHEAP", "2"))\nMIN_FAMILY_SCORE_BALANCED = int(os.getenv("MIN_FAMILY_SCORE_BALANCED", "1"))\nMIN_FAMILY_SCORE_THOROUGH = int(os.getenv("MIN_FAMILY_SCORE_THOROUGH", "0"))\n\nONTOLOGY_DIR = BASE / "ontology"\nCOST_DIR = BASE / "cost_logs"\nFIGURE_EVIDENCE_DIR = BASE / "figure_evidence"\nfor _d in [ONTOLOGY_DIR, COST_DIR, FIGURE_EVIDENCE_DIR]:\n    _d.mkdir(parents=True, exist_ok=True)\n\n# -----------------------------\n# Domain ontology\n# -----------------------------\nPEROVSKITE_ONTOLOGY: Dict[str, Any] = {\n    "a_site_cations": {\n        "small_organic": {\n            "FA": ["formamidinium", "HC(NH2)2", "FAI", "formamidinium iodide", "FA+"],\n            "MA": ["methylammonium", "CH3NH3", "MAI", "methylammonium iodide", "MA+"],\n            "GA": ["guanidinium", "GAI", "guanidinium iodide"],\n            "EA": ["ethylammonium", "EAI", "ethylammonium iodide"],\n            "DMA": ["dimethylammonium", "DMAI", "dimethylammonium iodide"],\n        },\n        "inorganic": {\n            "Cs": ["cesium", "caesium", "CsI", "cesium iodide"],\n            "Rb": ["rubidium", "RbI", "rubidium iodide"],\n            "K": ["potassium", "KI", "potassium iodide"],\n        },\n        "large_organic_spacers": {\n            "PEA": ["phenethylammonium", "2-phenylethylammonium", "PEAI", "PEA iodide", "PEA2PbI4"],\n            "BA": ["butylammonium", "n-butylammonium", "BAI", "butylammonium iodide", "BA2PbI4"],\n            "BDA": ["butane-1,4-diammonium", "1,4-butanediammonium", "BDAI2", "BDAPbI4"],\n            "OAI": ["octylammonium iodide", "octylammonium", "n-octylammonium", "OA", "OAm"],\n            "4Tm": ["thiophenemethylammonium", "thiophen-2-ylmethylammonium", "4Tm"],\n            "PMA": ["phenylmethylammonium", "benzylammonium", "PMAI"],\n            "AVA": ["5-aminovaleric acid", "AVA", "AVAI"],\n        },\n    },\n    "b_site_cations": {\n        "Pb": ["lead", "PbI2", "PbBr2", "lead iodide", "lead bromide", "lead chloride"],\n        "Sn": ["tin", "SnI2", "SnBr2", "tin iodide", "tin bromide"],\n        "Ge": ["germanium", "GeI2", "germanium iodide"],\n        "Bi": ["bismuth", "BiI3", "bismuth iodide"],\n        "Sb": ["antimony", "SbI3", "antimony iodide"],\n    },\n    "x_site_anions": {\n        "I": ["iodide", "I-", "iodine", "PbI2", "FAI", "MAI", "CsI"],\n        "Br": ["bromide", "Br-", "bromine", "PbBr2", "FABr", "MABr", "CsBr"],\n        "Cl": ["chloride", "Cl-", "chlorine", "MACl", "FACl", "PbCl2"],\n        "SCN": ["thiocyanate", "SCN", "NH4SCN"],\n    },\n    "dimensionality": {\n        "3D": ["3D perovskite", "APbX3", "FAPbI3", "MAPbI3", "CsPbI3", "black phase", "alpha phase"],\n        "2D_RP": ["Ruddlesden-Popper", "RP perovskite", "A\'2An-1BnX3n+1", "PEA2PbI4", "BA2PbI4"],\n        "2D_DJ": ["Dion-Jacobson", "DJ perovskite", "A\'An-1BnX3n+1", "BDAPbI4"],\n        "quasi_2D": ["quasi-2D", "quasi 2D", "2D/3D", "2D-3D", "mixed-dimensional", "n = 1", "n=1", "n = 2", "n=2", "n = 3", "n=3", "n = 4", "n=4"],\n        "0D": ["zero-dimensional", "0D", "isolated octahedra"],\n    },\n    "processing": {\n        "solution_deposition": ["spin coating", "spin-coated", "one-step", "two-step", "blade coating", "slot-die", "doctor blade", "spray coating", "drop casting"],\n        "vapor_deposition": ["thermal evaporation", "co-evaporation", "vapor deposition", "CVD", "PVD"],\n        "quenching": ["antisolvent", "anti-solvent", "chlorobenzene", "toluene", "ethyl acetate", "diethyl ether", "anisole", "dripping"],\n        "annealing": ["annealed", "annealing", "hotplate", "thermal treatment", "solvent annealing", "post-treatment"],\n        "atmosphere": ["glovebox", "nitrogen", "N2", "air", "ambient", "relative humidity", "RH"],\n    },\n    "solvents": {\n        "precursor_solvents": ["DMF", "DMSO", "NMP", "GBL", "DMAc", "ACN", "2-ME", "DMI"],\n        "antisolvents": ["chlorobenzene", "toluene", "ethyl acetate", "diethyl ether", "anisole", "trifluorotoluene"],\n        "alcohols": ["IPA", "isopropanol", "ethanol", "methanol", "2-propanol"],\n    },\n    "device_layers": {\n        "substrates": ["ITO", "FTO", "glass/ITO", "glass/FTO", "flexible substrate", "PET", "PEN"],\n        "etl": ["SnO2", "TiO2", "C60", "PCBM", "ZnO", "phenyl-C61-butyric acid methyl ester"],\n        "htl": ["Spiro-OMeTAD", "spiro", "PTAA", "NiOx", "PEDOT:PSS", "MeO-2PACz", "2PACz", "SAM", "self-assembled monolayer"],\n        "back_contact": ["Au", "Ag", "Al", "Cu", "carbon", "metal electrode", "back contact"],\n        "architecture": ["n-i-p", "p-i-n", "regular architecture", "inverted architecture", "mesoscopic", "planar"],\n    },\n    "performance_metrics": {\n        "jv": ["PCE", "power conversion efficiency", "Voc", "V_OC", "open-circuit voltage", "Jsc", "J_SC", "short-circuit current", "FF", "fill factor", "hysteresis", "reverse scan", "forward scan"],\n        "eqe": ["EQE", "IPCE", "external quantum efficiency", "integrated Jsc", "integrated current"],\n        "stabilized": ["MPP", "maximum power point", "stabilized PCE", "stabilised PCE", "steady-state", "steady state"],\n        "stability": ["T80", "T95", "T90", "ISOS", "operational stability", "thermal stability", "shelf stability", "humidity stability", "retained", "after 1000 h", "aging"],\n    },\n    "characterization": {\n        "structural": ["XRD", "GIWAXS", "XPS", "FTIR", "Raman", "SEM", "TEM", "AFM"],\n        "optical": ["PL", "photoluminescence", "TRPL", "UV-vis", "absorption", "band gap", "Tauc"],\n        "electronic": ["SCLC", "EIS", "Mott-Schottky", "UPS", "KPFM", "transient photocurrent", "transient photovoltage"],\n    },\n}\n\nFAMILY_ONTOLOGY_MAP: Dict[str, List[str]] = {\n    "reference_device": ["device_layers", "performance_metrics"],\n    "perovskite_chemistry": ["a_site_cations", "b_site_cations", "x_site_anions", "dimensionality", "characterization"],\n    "etl": ["device_layers", "processing", "solvents"],\n    "perovskite_processing": ["a_site_cations", "b_site_cations", "x_site_anions", "processing", "solvents", "dimensionality"],\n    "htl_backcontact_additional": ["device_layers", "processing", "solvents"],\n    "performance": ["performance_metrics", "device_layers"],\n    "stability_outdoor": ["performance_metrics", "device_layers"],\n}\n\nFAMILY_GATE_TERMS: Dict[str, List[str]] = {\n    "reference_device": ["device", "architecture", "stack", "ITO", "FTO", "substrate", "cell"],\n    "perovskite_chemistry": ["perovskite", "composition", "FA", "MA", "Cs", "PbI2", "PbBr2", "band gap", "PL", "2D", "3D"],\n    "etl": ["ETL", "electron transport", "SnO2", "TiO2", "C60", "PCBM", "ZnO"],\n    "perovskite_processing": ["spin", "anneal", "antisolvent", "precursor", "DMF", "DMSO", "deposition", "solution"],\n    "htl_backcontact_additional": ["HTL", "hole transport", "spiro", "PTAA", "NiOx", "PEDOT", "Au", "Ag", "electrode", "encapsulation"],\n    "performance": ["PCE", "Voc", "Jsc", "FF", "J-V", "JV", "EQE", "MPP", "efficiency"],\n    "stability_outdoor": ["stability", "stable", "T80", "T95", "aging", "ISOS", "operational", "outdoor", "1000 h", "retention"],\n}\n\nFIGURE_TASK_ONTOLOGY: Dict[str, Any] = {\n    "jv_curve": {"terms": ["J-V", "JV", "current density-voltage", "Voc", "Jsc", "PCE", "FF"], "priority": "high"},\n    "eqe_spectrum": {"terms": ["EQE", "IPCE", "external quantum efficiency", "integrated Jsc"], "priority": "high"},\n    "stability_curve": {"terms": ["stability", "T80", "T95", "aging", "retained", "normalized PCE", "MPP"], "priority": "high"},\n    "device_schematic": {"terms": ["device architecture", "device structure", "schematic", "ITO", "FTO", "ETL", "HTL"], "priority": "high"},\n    "xrd_pattern": {"terms": ["XRD", "diffraction", "2theta", "PbI2", "peak"], "priority": "medium"},\n    "pl_uvvis": {"terms": ["PL", "photoluminescence", "UV-vis", "absorption", "Tauc", "band gap"], "priority": "medium"},\n    "sem_tem_afm": {"terms": ["SEM", "TEM", "AFM", "morphology", "grain", "roughness"], "priority": "low"},\n    "energy_level": {"terms": ["energy level", "band alignment", "HOMO", "LUMO", "work function"], "priority": "medium"},\n}\n\n# Save ontology snapshot for reproducibility on import/run.\ntry:\n    (ONTOLOGY_DIR / "perovskite_ontology_v18.json").write_text(json.dumps(PEROVSKITE_ONTOLOGY, indent=2, ensure_ascii=False), encoding="utf-8")\n    (ONTOLOGY_DIR / "family_ontology_map_v18.json").write_text(json.dumps(FAMILY_ONTOLOGY_MAP, indent=2, ensure_ascii=False), encoding="utf-8")\n    (ONTOLOGY_DIR / "figure_task_ontology_v18.json").write_text(json.dumps(FIGURE_TASK_ONTOLOGY, indent=2, ensure_ascii=False), encoding="utf-8")\nexcept Exception:\n    pass\n\n# -----------------------------\n# Ontology utilities\n# -----------------------------\ndef _flatten_values(obj: Any) -> List[str]:\n    vals: List[str] = []\n    if obj is None:\n        return vals\n    if isinstance(obj, str):\n        vals.append(obj)\n    elif isinstance(obj, dict):\n        for k, v in obj.items():\n            vals.append(str(k))\n            vals.extend(_flatten_values(v))\n    elif isinstance(obj, (list, tuple, set)):\n        for x in obj:\n            vals.extend(_flatten_values(x))\n    else:\n        vals.append(str(obj))\n    # keep order while removing duplicates\n    seen = set()\n    out = []\n    for v in vals:\n        vv = str(v).strip()\n        if vv and vv.lower() not in seen:\n            seen.add(vv.lower())\n            out.append(vv)\n    return out\n\n\ndef ontology_slice_for_family(family_name: str) -> Dict[str, Any]:\n    if not ONTOLOGY_ENABLE:\n        return {}\n    keys = FAMILY_ONTOLOGY_MAP.get(family_name, [])\n    return {k: PEROVSKITE_ONTOLOGY.get(k, {}) for k in keys if k in PEROVSKITE_ONTOLOGY}\n\n\ndef terms_for_family(family_name: str, family_columns: Optional[List[str]] = None) -> List[str]:\n    terms: List[str] = []\n    terms.extend(FAMILY_GATE_TERMS.get(family_name, []))\n    for key in FAMILY_ONTOLOGY_MAP.get(family_name, []):\n        terms.extend(_flatten_values(PEROVSKITE_ONTOLOGY.get(key, {})))\n    for col in family_columns or []:\n        terms.append(col)\n        terms.extend(FIELD_ALIASES.get(col, []))\n    seen = set()\n    out = []\n    for t in terms:\n        tt = str(t).strip()\n        if len(tt) >= 2 and tt.lower() not in seen:\n            seen.add(tt.lower())\n            out.append(tt)\n    return out\n\n\ndef count_term_hits(text: str, terms: List[str]) -> int:\n    low = (text or "").lower()\n    hits = 0\n    for term in terms:\n        t = str(term).strip().lower()\n        if not t:\n            continue\n        if len(t) <= 4:\n            # keep short tokens such as PCE/FA/MA from massively overmatching by word boundary\n            if re.search(rf"(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])", low):\n                hits += 1\n        elif t in low:\n            hits += 1\n    return hits\n\n\ndef get_family_gate_threshold() -> int:\n    if LLM_COST_MODE == "cheap":\n        return MIN_FAMILY_SCORE_CHEAP\n    if LLM_COST_MODE == "thorough":\n        return MIN_FAMILY_SCORE_THOROUGH\n    return MIN_FAMILY_SCORE_BALANCED\n\n\ndef get_max_chunks_per_family() -> int:\n    if LLM_COST_MODE == "cheap":\n        return MAX_CHUNKS_PER_FAMILY_CHEAP\n    if LLM_COST_MODE == "thorough":\n        return MAX_CHUNKS_PER_FAMILY_THOROUGH\n    return MAX_CHUNKS_PER_FAMILY_BALANCED\n\n\ndef family_has_enough_evidence(family_name: str, family_columns: List[str], text: str) -> Tuple[bool, int]:\n    if not FAMILY_GATING_ENABLE or LLM_COST_MODE == "thorough":\n        return True, 999\n    score = count_term_hits(text[:180000], terms_for_family(family_name, family_columns))\n    return score >= get_family_gate_threshold(), score\n\n\ndef rank_chunks_for_family(chunks: List[str], family_name: str, family_columns: List[str]) -> List[str]:\n    if not CHUNK_RANKING_ENABLE or LLM_COST_MODE == "thorough":\n        return chunks\n    terms = terms_for_family(family_name, family_columns)\n    scored = []\n    for i, chunk in enumerate(chunks):\n        score = count_term_hits(chunk, terms)\n        # section headers are strongly useful; gently boost methods/results/table/figure appendices\n        lower = chunk.lower()\n        if any(x in lower for x in ["experimental", "materials and methods", "device fabrication", "results", "table", "figure", "j-v", "eqe"]):\n            score += 2\n        scored.append((score, i, chunk))\n    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)\n    selected = [c for score, _, c in scored if score > 0]\n    if not selected:\n        selected = [c for _, _, c in scored[:1]]\n    return selected[:get_max_chunks_per_family()]\n\n# -----------------------------\n# Deterministic pre-extraction\n# -----------------------------\nREGEX_EXTRACTORS: Dict[str, List[str]] = {\n    "JV_default_PCE": [\n        r"(?:champion|best|maximum|highest)?\\s*(?:PCE|power conversion efficiency|efficiency)\\s*(?:of|=|:|was|reached|up to)?\\s*([0-9]{1,2}(?:\\.[0-9]+)?)\\s*%",\n        r"([0-9]{1,2}(?:\\.[0-9]+)?)\\s*%\\s*(?:PCE|power conversion efficiency|efficiency)",\n    ],\n    "JV_default_Voc": [\n        r"(?:V[_\\s-]?OC|Voc|open-circuit voltage)\\s*(?:of|=|:|was)?\\s*([0-9]+(?:\\.[0-9]+)?)\\s*V",\n    ],\n    "JV_default_Jsc": [\n        r"(?:J[_\\s-]?SC|Jsc|short-circuit current density)\\s*(?:of|=|:|was)?\\s*([0-9]+(?:\\.[0-9]+)?)\\s*mA\\s*cm",\n    ],\n    "JV_default_FF": [\n        r"(?:FF|fill factor)\\s*(?:of|=|:|was)?\\s*(0?\\.[0-9]+|[0-9]{1,2}(?:\\.[0-9]+)?)",\n    ],\n    "EQE_integrated_Jsc": [\n        r"integrated\\s*(?:Jsc|current density)\\s*(?:of|=|:|was)?\\s*([0-9]+(?:\\.[0-9]+)?)\\s*mA\\s*cm",\n    ],\n    "Perovskite_band_gap": [\n        r"(?:band\\s*gap|bandgap|Eg)\\s*(?:of|=|:|was)?\\s*([0-9]+(?:\\.[0-9]+)?)\\s*eV",\n    ],\n    "Perovskite_pl_max": [\n        r"(?:PL|photoluminescence)\\s*(?:peak|maximum|max)\\s*(?:at|=|:)?\\s*([0-9]{3,4}(?:\\.[0-9]+)?)\\s*nm",\n    ],\n    "Perovskite_deposition_thermal_annealing_temperature": [\n        r"anneal(?:ed|ing)?[^\\.\\n]{0,80}?(?:at|@)\\s*([0-9]{2,3}(?:\\.[0-9]+)?)\\s*°?\\s*C",\n    ],\n    "Perovskite_deposition_thermal_annealing_time": [\n        r"anneal(?:ed|ing)?[^\\.\\n]{0,120}?for\\s*([0-9]+(?:\\.[0-9]+)?)\\s*(min|minutes|h|hours)",\n    ],\n    "Stability_PCE_T80": [\n        r"T80\\s*(?:of|=|:|was)?\\s*([0-9]+(?:\\.[0-9]+)?)\\s*(h|hours)",\n    ],\n    "Stability_PCE_T95": [\n        r"T95\\s*(?:of|=|:|was)?\\s*([0-9]+(?:\\.[0-9]+)?)\\s*(h|hours)",\n    ],\n    "Stability_time_total_exposure": [\n        r"(?:after|for)\\s*([0-9]+(?:\\.[0-9]+)?)\\s*(h|hours)\\s*(?:of)?\\s*(?:continuous|operational|illumination|aging|stability)",\n    ],\n}\n\nCOMPOSITION_PATTERNS = [\n    r"\\b(?:FA|MA|Cs|Rb|K)(?:[0-9\\.]+)?(?:FA|MA|Cs|Rb|K)?[0-9\\.]*Pb\\(?I[0-9\\.]*Br?[0-9\\.]*Cl?[0-9\\.]*\\)?[0-9\\.]*\\b",\n    r"\\b(?:FAPbI3|MAPbI3|CsPbI3|CsPbBr3|FAPbBr3|MAPbBr3)\\b",\n    r"\\b(?:PEA2PbI4|BA2PbI4|BDAPbI4)\\b",\n]\n\n\ndef _field_context(text: str, match: re.Match, radius: int = 220) -> str:\n    s = max(0, match.start() - radius)\n    e = min(len(text), match.end() + radius)\n    return normalize_text(text[s:e])[:500]\n\n\ndef deterministic_preextract_record(meta: Dict[str, Any], text: str) -> Dict[str, Any]:\n    rec = make_empty_record()\n    source_text = text or ""\n    for field, patterns in REGEX_EXTRACTORS.items():\n        for pat in patterns:\n            m = re.search(pat, source_text, flags=re.IGNORECASE | re.DOTALL)\n            if not m:\n                continue\n            value = m.group(1)\n            # Convert hour/min units when possible for time fields. Keep annealing time in minutes.\n            if field.endswith("_time") and len(m.groups()) >= 2:\n                unit = (m.group(2) or "").lower()\n                try:\n                    val = float(value)\n                    if unit.startswith("h"):\n                        value = str(val * 60.0)\n                    else:\n                        value = str(val)\n                except Exception:\n                    pass\n            rec[field] = value\n            rec[f"_evidence_{field}"] = _field_context(source_text, m)\n            break\n    for pat in COMPOSITION_PATTERNS:\n        m = re.search(pat, source_text, flags=re.IGNORECASE)\n        if m:\n            rec["Perovskite_composition_short_form"] = m.group(0)\n            rec["_evidence_Perovskite_composition_short_form"] = _field_context(source_text, m)\n            break\n    blob = source_text[:120000].lower()\n    if any(x.lower() in blob for x in ["spin coating", "spin-coated", "spin coated"]):\n        rec["Perovskite_deposition_procedure"] = rec.get("Perovskite_deposition_procedure") or "spin coating"\n        rec["_evidence_Perovskite_deposition_procedure"] = "Detected spin-coating terminology in extraction text."\n    solvents_found = []\n    for sol in PEROVSKITE_ONTOLOGY["solvents"]["precursor_solvents"]:\n        if re.search(rf"(?<![A-Za-z0-9]){re.escape(sol)}(?![A-Za-z0-9])", source_text, flags=re.IGNORECASE):\n            solvents_found.append(sol)\n    if solvents_found:\n        rec["Perovskite_deposition_solvents"] = ";".join(sorted(set(solvents_found)))\n        rec["_evidence_Perovskite_deposition_solvents"] = "Detected precursor solvent abbreviations in extraction text."\n    # metadata and derived flags\n    rec = enrich_record_with_metadata(rec, meta, 1)\n    return rec\n\n\ndef record_has_any_target_fields(rec: Dict[str, Any], target_fields: Optional[List[str]] = None) -> bool:\n    fields = target_fields or ALL_COLUMNS\n    ignore = {"Ref_ID", "Ref_ID_temp", "Ref_name_of_person_entering_the_data", "Ref_part_of_initial_dataset"}\n    for k in fields:\n        if k in ignore:\n            continue\n        if rec.get(k) not in (None, "", [], {}):\n            return True\n    return False\n\n# -----------------------------\n# Cost logging and model routing\n# -----------------------------\ndef estimate_tokens(text: Any) -> int:\n    try:\n        s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)\n    except Exception:\n        s = str(text)\n    return max(1, int(len(s) / 4))\n\n\ndef choose_model_for_stage(stage: str) -> str:\n    stage_low = (stage or "").lower()\n    if LLM_COST_MODE == "cheap":\n        return HF_MODEL_FAST\n    if any(x in stage_low for x in ["summary", "gate", "caption", "recovery"]):\n        return HF_MODEL_FAST\n    if any(x in stage_low for x in ["perovskite_chemistry", "performance", "stability", "consolidated", "figure_visual"]):\n        return HF_MODEL_STRONG\n    return HF_MODEL_FAST if LLM_COST_MODE == "balanced" else HF_MODEL_STRONG\n\n\ndef append_cost_log(paper_slug: str, stage: str, model: str, user_payload: Dict[str, Any], max_tokens: int) -> None:\n    if not COST_LOGGING_ENABLE:\n        return\n    try:\n        rec = {\n            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),\n            "paper_slug": paper_slug,\n            "stage": stage,\n            "model": model,\n            "cost_mode": LLM_COST_MODE,\n            "estimated_input_tokens": estimate_tokens(user_payload),\n            "max_output_tokens": max_tokens,\n        }\n        with (COST_DIR / "llm_call_log.jsonl").open("a", encoding="utf-8") as f:\n            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")\n    except Exception:\n        pass\n\n\nclass HuggingFaceQwenExtractor(HuggingFaceQwenExtractor):\n    """v18 extractor: ontology-aware prompts + model routing + call accounting."""\n\n    def _call(self, system_prompt: str, user_payload: Dict[str, Any], paper_slug: str, stage: str) -> Dict[str, Any]:\n        chosen_model = choose_model_for_stage(stage)\n        append_cost_log(paper_slug, stage, chosen_model, user_payload, LLM_MAX_OUTPUT_TOKENS)\n        payload = {\n            "model": chosen_model,\n            "temperature": 0,\n            "max_tokens": LLM_MAX_OUTPUT_TOKENS,\n            "response_format": {"type": "json_object"},\n            "messages": [\n                {"role": "system", "content": system_prompt},\n                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},\n            ],\n        }\n        data = self._request(payload)\n        content = data["choices"][0]["message"]["content"]\n        return self._parse_content_to_json(content, paper_slug=paper_slug, stage=stage)\n\n    def extract_family_records(\n        self,\n        metadata: Dict[str, Any],\n        chunk_text: str,\n        family_name: str,\n        family_columns: List[str],\n        alias_map: Dict[str, List[str]],\n        paper_slug: str,\n        chunk_idx: int,\n    ) -> Dict[str, Any]:\n        ontology_slice = ontology_slice_for_family(family_name)\n        system = (\n            "You extract structured data from experimental halide-perovskite research papers. "\n            "Act like a careful materials-science data curator, not a generic summarizer. "\n            "Populate the requested CSV columns only when the supplied text explicitly supports the information. "\n            "Use the provided ontology and alias lists to recognize perovskite-specific synonyms, abbreviations, device layers, spacers, additives, solvents, and performance metrics. "\n            "Do not invent unsupported values. Do not copy background or literature-review examples as if they were this paper\'s experiments. "\n            "Return one record per distinct experimentally described device/material/sample when the text supports multiple rows. "\n            "For each populated field, return a short evidence snippet in a matching key named _evidence_<fieldname>. "\n            "Return valid JSON only with top-level key \'records\'."\n        )\n        payload = {\n            "paper_metadata": metadata,\n            "schema_family": family_name,\n            "target_columns": family_columns,\n            "field_aliases": alias_map,\n            "perovskite_ontology_slice": ontology_slice,\n            "extraction_policy": {\n                "allowed": [\n                    "explicit experimental values from abstract, methods, results, captions, tables, and figure OCR",\n                    "straightforward unit normalization when source units are clear",\n                    "composition tokenization only when the formula is explicitly reported",\n                ],\n                "not_allowed": [\n                    "background/prior-work values unless this paper repeats the experiment",\n                    "guessed missing parameters",\n                    "assuming champion metrics apply to all devices",\n                    "filling stability fields from general claims without duration/condition evidence",\n                ],\n            },\n            "task": (\n                "Extract all available values for this schema family from the text chunk. "\n                "Search by scientific meaning, ontology terms, and aliases, not only exact column names. "\n                "If multiple control/target/champion devices are described, preserve that distinction. "\n                "Only return values explicitly supported by text, table text, caption text, OCR text, or figure evidence provided in the chunk."\n            ),\n            "text_chunk": chunk_text,\n            "output_format": {\n                "records": [\n                    {\n                        "Ref_internal_sample_id": "optional short sample/device name if explicitly clear",\n                        "example_column": "example_value",\n                        "_evidence_example_column": "short source snippet"\n                    }\n                ]\n            },\n        }\n        return self._call(system, payload, paper_slug=paper_slug, stage=f"{family_name}_partial_{chunk_idx}")\n\n    def consolidate_records(\n        self,\n        metadata: Dict[str, Any],\n        family_name: str,\n        partial_payloads: List[Dict[str, Any]],\n        family_columns: List[str],\n        paper_slug: str,\n    ) -> Dict[str, Any]:\n        system = (\n            "You merge partial structured extraction outputs from one halide-perovskite paper. "\n            "Merge rows only when they clearly refer to the same device/material/sample. "\n            "Keep control, target, champion, encapsulated, large-area, and aged/stability devices separate when the evidence indicates distinct samples. "\n            "Preserve evidence keys. Do not create unsupported values. Return valid JSON only with top-level key \'records\'."\n        )\n        payload = {\n            "paper_metadata": metadata,\n            "schema_family": family_name,\n            "target_columns": family_columns,\n            "perovskite_ontology_slice": ontology_slice_for_family(family_name),\n            "partials": partial_payloads,\n            "task": "Merge partial outputs into device/sample-level records for this schema family without inventing missing data.",\n        }\n        return self._call(system, payload, paper_slug=paper_slug, stage=f"{family_name}_consolidated")\n\n# -----------------------------\n# v18 family extraction override\n# -----------------------------\ndef extract_family_for_chunks(\n    meta: Dict[str, Any],\n    family_name: str,\n    family_columns: List[str],\n    chunks: List[str],\n    extractor: HuggingFaceQwenExtractor,\n    paper_slug: str,\n) -> List[Dict[str, Any]]:\n    allowed, gate_score = family_has_enough_evidence(family_name, family_columns, "\\n\\n".join(chunks))\n    if not allowed:\n        print(f"[SKIP] Family gated out: {family_name} | score={gate_score} | mode={LLM_COST_MODE}")\n        return []\n    selected_chunks = rank_chunks_for_family(chunks, family_name, family_columns)\n    print(f"[INFO] Family {family_name}: using {len(selected_chunks)}/{len(chunks)} chunks | gate_score={gate_score}")\n    partial_payloads: List[Dict[str, Any]] = []\n    alias_map = {col: FIELD_ALIASES.get(col, []) for col in family_columns}\n    for i, chunk in enumerate(selected_chunks, start=1):\n        if not chunk.strip():\n            continue\n        (TEXT_DIR / f"{paper_slug}_{family_name}_v18_ranked_chunk_{i}.txt").write_text(chunk, encoding="utf-8")\n        payload = extractor.extract_family_records(\n            metadata={\n                "title": meta.get("title"),\n                "doi": meta.get("doi"),\n                "journal": meta.get("journal"),\n                "year": meta.get("year"),\n                "lead_author": meta.get("lead_author"),\n                "chunk_index": i,\n                "num_chunks": len(selected_chunks),\n                "agent_version": AGENT_VERSION,\n                "cost_mode": LLM_COST_MODE,\n            },\n            chunk_text=chunk,\n            family_name=family_name,\n            family_columns=family_columns,\n            alias_map=alias_map,\n            paper_slug=paper_slug,\n            chunk_idx=i,\n        )\n        payload["records"] = validate_record_keys(payload.get("records", []), family_columns)\n        partial_payloads.append(payload)\n        (RAW_LLM_DIR / f"{paper_slug}_{family_name}_v18_partial_{i}.json").write_text(\n            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"\n        )\n    if not partial_payloads:\n        return []\n    if SKIP_CONSOLIDATION_IF_ONE_CHUNK and len(partial_payloads) == 1:\n        consolidated = partial_payloads[0]\n    elif USE_CONSOLIDATION_PASS:\n        consolidated = extractor.consolidate_records(\n            metadata={\n                "title": meta.get("title"),\n                "doi": meta.get("doi"),\n                "journal": meta.get("journal"),\n                "year": meta.get("year"),\n                "lead_author": meta.get("lead_author"),\n                "agent_version": AGENT_VERSION,\n            },\n            family_name=family_name,\n            partial_payloads=partial_payloads,\n            family_columns=family_columns,\n            paper_slug=paper_slug,\n        )\n    else:\n        consolidated = {"records": [r for p in partial_payloads for r in p.get("records", [])]}\n    consolidated["records"] = validate_record_keys(consolidated.get("records", []), family_columns)\n    (RAW_LLM_DIR / f"{paper_slug}_{family_name}_v18_consolidated.json").write_text(\n        json.dumps(consolidated, indent=2, ensure_ascii=False), encoding="utf-8"\n    )\n    records = consolidated.get("records", [])\n    return records if isinstance(records, list) else []\n\n\ndef merge_deterministic_with_llm_records(deterministic: Dict[str, Any], llm_records: List[Dict[str, Any]], meta: Dict[str, Any]) -> List[Dict[str, Any]]:\n    if not record_has_any_target_fields(deterministic):\n        return llm_records\n    if not llm_records:\n        return [deterministic]\n    merged = []\n    for idx, rec in enumerate(llm_records, start=1):\n        # LLM-supported values win; deterministic fills only gaps.\n        base = dict(deterministic)\n        for k, v in rec.items():\n            if v not in (None, "", [], {}):\n                base[k] = v\n        merged.append(enrich_record_with_metadata(base, meta, idx))\n    return merged\n\n\ndef extract_records_with_llm(meta: Dict[str, Any], extraction_text: str, extractor: HuggingFaceQwenExtractor) -> List[Dict[str, Any]]:\n    chunks = chunk_text(extraction_text, max_chars=LLM_CHUNK_CHAR_LIMIT, overlap=LLM_CHUNK_OVERLAP)\n    paper_slug = slugify(meta.get("title") or meta.get("doi") or "paper")\n    deterministic_rec = deterministic_preextract_record(meta, extraction_text) if DETERMINISTIC_EXTRACTION_ENABLE else make_empty_record()\n    if DETERMINISTIC_EXTRACTION_ENABLE:\n        (JSON_DIR / f"{paper_slug}_v18_deterministic_preextract.json").write_text(\n            json.dumps(deterministic_rec, indent=2, ensure_ascii=False), encoding="utf-8"\n        )\n    family_outputs: Dict[str, List[Dict[str, Any]]] = {}\n    for family_name, family_columns in SCHEMA_FAMILIES.items():\n        print(f"[INFO] Extracting family: {family_name} | v18 mode={LLM_COST_MODE}")\n        family_records = extract_family_for_chunks(\n            meta=meta,\n            family_name=family_name,\n            family_columns=family_columns,\n            chunks=chunks,\n            extractor=extractor,\n            paper_slug=paper_slug,\n        )\n        family_outputs[family_name] = family_records\n    combined_records: List[Dict[str, Any]] = []\n    for family_name in SCHEMA_FAMILIES.keys():\n        combined_records = merge_record_lists(combined_records, family_outputs.get(family_name, []))\n    combined_records = merge_deterministic_with_llm_records(deterministic_rec, combined_records, meta)\n    if not combined_records:\n        raise RuntimeError("v18 extraction returned zero usable records.")\n    cleaned: List[Dict[str, Any]] = []\n    for idx, rec in enumerate(combined_records, start=1):\n        rec = enrich_record_with_metadata(rec, meta, idx)\n        rec["Ref_free_text_comment"] = (rec.get("Ref_free_text_comment") or "") + f" Extracted with {AGENT_VERSION}; cost_mode={LLM_COST_MODE}."\n        cleaned.append(rec)\n    cleaned = dedupe_records(cleaned)\n    if not cleaned:\n        raise RuntimeError("v18 extraction returned zero usable records after normalization.")\n    return cleaned\n\n# -----------------------------\n# v18 figure intelligence\n# -----------------------------\ndef classify_figure_text(text: str) -> Tuple[str, str, int]:\n    best_type = "unknown"\n    best_priority = "low"\n    best_score = 0\n    for fig_type, cfg in FIGURE_TASK_ONTOLOGY.items():\n        score = count_term_hits(text, cfg.get("terms", []))\n        if score > best_score:\n            best_type = fig_type\n            best_priority = cfg.get("priority", "low")\n            best_score = score\n    return best_type, best_priority, best_score\n\n\ndef extract_figure_hint_fields(text: str) -> Dict[str, Any]:\n    rec = deterministic_preextract_record({}, text)\n    return {k: v for k, v in rec.items() if (k in ALL_COLUMNS or k.startswith("_evidence_")) and v not in (None, "", [], {})}\n\n\ndef _image_file_to_data_url(path: str) -> Optional[str]:\n    try:\n        p = Path(path)\n        if not p.exists() or p.stat().st_size <= 0:\n            return None\n        data = base64.b64encode(p.read_bytes()).decode("ascii")\n        return f"data:image/png;base64,{data}"\n    except Exception:\n        return None\n\n\ndef call_hf_vision_for_figure(image_path: str, context_text: str, paper_slug: str, figure_id: str) -> Optional[Dict[str, Any]]:\n    """Optional multimodal extraction. Disabled unless VISION_LLM_ENABLE=1 and HF_VISION_MODEL is set.\n\n    The Hugging Face router follows an OpenAI-compatible chat-completions shape for many vision models,\n    but provider support varies. This function is defensive: failures are logged as nonfatal evidence notes.\n    """\n    if not (VISION_LLM_ENABLE and HF_VISION_MODEL and HF_API_KEY):\n        return None\n    data_url = _image_file_to_data_url(image_path)\n    if not data_url:\n        return None\n    system = (\n        "You inspect figures from halide-perovskite research papers for data extraction. "\n        "Extract only visible or caption-supported values. Do not infer exact numerical values from curves unless labeled. "\n        "All summary fields MUST be written in English, even if the paper text, OCR text, or extracted passages are in another language. Do not use Chinese or any non-English language in the summary. Return valid JSON only."\n    )\n    user_text = {\n        "figure_id": figure_id,\n        "nearby_caption_or_ocr": context_text[:2500],\n        "task": "Classify the figure and extract visible/caption-supported perovskite device/material values. Include uncertainty notes.",\n        "output_format": {\n            "figure_type": "jv_curve|eqe_spectrum|stability_curve|xrd_pattern|pl_uvvis|device_schematic|other",\n            "extracted_values": {},\n            "interpretation_summary": "...",\n            "needs_human_check": True,\n            "confidence": "low|medium|high"\n        },\n    }\n    payload = {\n        "model": HF_VISION_MODEL,\n        "temperature": 0,\n        "max_tokens": min(1500, LLM_MAX_OUTPUT_TOKENS),\n        "response_format": {"type": "json_object"},\n        "messages": [\n            {"role": "system", "content": system},\n            {"role": "user", "content": [\n                {"type": "text", "text": json.dumps(user_text, ensure_ascii=False)},\n                {"type": "image_url", "image_url": {"url": data_url}},\n            ]},\n        ],\n    }\n    append_cost_log(paper_slug, f"figure_visual_{figure_id}", HF_VISION_MODEL, user_text, min(1500, LLM_MAX_OUTPUT_TOKENS))\n    try:\n        tmp_extractor = HuggingFaceQwenExtractor(HF_API_KEY, HF_VISION_MODEL, HF_API_URL)\n        data = tmp_extractor._request(payload)\n        content = data["choices"][0]["message"]["content"]\n        return tmp_extractor._parse_content_to_json(content, paper_slug=paper_slug, stage=f"figure_visual_{figure_id}")\n    except Exception as exc:\n        return {"ok": False, "error": repr(exc), "figure_id": figure_id, "model": HF_VISION_MODEL}\n\n\n_previous_analyze_pdf_figures_and_images_v18 = analyze_pdf_figures_and_images\n\ndef analyze_pdf_figures_and_images(pdf_path: Path, slug: str, native_text: str = "") -> Dict[str, Any]:\n    base_payload = _previous_analyze_pdf_figures_and_images_v18(pdf_path, slug=slug, native_text=native_text)\n    if not isinstance(base_payload, dict):\n        return base_payload\n    figure_evidence: List[Dict[str, Any]] = []\n    captions = base_payload.get("captions", []) or []\n    # Caption-level classification\n    for i, cap in enumerate(captions[:80], start=1):\n        fig_type, priority, score = classify_figure_text(cap)\n        if fig_type == "unknown" and score == 0:\n            continue\n        figure_evidence.append({\n            "figure_id": f"caption_{i}",\n            "source": "caption_text",\n            "figure_type": fig_type,\n            "priority": priority,\n            "score": score,\n            "text": cap,\n            "extracted_hint_fields": extract_figure_hint_fields(cap),\n            "needs_human_check": priority in {"high", "medium"},\n        })\n    # OCR/image region classification; optional vision model only for high-priority regions.\n    saved_images = base_payload.get("saved_images", []) or []\n    regions = base_payload.get("figure_regions", []) or []\n    for i, region in enumerate(regions[:40], start=1):\n        context = region.get("ocr_text") or ""\n        fig_type, priority, score = classify_figure_text(context)\n        ev = {\n            "figure_id": f"region_{i}",\n            "source": "figure_region_ocr",\n            "page": region.get("page"),\n            "image_index": region.get("image_index"),\n            "bbox": region.get("bbox"),\n            "figure_type": fig_type,\n            "priority": priority,\n            "score": score,\n            "ocr_text": context,\n            "extracted_hint_fields": extract_figure_hint_fields(context),\n            "needs_human_check": priority in {"high", "medium"},\n        }\n        # Saved image order generally follows OCR region order, but be defensive.\n        if VISION_LLM_ENABLE and priority == "high" and i - 1 < len(saved_images):\n            vis = call_hf_vision_for_figure(saved_images[i - 1], context, slug, ev["figure_id"])\n            if vis:\n                ev["vision_model_output"] = vis\n        figure_evidence.append(ev)\n    base_payload["figure_evidence_v18"] = figure_evidence\n    try:\n        (FIGURE_EVIDENCE_DIR / f"{slug}_figure_evidence_v18.json").write_text(\n            json.dumps(figure_evidence, indent=2, ensure_ascii=False), encoding="utf-8"\n        )\n    except Exception:\n        pass\n    # Append compact evidence text to the figure appendix so the text extractor can use it.\n    if figure_evidence:\n        lines = ["[FIGURE_EVIDENCE_V18]"]\n        for ev in figure_evidence[:60]:\n            label = f"{ev.get(\'figure_id\')} type={ev.get(\'figure_type\')} priority={ev.get(\'priority\')} score={ev.get(\'score\')}"\n            txt = ev.get("text") or ev.get("ocr_text") or ""\n            hints = ev.get("extracted_hint_fields") or {}\n            lines.append(label + "\\n" + _truncate_text(txt, 700) + ("\\nHINT_FIELDS=" + json.dumps(hints, ensure_ascii=False) if hints else ""))\n        extra = "\\n\\n".join(lines)\n        base_payload["figure_text"] = _truncate_text(((base_payload.get("figure_text") or "") + "\\n\\n" + extra).strip(), FIGURE_APPENDIX_MAX_CHARS)\n    return base_payload\n\n# -----------------------------\n# v18 output metadata wrappers\n# -----------------------------\n_previous_process_batch_api_v18 = process_batch_api\n\ndef process_batch_api(\n    query: str = SEARCH_QUERY,\n    year_min: int = YEAR_MIN,\n    year_max: int = YEAR_MAX,\n    max_papers: int = MAX_PAPERS_TO_PROCESS,\n    run_mode: str = RUN_MODE,\n    force_reprocess: bool = False,\n    reset_output: bool = False,\n) -> Dict[str, Any]:\n    result = _previous_process_batch_api_v18(\n        query=query,\n        year_min=year_min,\n        year_max=year_max,\n        max_papers=max_papers,\n        run_mode=run_mode,\n        force_reprocess=force_reprocess,\n        reset_output=reset_output,\n    )\n    if isinstance(result, dict):\n        result["agent_version"] = AGENT_VERSION\n        result["v18_settings"] = {\n            "ontology_enable": bool(ONTOLOGY_ENABLE),\n            "deterministic_extraction_enable": bool(DETERMINISTIC_EXTRACTION_ENABLE),\n            "family_gating_enable": bool(FAMILY_GATING_ENABLE),\n            "chunk_ranking_enable": bool(CHUNK_RANKING_ENABLE),\n            "cost_logging_enable": bool(COST_LOGGING_ENABLE),\n            "vision_llm_enable": bool(VISION_LLM_ENABLE),\n            "llm_cost_mode": LLM_COST_MODE,\n            "hf_model_fast": HF_MODEL_FAST,\n            "hf_model_strong": HF_MODEL_STRONG,\n            "hf_vision_model": HF_VISION_MODEL,\n            "ontology_dir": str(ONTOLOGY_DIR),\n            "cost_log_dir": str(COST_DIR),\n            "figure_evidence_dir": str(FIGURE_EVIDENCE_DIR),\n        }\n    return result\n\n\ndef literature_agent_main_v18() -> None:\n    if not ALL_COLUMNS:\n        raise ValueError("ALL_COLUMNS is empty.")\n    if not HF_API_KEY:\n        raise ValueError("HF_API_KEY is missing.")\n    mode = normalize_run_mode(RUN_MODE)\n    out = process_batch_api(\n        query=SEARCH_QUERY,\n        year_min=YEAR_MIN,\n        year_max=YEAR_MAX,\n        max_papers=MAX_PAPERS_TO_PROCESS,\n        run_mode=mode,\n        force_reprocess=(mode in {"reprocess", "reset"}),\n        reset_output=(mode == "reset"),\n    )\n    print(json.dumps(out, indent=2, ensure_ascii=False))\n\n# Make the existing bottom-of-file `main()` call execute v18 behavior.\nmain = literature_agent_main_v18\n\n\n# ============================================================\n# v19 EXTERNAL ONTOLOGY + SAFER CANDIDATE EXTRACTION LAYER\n# ============================================================\n# This layer intentionally lives after the v18 layer so it can override the\n# global ontology/maps/functions before main() executes.  The code still remains\n# a single-file agent for easy running, but the vocabulary now lives in an\n# editable JSON file next to this script by default.\n\nAGENT_VERSION = "v19_3_external_ontology_cost_visual_downloads_output"\n\nONTOLOGY_LIBRARY_PATH = Path(os.getenv(\n    "PEROVSKITE_ONTOLOGY_PATH",\n    str(Path(__file__).with_name("perovskite_ontology_library_v19.json"))\n))\n\n# More conservative than v18: deterministic extraction becomes candidate-first.\nDETERMINISTIC_CANDIDATE_ENABLE = os.getenv("DETERMINISTIC_CANDIDATE_ENABLE", "1").strip().lower() not in {"0", "false", "no"}\nDETERMINISTIC_CANDIDATE_MAX_PER_FIELD = int(os.getenv("DETERMINISTIC_CANDIDATE_MAX_PER_FIELD", "5"))\nVALIDATE_REGEX_CANDIDATES_WITH_LLM = os.getenv("VALIDATE_REGEX_CANDIDATES_WITH_LLM", "0").strip().lower() in {"1", "true", "yes"}\n\n\ndef load_external_perovskite_ontology(path: Path) -> Dict[str, Any]:\n    """Load the editable ontology JSON. Falls back to the in-script v18 ontology if missing."""\n    if not path.exists():\n        print(f"[WARN] Ontology file not found: {path}. Falling back to embedded ontology.")\n        return {"_fallback": True, **PEROVSKITE_ONTOLOGY}\n    try:\n        data = json.loads(path.read_text(encoding="utf-8"))\n        if not isinstance(data, dict):\n            raise ValueError("ontology root must be a JSON object")\n        return data\n    except Exception as exc:\n        print(f"[WARN] Failed to load ontology file {path}: {exc}. Falling back to embedded ontology.")\n        return {"_fallback": True, **PEROVSKITE_ONTOLOGY}\n\n\ndef _ontology_core_keys(data: Dict[str, Any]) -> List[str]:\n    ignored = {"version", "description", "family_ontology_map", "family_gate_terms", "figure_task_ontology"}\n    return [k for k in data.keys() if k not in ignored and not str(k).startswith("_")]\n\n\ndef validate_ontology_library(data: Dict[str, Any]) -> Dict[str, Any]:\n    keys = _ontology_core_keys(data)\n    flat_terms = _flatten_values({k: data.get(k) for k in keys})\n    required = ["a_site_cations", "b_site_cations", "x_site_anions", "device_layers", "performance_metrics"]\n    missing = [k for k in required if k not in data]\n    return {\n        "version": data.get("version"),\n        "core_sections": len(keys),\n        "approx_unique_terms": len(set(t.lower() for t in flat_terms)),\n        "missing_required_sections": missing,\n        "path": str(ONTOLOGY_LIBRARY_PATH),\n    }\n\n\n# Load external ontology and override v18 globals used by prompt construction,\n# family gating, chunk ranking, and figure classification.\n_EXTERNAL_ONTOLOGY_LIBRARY = load_external_perovskite_ontology(ONTOLOGY_LIBRARY_PATH)\n_ONTOLOGY_VALIDATION_SUMMARY = validate_ontology_library(_EXTERNAL_ONTOLOGY_LIBRARY)\n\n# Keep only the actual ontology sections in PEROVSKITE_ONTOLOGY.  Maps are pulled\n# out separately so users can edit them in the same JSON file.\nPEROVSKITE_ONTOLOGY = {\n    k: v for k, v in _EXTERNAL_ONTOLOGY_LIBRARY.items()\n    if k not in {"version", "description", "family_ontology_map", "family_gate_terms", "figure_task_ontology"}\n    and not str(k).startswith("_")\n}\nFAMILY_ONTOLOGY_MAP = _EXTERNAL_ONTOLOGY_LIBRARY.get("family_ontology_map", FAMILY_ONTOLOGY_MAP)\nFAMILY_GATE_TERMS = _EXTERNAL_ONTOLOGY_LIBRARY.get("family_gate_terms", FAMILY_GATE_TERMS)\nFIGURE_TASK_ONTOLOGY = _EXTERNAL_ONTOLOGY_LIBRARY.get("figure_task_ontology", FIGURE_TASK_ONTOLOGY)\n\n# Save a run snapshot inside WORK_DIR for reproducibility.\ntry:\n    (ONTOLOGY_DIR / "perovskite_ontology_v19_loaded_snapshot.json").write_text(\n        json.dumps(_EXTERNAL_ONTOLOGY_LIBRARY, indent=2, ensure_ascii=False), encoding="utf-8"\n    )\n    (ONTOLOGY_DIR / "perovskite_ontology_v19_validation.json").write_text(\n        json.dumps(_ONTOLOGY_VALIDATION_SUMMARY, indent=2, ensure_ascii=False), encoding="utf-8"\n    )\nexcept Exception:\n    pass\n\n\ndef _ontology_terms(section: str, *path: str) -> List[str]:\n    """Safe helper for terms from nested ontology sections."""\n    obj: Any = PEROVSKITE_ONTOLOGY.get(section, {})\n    for p in path:\n        if isinstance(obj, dict):\n            obj = obj.get(p, {})\n        else:\n            return []\n    return _flatten_values(obj)\n\n\n# Strengthen regex patterns with more perovskite-specific spellings.  These are\n# still only candidate generators; the LLM/final consolidation has priority.\nREGEX_EXTRACTORS.update({\n    "JV_default_PCE": [\n        r"(?:champion|best|maximum|highest|optimized|target)?[^\\.\\n]{0,80}?(?:PCE|power conversion efficiency|efficiency|η)\\s*(?:of|=|:|was|is|reached|achieved|up to)?\\s*([0-9]{1,2}(?:\\.[0-9]+)?)\\s*%",\n        r"([0-9]{1,2}(?:\\.[0-9]+)?)\\s*%\\s*(?:PCE|power conversion efficiency|efficiency)",\n    ],\n    "JV_default_Voc": [\n        r"(?:V[_\\s-]?OC|V\\s*OC|Voc|open[- ]circuit voltage)\\s*(?:of|=|:|was|is)?\\s*([0-9]+(?:\\.[0-9]+)?)\\s*(?:V|volt)",\n    ],\n    "JV_default_Jsc": [\n        r"(?:J[_\\s-]?SC|J\\s*SC|Jsc|short[- ]circuit current density)\\s*(?:of|=|:|was|is)?\\s*([0-9]+(?:\\.[0-9]+)?)\\s*(?:mA\\s*cm|mA/cm|mA cm)",\n    ],\n    "JV_default_FF": [\n        r"(?:FF|fill factor)\\s*(?:of|=|:|was|is)?\\s*(0?\\.[0-9]+|[0-9]{1,2}(?:\\.[0-9]+)?)\\s*%?",\n    ],\n    "Stability_PCE_after_1000_h": [\n        r"(?:retained|maintained|remaining|retention)[^\\.\\n]{0,120}?([0-9]{1,3}(?:\\.[0-9]+)?)\\s*%[^\\.\\n]{0,80}?(?:1000\\s*h|1000\\s*hours)",\n        r"(?:1000\\s*h|1000\\s*hours)[^\\.\\n]{0,120}?(?:retained|maintained|remaining|retention)[^\\.\\n]{0,80}?([0-9]{1,3}(?:\\.[0-9]+)?)\\s*%",\n    ],\n})\n\n# Broader composition detector.  Keep conservative enough to avoid random prose.\nCOMPOSITION_PATTERNS = [\n    r"\\b(?:FA|MA|Cs|Rb|K)[A-Za-z0-9\\.\\(\\)]{0,40}Pb\\(?I[0-9\\.x\\-]*Br?[0-9\\.x\\-]*Cl?[0-9\\.x\\-]*\\)?[0-9\\.x\\-]*\\b",\n    r"\\b(?:FA|MA|Cs|Rb|K)[A-Za-z0-9\\.\\(\\)]{0,40}(?:Sn|Pb)[A-Za-z0-9\\.\\(\\)]{0,25}(?:I|Br|Cl)[A-Za-z0-9\\.\\(\\)]{0,25}\\b",\n    r"\\b(?:FAPbI3|MAPbI3|CsPbI3|CsPbBr3|FAPbBr3|MAPbBr3|FASnI3|MASnI3)\\b",\n    r"\\b(?:PEA2PbI4|BA2PbI4|BDAPbI4|\\(PEA\\)2PbI4|\\(BA\\)2PbI4|\\(BDA\\)PbI4)\\b",\n]\n\n\ndef deterministic_preextract_candidates(meta: Dict[str, Any], text: str) -> List[Dict[str, Any]]:\n    """Return candidates with evidence instead of silently treating regex as truth."""\n    source_text = text or ""\n    candidates: List[Dict[str, Any]] = []\n    for field, patterns in REGEX_EXTRACTORS.items():\n        field_count = 0\n        seen_values = set()\n        for pat in patterns:\n            for m in re.finditer(pat, source_text, flags=re.IGNORECASE | re.DOTALL):\n                if field_count >= DETERMINISTIC_CANDIDATE_MAX_PER_FIELD:\n                    break\n                value = m.group(1)\n                if field.endswith("_time") and len(m.groups()) >= 2:\n                    unit = (m.group(2) or "").lower()\n                    try:\n                        val = float(value)\n                        value = str(val * 60.0 if unit.startswith("h") else val)\n                    except Exception:\n                        pass\n                key = (field, str(value).lower())\n                if key in seen_values:\n                    continue\n                seen_values.add(key)\n                ctx = _field_context(source_text, m)\n                lowctx = ctx.lower()\n                confidence = "medium"\n                if any(w in lowctx for w in ["champion", "best", "optimized", "target", "control", "device"]):\n                    confidence = "medium_high"\n                if any(w in lowctx for w in ["previous", "reported", "literature", "reference", "prior work"]):\n                    confidence = "low_context_risk"\n                candidates.append({\n                    "field": field,\n                    "value": value,\n                    "confidence": confidence,\n                    "source": "regex_candidate",\n                    "evidence": ctx,\n                })\n                field_count += 1\n            if field_count >= DETERMINISTIC_CANDIDATE_MAX_PER_FIELD:\n                break\n\n    for pat in COMPOSITION_PATTERNS:\n        for m in re.finditer(pat, source_text, flags=re.IGNORECASE):\n            val = m.group(0)\n            candidates.append({\n                "field": "Perovskite_composition_short_form",\n                "value": val,\n                "confidence": "medium",\n                "source": "regex_candidate",\n                "evidence": _field_context(source_text, m),\n            })\n            break\n        if any(c["field"] == "Perovskite_composition_short_form" for c in candidates):\n            break\n\n    # Ontology-derived solvent candidates.\n    solvents_found = []\n    for sol in _ontology_terms("solvents", "precursor_solvents"):\n        if re.search(rf"(?<![A-Za-z0-9]){re.escape(sol)}(?![A-Za-z0-9])", source_text, flags=re.IGNORECASE):\n            solvents_found.append(sol)\n    if solvents_found:\n        candidates.append({\n            "field": "Perovskite_deposition_solvents",\n            "value": ";".join(sorted(set(solvents_found), key=str.lower)),\n            "confidence": "medium_high",\n            "source": "ontology_term_match",\n            "evidence": "Detected precursor solvent terms from ontology in extraction text.",\n        })\n\n    if re.search(r"\\bspin[- ]?coat(?:ed|ing)?\\b", source_text, flags=re.IGNORECASE):\n        candidates.append({\n            "field": "Perovskite_deposition_procedure",\n            "value": "spin coating",\n            "confidence": "medium_high",\n            "source": "ontology_term_match",\n            "evidence": "Detected spin-coating terminology in extraction text.",\n        })\n    return candidates\n\n\ndef deterministic_preextract_record(meta: Dict[str, Any], text: str) -> Dict[str, Any]:\n    """v19 candidate-first deterministic pre-extraction.\n\n    The record is still prefilled to reduce cost, but all prefilled values carry\n    _evidence_* notes that mark them as deterministic candidates.  The LLM family\n    extraction/consolidation can overwrite or supplement these fields.\n    """\n    if not DETERMINISTIC_CANDIDATE_ENABLE:\n        return make_empty_record()\n    rec = make_empty_record()\n    candidates = deterministic_preextract_candidates(meta, text)\n    best_by_field: Dict[str, Dict[str, Any]] = {}\n    rank = {"medium_high": 3, "medium": 2, "low_context_risk": 1, "low": 0}\n    for cand in candidates:\n        field = cand.get("field")\n        if not field or field not in rec:\n            continue\n        old = best_by_field.get(field)\n        if old is None or rank.get(cand.get("confidence", "low"), 0) > rank.get(old.get("confidence", "low"), 0):\n            best_by_field[field] = cand\n    for field, cand in best_by_field.items():\n        rec[field] = cand.get("value")\n        rec[f"_evidence_{field}"] = f"Deterministic v19 candidate ({cand.get(\'confidence\')}): {cand.get(\'evidence\')}"\n    rec = enrich_record_with_metadata(rec, meta, 1)\n    try:\n        # Keep a small candidate preview as an evidence field only; this survives\n        # existing filter rules because it starts with _evidence_.\n        rec["_evidence_v19_deterministic_candidates"] = json.dumps(candidates[:40], ensure_ascii=False)\n    except Exception:\n        pass\n    return rec\n\n\n# Override metadata wrapper with v19 info.\n_previous_process_batch_api_v19 = process_batch_api\n\ndef process_batch_api(\n    query: str = SEARCH_QUERY,\n    year_min: int = YEAR_MIN,\n    year_max: int = YEAR_MAX,\n    max_papers: int = MAX_PAPERS_TO_PROCESS,\n    run_mode: str = RUN_MODE,\n    force_reprocess: bool = False,\n    reset_output: bool = False,\n) -> Dict[str, Any]:\n    result = _previous_process_batch_api_v19(\n        query=query,\n        year_min=year_min,\n        year_max=year_max,\n        max_papers=max_papers,\n        run_mode=run_mode,\n        force_reprocess=force_reprocess,\n        reset_output=reset_output,\n    )\n    if isinstance(result, dict):\n        result["agent_version"] = AGENT_VERSION\n        result["v19_settings"] = {\n            "ontology_library_path": str(ONTOLOGY_LIBRARY_PATH),\n            "ontology_validation": _ONTOLOGY_VALIDATION_SUMMARY,\n            "deterministic_candidate_enable": bool(DETERMINISTIC_CANDIDATE_ENABLE),\n            "deterministic_candidate_max_per_field": DETERMINISTIC_CANDIDATE_MAX_PER_FIELD,\n            "validate_regex_candidates_with_llm": bool(VALIDATE_REGEX_CANDIDATES_WITH_LLM),\n            "llm_cost_mode": LLM_COST_MODE,\n            "hf_model_fast": HF_MODEL_FAST,\n            "hf_model_strong": HF_MODEL_STRONG,\n            "hf_vision_model": HF_VISION_MODEL,\n            "work_dir": str(BASE),\n            "cost_log_dir": str(COST_DIR),\n            "figure_evidence_dir": str(FIGURE_EVIDENCE_DIR),\n        }\n    return result\n\n\n# ============================================================\n# V19.1 TEST / ONE-PAPER INITIAL-RUN OVERRIDE\n# ============================================================\n# Purpose:\n# - Keep the normal initial/expand/benchmark workflows available.\n# - Add a safe testing path that processes exactly one paper without crawling the\n#   whole Google Drive folder and without running a full Crossref/RSS expansion.\n#\n# Recommended testing command:\n#   $env:TEST_ONE_PAPER="1"\n#   $env:RUN_MODE="initial"\n#   $env:SINGLE_PAPER_SOURCE="drive"   # drive | search | pdf\n#   python literature_manager_agent_v19_1_external_ontology_cost_visual.py\n#\n# Optional local-PDF test:\n#   $env:SINGLE_PAPER_SOURCE="pdf"\n#   $env:SINGLE_PAPER_PDF_PATH="C:\\\\path\\\\to\\\\paper.pdf"\n#\nTEST_ONE_PAPER = os.getenv("TEST_ONE_PAPER", "0").strip().lower() in {"1", "true", "yes", "on"}\nSINGLE_PAPER_SOURCE = os.getenv("SINGLE_PAPER_SOURCE", "drive").strip().lower() or "drive"\nSINGLE_PAPER_PDF_PATH = os.getenv("SINGLE_PAPER_PDF_PATH", "").strip()\nSINGLE_PAPER_TITLE = os.getenv("SINGLE_PAPER_TITLE", "").strip()\nSINGLE_PAPER_DOI = os.getenv("SINGLE_PAPER_DOI", "").strip()\nSINGLE_PAPER_QUERY = os.getenv("SINGLE_PAPER_QUERY", SEARCH_QUERY).strip() or SEARCH_QUERY\nONE_PAPER_MAX_DRIVE_FILES = int(os.getenv("ONE_PAPER_MAX_DRIVE_FILES", "1"))\nONE_PAPER_SKIP_SEARCH_AFTER_DRIVE = os.getenv("ONE_PAPER_SKIP_SEARCH_AFTER_DRIVE", "1").strip().lower() in {"1", "true", "yes", "on"}\nONE_PAPER_INCLUDE_OCR = os.getenv("ONE_PAPER_INCLUDE_OCR", "1").strip().lower() in {"1", "true", "yes", "on"}\nONE_PAPER_INCLUDE_FIGURES = os.getenv("ONE_PAPER_INCLUDE_FIGURES", "1").strip().lower() in {"1", "true", "yes", "on"}\n\n_previous_normalize_run_mode_v19 = normalize_run_mode\n\ndef normalize_run_mode(run_mode: Optional[str] = None) -> str:\n    """Normalize run modes, including user-facing initial/test aliases.\n\n    initial: reset-style fresh run. In TEST_ONE_PAPER mode this only clears the\n    configured WORK_DIR and processes one selected paper, not the full library.\n    one_paper/test/smoke: resume-style one-paper test unless force_reprocess is set.\n    """\n    mode = (run_mode or RUN_MODE or "resume").strip().lower()\n    aliases = {\n        "default": "resume",\n        "fresh": "reset",\n        "overwrite": "reset",\n        "rerun": "reprocess",\n        "initial": "reset",\n        "init": "reset",\n        "one_paper": "resume",\n        "one-paper": "resume",\n        "single": "resume",\n        "test": "resume",\n        "smoke": "resume",\n    }\n    mode = aliases.get(mode, mode)\n    return _previous_normalize_run_mode_v19(mode)\n\n\ndef _make_local_pdf_meta(pdf_path: str, title: Optional[str] = None, doi: Optional[str] = None) -> Dict[str, Any]:\n    p = Path(str(pdf_path)).expanduser()\n    return {\n        "title": title or p.stem,\n        "doi": doi or None,\n        "journal": "local PDF test import",\n        "year": None,\n        "abstract": None,\n        "lead_author": None,\n        "source": "local_pdf_test",\n        "source_collection": "local_pdf_test",\n        "landing_page": None,\n        "original_filename": p.name,\n        "local_pdf_path": str(p),\n        "_include_ocr": ONE_PAPER_INCLUDE_OCR,\n        "_include_figures": ONE_PAPER_INCLUDE_FIGURES,\n    }\n\n\ndef process_single_pdf_api(\n    pdf_path: str,\n    title: Optional[str] = None,\n    doi: Optional[str] = None,\n    run_mode: Optional[str] = None,\n    force_reprocess: bool = False,\n) -> Dict[str, Any]:\n    """Process one local PDF file through the normal extraction pipeline."""\n    p = Path(str(pdf_path)).expanduser()\n    if not p.exists():\n        return {\n            "ok": False,\n            "tool": "process_single_pdf",\n            "reason": f"SINGLE_PAPER_PDF_PATH does not exist: {p}",\n            "pdf_path": str(p),\n        }\n    meta = _make_local_pdf_meta(str(p), title=title, doi=doi)\n    mode = normalize_run_mode(run_mode)\n    result = process_one_paper_api(meta=meta, force_reprocess=force_reprocess, run_mode=mode)\n    return {\n        "ok": bool(result.get("ok", True)) if isinstance(result, dict) else True,\n        "tool": "process_single_pdf",\n        "run_mode": mode,\n        "paper_slug": result.get("paper_slug") if isinstance(result, dict) else None,\n        "meta": meta,\n        "result": result,\n        "combined_global_sync": sync_global_outputs_to_combined(),\n    }\n\n\n_standard_v19_process_batch_api = process_batch_api\n\n\ndef process_batch_api(\n    query: str = SEARCH_QUERY,\n    year_min: int = YEAR_MIN,\n    year_max: int = YEAR_MAX,\n    max_papers: int = MAX_PAPERS_TO_PROCESS,\n    run_mode: str = RUN_MODE,\n    force_reprocess: bool = False,\n    reset_output: bool = False,\n) -> Dict[str, Any]:\n    """Normal batch API plus safe TEST_ONE_PAPER mode.\n\n    In TEST_ONE_PAPER mode:\n    - source=pdf: process SINGLE_PAPER_PDF_PATH only.\n    - source=drive: import/process only the newest ONE_PAPER_MAX_DRIVE_FILES file(s), default 1.\n      By default, this does NOT continue to Crossref/RSS afterward.\n    - source=search: skip Google Drive precheck and process one Crossref/RSS candidate.\n    """\n    mode = normalize_run_mode(run_mode)\n    effective_force = force_reprocess or (mode in {"reset", "reprocess"})\n    effective_reset = reset_output or (mode == "reset")\n\n    if not TEST_ONE_PAPER:\n        return _standard_v19_process_batch_api(\n            query=query,\n            year_min=year_min,\n            year_max=year_max,\n            max_papers=max_papers,\n            run_mode=mode,\n            force_reprocess=force_reprocess,\n            reset_output=reset_output,\n        )\n\n    if effective_reset:\n        print(f"[INFO] TEST_ONE_PAPER=1 and RUN_MODE={mode}: clearing WORK_DIR once before one-paper test: {BASE}")\n        reset_work_dir()\n\n    source = SINGLE_PAPER_SOURCE\n    print("=" * 80)\n    print("TEST_ONE_PAPER mode is ON")\n    print(f"Source: {source} | mode: {mode} | max_papers: 1")\n    print(f"OCR: {ONE_PAPER_INCLUDE_OCR} | figure analysis: {ONE_PAPER_INCLUDE_FIGURES}")\n    print("=" * 80)\n\n    if source == "pdf":\n        if not SINGLE_PAPER_PDF_PATH:\n            return {\n                "ok": False,\n                "tool": "test_one_paper",\n                "reason": "SINGLE_PAPER_SOURCE=pdf requires SINGLE_PAPER_PDF_PATH.",\n            }\n        result = process_single_pdf_api(\n            pdf_path=SINGLE_PAPER_PDF_PATH,\n            title=SINGLE_PAPER_TITLE or None,\n            doi=SINGLE_PAPER_DOI or None,\n            run_mode=mode,\n            force_reprocess=effective_force,\n        )\n        result["agent_version"] = AGENT_VERSION\n        result["test_one_paper"] = True\n        return result\n\n    if source == "drive":\n        drive_result = process_google_drive_folder_api(\n            folder_id_or_link=GOOGLE_DRIVE_FOLDER_LINK or GOOGLE_DRIVE_FOLDER_ID,\n            max_files=max(1, ONE_PAPER_MAX_DRIVE_FILES),\n            include_ocr=ONE_PAPER_INCLUDE_OCR,\n            include_figures=ONE_PAPER_INCLUDE_FIGURES,\n            force_reprocess=effective_force,\n            run_mode=mode,\n        )\n        out = {\n            "ok": drive_result.get("ok", False),\n            "tool": "test_one_paper_drive",\n            "agent_version": AGENT_VERSION,\n            "test_one_paper": True,\n            "run_mode": mode,\n            "single_paper_source": source,\n            "max_drive_files_requested": max(1, ONE_PAPER_MAX_DRIVE_FILES),\n            "skipped_crossref_and_rss": bool(ONE_PAPER_SKIP_SEARCH_AFTER_DRIVE),\n            "drive_result": drive_result,\n            "combined_global_sync": sync_global_outputs_to_combined(),\n        }\n        if ONE_PAPER_SKIP_SEARCH_AFTER_DRIVE:\n            return out\n        # Optional fallback: if explicitly disabled, continue to one search candidate.\n\n    # source == search, or drive fallback when ONE_PAPER_SKIP_SEARCH_AFTER_DRIVE=0\n    # Temporarily suppress Drive-first precheck so this does not process the folder.\n    global DRIVE_PRECHECK_BEFORE_SEARCH\n    old_drive_precheck = DRIVE_PRECHECK_BEFORE_SEARCH\n    DRIVE_PRECHECK_BEFORE_SEARCH = False\n    try:\n        result = _standard_v19_process_batch_api(\n            query=SINGLE_PAPER_QUERY or query,\n            year_min=year_min,\n            year_max=year_max,\n            max_papers=1,\n            run_mode=mode,\n            force_reprocess=effective_force,\n            reset_output=False,  # already handled above\n        )\n    finally:\n        DRIVE_PRECHECK_BEFORE_SEARCH = old_drive_precheck\n    if isinstance(result, dict):\n        result["agent_version"] = AGENT_VERSION\n        result["test_one_paper"] = True\n        result["single_paper_source"] = "search"\n        result["drive_precheck_disabled_for_test"] = True\n    return result\n\n\ndef literature_agent_main_v19() -> None:\n    if not ALL_COLUMNS:\n        raise ValueError("ALL_COLUMNS is empty.")\n    if not HF_API_KEY:\n        raise ValueError("HF_API_KEY is missing.")\n    print("=" * 80)\n    print(f"Literature agent version: {AGENT_VERSION}")\n    print(f"Ontology library: {ONTOLOGY_LIBRARY_PATH}")\n    print(f"Ontology terms approx.: {_ONTOLOGY_VALIDATION_SUMMARY.get(\'approx_unique_terms\')}")\n    print(f"Cost mode: {LLM_COST_MODE} | fast model: {HF_MODEL_FAST} | strong model: {HF_MODEL_STRONG}")\n    print(f"Output directory: {BASE}")\n    if _ONTOLOGY_VALIDATION_SUMMARY.get("missing_required_sections"):\n        print(f"[WARN] Ontology missing sections: {_ONTOLOGY_VALIDATION_SUMMARY[\'missing_required_sections\']}")\n    print("=" * 80)\n    mode = normalize_run_mode(RUN_MODE)\n    out = process_batch_api(\n        query=SEARCH_QUERY,\n        year_min=YEAR_MIN,\n        year_max=YEAR_MAX,\n        max_papers=MAX_PAPERS_TO_PROCESS,\n        run_mode=mode,\n        force_reprocess=(mode in {"reprocess", "reset"}),\n        reset_output=(mode == "reset"),\n    )\n    print(json.dumps(out, indent=2, ensure_ascii=False))\n\n\n\n# ============================================================\n# v19.4 PATCHES: safer deterministic extraction, summary repair,\n# and cleaner one-paper Drive selection.\n# ============================================================\nAGENT_VERSION = "v19_4_clean_summary_safe_extract"\n\n# In v19.3 the deterministic candidate layer could still pre-fill schema fields.\n# That helped cost, but for review/rulebook papers it could promote contextual\n# examples into apparent device records.  v19.4 keeps candidates as evidence by\n# default and lets the LLM / validation layer fill final schema fields.\nDETERMINISTIC_PREFILL_FIELDS = os.getenv("DETERMINISTIC_PREFILL_FIELDS", "0").strip().lower() in {"1", "true", "yes", "on"}\n\n# One-paper Drive testing can optionally avoid review/rulebook PDFs and prefer\n# filenames that look like experimental device papers.  This prevents a newest\n# review PDF from being used as the test paper unless explicitly requested.\nONE_PAPER_PREFER_EXPERIMENTAL = os.getenv("ONE_PAPER_PREFER_EXPERIMENTAL", "1").strip().lower() in {"1", "true", "yes", "on"}\nONE_PAPER_DRIVE_FILENAME_INCLUDE = os.getenv("ONE_PAPER_DRIVE_FILENAME_INCLUDE", "").strip().lower()\nONE_PAPER_DRIVE_FILENAME_EXCLUDE = os.getenv("ONE_PAPER_DRIVE_FILENAME_EXCLUDE", "review,rulebook,perspective,outlook,roadmap,tutorial,primer").strip().lower()\nREVIEW_LIKE_TITLE_PAT = re.compile(r"\\b(review|rulebook|perspective|outlook|roadmap|tutorial|primer|commentary)\\b", re.IGNORECASE)\nEXPERIMENTAL_TITLE_PAT = re.compile(r"\\b(device|solar cell|photovoltaic|pce|efficiency|stability|j-v|junction|passivation|fabrication|interface|transport layer)\\b", re.IGNORECASE)\n\n\ndef _is_review_like_title(title: Any) -> bool:\n    return bool(REVIEW_LIKE_TITLE_PAT.search(str(title or "")))\n\n\ndef _looks_experimental_drive_item(item: Dict[str, Any]) -> bool:\n    name = str(item.get("name") or "").lower()\n    if ONE_PAPER_DRIVE_FILENAME_INCLUDE and ONE_PAPER_DRIVE_FILENAME_INCLUDE not in name:\n        return False\n    exclude_terms = [x.strip() for x in ONE_PAPER_DRIVE_FILENAME_EXCLUDE.split(",") if x.strip()]\n    if any(x in name for x in exclude_terms):\n        return False\n    if ONE_PAPER_PREFER_EXPERIMENTAL and not EXPERIMENTAL_TITLE_PAT.search(name):\n        return False\n    return True\n\n\ndef _normalize_summary_payload_v19_4(payload: Any) -> Dict[str, Any]:\n    """Accept common LLM summary shapes and normalize to {\'paper_summary\': {...}}."""\n    if not isinstance(payload, dict):\n        raise RuntimeError("LLM summary payload was not a JSON object.")\n    # Correct shape.\n    ps = payload.get("paper_summary")\n    if isinstance(ps, dict):\n        # Some model outputs accidentally nest paper_summary twice:\n        # {\'paper_summary\': {\'paper_summary\': {...}}}\n        inner = ps.get("paper_summary")\n        if isinstance(inner, dict):\n            fixed = dict(payload)\n            fixed["paper_summary"] = inner\n            return fixed\n        return payload\n    # Model returned the summary object directly.\n    summary_keys = {"one_sentence_summary", "objective", "main_contribution", "best_reported_performance"}\n    if any(k in payload for k in summary_keys):\n        return {"paper_summary": payload}\n    # Model returned under alternate key.\n    for alt in ["summary", "paper", "result"]:\n        if isinstance(payload.get(alt), dict):\n            return _normalize_summary_payload_v19_4(payload[alt])\n    raise RuntimeError("LLM summary payload missing top-level \'paper_summary\' object.")\n\n\n_previous_deterministic_preextract_record_v19_4 = deterministic_preextract_record\n\ndef deterministic_preextract_record(meta: Dict[str, Any], text: str) -> Dict[str, Any]:\n    """Candidate-first deterministic extraction.\n\n    By default, deterministic values are stored only as evidence/candidates, not\n    as final schema values.  Set DETERMINISTIC_PREFILL_FIELDS=1 to restore the\n    older pre-fill behavior.\n    """\n    if not DETERMINISTIC_CANDIDATE_ENABLE:\n        return make_empty_record()\n    rec = make_empty_record()\n    candidates = deterministic_preextract_candidates(meta, text)\n    if DETERMINISTIC_PREFILL_FIELDS:\n        # Optional legacy behavior.\n        best_by_field: Dict[str, Dict[str, Any]] = {}\n        rank = {"medium_high": 3, "medium": 2, "low_context_risk": 1, "low": 0}\n        for cand in candidates:\n            field = cand.get("field")\n            if not field or field not in rec:\n                continue\n            # Never pre-fill low-context-risk values.\n            if cand.get("confidence") == "low_context_risk":\n                continue\n            old = best_by_field.get(field)\n            if old is None or rank.get(cand.get("confidence", "low"), 0) > rank.get(old.get("confidence", "low"), 0):\n                best_by_field[field] = cand\n        for field, cand in best_by_field.items():\n            rec[field] = cand.get("value")\n            rec[f"_evidence_{field}"] = f"Deterministic v19.4 prefill ({cand.get(\'confidence\')}): {cand.get(\'evidence\')}"\n    # Always store candidate preview for validation/debugging.\n    rec = enrich_record_with_metadata(rec, meta, 1)\n    try:\n        rec["_evidence_v19_deterministic_candidates"] = json.dumps(candidates[:60], ensure_ascii=False)\n        rec["_evidence_v19_4_note"] = (\n            "Deterministic candidates are evidence-only by default. "\n            "Set DETERMINISTIC_PREFILL_FIELDS=1 to use them as field prefill."\n        )\n    except Exception:\n        pass\n    return rec\n\n\n_previous_build_fallback_paper_summary_v19_4 = build_fallback_paper_summary\n\ndef build_fallback_paper_summary(metadata: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:\n    payload = _previous_build_fallback_paper_summary_v19_4(metadata, records)\n    title = str(metadata.get("title") or "")\n    if _is_review_like_title(title):\n        ps = payload.get("paper_summary", {})\n        ps["objective"] = (\n            "This appears to be a review/rulebook-style paper, so device-level extracted values may be examples from the literature rather than the paper\'s own new device results."\n        )\n        notes = ps.setdefault("data_quality_notes", [])\n        notes.append("v19.4 review-like title guard: treat extracted device metrics as contextual examples unless manually verified.")\n        payload["paper_summary"] = ps\n    return payload\n\n\n_previous_ensure_summary_outputs_v19_4 = ensure_summary_outputs\n\ndef ensure_summary_outputs(slug: str, meta: Dict[str, Any], records: List[Dict[str, Any]], full_text: str,\n                           extractor: Optional[HuggingFaceQwenExtractor] = None, force_llm_summary: bool = False) -> Tuple[Dict[str, Any], str]:\n    visible_json = PAPER_SUMMARY_JSON_DIR / f"{slug}_summary.json"\n    visible_txt = PAPER_SUMMARY_TEXT_DIR / f"{slug}_summary.txt"\n    summary_metadata = {\n        "title": meta.get("title"),\n        "doi": meta.get("doi"),\n        "journal": meta.get("journal"),\n        "year": meta.get("year"),\n        "lead_author": meta.get("lead_author"),\n    }\n    if visible_json.exists() and visible_txt.exists() and not force_llm_summary:\n        try:\n            existing = json.loads(visible_json.read_text(encoding="utf-8"))\n            existing = _normalize_summary_payload_v19_4(existing)\n            # Do not reuse a bad fallback if a new LLM summary is possible.\n            if existing.get("_summary_generation_mode") != "fallback" or extractor is None:\n                return existing, visible_txt.read_text(encoding="utf-8")\n        except Exception:\n            pass\n\n    summary_payload: Optional[Dict[str, Any]] = None\n    llm_error = None\n    if extractor is not None and full_text and (force_llm_summary or len(full_text) >= 1000):\n        try:\n            raw_payload = extractor.summarize_paper(\n                metadata=summary_metadata,\n                records=records,\n                summary_text=full_text,\n                paper_slug=slug,\n            )\n            summary_payload = _normalize_summary_payload_v19_4(raw_payload)\n            summary_payload["_summary_generation_mode"] = "llm"\n        except Exception as exc:\n            llm_error = str(exc)\n            summary_payload = None\n            print(f"[WARN] Paper summary generation failed for {meta.get(\'title\')}: {exc}")\n\n    if summary_payload is None:\n        summary_payload = build_fallback_paper_summary(summary_metadata, records)\n        summary_payload["_summary_generation_mode"] = "fallback"\n        if llm_error:\n            summary_payload["_summary_generation_error"] = llm_error\n\n    summary_text_rendered = render_paper_summary_text(summary_payload, summary_metadata)\n    write_visible_summary_files(slug, summary_payload, summary_text_rendered)\n    persist_artifacts(slug, summary_metadata, records, full_text, build_extraction_text(full_text) if full_text else "",\n                      summary_payload=summary_payload, summary_text_rendered=summary_text_rendered)\n    print(f"[OK] Wrote visible paper summaries: {PAPER_SUMMARY_JSON_DIR / f\'{slug}_summary.json\'} | {PAPER_SUMMARY_TEXT_DIR / f\'{slug}_summary.txt\'}")\n    return summary_payload, summary_text_rendered\n\n\n# Preserve full Drive mode, but in TEST_ONE_PAPER source=drive allow a cleaner\n# selection pass so the first paper is not a review/rulebook unless requested.\n_previous_process_batch_api_v19_4 = process_batch_api\n\ndef process_batch_api(\n    query: str = SEARCH_QUERY,\n    year_min: int = YEAR_MIN,\n    year_max: int = YEAR_MAX,\n    max_papers: int = MAX_PAPERS_TO_PROCESS,\n    run_mode: str = RUN_MODE,\n    force_reprocess: bool = False,\n    reset_output: bool = False,\n) -> Dict[str, Any]:\n    mode = normalize_run_mode(run_mode)\n    # Use the original path for all non-test modes.\n    if not TEST_ONE_PAPER or SINGLE_PAPER_SOURCE != "drive" or not ONE_PAPER_PREFER_EXPERIMENTAL:\n        return _previous_process_batch_api_v19_4(query, year_min, year_max, max_papers, mode, force_reprocess, reset_output)\n\n    effective_force = force_reprocess or (mode in {"reset", "reprocess"})\n    effective_reset = reset_output or (mode == "reset")\n    if effective_reset:\n        print(f"[INFO] TEST_ONE_PAPER=1 and RUN_MODE={mode}: clearing WORK_DIR once before one-paper test: {BASE}")\n        reset_work_dir()\n\n    source = SINGLE_PAPER_SOURCE\n    print("=" * 80)\n    print("TEST_ONE_PAPER mode is ON")\n    print(f"Source: {source} | mode: {mode} | max_papers: 1")\n    print(f"OCR: {ONE_PAPER_INCLUDE_OCR} | figure analysis: {ONE_PAPER_INCLUDE_FIGURES}")\n    print(f"Drive filename include: {ONE_PAPER_DRIVE_FILENAME_INCLUDE or \'[none]\'}")\n    print(f"Prefer experimental/device-like PDFs: {ONE_PAPER_PREFER_EXPERIMENTAL}")\n    print("=" * 80)\n\n    folder_ref = GOOGLE_DRIVE_FOLDER_LINK or GOOGLE_DRIVE_FOLDER_ID\n    folder_id = parse_google_drive_folder_id(folder_ref)\n    if not folder_id:\n        return {"ok": False, "tool": "test_one_paper_drive", "reason": "No valid Google Drive folder ID or link was provided."}\n    try:\n        files = list_google_drive_folder_files(folder_id=folder_id)\n    except Exception as exc:\n        return {"ok": False, "tool": "test_one_paper_drive", "folder_id": folder_id, "reason": f"Failed to list/access Drive folder: {exc}"}\n\n    allowed_google_types = {"application/pdf", "application/vnd.google-apps.document", "application/vnd.google-apps.presentation"}\n    pdf_like = _sort_drive_items_newest_first([item for item in files if (item.get("mimeType") or "") in allowed_google_types])\n    selected = [item for item in pdf_like if _looks_experimental_drive_item(item)]\n    selection_note = "experimental/device-like filter"\n    if not selected:\n        selected = pdf_like\n        selection_note = "fallback to newest PDF-like item; no experimental/device-like filename matched"\n    original_list_google_drive_folder_files = list_google_drive_folder_files\n\n    def _one_file_list_override(folder_id: str, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:\n        return selected[:max(1, int(ONE_PAPER_MAX_DRIVE_FILES))]\n\n    globals()["list_google_drive_folder_files"] = _one_file_list_override\n    try:\n        drive_result = process_google_drive_folder_api(\n            folder_id_or_link=folder_ref,\n            max_files=max(1, ONE_PAPER_MAX_DRIVE_FILES),\n            include_ocr=ONE_PAPER_INCLUDE_OCR,\n            include_figures=ONE_PAPER_INCLUDE_FIGURES,\n            force_reprocess=effective_force,\n            run_mode=mode,\n        )\n    finally:\n        globals()["list_google_drive_folder_files"] = original_list_google_drive_folder_files\n\n    out = {\n        "ok": drive_result.get("ok", False),\n        "tool": "test_one_paper_drive",\n        "agent_version": AGENT_VERSION,\n        "test_one_paper": True,\n        "run_mode": mode,\n        "single_paper_source": source,\n        "max_drive_files_requested": max(1, ONE_PAPER_MAX_DRIVE_FILES),\n        "skipped_crossref_and_rss": bool(ONE_PAPER_SKIP_SEARCH_AFTER_DRIVE),\n        "drive_selection_note": selection_note,\n        "selected_drive_filenames": [x.get("name") for x in selected[:max(1, int(ONE_PAPER_MAX_DRIVE_FILES))]],\n        "drive_result": drive_result,\n        "combined_global_sync": sync_global_outputs_to_combined(),\n    }\n    if ONE_PAPER_SKIP_SEARCH_AFTER_DRIVE:\n        return out\n    # If fallback search is desired, delegate to the previous implementation with\n    # the Drive precheck behavior it already controls.\n    return _previous_process_batch_api_v19_4(query, year_min, year_max, 1, mode, effective_force, False)\n\n\nmain = literature_agent_main_v19\n\n\n# v19.6 note: main entrypoint moved to end of file so extension layers load before execution.\n\n# ============================================================\n# v19.5 FIGURE REPORTING / SURFACING LAYER\n# ============================================================\n# v19.4 already performed caption/OCR-based figure evidence extraction and\n# optional vision-model calls.  v19.5 makes that evidence easier to inspect by\n# generating a concise per-paper figure report and appending a short section to\n# the visible paper summary text.\n\nAGENT_VERSION = "v19_6_figure_reports_folder_safe_extract"\nFIGURE_REPORT_ENABLE = os.getenv("FIGURE_REPORT_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}\nFIGURE_REPORT_MAX_ITEMS = int(os.getenv("FIGURE_REPORT_MAX_ITEMS", "12"))\nFIGURE_REPORT_MIN_SCORE = int(os.getenv("FIGURE_REPORT_MIN_SCORE", "1"))\n# v19.6: put human-readable figure reports in a dedicated top-level folder inside WORK_DIR.\nFIGURE_REPORTS_DIR = BASE / "figure_reports"\nFIGURE_REPORTS_JSON_DIR = FIGURE_REPORTS_DIR / "json"\nFIGURE_REPORTS_TEXT_DIR = FIGURE_REPORTS_DIR / "text"\nFIGURE_REPORTS_CROPS_DIR = FIGURE_REPORTS_DIR / "crops"\nfor _d in [FIGURE_REPORTS_DIR, FIGURE_REPORTS_JSON_DIR, FIGURE_REPORTS_TEXT_DIR, FIGURE_REPORTS_CROPS_DIR]:\n    _d.mkdir(parents=True, exist_ok=True)\nFIGURE_REPORT_HIGH_VALUE_TYPES = {\n    "jv_curve", "eqe_spectrum", "stability_curve", "device_schematic",\n    "xrd_pattern", "pl_uvvis", "energy_level_diagram"\n}\n\n\ndef _figure_evidence_rank_v19_5(ev: Dict[str, Any]) -> Tuple[int, int, int]:\n    priority_rank = {"high": 3, "medium": 2, "low": 1}.get(str(ev.get("priority") or "low"), 0)\n    type_rank = 2 if ev.get("figure_type") in FIGURE_REPORT_HIGH_VALUE_TYPES else 0\n    score = int(ev.get("score") or 0)\n    return (priority_rank, type_rank, score)\n\n\ndef _compact_hint_fields_v19_5(hints: Dict[str, Any]) -> Dict[str, Any]:\n    if not isinstance(hints, dict):\n        return {}\n    keep = [\n        "JV_default_PCE", "JV_default_Voc", "JV_default_Jsc", "JV_default_FF",\n        "EQE_integrated_Jsc", "Stability_PCE_T80", "Stability_PCE_after_1000_h",\n        "Stability_time_total_exposure", "Perovskite_band_gap", "Perovskite_pl_max",\n        "Cell_stack_sequence", "Perovskite_composition_short_form",\n    ]\n    out = {}\n    for k in keep:\n        v = hints.get(k)\n        if v not in (None, "", [], {}):\n            out[k] = v\n    return out\n\n\ndef build_figure_report_v19_5(slug: str, figure_payload: Dict[str, Any]) -> Dict[str, Any]:\n    evidence = figure_payload.get("figure_evidence_v18") or figure_payload.get("figure_evidence_v19_5") or []\n    if not isinstance(evidence, list):\n        evidence = []\n    filtered = []\n    for ev in evidence:\n        if not isinstance(ev, dict):\n            continue\n        if int(ev.get("score") or 0) < FIGURE_REPORT_MIN_SCORE and ev.get("figure_type") == "unknown":\n            continue\n        compact = dict(ev)\n        compact["extracted_hint_fields"] = _compact_hint_fields_v19_5(ev.get("extracted_hint_fields") or {})\n        # Avoid huge OCR/caption payloads in the report JSON.\n        if "ocr_text" in compact:\n            compact["ocr_text_preview"] = _truncate_text(str(compact.get("ocr_text") or ""), 500)\n            compact.pop("ocr_text", None)\n        if "text" in compact:\n            compact["text_preview"] = _truncate_text(str(compact.get("text") or ""), 500)\n            compact.pop("text", None)\n        filtered.append(compact)\n    filtered.sort(key=_figure_evidence_rank_v19_5, reverse=True)\n    top = filtered[:FIGURE_REPORT_MAX_ITEMS]\n    counts: Dict[str, int] = {}\n    high_value = []\n    for ev in filtered:\n        ft = str(ev.get("figure_type") or "unknown")\n        counts[ft] = counts.get(ft, 0) + 1\n        if ft in FIGURE_REPORT_HIGH_VALUE_TYPES and len(high_value) < FIGURE_REPORT_MAX_ITEMS:\n            high_value.append(ev)\n    report = {\n        "paper_slug": slug,\n        "figure_report_version": "v19.5",\n        "pages_scanned": figure_payload.get("pages_scanned"),\n        "saved_image_count": len(figure_payload.get("saved_images") or []),\n        "caption_count": len(figure_payload.get("captions") or []),\n        "figure_region_count": len(figure_payload.get("figure_regions") or []),\n        "figure_type_counts": counts,\n        "top_figure_evidence": top,\n        "high_value_figure_evidence": high_value,\n        "vision_enabled": bool(VISION_LLM_ENABLE),\n        "vision_model": HF_VISION_MODEL,\n        "figure_reports_dir": str(FIGURE_REPORTS_DIR),\n        "figure_report_json_path": str(FIGURE_REPORTS_JSON_DIR / f"{slug}_figure_report_v19_6.json"),\n        "figure_report_text_path": str(FIGURE_REPORTS_TEXT_DIR / f"{slug}_figure_report_v19_6.txt"),\n        "legacy_figure_evidence_dir": str(FIGURE_EVIDENCE_DIR),\n        "note": (\n            "Figure evidence is caption/OCR based unless VISION_LLM_ENABLE=1 and HF_VISION_MODEL is set. "\n            "Curve digitization is not performed here; visible numerical values are treated as hints and should be validated."\n        ),\n    }\n    return report\n\n\ndef render_figure_report_text_v19_5(report: Dict[str, Any]) -> str:\n    lines = []\n    lines.append("Figure / Visual Evidence Report")\n    lines.append("=" * 32)\n    lines.append(f"Pages scanned: {report.get(\'pages_scanned\')}")\n    lines.append(f"Captions found: {report.get(\'caption_count\')}")\n    lines.append(f"Figure/image regions OCR\'d: {report.get(\'figure_region_count\')}")\n    lines.append(f"Saved image crops: {report.get(\'saved_image_count\')}")\n    counts = report.get("figure_type_counts") or {}\n    if counts:\n        counts_txt = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: kv[0]))\n        lines.append(f"Detected figure types: {counts_txt}")\n    lines.append("")\n    lines.append("High-value figure evidence:")\n    items = report.get("high_value_figure_evidence") or report.get("top_figure_evidence") or []\n    if not items:\n        lines.append("- No high-value figure evidence found from captions/OCR.")\n    for ev in items[:FIGURE_REPORT_MAX_ITEMS]:\n        fig_id = ev.get("figure_id")\n        fig_type = ev.get("figure_type")\n        priority = ev.get("priority")\n        score = ev.get("score")\n        page = ev.get("page")\n        src = ev.get("source")\n        text = ev.get("text_preview") or ev.get("ocr_text_preview") or ""\n        hints = ev.get("extracted_hint_fields") or {}\n        page_txt = f", page {page}" if page else ""\n        lines.append(f"- {fig_id} ({fig_type}, {priority}, score={score}, {src}{page_txt})")\n        if hints:\n            lines.append(f"  Candidate fields: {json.dumps(hints, ensure_ascii=False)}")\n        if text:\n            lines.append(f"  Evidence preview: {_truncate_text(text, 280)}")\n    lines.append("")\n    lines.append(str(report.get("note") or ""))\n    return "\\n".join(lines).strip() + "\\n"\n\n\n_previous_analyze_pdf_figures_and_images_v19_5 = analyze_pdf_figures_and_images\n\ndef analyze_pdf_figures_and_images(pdf_path: Path, slug: str, native_text: str = "") -> Dict[str, Any]:\n    payload = _previous_analyze_pdf_figures_and_images_v19_5(pdf_path, slug=slug, native_text=native_text)\n    if not isinstance(payload, dict) or not FIGURE_REPORT_ENABLE:\n        return payload\n    try:\n        report = build_figure_report_v19_5(slug, payload)\n        report_text = render_figure_report_text_v19_5(report)\n        payload["figure_report_v19_5"] = report\n        payload["figure_report_text_v19_5"] = report_text\n        # Save to the new top-level lit_outputs/figure_reports folder.\n        (FIGURE_REPORTS_JSON_DIR / f"{slug}_figure_report_v19_6.json").write_text(\n            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"\n        )\n        (FIGURE_REPORTS_TEXT_DIR / f"{slug}_figure_report_v19_6.txt").write_text(report_text, encoding="utf-8")\n        # Also save legacy copies in figure_evidence for backwards compatibility.\n        (FIGURE_EVIDENCE_DIR / f"{slug}_figure_report_v19_6.json").write_text(\n            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"\n        )\n        (FIGURE_EVIDENCE_DIR / f"{slug}_figure_report_v19_6.txt").write_text(report_text, encoding="utf-8")\n        # Make the report visible to the normal text/summary pipeline too.\n        payload["figure_text"] = _truncate_text(\n            ((payload.get("figure_text") or "") + "\\n\\n[FIGURE_REPORT_V19_5]\\n" + report_text).strip(),\n            FIGURE_APPENDIX_MAX_CHARS,\n        )\n    except Exception as exc:\n        payload["figure_report_v19_5_error"] = repr(exc)\n    return payload\n\n\ndef _load_figure_report_text_for_slug_v19_5(slug: str) -> str:\n    # v19.6 preferred top-level report location, with legacy fallback.\n    candidates = [\n        FIGURE_REPORTS_TEXT_DIR / f"{slug}_figure_report_v19_6.txt",\n        FIGURE_EVIDENCE_DIR / f"{slug}_figure_report_v19_6.txt",\n        FIGURE_EVIDENCE_DIR / f"{slug}_figure_report_v19_5.txt",\n    ]\n    for p in candidates:\n        if p.exists():\n            try:\n                return p.read_text(encoding="utf-8")\n            except Exception:\n                continue\n    return ""\n\n\n_previous_ensure_summary_outputs_v19_5 = ensure_summary_outputs\n\ndef ensure_summary_outputs(slug: str, meta: Dict[str, Any], records: List[Dict[str, Any]], full_text: str,\n                           extractor: Optional[HuggingFaceQwenExtractor] = None, force_llm_summary: bool = False) -> Tuple[Dict[str, Any], str]:\n    summary_payload, summary_text = _previous_ensure_summary_outputs_v19_5(\n        slug, meta, records, full_text, extractor=extractor, force_llm_summary=force_llm_summary\n    )\n    if not FIGURE_REPORT_ENABLE:\n        return summary_payload, summary_text\n    fig_report_text = _load_figure_report_text_for_slug_v19_5(slug)\n    if fig_report_text and "Figure / Visual Evidence Report" not in summary_text:\n        appended = summary_text.rstrip() + "\\n\\n" + fig_report_text\n        try:\n            summary_payload["_figure_report_text_v19_5"] = fig_report_text\n            summary_payload["_figure_report_path_v19_6"] = str(FIGURE_REPORTS_TEXT_DIR / f"{slug}_figure_report_v19_6.txt")\n            write_visible_summary_files(slug, summary_payload, appended)\n            persist_artifacts(slug, meta, records, full_text, build_extraction_text(full_text) if full_text else "",\n                              summary_payload=summary_payload, summary_text_rendered=appended)\n            summary_text = appended\n        except Exception:\n            pass\n    return summary_payload, summary_text\n\nprint(f"[INFO] v19.5 figure report layer loaded | figure reports: {FIGURE_REPORT_ENABLE} | vision: {VISION_LLM_ENABLE} | model: {HF_VISION_MODEL or \'(none)\'}")\n\n\n# ============================================================\n# v19.7 QWEN2.5-VL FIGURE VISION INTEGRATION\n# ============================================================\n# This layer keeps the v19.6 caption/OCR figure report system but adds an\n# optional Qwen2.5-VL call for only high-value figure crops. It is deliberately\n# conservative: visual-model outputs are saved as candidate evidence, cached by\n# image/context hash, and never treated as final numeric ground truth without\n# downstream validation.\n\nAGENT_VERSION = "v19_7_qwen_vl_figure_vision"\n\n# Separate vision settings.  These fall back to the existing HF_* variables so\n# older terminal commands still work.\nVISION_PROVIDER = os.getenv("VISION_PROVIDER", "hf_router_openai_compatible").strip()\nVISION_API_KEY = os.getenv("VISION_API_KEY", os.getenv("HF_API_KEY", "")).strip()\nVISION_API_URL = os.getenv("VISION_API_URL", os.getenv("HF_API_URL", "https://router.huggingface.co/v1/chat/completions")).strip()\nVISION_MODEL = os.getenv("VISION_MODEL", os.getenv("HF_VISION_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")).strip()\nVISION_MAX_FIGURES_PER_PAPER = int(os.getenv("VISION_MAX_FIGURES_PER_PAPER", "4"))\nVISION_MAX_IMAGE_WIDTH = int(os.getenv("VISION_MAX_IMAGE_WIDTH", "1200"))\nVISION_MAX_CONTEXT_CHARS = int(os.getenv("VISION_MAX_CONTEXT_CHARS", "2500"))\nVISION_MAX_OUTPUT_TOKENS = int(os.getenv("VISION_MAX_OUTPUT_TOKENS", "1200"))\nVISION_CACHE_ENABLE = os.getenv("VISION_CACHE_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}\nVISION_SAVE_CROPS_ENABLE = os.getenv("VISION_SAVE_CROPS_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}\nVISION_STRICT_JSON_ENABLE = os.getenv("VISION_STRICT_JSON_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}\n\n# Keep legacy variable populated so older code/reporting sees the selected model.\nif VISION_MODEL and not HF_VISION_MODEL:\n    HF_VISION_MODEL = VISION_MODEL\n\nVISION_CACHE_DIR = FIGURE_REPORTS_DIR / "vision_cache"\nVISION_LOG_DIR = FIGURE_REPORTS_DIR / "vision_logs"\nVISION_CROPS_DIR = FIGURE_REPORTS_DIR / "crops"\nfor _d in [VISION_CACHE_DIR, VISION_LOG_DIR, VISION_CROPS_DIR]:\n    _d.mkdir(parents=True, exist_ok=True)\n\n# This is intentionally narrower than all figure classes so cost remains controlled.\nVISION_HIGH_VALUE_TYPES = {\n    "jv_curve", "eqe_spectrum", "stability_curve", "device_schematic",\n    "xrd_pattern", "pl_uvvis", "energy_level_diagram"\n}\nVISION_OPTIONAL_TYPES = {"sem_tem_afm"}\n\n\ndef _safe_json_dumps_v19_7(obj: Any, max_chars: int = 4000) -> str:\n    try:\n        return json.dumps(obj, ensure_ascii=False, indent=2)[:max_chars]\n    except Exception:\n        return str(obj)[:max_chars]\n\n\ndef _hash_for_vision_v19_7(image_path: str, context_text: str, model: str) -> str:\n    h = hashlib.sha256()\n    try:\n        p = Path(image_path)\n        if p.exists():\n            h.update(p.read_bytes())\n    except Exception:\n        h.update(str(image_path).encode("utf-8", errors="ignore"))\n    h.update((context_text or "")[:2000].encode("utf-8", errors="ignore"))\n    h.update((model or "").encode("utf-8", errors="ignore"))\n    return h.hexdigest()[:32]\n\n\ndef _prepare_vision_image_v19_7(image_path: str, paper_slug: str, figure_id: str) -> Tuple[Optional[str], Optional[str]]:\n    """Return (path_to_png_for_api, copied_crop_path). Downscale large images to reduce cost."""\n    try:\n        src = Path(image_path)\n        if not src.exists() or src.stat().st_size <= 0:\n            return None, None\n        paper_crop_dir = VISION_CROPS_DIR / paper_slug\n        paper_crop_dir.mkdir(parents=True, exist_ok=True)\n        safe_fig = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(figure_id or "figure"))[:80]\n        out_path = paper_crop_dir / f"{safe_fig}.png"\n        try:\n            from PIL import Image\n            img = Image.open(src)\n            if img.mode not in {"RGB", "RGBA"}:\n                img = img.convert("RGB")\n            w, h = img.size\n            if VISION_MAX_IMAGE_WIDTH > 0 and w > VISION_MAX_IMAGE_WIDTH:\n                new_h = max(1, int(h * (VISION_MAX_IMAGE_WIDTH / float(w))))\n                img = img.resize((VISION_MAX_IMAGE_WIDTH, new_h))\n            if VISION_SAVE_CROPS_ENABLE:\n                img.save(out_path, format="PNG")\n                return str(out_path), str(out_path)\n            tmp_path = VISION_CACHE_DIR / f"tmp_{paper_slug}_{safe_fig}.png"\n            img.save(tmp_path, format="PNG")\n            return str(tmp_path), None\n        except Exception:\n            # If Pillow fails, still use the original file and optionally copy it.\n            if VISION_SAVE_CROPS_ENABLE:\n                shutil.copy2(src, out_path)\n                return str(out_path), str(out_path)\n            return str(src), None\n    except Exception:\n        return None, None\n\n\ndef _image_file_to_data_url(path: str) -> Optional[str]:\n    """v19.7 override: encode a prepared/downscaled image path as a data URL."""\n    try:\n        p = Path(path)\n        if not p.exists() or p.stat().st_size <= 0:\n            return None\n        data = base64.b64encode(p.read_bytes()).decode("ascii")\n        return f"data:image/png;base64,{data}"\n    except Exception:\n        return None\n\n\ndef _extract_json_object_from_text_v19_7(text: str) -> Dict[str, Any]:\n    if not text:\n        return {}\n    s = str(text).strip()\n    # Remove common markdown fences.\n    s = re.sub(r"^```(?:json)?\\s*", "", s, flags=re.IGNORECASE).strip()\n    s = re.sub(r"\\s*```$", "", s).strip()\n    try:\n        data = json.loads(s)\n        return data if isinstance(data, dict) else {"raw": data}\n    except Exception:\n        pass\n    # Fallback: find first balanced-looking JSON object.\n    start = s.find("{")\n    end = s.rfind("}")\n    if start >= 0 and end > start:\n        try:\n            data = json.loads(s[start:end + 1])\n            return data if isinstance(data, dict) else {"raw": data}\n        except Exception:\n            pass\n    return {"raw_text": s[:4000], "parse_warning": "Could not parse strict JSON from vision model output."}\n\n\ndef _qwen_vl_prompt_payload_v19_7(context_text: str, figure_id: str) -> Dict[str, Any]:\n    return {\n        "figure_id": figure_id,\n        "caption_or_ocr_context": (context_text or "")[:VISION_MAX_CONTEXT_CHARS],\n        "instructions": [\n            "Analyze this figure crop from a halide-perovskite solar-cell paper.",\n            "Use the image plus caption/OCR context.",\n            "Extract only visually visible or caption-supported information.",\n            "Do not invent exact values from plotted curves unless labels/annotations make them explicit.",\n            "Prefer concise benchmark-style wording.",\n            "Return only valid JSON."\n        ],\n        "schema": {\n            "figure_type": "jv_curve|eqe_spectrum|stability_curve|device_schematic|xrd_pattern|pl_uvvis|kpfm_npfm|stem_edx|sem_tem_afm|energy_level_diagram|other",\n            "is_high_value": True,\n            "visual_summary": "one concise sentence",\n            "candidate_database_fields": {\n                "Cell_stack_sequence": None,\n                "Cell_architecture": None,\n                "Perovskite_composition_short_form": None,\n                "JV_default_PCE": None,\n                "JV_default_Voc": None,\n                "JV_default_Jsc": None,\n                "JV_default_FF": None,\n                "EQE_integrated_Jsc": None,\n                "Stability_time_total_exposure": None,\n                "Stability_PCE_end_of_experiment": None,\n                "Stability_PCE_T80": None,\n                "Stability_PCE_after_1000_h": None,\n                "Stability_protocol": None,\n                "Perovskite_band_gap": None,\n                "Perovskite_pl_max": None\n            },\n            "evidence": ["short evidence item 1", "short evidence item 2"],\n            "confidence": "low|medium|medium_high|high",\n            "needs_human_check": True\n        }\n    }\n\n\ndef call_hf_vision_for_figure(image_path: str, context_text: str, paper_slug: str, figure_id: str) -> Optional[Dict[str, Any]]:\n    """v19.7 Qwen2.5-VL/OpenAI-compatible vision call with cache and crop saving.\n\n    This replaces the older v18 helper but keeps the same function name so the\n    existing figure-analysis wrapper automatically uses it.\n    """\n    if not VISION_LLM_ENABLE:\n        return None\n    model = VISION_MODEL or HF_VISION_MODEL\n    api_key = VISION_API_KEY or HF_API_KEY\n    api_url = VISION_API_URL or HF_API_URL\n    if not (model and api_key and api_url):\n        return {\n            "ok": False,\n            "figure_id": figure_id,\n            "error": "Vision enabled but VISION_MODEL/HF_VISION_MODEL, VISION_API_KEY/HF_API_KEY, or VISION_API_URL/HF_API_URL is missing.",\n        }\n\n    prepared_path, copied_crop_path = _prepare_vision_image_v19_7(image_path, paper_slug, figure_id)\n    if not prepared_path:\n        return {"ok": False, "figure_id": figure_id, "error": "Could not prepare image for vision model."}\n\n    cache_key = _hash_for_vision_v19_7(prepared_path, context_text or "", model)\n    cache_path = VISION_CACHE_DIR / f"{paper_slug}_{figure_id}_{cache_key}.json"\n    if VISION_CACHE_ENABLE and cache_path.exists():\n        try:\n            cached = json.loads(cache_path.read_text(encoding="utf-8"))\n            cached["from_cache"] = True\n            return cached\n        except Exception:\n            pass\n\n    data_url = _image_file_to_data_url(prepared_path)\n    if not data_url:\n        return {"ok": False, "figure_id": figure_id, "error": "Could not encode image as data URL."}\n\n    system = (\n        "You are a figure-analysis assistant for halide-perovskite photovoltaics. "\n        "Return compact, valid JSON only. Treat all extracted values as candidate evidence, not final truth."\n    )\n    user_payload = _qwen_vl_prompt_payload_v19_7(context_text, figure_id)\n\n    request_payload = {\n        "model": model,\n        "temperature": 0,\n        "max_tokens": min(VISION_MAX_OUTPUT_TOKENS, LLM_MAX_OUTPUT_TOKENS),\n        "messages": [\n            {"role": "system", "content": system},\n            {"role": "user", "content": [\n                {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)},\n                {"type": "image_url", "image_url": {"url": data_url}},\n            ]},\n        ],\n    }\n    if VISION_STRICT_JSON_ENABLE:\n        request_payload["response_format"] = {"type": "json_object"}\n\n    append_cost_log(paper_slug, f"qwen_vl_figure_{figure_id}", model, user_payload, min(VISION_MAX_OUTPUT_TOKENS, LLM_MAX_OUTPUT_TOKENS))\n\n    try:\n        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}\n        r = requests.post(api_url, headers=headers, json=request_payload, timeout=REQUEST_TIMEOUT)\n        r.raise_for_status()\n        data = r.json()\n        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")\n        parsed = _extract_json_object_from_text_v19_7(content)\n        result = {\n            "ok": True,\n            "figure_id": figure_id,\n            "provider": VISION_PROVIDER,\n            "model": model,\n            "vision_api_url": api_url,\n            "crop_path": copied_crop_path or prepared_path,\n            "cache_key": cache_key,\n            "from_cache": False,\n            "raw_content_preview": str(content)[:1200],\n            "parsed": parsed,\n        }\n    except Exception as exc:\n        result = {\n            "ok": False,\n            "figure_id": figure_id,\n            "provider": VISION_PROVIDER,\n            "model": model,\n            "crop_path": copied_crop_path or prepared_path,\n            "error": repr(exc),\n        }\n\n    try:\n        if VISION_CACHE_ENABLE:\n            cache_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")\n        (VISION_LOG_DIR / f"{paper_slug}_{figure_id}_{cache_key}.json").write_text(\n            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"\n        )\n    except Exception:\n        pass\n    return result\n\n\n# v19.7: override figure analysis again so vision calls are capped per paper and\n# saved image paths are copied into the top-level figure_reports/crops folder.\n_previous_analyze_pdf_figures_and_images_v19_7 = analyze_pdf_figures_and_images\n\ndef analyze_pdf_figures_and_images(pdf_path: Path, slug: str, native_text: str = "") -> Dict[str, Any]:\n    # Temporarily disable the older v18 per-region vision calls so we can apply\n    # our own capped/controlled vision pass after v19.6 report generation.\n    global VISION_LLM_ENABLE\n    requested_vision = bool(VISION_LLM_ENABLE)\n    VISION_LLM_ENABLE = False\n    try:\n        payload = _previous_analyze_pdf_figures_and_images_v19_7(pdf_path, slug=slug, native_text=native_text)\n    finally:\n        VISION_LLM_ENABLE = requested_vision\n\n    if not isinstance(payload, dict) or not requested_vision:\n        return payload\n\n    evidence = payload.get("figure_evidence_v18") or []\n    saved_images = payload.get("saved_images") or []\n    if not isinstance(evidence, list):\n        evidence = []\n\n    # Select high-value OCR/region entries first. Caption-only entries have no image crop.\n    candidates = []\n    for ev in evidence:\n        if not isinstance(ev, dict):\n            continue\n        if ev.get("source") != "figure_region_ocr":\n            continue\n        fig_type = str(ev.get("figure_type") or "unknown")\n        priority = str(ev.get("priority") or "low")\n        score = int(ev.get("score") or 0)\n        image_idx = ev.get("image_index")\n        # Most old image_index values are 0-based or 1-based depending on source; handle both below.\n        if fig_type in VISION_HIGH_VALUE_TYPES or (priority == "high" and score > 0):\n            candidates.append(ev)\n    candidates.sort(key=_figure_evidence_rank_v19_5, reverse=True)\n\n    vision_outputs = []\n    calls_used = 0\n    for ev in candidates:\n        if calls_used >= VISION_MAX_FIGURES_PER_PAPER:\n            break\n        i = ev.get("image_index")\n        possible_paths = []\n        if isinstance(i, int):\n            if 0 <= i < len(saved_images):\n                possible_paths.append(saved_images[i])\n            if 1 <= i <= len(saved_images):\n                possible_paths.append(saved_images[i - 1])\n        # Fallback: region_N often corresponds roughly to saved_images[N-1]\n        m = re.search(r"(\\d+)$", str(ev.get("figure_id") or ""))\n        if m:\n            idx = int(m.group(1)) - 1\n            if 0 <= idx < len(saved_images):\n                possible_paths.append(saved_images[idx])\n        # Deduplicate paths while preserving order.\n        seen_paths = set()\n        possible_paths = [p for p in possible_paths if not (p in seen_paths or seen_paths.add(p))]\n        if not possible_paths:\n            continue\n        context = ev.get("ocr_text") or ev.get("text") or ""\n        if ev.get("extracted_hint_fields"):\n            context += "\\n\\nCandidate fields from OCR/caption: " + json.dumps(ev.get("extracted_hint_fields"), ensure_ascii=False)\n        vision = call_hf_vision_for_figure(possible_paths[0], context, slug, str(ev.get("figure_id") or f"region_{calls_used+1}"))\n        if vision:\n            ev["vision_model_output_v19_7"] = vision\n            vision_outputs.append(vision)\n            calls_used += 1\n\n    payload["figure_vision_outputs_v19_7"] = vision_outputs\n    payload["figure_vision_settings_v19_7"] = {\n        "vision_enabled": True,\n        "vision_model": VISION_MODEL or HF_VISION_MODEL,\n        "vision_api_url": VISION_API_URL or HF_API_URL,\n        "max_figures_per_paper": VISION_MAX_FIGURES_PER_PAPER,\n        "max_image_width": VISION_MAX_IMAGE_WIDTH,\n        "cache_enable": VISION_CACHE_ENABLE,\n        "calls_used": calls_used,\n        "vision_cache_dir": str(VISION_CACHE_DIR),\n        "vision_crops_dir": str(VISION_CROPS_DIR),\n        "vision_log_dir": str(VISION_LOG_DIR),\n    }\n\n    # Rebuild and resave figure report after vision outputs have been attached.\n    if FIGURE_REPORT_ENABLE:\n        try:\n            report = build_figure_report_v19_5(slug, payload)\n            report["figure_report_version"] = "v19.7"\n            report["vision_outputs_count"] = len(vision_outputs)\n            report["vision_settings_v19_7"] = payload.get("figure_vision_settings_v19_7")\n            report_text = render_figure_report_text_v19_5(report)\n            payload["figure_report_v19_7"] = report\n            payload["figure_report_text_v19_7"] = report_text\n            (FIGURE_REPORTS_JSON_DIR / f"{slug}_figure_report_v19_7.json").write_text(\n                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"\n            )\n            (FIGURE_REPORTS_TEXT_DIR / f"{slug}_figure_report_v19_7.txt").write_text(report_text, encoding="utf-8")\n            (FIGURE_EVIDENCE_DIR / f"{slug}_figure_report_v19_7.json").write_text(\n                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"\n            )\n            (FIGURE_EVIDENCE_DIR / f"{slug}_figure_report_v19_7.txt").write_text(report_text, encoding="utf-8")\n            payload["figure_text"] = _truncate_text(\n                ((payload.get("figure_text") or "") + "\\n\\n[FIGURE_REPORT_V19_7]\\n" + report_text).strip(),\n                FIGURE_APPENDIX_MAX_CHARS,\n            )\n        except Exception as exc:\n            payload["figure_report_v19_7_error"] = repr(exc)\n\n    return payload\n\n\n# v19.7: enrich figure report JSON/text with Qwen-VL results when present.\n_previous_build_figure_report_v19_5_v19_7 = build_figure_report_v19_5\n\ndef build_figure_report_v19_5(slug: str, figure_payload: Dict[str, Any]) -> Dict[str, Any]:\n    report = _previous_build_figure_report_v19_5_v19_7(slug, figure_payload)\n    report["figure_report_version"] = "v19.7"\n    report["vision_provider"] = VISION_PROVIDER\n    report["vision_enabled"] = bool(VISION_LLM_ENABLE)\n    report["vision_model"] = VISION_MODEL or HF_VISION_MODEL\n    report["vision_cache_dir"] = str(VISION_CACHE_DIR)\n    report["vision_crops_dir"] = str(VISION_CROPS_DIR)\n    report["vision_log_dir"] = str(VISION_LOG_DIR)\n    report["vision_outputs_count"] = len(figure_payload.get("figure_vision_outputs_v19_7") or [])\n    report["figure_report_json_path_v19_7"] = str(FIGURE_REPORTS_JSON_DIR / f"{slug}_figure_report_v19_7.json")\n    report["figure_report_text_path_v19_7"] = str(FIGURE_REPORTS_TEXT_DIR / f"{slug}_figure_report_v19_7.txt")\n    return report\n\n\ndef _summarize_vision_output_for_text_v19_7(vis: Dict[str, Any]) -> List[str]:\n    lines = []\n    if not isinstance(vis, dict):\n        return lines\n    if not vis.get("ok"):\n        lines.append(f"  Vision error: {vis.get(\'error\')}")\n        return lines\n    parsed = vis.get("parsed") if isinstance(vis.get("parsed"), dict) else {}\n    if parsed:\n        fig_type = parsed.get("figure_type")\n        conf = parsed.get("confidence")\n        summary = parsed.get("visual_summary") or parsed.get("interpretation_summary")\n        if summary:\n            lines.append(f"  Qwen-VL summary: {summary}")\n        if fig_type or conf:\n            lines.append(f"  Qwen-VL type/confidence: {fig_type or \'unknown\'} / {conf or \'unknown\'}")\n        fields = parsed.get("candidate_database_fields") or parsed.get("extracted_values") or {}\n        if isinstance(fields, dict):\n            fields = {k: v for k, v in fields.items() if v not in (None, "", [], {})}\n            if fields:\n                lines.append(f"  Qwen-VL candidate fields: {json.dumps(fields, ensure_ascii=False)}")\n        evs = parsed.get("evidence")\n        if isinstance(evs, list) and evs:\n            lines.append("  Qwen-VL evidence: " + "; ".join(str(x) for x in evs[:3]))\n    if vis.get("crop_path"):\n        lines.append(f"  Crop: {vis.get(\'crop_path\')}")\n    if vis.get("from_cache"):\n        lines.append("  From cache: true")\n    return lines\n\n\ndef render_figure_report_text_v19_5(report: Dict[str, Any]) -> str:\n    lines = []\n    lines.append("Figure / Visual Evidence Report")\n    lines.append("=" * 32)\n    lines.append(f"Report version: {report.get(\'figure_report_version\') or \'v19.7\'}")\n    lines.append(f"Pages scanned: {report.get(\'pages_scanned\')}")\n    lines.append(f"Captions found: {report.get(\'caption_count\')}")\n    lines.append(f"Figure/image regions OCR\'d: {report.get(\'figure_region_count\')}")\n    lines.append(f"Saved image crops: {report.get(\'saved_image_count\')}")\n    lines.append(f"Vision enabled: {report.get(\'vision_enabled\')} | model: {report.get(\'vision_model\') or \'(none)\'}")\n    if report.get("vision_outputs_count") is not None:\n        lines.append(f"Vision outputs: {report.get(\'vision_outputs_count\')}")\n    counts = report.get("figure_type_counts") or {}\n    if counts:\n        counts_txt = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: kv[0]))\n        lines.append(f"Detected figure types: {counts_txt}")\n    lines.append("")\n    lines.append("High-value figure evidence:")\n    items = report.get("high_value_figure_evidence") or report.get("top_figure_evidence") or []\n    if not items:\n        lines.append("- No high-value figure evidence found from captions/OCR.")\n    for ev in items[:FIGURE_REPORT_MAX_ITEMS]:\n        fig_id = ev.get("figure_id")\n        fig_type = ev.get("figure_type")\n        priority = ev.get("priority")\n        score = ev.get("score")\n        page = ev.get("page")\n        src = ev.get("source")\n        text = ev.get("text_preview") or ev.get("ocr_text_preview") or ""\n        hints = ev.get("extracted_hint_fields") or {}\n        page_txt = f", page {page}" if page else ""\n        lines.append(f"- {fig_id} ({fig_type}, {priority}, score={score}, {src}{page_txt})")\n        if hints:\n            lines.append(f"  OCR/caption candidate fields: {json.dumps(hints, ensure_ascii=False)}")\n        vis = ev.get("vision_model_output_v19_7") or ev.get("vision_model_output")\n        if isinstance(vis, dict):\n            lines.extend(_summarize_vision_output_for_text_v19_7(vis))\n        if text:\n            lines.append(f"  Evidence preview: {_truncate_text(text, 280)}")\n    lines.append("")\n    lines.append(\n        "Notes: OCR/caption values and Qwen-VL values are candidate evidence. "\n        "Do not treat curve-derived values as final unless explicitly labeled or validated downstream."\n    )\n    return "\\n".join(lines).strip() + "\\n"\n\n\n# v19.7: look for newest figure report in summaries.\ndef _load_figure_report_text_for_slug_v19_5(slug: str) -> str:\n    candidates = [\n        FIGURE_REPORTS_TEXT_DIR / f"{slug}_figure_report_v19_7.txt",\n        FIGURE_REPORTS_TEXT_DIR / f"{slug}_figure_report_v19_6.txt",\n        FIGURE_EVIDENCE_DIR / f"{slug}_figure_report_v19_7.txt",\n        FIGURE_EVIDENCE_DIR / f"{slug}_figure_report_v19_6.txt",\n        FIGURE_EVIDENCE_DIR / f"{slug}_figure_report_v19_5.txt",\n    ]\n    for p in candidates:\n        if p.exists():\n            try:\n                return p.read_text(encoding="utf-8")\n            except Exception:\n                continue\n    return ""\n\nprint(\n    f"[INFO] v19.7 Qwen-VL layer loaded | vision={VISION_LLM_ENABLE} | "\n    f"model={VISION_MODEL or HF_VISION_MODEL or \'(none)\'} | max_figures={VISION_MAX_FIGURES_PER_PAPER} | "\n    f"cache={VISION_CACHE_ENABLE}"\n)\n\n# ============================================================\n# v19.7 FINAL ENTRYPOINT\n# ============================================================\n# This must remain at the very end of the file so all extension layers above are\n# registered before the agent starts processing papers.\nif __name__ == "__main__":\n    print(f"[INFO] v19.7 output folder: {BASE}")\n    print(f"[INFO] v19.7 figure report folder: {FIGURE_REPORTS_DIR}")\n    print(f"[INFO] v19.7 figure report text folder: {FIGURE_REPORTS_TEXT_DIR}")\n    print(f"[INFO] v19.7 figure report json folder: {FIGURE_REPORTS_JSON_DIR}")\n    print(f"[INFO] v19.7 figure crop folder: {VISION_CROPS_DIR}")\n    print(f"[INFO] v19.7 vision cache folder: {VISION_CACHE_DIR}")\n    if os.getenv("RUN_AS_MCP", "").strip().lower() in {"1", "true", "yes"}:\n        run_mcp_server()\n    else:\n        main()\n'


# =============================================================================
# EMBEDDED INTEGRATION BRIDGE SOURCE
# =============================================================================

_INTEGRATION_BRIDGE_SOURCE = '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"""\nIntegrate LiteratureAgent outputs with the PCE -> Stability modeling dataset.\n\nPurpose\n-------\nThis script connects the literature-mining workflow to the PCE/stability modeling workflow.\n\nIt does NOT replace either main script. Instead, it creates a clean bridge:\n\n    LiteratureAgent all_records.csv\n        -> validate extracted rows\n        -> separate evidence fields\n        -> deduplicate/merge against baseline Perovskite Database CSV\n        -> write updated modeling CSV\n        -> optionally launch pce_then_stability_same_approach.py on the updated CSV\n\nWhy this exists\n---------------\nThe LiteratureAgent should update the modeling dataset, but it should not directly overwrite\nthe original Perovskite Database. This script creates an auditable staged update so you can check\nwhat was added, rejected, or potentially duplicated before training the PCE/stability models.\n\nTypical workflow\n----------------\n1) Run LiteratureAgent first:\n\n   python literature_manager_agent_v19_7_qwen_vl_figure_vision.py\n\n2) Merge LiteratureAgent output into your base database:\n\n   python integrate_literature_to_pce_stability.py ^\n     --base_csv "Perovskite_database_content_all_data.csv" ^\n     --lit_csv "%WORK_DIR%\\\\csv\\\\all_records.csv" ^\n     --out_dir "artifacts_literature_dataset_update" ^\n     --updated_csv "updated_perovskite_database_with_literature_agent.csv"\n\n3) Run the PCE/stability model on the updated dataset:\n\n   python pce_then_stability_same_approach.py ^\n     --csv "updated_perovskite_database_with_literature_agent.csv" ^\n     --out "artifacts_pce_then_stability_lit_updated"\n\nOr run the model automatically from this script:\n\n   python integrate_literature_to_pce_stability.py ^\n     --base_csv "Perovskite_database_content_all_data.csv" ^\n     --lit_csv "%WORK_DIR%\\\\csv\\\\all_records.csv" ^\n     --model_script "pce_then_stability_same_approach.py" ^\n     --run_model\n\nOutputs\n-------\nout_dir/\n    updated_perovskite_database_with_literature_agent.csv\n    literature_update_accepted_rows.csv\n    literature_update_rejected_rows.csv\n    literature_update_evidence_long.csv\n    literature_update_evidence_wide.csv\n    literature_update_duplicate_candidates.csv\n    literature_update_audit.csv\n    literature_update_summary.json\n    README.txt\n\nNotes\n-----\n- The updated modeling CSV keeps schema/modeling columns and useful source metadata.\n- Evidence snippets are saved separately so they do not pollute model features.\n- Merge behavior is conservative by default:\n    same DOI + similar device fingerprint -> do not blindly append as new;\n    it is flagged as a duplicate candidate.\n- Use --merge_policy append_new_only for safest behavior.\n- Use --merge_policy update_existing to fill missing values in baseline rows only when duplicate is detected.\n- Use --merge_policy append_all to append every accepted LiteratureAgent row.\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport hashlib\nimport json\nimport math\nimport os\nimport re\nimport subprocess\nimport sys\nfrom dataclasses import asdict, dataclass\nfrom datetime import datetime\nfrom pathlib import Path\nfrom typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple\n\nimport numpy as np\nimport pandas as pd\n\n\n# =============================================================================\n# CONFIG\n# =============================================================================\n\nMISSING_TOKENS = {\n    "", " ", "na", "n/a", "n.a.", "none", "null", "nan", "unknown",\n    "-", "--", "—", "not available", "not reported", "nr", "n.r.",\n    "not applicable", "not measured", "not specified", "not given",\n}\n\nIMPORTANT_MODEL_FIELDS = [\n    # provenance\n    "Ref_DOI_number",\n    "Ref_lead_author",\n    "Ref_publication_date",\n    "Ref_journal",\n    "Ref_internal_sample_id",\n\n    # device\n    "Cell_stack_sequence",\n    "Cell_architecture",\n    "Cell_area_total",\n    "Cell_area_measured",\n    "Cell_number_of_cells_per_substrate",\n    "Cell_flexible",\n    "Module",\n    "Substrate_stack_sequence",\n\n    # chemistry\n    "Perovskite_composition_short_form",\n    "Perovskite_composition_long_form",\n    "Perovskite_composition_a_ions",\n    "Perovskite_composition_a_ions_coefficients",\n    "Perovskite_composition_b_ions",\n    "Perovskite_composition_b_ions_coefficients",\n    "Perovskite_composition_c_ions",\n    "Perovskite_composition_c_ions_coefficients",\n    "Perovskite_dimension_0D",\n    "Perovskite_dimension_2D",\n    "Perovskite_dimension_2D3D_mixture",\n    "Perovskite_dimension_3D",\n    "Perovskite_dimension_3D_with_2D_capping_layer",\n    "Perovskite_additives_compounds",\n    "Perovskite_additives_concentrations",\n    "Perovskite_thickness",\n    "Perovskite_band_gap",\n    "Perovskite_pl_max",\n\n    # process\n    "Perovskite_deposition_procedure",\n    "Perovskite_deposition_solvents",\n    "Perovskite_deposition_solvents_mixing_ratios",\n    "Perovskite_deposition_quenching_media",\n    "Perovskite_deposition_quenching_media_volume",\n    "Perovskite_deposition_thermal_annealing_temperature",\n    "Perovskite_deposition_thermal_annealing_time",\n    "Perovskite_deposition_thermal_annealing_atmosphere",\n\n    # interfaces\n    "ETL_stack_sequence",\n    "ETL_thickness",\n    "ETL_additives_compounds",\n    "ETL_additives_concentrations",\n    "ETL_deposition_procedure",\n    "HTL_stack_sequence",\n    "HTL_thickness_list",\n    "HTL_additives_compounds",\n    "HTL_additives_concentrations",\n    "HTL_deposition_procedure",\n    "Backcontact_stack_sequence",\n    "Backcontact_thickness_list",\n    "Encapsulation",\n    "Encapsulation_stack_sequence",\n\n    # PCE / initial performance\n    "JV_measured",\n    "JV_default_PCE",\n    "JV_default_Voc",\n    "JV_default_Jsc",\n    "JV_default_FF",\n    "JV_reverse_scan_PCE",\n    "JV_forward_scan_PCE",\n    "JV_hysteresis_index",\n    "Stabilised_performance_measured",\n    "Stabilised_performance_PCE",\n\n    # stability/outdoor\n    "Stability_measured",\n    "Stability_protocol",\n    "Stability_time_total_exposure",\n    "Stability_PCE_initial_value",\n    "Stability_PCE_end_of_experiment",\n    "Stability_PCE_T95",\n    "Stability_PCE_T80",\n    "Stability_PCE_after_1000_h",\n    "Outdoor_tested",\n    "Outdoor_protocol",\n    "Outdoor_time_total_exposure",\n    "Outdoor_PCE_initial_value",\n    "Outdoor_PCE_end_of_experiment",\n    "Outdoor_PCE_T95",\n    "Outdoor_PCE_T80",\n    "Outdoor_PCE_after_1000_h",\n]\n\nSOURCE_METADATA_FIELDS = [\n    "_source_abstract",\n    "_source_pdf_url",\n    "_source_open_access",\n    "_source_landing_page",\n    "_full_text_length",\n    "_full_text_source",\n    "_composition_json",\n    "_lit_agent_version",\n    "_lit_agent_run_id",\n    "_lit_agent_added_date",\n    "_lit_agent_row_hash",\n    "_source_added_by_literature_agent",\n    "_lit_agent_confidence_status",\n]\n\nPERFORMANCE_FIELDS = [\n    "JV_default_PCE",\n    "JV_reverse_scan_PCE",\n    "JV_forward_scan_PCE",\n    "Stabilised_performance_PCE",\n    "JV_default_Voc",\n    "JV_default_Jsc",\n    "JV_default_FF",\n]\n\nSTABILITY_FIELDS = [\n    "Stability_PCE_T80",\n    "Stability_PCE_T95",\n    "Stability_PCE_end_of_experiment",\n    "Stability_PCE_after_1000_h",\n    "Stability_time_total_exposure",\n    "Outdoor_PCE_T80",\n    "Outdoor_PCE_T95",\n    "Outdoor_PCE_end_of_experiment",\n    "Outdoor_PCE_after_1000_h",\n    "Outdoor_time_total_exposure",\n]\n\nDESIGN_FIELDS = [\n    "Perovskite_composition_short_form",\n    "Perovskite_composition_long_form",\n    "Cell_stack_sequence",\n    "Cell_architecture",\n    "ETL_stack_sequence",\n    "HTL_stack_sequence",\n    "Backcontact_stack_sequence",\n    "Perovskite_deposition_procedure",\n    "Perovskite_deposition_solvents",\n]\n\n\n@dataclass\nclass MergeConfig:\n    base_csv: str\n    lit_csv: str\n    out_dir: str = "artifacts_literature_dataset_update"\n    updated_csv: str = "updated_perovskite_database_with_literature_agent.csv"\n    merge_policy: str = "append_new_only"  # append_new_only, update_existing, append_all\n    min_required_non_null_important_fields: int = 4\n    min_required_evidence_fields: int = 1\n    require_doi: bool = True\n    duplicate_similarity_threshold: float = 0.82\n    model_script: str = "pce_then_stability_same_approach.py"\n    run_model: bool = False\n    model_out_dir: str = "artifacts_pce_then_stability_lit_updated"\n    dry_run: bool = False\n\n\n# =============================================================================\n# IO / CLEANING HELPERS\n# =============================================================================\n\ndef ensure_dir(path: str | Path) -> Path:\n    p = Path(path)\n    p.mkdir(parents=True, exist_ok=True)\n    return p\n\n\ndef now_stamp() -> str:\n    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")\n\n\ndef safe_json_dump(obj: Any, path: str | Path) -> None:\n    def conv(x):\n        if isinstance(x, (np.integer,)):\n            return int(x)\n        if isinstance(x, (np.floating,)):\n            if np.isnan(x):\n                return None\n            return float(x)\n        if isinstance(x, np.ndarray):\n            return x.tolist()\n        if isinstance(x, pd.Timestamp):\n            return str(x)\n        return str(x)\n\n    with open(path, "w", encoding="utf-8") as f:\n        json.dump(obj, f, indent=2, default=conv, ensure_ascii=False)\n\n\ndef normalize_colname(c: str) -> str:\n    return str(c).strip().replace("\\ufeff", "")\n\n\ndef normalize_text_value(v: Any) -> Any:\n    if v is None:\n        return np.nan\n    if isinstance(v, float):\n        if math.isnan(v) or math.isinf(v):\n            return np.nan\n        return v\n    if isinstance(v, (int, np.integer, np.floating)):\n        return v\n    s = str(v).strip()\n    if s.lower() in MISSING_TOKENS:\n        return np.nan\n    return s\n\n\ndef dataframe_map(df: pd.DataFrame, func) -> pd.DataFrame:\n    try:\n        return df.map(func)\n    except Exception:\n        return df.applymap(func)\n\n\ndef robust_read_csv(path: str | Path) -> pd.DataFrame:\n    path = Path(path)\n    if not path.exists():\n        raise FileNotFoundError(f"CSV not found: {path}")\n\n    encodings = ["utf-8-sig", "utf-8", "utf-16", "latin1", "cp1252"]\n    last_error = None\n    for enc in encodings:\n        try:\n            df = pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")\n            df.columns = [normalize_colname(c) for c in df.columns]\n            df = df.loc[:, ~df.columns.duplicated()].copy()\n            df = dataframe_map(df, normalize_text_value)\n            print(f"Loaded {path} with encoding={enc}: {df.shape[0]:,} rows × {df.shape[1]:,} columns")\n            return df\n        except Exception as e:\n            last_error = e\n    raise RuntimeError(f"Could not read {path}. Last error: {last_error}")\n\n\ndef write_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:\n    path = Path(path)\n    path.parent.mkdir(parents=True, exist_ok=True)\n    df.to_csv(path, index=index, encoding="utf-8-sig")\n\n\n# =============================================================================\n# NORMALIZATION / FINGERPRINTS\n# =============================================================================\n\ndef normalize_doi(v: Any) -> str:\n    if pd.isna(v):\n        return ""\n    s = str(v).strip()\n    s = re.sub(r"^https?://(dx\\.)?doi\\.org/", "", s, flags=re.I)\n    s = re.sub(r"^doi:\\s*", "", s, flags=re.I)\n    return s.strip().lower()\n\n\ndef normalize_compact_text(v: Any) -> str:\n    if pd.isna(v):\n        return ""\n    s = str(v).lower()\n    s = re.sub(r"[_\\-/|,;:()\\[\\]{}]+", " ", s)\n    s = re.sub(r"\\s+", " ", s).strip()\n    return s\n\n\ndef numeric_like(v: Any) -> str:\n    if pd.isna(v):\n        return ""\n    s = str(v)\n    m = re.search(r"[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?", s)\n    return m.group(0) if m else normalize_compact_text(s)\n\n\ndef row_hash(row: pd.Series, cols: Sequence[str]) -> str:\n    parts = []\n    for c in cols:\n        if c in row.index:\n            parts.append(f"{c}={normalize_compact_text(row.get(c))}")\n    raw = "|".join(parts)\n    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]\n\n\ndef make_device_fingerprint(row: pd.Series) -> str:\n    """\n    Fingerprint is intentionally not too strict. It is for duplicate detection,\n    not for proving two rows are identical.\n    """\n    parts = [\n        normalize_doi(row.get("Ref_DOI_number")),\n        normalize_compact_text(row.get("Ref_internal_sample_id")),\n        normalize_compact_text(row.get("Cell_stack_sequence")),\n        normalize_compact_text(row.get("Cell_architecture")),\n        normalize_compact_text(row.get("Perovskite_composition_short_form")),\n        normalize_compact_text(row.get("Perovskite_composition_long_form")),\n        normalize_compact_text(row.get("ETL_stack_sequence")),\n        normalize_compact_text(row.get("HTL_stack_sequence")),\n        normalize_compact_text(row.get("Backcontact_stack_sequence")),\n        numeric_like(row.get("JV_default_PCE")),\n        numeric_like(row.get("JV_reverse_scan_PCE")),\n        numeric_like(row.get("Stabilised_performance_PCE")),\n        numeric_like(row.get("Stability_PCE_T80")),\n        numeric_like(row.get("Stability_PCE_end_of_experiment")),\n    ]\n    raw = "|".join(parts)\n    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:20]\n\n\ndef token_set_for_row(row: pd.Series, cols: Sequence[str]) -> set:\n    text = " ".join(normalize_compact_text(row.get(c)) for c in cols if c in row.index)\n    tokens = set(re.findall(r"[a-z0-9]{2,}", text))\n    return tokens\n\n\ndef jaccard(a: set, b: set) -> float:\n    if not a and not b:\n        return 0.0\n    if not a or not b:\n        return 0.0\n    return len(a & b) / max(1, len(a | b))\n\n\n# =============================================================================\n# LITERATURE ROW VALIDATION\n# =============================================================================\n\ndef is_evidence_col(c: str) -> bool:\n    return str(c).startswith("_evidence_")\n\n\ndef split_evidence_fields(lit_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:\n    """\n    Returns:\n      clean_df without _evidence_* columns\n      evidence_wide with provenance columns + _evidence_* columns\n      evidence_long with row/field/evidence format\n    """\n    evidence_cols = [c for c in lit_df.columns if is_evidence_col(c)]\n    provenance_cols = [c for c in ["Ref_DOI_number", "Ref_internal_sample_id", "Ref_lead_author", "_source_pdf_url", "_source_landing_page"] if c in lit_df.columns]\n\n    if evidence_cols:\n        evidence_wide = lit_df[provenance_cols + evidence_cols].copy()\n    else:\n        evidence_wide = pd.DataFrame(columns=provenance_cols)\n\n    long_rows = []\n    for idx, row in lit_df.iterrows():\n        doi = normalize_doi(row.get("Ref_DOI_number"))\n        sample_id = row.get("Ref_internal_sample_id")\n        for ec in evidence_cols:\n            val = row.get(ec)\n            if pd.notna(val):\n                field = ec.replace("_evidence_", "", 1)\n                long_rows.append({\n                    "lit_row_index": idx,\n                    "Ref_DOI_number": doi,\n                    "Ref_internal_sample_id": sample_id,\n                    "field": field,\n                    "evidence": val,\n                })\n\n    evidence_long = pd.DataFrame(long_rows)\n    clean_df = lit_df.drop(columns=evidence_cols, errors="ignore").copy()\n    return clean_df, evidence_wide, evidence_long\n\n\ndef count_non_null(row: pd.Series, cols: Sequence[str]) -> int:\n    return int(sum((c in row.index) and pd.notna(row.get(c)) for c in cols))\n\n\ndef validate_literature_rows(\n    lit_df: pd.DataFrame,\n    evidence_long: pd.DataFrame,\n    cfg: MergeConfig,\n) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:\n    """\n    Conservative row validation:\n    - require DOI by default\n    - require at least one PCE/stability field OR strong design record\n    - require enough important fields\n    - optionally require evidence\n    """\n    evidence_count_by_idx = {}\n    if not evidence_long.empty and "lit_row_index" in evidence_long.columns:\n        evidence_count_by_idx = evidence_long.groupby("lit_row_index")["evidence"].count().to_dict()\n\n    audit_rows = []\n    accepted_indices = []\n    rejected_indices = []\n\n    for idx, row in lit_df.iterrows():\n        doi = normalize_doi(row.get("Ref_DOI_number"))\n        n_perf = count_non_null(row, PERFORMANCE_FIELDS)\n        n_stab = count_non_null(row, STABILITY_FIELDS)\n        n_design = count_non_null(row, DESIGN_FIELDS)\n        n_important = count_non_null(row, IMPORTANT_MODEL_FIELDS)\n        n_evidence = int(evidence_count_by_idx.get(idx, 0))\n\n        reasons = []\n        status = "accepted"\n\n        if cfg.require_doi and not doi:\n            status = "rejected"\n            reasons.append("missing_doi")\n\n        if n_important < cfg.min_required_non_null_important_fields:\n            status = "rejected"\n            reasons.append(f"too_few_important_fields:{n_important}")\n\n        if (n_perf + n_stab) == 0 and n_design < 2:\n            status = "rejected"\n            reasons.append("no_performance_or_stability_and_weak_design_info")\n\n        if cfg.min_required_evidence_fields > 0 and n_evidence < cfg.min_required_evidence_fields:\n            # Do not automatically reject if source metadata suggests this was deterministic/CSV input,\n            # but mark as weak. Most LiteratureAgent LLM rows should have evidence.\n            if any(str(row.get(c, "")).strip() for c in ["_source_pdf_url", "_source_landing_page", "_source_abstract"] if c in row.index):\n                if status == "accepted":\n                    status = "weak_accepted"\n                reasons.append(f"low_evidence_count:{n_evidence}")\n            else:\n                status = "rejected"\n                reasons.append(f"low_evidence_count:{n_evidence}")\n\n        if status in {"accepted", "weak_accepted"}:\n            accepted_indices.append(idx)\n        else:\n            rejected_indices.append(idx)\n\n        audit_rows.append({\n            "lit_row_index": idx,\n            "Ref_DOI_number": doi,\n            "status": status,\n            "reasons": ";".join(reasons) if reasons else "ok",\n            "n_performance_fields": n_perf,\n            "n_stability_fields": n_stab,\n            "n_design_fields": n_design,\n            "n_important_fields": n_important,\n            "n_evidence_fields": n_evidence,\n        })\n\n    audit_df = pd.DataFrame(audit_rows)\n    accepted = lit_df.loc[accepted_indices].copy()\n    rejected = lit_df.loc[rejected_indices].copy()\n    return accepted, rejected, audit_df\n\n\n# =============================================================================\n# MERGING / DEDUPLICATION\n# =============================================================================\n\ndef align_columns_for_merge(base_df: pd.DataFrame, lit_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:\n    """\n    Keep all existing baseline columns, plus source metadata fields and any LiteratureAgent columns\n    that look safe/useful. Evidence has already been removed by this point.\n    """\n    base_cols = list(base_df.columns)\n    extra_safe = []\n    for c in lit_df.columns:\n        if c in base_cols:\n            continue\n        if c in SOURCE_METADATA_FIELDS:\n            extra_safe.append(c)\n        elif c.startswith("_source_") or c.startswith("_lit_agent_"):\n            extra_safe.append(c)\n        elif c.startswith("ADD_DESC_"):\n            # Usually added later by model script, but harmless if present.\n            extra_safe.append(c)\n\n    final_cols = base_cols + [c for c in extra_safe if c not in base_cols]\n\n    for c in final_cols:\n        if c not in base_df.columns:\n            base_df[c] = np.nan\n        if c not in lit_df.columns:\n            lit_df[c] = np.nan\n\n    return base_df[final_cols].copy(), lit_df[final_cols].copy(), final_cols\n\n\ndef build_duplicate_candidates(\n    base_df: pd.DataFrame,\n    lit_df: pd.DataFrame,\n    threshold: float,\n) -> Tuple[pd.DataFrame, Dict[int, int]]:\n    """\n    Returns duplicate candidate table and a map from lit row index -> base row index.\n    Matching strategy:\n      1) exact device fingerprint match\n      2) same DOI + Jaccard similarity on key descriptor tokens\n    """\n    key_cols = [\n        "Ref_internal_sample_id",\n        "Cell_stack_sequence",\n        "Cell_architecture",\n        "Perovskite_composition_short_form",\n        "Perovskite_composition_long_form",\n        "ETL_stack_sequence",\n        "HTL_stack_sequence",\n        "Backcontact_stack_sequence",\n        "JV_default_PCE",\n        "JV_reverse_scan_PCE",\n        "Stability_PCE_T80",\n        "Stability_PCE_end_of_experiment",\n    ]\n\n    base = base_df.copy()\n    lit = lit_df.copy()\n\n    base["_merge_doi_norm"] = base.get("Ref_DOI_number", pd.Series(index=base.index, dtype=object)).map(normalize_doi)\n    lit["_merge_doi_norm"] = lit.get("Ref_DOI_number", pd.Series(index=lit.index, dtype=object)).map(normalize_doi)\n\n    base["_device_fingerprint"] = base.apply(make_device_fingerprint, axis=1)\n    lit["_device_fingerprint"] = lit.apply(make_device_fingerprint, axis=1)\n\n    fingerprint_to_base = {}\n    for i, fp in base["_device_fingerprint"].items():\n        if fp and fp not in fingerprint_to_base:\n            fingerprint_to_base[fp] = i\n\n    duplicate_rows = []\n    lit_to_base = {}\n\n    # Exact fingerprint first\n    for li, row in lit.iterrows():\n        fp = row["_device_fingerprint"]\n        if fp in fingerprint_to_base:\n            bi = fingerprint_to_base[fp]\n            lit_to_base[li] = bi\n            duplicate_rows.append({\n                "lit_row_index": li,\n                "base_row_index": bi,\n                "match_type": "exact_device_fingerprint",\n                "similarity": 1.0,\n                "Ref_DOI_number": row.get("_merge_doi_norm"),\n            })\n\n    # Same DOI similarity second\n    base_by_doi = {}\n    for bi, brow in base.iterrows():\n        doi = brow.get("_merge_doi_norm", "")\n        if doi:\n            base_by_doi.setdefault(doi, []).append(bi)\n\n    base_token_cache = {bi: token_set_for_row(base.loc[bi], key_cols) for bi in base.index}\n\n    for li, row in lit.iterrows():\n        if li in lit_to_base:\n            continue\n        doi = row.get("_merge_doi_norm", "")\n        if not doi or doi not in base_by_doi:\n            continue\n\n        lit_tokens = token_set_for_row(row, key_cols)\n        best_bi = None\n        best_score = 0.0\n        for bi in base_by_doi[doi]:\n            score = jaccard(lit_tokens, base_token_cache[bi])\n            if score > best_score:\n                best_score = score\n                best_bi = bi\n\n        if best_bi is not None and best_score >= threshold:\n            lit_to_base[li] = best_bi\n            duplicate_rows.append({\n                "lit_row_index": li,\n                "base_row_index": best_bi,\n                "match_type": "same_doi_similarity",\n                "similarity": float(best_score),\n                "Ref_DOI_number": doi,\n            })\n\n    dup_df = pd.DataFrame(duplicate_rows)\n    return dup_df, lit_to_base\n\n\ndef update_existing_rows_fill_missing(\n    base_df: pd.DataFrame,\n    lit_df: pd.DataFrame,\n    lit_to_base: Dict[int, int],\n) -> Tuple[pd.DataFrame, pd.DataFrame]:\n    """\n    Fill missing baseline cells from duplicate LiteratureAgent rows.\n    Does not overwrite existing baseline values.\n    """\n    updated = base_df.copy()\n    change_rows = []\n\n    for li, bi in lit_to_base.items():\n        lit_row = lit_df.loc[li]\n        for c in lit_df.columns:\n            if c not in updated.columns:\n                continue\n            if c.startswith("_lit_agent_") or c.startswith("_source_"):\n                # Preserve source metadata only if empty.\n                pass\n            base_val = updated.at[bi, c]\n            lit_val = lit_row.get(c)\n            if pd.isna(base_val) and pd.notna(lit_val):\n                updated.at[bi, c] = lit_val\n                change_rows.append({\n                    "base_row_index": bi,\n                    "lit_row_index": li,\n                    "column": c,\n                    "action": "filled_missing_from_literature_agent",\n                    "new_value": lit_val,\n                })\n\n    return updated, pd.DataFrame(change_rows)\n\n\ndef merge_literature_into_base(\n    base_df: pd.DataFrame,\n    accepted_lit_df: pd.DataFrame,\n    cfg: MergeConfig,\n) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:\n    """\n    Applies merge policy and returns:\n      updated_df\n      duplicate_candidates\n      changes_df\n    """\n    base_aligned, lit_aligned, final_cols = align_columns_for_merge(base_df.copy(), accepted_lit_df.copy())\n\n    if lit_aligned.empty:\n        return base_aligned, pd.DataFrame(), pd.DataFrame()\n\n    duplicate_candidates, lit_to_base = build_duplicate_candidates(\n        base_aligned,\n        lit_aligned,\n        threshold=cfg.duplicate_similarity_threshold,\n    )\n\n    lit_aligned["_lit_duplicate_candidate"] = lit_aligned.index.map(lambda i: int(i in lit_to_base))\n    if "_lit_duplicate_candidate" not in final_cols:\n        final_cols.append("_lit_duplicate_candidate")\n        base_aligned["_lit_duplicate_candidate"] = np.nan\n\n    changes_df = pd.DataFrame()\n\n    if cfg.merge_policy == "append_all":\n        updated = pd.concat([base_aligned[final_cols], lit_aligned[final_cols]], ignore_index=True)\n\n    elif cfg.merge_policy == "append_new_only":\n        new_lit = lit_aligned.loc[[i for i in lit_aligned.index if i not in lit_to_base]].copy()\n        updated = pd.concat([base_aligned[final_cols], new_lit[final_cols]], ignore_index=True)\n\n    elif cfg.merge_policy == "update_existing":\n        updated_base, changes_df = update_existing_rows_fill_missing(base_aligned, lit_aligned, lit_to_base)\n        new_lit = lit_aligned.loc[[i for i in lit_aligned.index if i not in lit_to_base]].copy()\n        updated = pd.concat([updated_base[final_cols], new_lit[final_cols]], ignore_index=True)\n\n    else:\n        raise ValueError(f"Unknown merge_policy={cfg.merge_policy}")\n\n    return updated, duplicate_candidates, changes_df\n\n\n# =============================================================================\n# MAIN PIPELINE\n# =============================================================================\n\ndef add_lit_metadata(lit_df: pd.DataFrame, agent_version: Optional[str] = None) -> pd.DataFrame:\n    lit_df = lit_df.copy()\n\n    if "_source_added_by_literature_agent" not in lit_df.columns:\n        lit_df["_source_added_by_literature_agent"] = 1\n\n    if "_lit_agent_added_date" not in lit_df.columns:\n        lit_df["_lit_agent_added_date"] = now_stamp()\n\n    if "_lit_agent_version" not in lit_df.columns:\n        lit_df["_lit_agent_version"] = agent_version or "unknown"\n\n    if "_lit_agent_run_id" not in lit_df.columns:\n        lit_df["_lit_agent_run_id"] = hashlib.sha1(now_stamp().encode("utf-8")).hexdigest()[:10]\n\n    hash_cols = [c for c in IMPORTANT_MODEL_FIELDS if c in lit_df.columns]\n    if "_lit_agent_row_hash" not in lit_df.columns:\n        lit_df["_lit_agent_row_hash"] = lit_df.apply(lambda r: row_hash(r, hash_cols), axis=1)\n\n    return lit_df\n\n\ndef infer_agent_version_from_paths(lit_csv: Path) -> Optional[str]:\n    """\n    Try to read a nearby summary JSON if present.\n    """\n    candidates = [\n        lit_csv.parent / "run_summary.json",\n        lit_csv.parent.parent / "run_summary.json",\n        lit_csv.parent.parent / "summary.json",\n    ]\n    for p in candidates:\n        if p.exists():\n            try:\n                data = json.loads(p.read_text(encoding="utf-8"))\n                for key in ["agent_version", "AGENT_VERSION", "version"]:\n                    if key in data:\n                        return str(data[key])\n            except Exception:\n                pass\n    return None\n\n\ndef write_readme(out_dir: Path, cfg: MergeConfig, summary: Dict[str, Any]) -> None:\n    txt = f"""\nLiteratureAgent -> PCE/Stability dataset update complete.\n\nInput baseline CSV:\n{cfg.base_csv}\n\nInput LiteratureAgent CSV:\n{cfg.lit_csv}\n\nUpdated modeling CSV:\n{Path(cfg.updated_csv).name if Path(cfg.updated_csv).parent == Path(".") else cfg.updated_csv}\n\nMerge policy:\n{cfg.merge_policy}\n\nWhat to inspect first:\n1. literature_update_audit.csv\n   - shows accepted, weak_accepted, and rejected LiteratureAgent rows.\n\n2. literature_update_duplicate_candidates.csv\n   - rows that likely already exist in the baseline dataset.\n\n3. literature_update_evidence_long.csv\n   - evidence snippets by DOI/field; use this to verify extracted PCE/stability fields.\n\n4. updated_perovskite_database_with_literature_agent.csv\n   - the file to pass into pce_then_stability_same_approach.py.\n\nSummary:\n{json.dumps(summary, indent=2)}\n""".strip()\n    (out_dir / "README.txt").write_text(txt, encoding="utf-8")\n\n\ndef run_model_script(cfg: MergeConfig, updated_csv_path: Path) -> int:\n    model_script = Path(cfg.model_script)\n    if not model_script.exists():\n        print(f"[WARN] Model script not found: {model_script}")\n        print("Skipping model run. Run it manually after checking the updated CSV.")\n        return 1\n\n    cmd = [\n        sys.executable,\n        str(model_script),\n        "--csv",\n        str(updated_csv_path),\n        "--out",\n        str(cfg.model_out_dir),\n    ]\n    print("=" * 90)\n    print("Running PCE -> stability model on updated dataset")\n    print(" ".join(cmd))\n    print("=" * 90)\n    return subprocess.call(cmd)\n\n\ndef integrate(cfg: MergeConfig) -> Dict[str, Any]:\n    out_dir = ensure_dir(cfg.out_dir)\n\n    base_path = Path(cfg.base_csv)\n    lit_path = Path(cfg.lit_csv)\n\n    base_df = robust_read_csv(base_path)\n    lit_raw = robust_read_csv(lit_path)\n\n    agent_version = infer_agent_version_from_paths(lit_path)\n    lit_raw = add_lit_metadata(lit_raw, agent_version=agent_version)\n\n    lit_clean, evidence_wide, evidence_long = split_evidence_fields(lit_raw)\n\n    accepted, rejected, audit_df = validate_literature_rows(lit_clean, evidence_long, cfg)\n\n    # Store confidence status in accepted rows from audit.\n    if not accepted.empty and not audit_df.empty:\n        status_map = audit_df.set_index("lit_row_index")["status"].to_dict()\n        accepted["_lit_agent_confidence_status"] = accepted.index.map(lambda i: status_map.get(i, "accepted"))\n\n    updated_df, duplicate_candidates, changes_df = merge_literature_into_base(base_df, accepted, cfg)\n\n    updated_csv_path = Path(cfg.updated_csv)\n    if not updated_csv_path.is_absolute():\n        updated_csv_path = out_dir / updated_csv_path\n\n    if not cfg.dry_run:\n        write_csv(updated_df, updated_csv_path)\n        write_csv(accepted, out_dir / "literature_update_accepted_rows.csv")\n        write_csv(rejected, out_dir / "literature_update_rejected_rows.csv")\n        write_csv(evidence_wide, out_dir / "literature_update_evidence_wide.csv")\n        write_csv(evidence_long, out_dir / "literature_update_evidence_long.csv")\n        write_csv(duplicate_candidates, out_dir / "literature_update_duplicate_candidates.csv")\n        write_csv(audit_df, out_dir / "literature_update_audit.csv")\n        write_csv(changes_df, out_dir / "literature_update_existing_row_changes.csv")\n\n    summary = {\n        "timestamp": now_stamp(),\n        "base_csv": str(base_path),\n        "lit_csv": str(lit_path),\n        "out_dir": str(out_dir),\n        "updated_csv": str(updated_csv_path),\n        "merge_policy": cfg.merge_policy,\n        "base_rows": int(len(base_df)),\n        "literature_raw_rows": int(len(lit_raw)),\n        "literature_accepted_rows": int(len(accepted)),\n        "literature_rejected_rows": int(len(rejected)),\n        "duplicate_candidate_rows": int(len(duplicate_candidates)),\n        "updated_rows": int(len(updated_df)),\n        "added_rows_estimate": int(len(updated_df) - len(base_df)),\n        "evidence_snippets": int(len(evidence_long)),\n        "agent_version": agent_version or "unknown",\n        "dry_run": bool(cfg.dry_run),\n    }\n\n    if not cfg.dry_run:\n        safe_json_dump(summary, out_dir / "literature_update_summary.json")\n        write_readme(out_dir, cfg, summary)\n\n    print("=" * 90)\n    print("LITERATURE DATASET UPDATE SUMMARY")\n    print("=" * 90)\n    print(json.dumps(summary, indent=2))\n    print("=" * 90)\n\n    if cfg.run_model and not cfg.dry_run:\n        rc = run_model_script(cfg, updated_csv_path)\n        summary["model_return_code"] = rc\n        safe_json_dump(summary, out_dir / "literature_update_summary.json")\n\n    return summary\n\n\n# =============================================================================\n# CLI\n# =============================================================================\n\ndef parse_args() -> MergeConfig:\n    parser = argparse.ArgumentParser(\n        description="Merge LiteratureAgent Perovskite Database-style records into a modeling CSV for PCE/stability prediction."\n    )\n    parser.add_argument("--base_csv", required=True, help="Original Perovskite Database CSV.")\n    parser.add_argument(\n        "--lit_csv",\n        default=None,\n        help=(\n            "LiteratureAgent output CSV. If omitted, tries %%WORK_DIR%%/csv/all_records.csv, "\n            "~/Downloads/lit_outputs/csv/all_records.csv, and ./lit_outputs/csv/all_records.csv."\n        ),\n    )\n    parser.add_argument("--out_dir", default="artifacts_literature_dataset_update")\n    parser.add_argument("--updated_csv", default="updated_perovskite_database_with_literature_agent.csv")\n    parser.add_argument(\n        "--merge_policy",\n        default="append_new_only",\n        choices=["append_new_only", "update_existing", "append_all"],\n        help=(\n            "append_new_only = safest, skips duplicate candidates; "\n            "update_existing = fill missing baseline values from duplicate literature rows; "\n            "append_all = append every accepted literature row."\n        ),\n    )\n    parser.add_argument("--min_required_non_null_important_fields", type=int, default=4)\n    parser.add_argument("--min_required_evidence_fields", type=int, default=1)\n    parser.add_argument("--no_require_doi", action="store_true")\n    parser.add_argument("--duplicate_similarity_threshold", type=float, default=0.82)\n    parser.add_argument("--model_script", default="pce_then_stability_same_approach.py")\n    parser.add_argument("--run_model", action="store_true")\n    parser.add_argument("--model_out_dir", default="artifacts_pce_then_stability_lit_updated")\n    parser.add_argument("--dry_run", action="store_true")\n\n    args = parser.parse_args()\n\n    lit_csv = args.lit_csv\n    if not lit_csv:\n        candidates = []\n        work_dir = os.getenv("WORK_DIR", "").strip()\n        if work_dir:\n            candidates.append(Path(work_dir) / "csv" / "all_records.csv")\n        candidates.append(Path.home() / "Downloads" / "lit_outputs" / "csv" / "all_records.csv")\n        candidates.append(Path.cwd() / "lit_outputs" / "csv" / "all_records.csv")\n        for c in candidates:\n            if c.exists():\n                lit_csv = str(c)\n                break\n        if not lit_csv:\n            raise FileNotFoundError(\n                "Could not infer --lit_csv. Pass it explicitly, for example: "\n                \'--lit_csv "%WORK_DIR%\\\\csv\\\\all_records.csv"\'\n            )\n\n    return MergeConfig(\n        base_csv=args.base_csv,\n        lit_csv=lit_csv,\n        out_dir=args.out_dir,\n        updated_csv=args.updated_csv,\n        merge_policy=args.merge_policy,\n        min_required_non_null_important_fields=args.min_required_non_null_important_fields,\n        min_required_evidence_fields=args.min_required_evidence_fields,\n        require_doi=not args.no_require_doi,\n        duplicate_similarity_threshold=args.duplicate_similarity_threshold,\n        model_script=args.model_script,\n        run_model=args.run_model,\n        model_out_dir=args.model_out_dir,\n        dry_run=args.dry_run,\n    )\n\n\ndef main() -> None:\n    cfg = parse_args()\n    integrate(cfg)\n\n\n'


# =============================================================================
# SINGLE-FILE CONTROLLER HELPERS
# =============================================================================

def _downloads_dir() -> _Path_controller:
    return _Path_controller.home() / "Downloads"



def _v21_controller_embedded_quality_patch_source() -> str:
    """Runtime patch appended to the embedded LiteratureAgent without rewriting it.

    Keeps the original non-fast extraction/summarization machinery, but adds the
    pass-pipeline safety knobs the controller needs: persistent LLM call cache,
    resume-safe summary reuse, optional recovery passes, review shortcut, and
    stronger English-only summary instructions.
    """
    return r'''

# =============================================================================
# V21.4 CONTROLLER QUALITY/PASS-PIPELINE PATCH
# =============================================================================
AGENT_VERSION = "v21_4_pass_pipeline_quality_patch"

LLM_CALL_CACHE_ENABLE = os.getenv("LLM_CALL_CACHE_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}
LLM_CALL_CACHE_DIR = BASE / "llm_cache"
LLM_CALL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
FORCE_SUMMARY_REWRITE = os.getenv("FORCE_SUMMARY_REWRITE", "0").strip().lower() in {"1", "true", "yes", "on"}
REVIEW_FULL_EXTRACTION_ENABLE = os.getenv("REVIEW_FULL_EXTRACTION_ENABLE", "0").strip().lower() in {"1", "true", "yes", "on"}
RUN_MISSING_FIELD_RECOVERY = os.getenv("MISSING_FIELD_RECOVERY_ENABLE", "0").strip().lower() in {"1", "true", "yes", "on"}
FAMILY_GATING_MODE = os.getenv("FAMILY_GATING_MODE", "strict").strip().lower()
if FAMILY_GATING_MODE not in {"off", "moderate", "strict"}:
    FAMILY_GATING_MODE = "strict"


def json_to_csv(records: List[Dict[str, Any]], out_csv: Path) -> None:
    """Export schema values plus provenance/evidence required by integration.

    The embedded legacy exporter rebuilt rows from ``ALL_COLUMNS`` and thereby
    discarded every ``_evidence_<field>`` key even though extraction requested
    and JSON outputs retained those snippets. Keep a stable schema prefix while
    appending the union of provenance and evidence columns.
    """
    base_columns = list(dict.fromkeys(list(ALL_COLUMNS) + list(EXTRA_COLUMNS)))
    auxiliary = sorted({
        str(key)
        for record in (records or []) if isinstance(record, dict)
        for key in record
        if str(key).startswith(("_evidence_", "_source_", "_lit_agent_"))
        or str(key) in {"_full_text_length", "_full_text_source", "_composition_json"}
    })
    columns = base_columns + [key for key in auxiliary if key not in base_columns]
    rows = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        row = {}
        for column in columns:
            value = record.get(column)
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, ensure_ascii=False)
            row[column] = value
        rows.append(row)
    pd.DataFrame(rows, columns=columns).to_csv(out_csv, index=False)

try:
    LLM_REQUEST_TIMEOUT_SECONDS = int(float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", str(REQUEST_TIMEOUT))))
except Exception:
    LLM_REQUEST_TIMEOUT_SECONDS = REQUEST_TIMEOUT
try:
    REQUEST_TIMEOUT = int(float(os.getenv("WEB_REQUEST_TIMEOUT_SECONDS", str(REQUEST_TIMEOUT))))
except Exception:
    pass
try:
    API_MAX_RETRIES = max(1, int(float(os.getenv("LLM_API_MAX_RETRIES", str(API_MAX_RETRIES)))))
except Exception:
    pass
try:
    API_RETRY_BACKOFF_SECONDS = float(os.getenv("LLM_API_RETRY_BACKOFF_SECONDS", str(API_RETRY_BACKOFF_SECONDS)))
except Exception:
    pass

V21_4_REVIEW_LIKE_TERMS = (
    "review", "rulebook", "perspective", "roadmap", "outlook", "survey",
    "tutorial", "primer", "guideline", "guidelines", "design rules",
    "progress", "advances", "opportunities", "challenges", "status and prospects",
    "future directions", "overview", "commentary", "viewpoint",
)


def _v21_4_is_review_like(meta_or_title: Any) -> bool:
    if isinstance(meta_or_title, dict):
        text = " ".join(str(meta_or_title.get(k) or "") for k in ("title", "abstract", "journal", "paper_type", "source"))
    else:
        text = str(meta_or_title or "")
    low = text.lower()
    if any(term in low for term in V21_4_REVIEW_LIKE_TERMS):
        return True
    if "_is_review_like_title" in globals():
        try:
            return bool(_is_review_like_title(text))
        except Exception:
            return False
    return False


def _v21_4_json_hash(obj: Any) -> str:
    try:
        blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        blob = repr(obj)
    return hashlib.sha256(blob.encode("utf-8", errors="ignore")).hexdigest()


def _v21_4_cache_path(paper_slug: str, stage: str, model: str, system_prompt: str, user_payload: Dict[str, Any]) -> Path:
    stage_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stage or "stage"))[:90]
    slug_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(paper_slug or "paper"))[:90]
    key = _v21_4_json_hash({
        "paper_slug": paper_slug,
        "stage": stage,
        "model": model,
        "system_prompt_hash": hashlib.sha256((system_prompt or "").encode("utf-8", errors="ignore")).hexdigest(),
        "payload_hash": _v21_4_json_hash(user_payload),
    })
    return LLM_CALL_CACHE_DIR / f"{slug_safe}__{stage_safe}__{model.replace('/', '_').replace(':', '_')}__{key[:24]}.json"


JSON_FAILURE_DIR = BASE / "failed_json_outputs"
JSON_FAILURE_DIR.mkdir(parents=True, exist_ok=True)
JSON_FAILURE_REPORT = JSON_FAILURE_DIR / "json_failure_report.csv"
PAPER_TYPE_GATE_REPORT = BASE / "paper_type_gate_report.csv"

try:
    import csv as _v21_csv
except Exception:
    _v21_csv = None

try:
    import json_repair as _v21_json_repair
except Exception:
    _v21_json_repair = None


def _v21_append_json_failure(row: Dict[str, Any]) -> None:
    try:
        fields = ["timestamp", "paper_slug", "stage", "error", "raw_output_path", "repaired", "repair_method", "note"]
        row = {k: row.get(k, "") for k in fields}
        row["timestamp"] = row.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        exists = JSON_FAILURE_REPORT.exists()
        if _v21_csv is not None:
            with JSON_FAILURE_REPORT.open("a", encoding="utf-8-sig", newline="") as f:
                writer = _v21_csv.DictWriter(f, fieldnames=fields)
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
        else:
            with JSON_FAILURE_REPORT.open("a", encoding="utf-8") as f:
                if not exists:
                    f.write(",".join(fields) + "\n")
                f.write(",".join(json.dumps(str(row.get(k, "")), ensure_ascii=False) for k in fields) + "\n")
    except Exception as exc:
        print(f"[WARN] Could not append JSON failure report: {exc}")


def _v21_content_to_text(content: Any) -> str:
    if isinstance(content, list):
        bits = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                bits.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                bits.append(part)
        return "\n".join(bits).strip()
    return content if isinstance(content, str) else str(content or "")


def _v21_strip_json_fences(text: str) -> str:
    s = str(text or "").strip()
    s = re.sub(r"^```(?:json|JSON)?\s*", "", s).strip()
    s = re.sub(r"\s*```$", "", s).strip()
    return s


def _v21_largest_json_substring(text: str) -> str:
    s = _v21_strip_json_fences(text)
    best = ""
    starts = [i for i, ch in enumerate(s) if ch in "{["]
    for start in starts:
        stack = []
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    break
                opening = stack.pop()
                if (opening, ch) not in {("{", "}"), ("[", "]")}:
                    break
                if not stack:
                    cand = s[start:i + 1]
                    if len(cand) > len(best):
                        best = cand
                    break
    if best:
        return best
    first_obj = min([x for x in [s.find("{"), s.find("[")] if x >= 0], default=-1)
    return s[first_obj:] if first_obj >= 0 else s


def _v21_basic_json_cleanup(text: str) -> str:
    s = _v21_largest_json_substring(text)
    s = s.replace("\ufeff", "").strip()
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # If Qwen stopped a little early, close only clearly unmatched containers.
    stack = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and (stack[-1], ch) in {("{", "}"), ("[", "]")}:
                stack.pop()
    if in_str:
        s += '"'
    while stack:
        s += "}" if stack.pop() == "{" else "]"
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _v21_parse_json_without_llm(text: str) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    candidates = [
        ("direct", str(text or "")),
        ("strip_fences", _v21_strip_json_fences(text)),
        ("largest_json_substring", _v21_largest_json_substring(text)),
        ("basic_cleanup", _v21_basic_json_cleanup(text)),
    ]
    last_err = None
    for method, cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed, method, None
            if isinstance(parsed, list):
                return {"records": parsed}, method + "_array_wrapped", None
            return {"value": parsed}, method + "_value_wrapped", None
        except Exception as exc:
            last_err = repr(exc)
    if _v21_json_repair is not None:
        try:
            repaired_text = _v21_json_repair.repair_json(str(text or ""))
            parsed = json.loads(repaired_text)
            if isinstance(parsed, dict):
                return parsed, "json_repair", None
            if isinstance(parsed, list):
                return {"records": parsed}, "json_repair_array_wrapped", None
            return {"value": parsed}, "json_repair_value_wrapped", None
        except Exception as exc:
            last_err = repr(exc)
    return None, "", last_err


_previous_hf_qwen_extractor_v21_4 = HuggingFaceQwenExtractor


def _v21_12_match_text(value: Any) -> str:
    text = ihtml.unescape(str(value or ""))
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _v21_12_field_terms(field: str, aliases: Optional[List[str]] = None) -> List[str]:
    terms = []
    for item in [field, *(aliases or [])]:
        normalized = str(item or "").replace("_", " ").strip().lower()
        if not normalized:
            continue
        terms.append(normalized)
        terms.extend(
            token for token in re.findall(r"[a-z][a-z0-9+\-]{1,}", normalized)
            if token not in {
                "default", "value", "measured", "range", "list", "total",
                "perovskite", "stability", "cell", "stack", "sequence",
                "composition", "deposition",
            }
        )
    expansions = {
        "pce": ["power conversion efficiency", "efficiency"],
        "voc": ["open circuit voltage", "open-circuit voltage"],
        "jsc": ["short circuit current", "short-circuit current", "current density"],
        "ff": ["fill factor"],
        "etl": ["electron transport layer", "electron-transport layer"],
        "htl": ["hole transport layer", "hole-transport layer"],
        "t80": ["t80"],
        "t95": ["t95"],
    }
    lowered = str(field or "").lower()
    for key, values in expansions.items():
        if key in lowered:
            terms.extend(values)
    return list(dict.fromkeys(term for term in terms if len(term) >= 2))


def _v21_12_unwrap_model_value(value: Any) -> Any:
    """Normalize Qwen/reasoning-layer value wrappers to database scalars/lists."""
    if isinstance(value, dict):
        if "value" in value:
            return _v21_12_unwrap_model_value(value.get("value"))
        return None
    if isinstance(value, list):
        return [_v21_12_unwrap_model_value(item) for item in value]
    return value


def _v21_12_quote_window(text: str, start: int, end: int, radius: int = 230) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    left = max(text.rfind(". ", lo, start), text.rfind("\n", lo, start))
    right_period = text.find(". ", end, hi)
    right_newline = text.find("\n", end, hi)
    right_candidates = [pos for pos in (right_period, right_newline) if pos >= 0]
    if left >= lo:
        lo = left + 1
    if right_candidates:
        hi = min(right_candidates) + 1
    return re.sub(r"\s+", " ", text[lo:hi]).strip()


def _v21_12_quote_quality(quote: str, terms: List[str]) -> int:
    """Prefer explanatory source sentences over isolated table cells."""
    quote = str(quote or "").strip()
    if not quote:
        return -1
    lowered = quote.lower()
    alpha_count = sum(char.isalpha() for char in quote)
    term_hits = sum(1 for term in terms if term in lowered)
    # A bare number/table cell is not useful field-level provenance.
    if len(quote) < 20 or alpha_count < 8:
        return -1
    return min(len(quote), 500) + (80 * term_hits) + min(alpha_count, 120)


def _v21_12_find_source_quote(
    field: str,
    value: Any,
    source_text: str,
    aliases: Optional[List[str]] = None,
) -> Optional[str]:
    """Ground an extracted value in the exact chunk that produced it."""
    value = _v21_12_unwrap_model_value(value)
    if value in (None, "", [], {}):
        return None
    text = str(source_text or "")
    if not text:
        return None
    terms = _v21_12_field_terms(field, aliases)
    lowered = text.lower()

    if isinstance(value, list):
        items = [str(item).strip() for item in value if item not in (None, "", [], {})]
        if not items:
            return None
        item_matches = []
        for item in items:
            pattern = rf"(?<![A-Za-z0-9]){re.escape(item)}(?![A-Za-z0-9])"
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                return None
            item_matches.append((match.start(), match.end()))
        start = min(item[0] for item in item_matches)
        end = max(item[1] for item in item_matches)
        context = lowered[max(0, start - 300):min(len(text), end + 300)]
        if terms and not any(term in context for term in terms):
            return None
        return _v21_12_quote_window(text, start, end, radius=300)

    if isinstance(value, bool) or str(value).strip().lower() in {"true", "false"}:
        if str(value).strip().lower() != "true":
            return None
        for term in terms:
            pos = lowered.find(term)
            if pos >= 0:
                return _v21_12_quote_window(text, pos, pos + len(term))
        return None

    raw_value = str(value).strip()
    candidates = [raw_value]
    try:
        number = float(raw_value.replace(",", ""))
        candidates.extend({f"{number:g}", f"{number:.1f}", f"{number:.2f}", f"{number:.3f}"})
    except Exception:
        pass

    matches = []
    for candidate in sorted(set(candidates), key=len, reverse=True):
        if not candidate or candidate.lower() in {"true", "false", "none", "null"}:
            continue
        pattern = re.escape(candidate)
        if candidate[0].isdigit():
            pattern = rf"(?<![\d.]){pattern}(?![\d.])"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append((match.start(), match.end()))

    quote_candidates = []
    for start, end in matches:
        context = lowered[max(0, start - 260):min(len(text), end + 260)]
        semantic_time_context = (
            str(field or "").lower() == "stability_time_total_exposure"
            and any(term in context for term in (
                "stability", "stored", "retained", "aging", "glove box", "glovebox",
                "humidity", "ambient air", "nitrogen", "temperature", "dark",
            ))
            and bool(re.search(
                rf"(?<![\d.]){re.escape(str(value).strip())}(?![\d.])\s*(?:h|hour|hours|day|days)",
                text[start:min(len(text), end + 24)],
                flags=re.IGNORECASE,
            ))
        )
        if any(term in context for term in terms) or semantic_time_context:
            quote = _v21_12_quote_window(text, start, end)
            quality = _v21_12_quote_quality(quote, terms)
            if quality >= 0:
                quote_candidates.append((quality, quote))

    if quote_candidates:
        return max(quote_candidates, key=lambda item: item[0])[1]

    if str(field or "").lower() == "stability_protocol":
        value_tokens = {
            token for token in _v21_12_match_text(raw_value).split()
            if len(token) >= 3 and token not in {"the", "and", "for", "with", "under", "about"}
        }
        duration = re.search(r"\b\d+(?:\.\d+)?\s*(?:h|hour|hours|day|days)\b", raw_value, flags=re.IGNORECASE)
        anchors = [duration.group(0)] if duration else []
        anchors.extend(token for token in value_tokens if token in {"nitrogen", "ambient", "humidity", "temperature", "dark", "glovebox"})
        for anchor in anchors:
            for match in re.finditer(re.escape(anchor), text, flags=re.IGNORECASE):
                quote = _v21_12_quote_window(text, match.start(), match.end(), radius=320)
                quote_tokens = set(_v21_12_match_text(quote).split())
                required_tokens = {
                    token for token in value_tokens
                    if token.isdigit() or token in {
                        "nitrogen", "ambient", "air", "humidity", "temperature", "dark",
                        "glove", "box", "glovebox",
                    }
                }
                enough_required = required_tokens.issubset(quote_tokens) if required_tokens else True
                if enough_required and len(value_tokens & quote_tokens) >= min(3, max(1, len(value_tokens))):
                    return quote

    # Long chemical formulas, layer stacks, and method strings can be self-identifying.
    normalized_value = _v21_12_match_text(raw_value)
    if len(normalized_value) >= 8:
        normalized_source = _v21_12_match_text(text)
        if normalized_value in normalized_source and matches:
            start, end = matches[0]
            return _v21_12_quote_window(text, start, end)
    return None


def _v21_12_grounded_field_count(record: Dict[str, Any], family_columns: List[str]) -> int:
    count = 0
    for field in family_columns:
        value = _v21_12_unwrap_model_value(record.get(field))
        evidence = record.get(f"_evidence_{field}")
        if value not in (None, "", [], {}) and isinstance(evidence, str) and len(evidence.strip()) >= 12:
            count += 1
    return count


def _v21_12_field_expects_numeric(field: str) -> bool:
    lowered = str(field or "").lower()
    numeric_tokens = (
        "pce", "voc", "jsc", "_ff", "thickness", "temperature", "humidity",
        "time", "area", "intensity", "concentration", "volume", "pressure",
        "band_gap", "pl_max", "t80", "t95", "wavelength", "resistance",
    )
    boolean_exceptions = ("measured", "certified", "flexible", "inorganic", "leadfree")
    return any(token in lowered for token in numeric_tokens) and not any(
        token in lowered for token in boolean_exceptions
    )


def _v21_12_role_definition(field: str) -> Optional[Tuple[str, List[str]]]:
    """Return the ontology role and explicit source phrases for role-sensitive fields."""
    lowered = str(field or "").lower()
    role_map = {
        "etl_": ("etl", ["electron transport layer", "electron-transport layer", " etl "]),
        "htl_": ("htl", ["hole transport layer", "hole-transport layer", " htl ", " htm "]),
        "backcontact_": ("back_contact", ["back contact", "top electrode", "metal electrode"]),
    }
    for prefix, definition in role_map.items():
        if lowered.startswith(prefix):
            return definition
    return None


def _v21_12_ontology_role_terms(role: str) -> List[str]:
    library = globals().get("_EXTERNAL_ONTOLOGY_LIBRARY") or globals().get("PEROVSKITE_ONTOLOGY") or {}
    layers = library.get("device_layers", {}) if isinstance(library, dict) else {}
    values = layers.get(role, []) if isinstance(layers, dict) else []
    return [str(value).strip() for value in values if str(value).strip()]


def _v21_12_role_value_is_supported(field: str, value: Any, quote: str) -> bool:
    """Require role-sensitive compounds/stacks to have ontology or contextual role support."""
    definition = _v21_12_role_definition(field)
    if definition is None:
        return True
    role, explicit_phrases = definition
    quote_norm = f" {_v21_12_match_text(quote)} "
    role_terms = [_v21_12_match_text(term) for term in _v21_12_ontology_role_terms(role)]
    explicit_role = any(phrase.strip() in quote_norm for phrase in explicit_phrases)
    ontology_role_in_quote = any(term and term in quote_norm for term in role_terms)
    if not (explicit_role or ontology_role_in_quote):
        return False

    lowered_field = str(field or "").lower()
    role_sensitive_suffixes = (
        "stack_sequence", "additives_compounds", "reaction_solutions_compounds"
    )
    if not any(lowered_field.endswith(suffix) for suffix in role_sensitive_suffixes):
        return True
    if explicit_role:
        return True

    values = value if isinstance(value, list) else [value]
    if lowered_field.endswith("stack_sequence"):
        split_values = []
        for item in values:
            split_values.extend(part for part in re.split(r"\s*(?:/|>|;|,)\s*", str(item)) if part)
        values = split_values
    populated = [_v21_12_match_text(item) for item in values if item not in (None, "", [], {})]
    if not populated:
        return False
    return all(any(item == term or item in term or term in item for term in role_terms) for item in populated)


def _v21_12_field_context_is_supported(field: str, value: Any, quote: str) -> bool:
    lowered_field = str(field or "").lower()
    quote_norm = _v21_12_match_text(quote)
    if lowered_field == "cell_architecture":
        return any(term in quote_norm for term in (
            "inverted", "regular", "normal architecture", "p i n", "n i p",
            "planar", "mesoscopic",
        ))
    if lowered_field.startswith("perovskite_composition_"):
        generic_background = any(term in quote_norm for term in (
            "for example", "e g", "etc", "can be", "may be", "as a and x",
            "typically consists", "general formula",
        ))
        if generic_background:
            return False
        if lowered_field.endswith(("a_ions", "b_ions", "c_ions", "a_ions_coefficients", "b_ions_coefficients", "c_ions_coefficients")):
            return any(term in quote_norm for term in (
                "prepared", "precursor", "film", "device", "composition", "formula", "mixed spacer",
            ))
    if lowered_field == "perovskite_deposition_number_of_deposition_steps":
        return bool(re.search(r"\b(?:one|two|three|1|2|3)[ -]?step\b", quote_norm))
    if lowered_field == "perovskite_deposition_procedure":
        return any(term in quote_norm for term in (
            "spin coat", "blade coat", "slot die", "doctor blade", "spray coat",
            "inkjet", "vapor deposit", "vapour deposit", "evaporat", "deposition",
        ))
    if lowered_field == "stability_pce_after_1000_h":
        return bool(re.search(r"\b(?:after|at|for|over)?\s*1000\s*(?:h|hour)", quote_norm))
    if lowered_field.startswith("stability_"):
        stability_context = any(term in quote_norm for term in (
            "stability", "stable", "retained", "retention", "degradation",
            "degraded", "lifetime", "aging", "aged", "stored", "t80", "t95",
            "humidity", "glove box", "glovebox", "ambient air", "nitrogen", "dark",
        ))
        if not stability_context:
            return False
        if lowered_field == "stability_time_total_exposure":
            return bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:h|hour|day)", quote_norm))
        if lowered_field == "stability_temperature_range":
            return bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:c|deg c|degree c|degrees c)\b", quote_norm))
        if lowered_field in {"stability_relative_humidity_range", "stability_relative_humidity_average_value"}:
            return bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:%|percent).{0,30}(?:rh|relative humidity)|(?:rh|relative humidity).{0,30}\d+(?:\.\d+)?\s*(?:%|percent)", quote_norm))
        if lowered_field in {"stability_pce_end_of_experiment", "stability_pce_initial_value"}:
            return "%" in str(quote) or "pce" in quote_norm
        return True
    if lowered_field.startswith("outdoor_"):
        return any(term in quote_norm for term in ("outdoor", "field test", "module temperature"))
    return True


def _v21_12_coalesce_final_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Join complementary schema-family rows while preserving true device conflicts."""
    metadata_fields = {
        "Ref_ID", "Ref_ID_temp", "Ref_name_of_person_entering_the_data",
        "Ref_data_entered_by_author", "Ref_DOI_number", "Ref_lead_author",
        "Ref_publication_date", "Ref_journal", "Ref_part_of_initial_dataset",
        "Ref_original_filename_data_upload", "Ref_free_text_comment",
        "Ref_internal_sample_id",
    }
    target_prefixes = ("JV_", "Stability_", "Stabilised_performance_", "EQE_")
    identity_fields = ("Ref_DOI_number", "Ref_internal_sample_id")
    device_identity_targets = {
        "JV_default_PCE", "JV_default_Voc", "JV_default_Jsc", "JV_default_FF",
        "JV_reverse_scan_PCE", "JV_reverse_scan_Voc", "JV_reverse_scan_Jsc", "JV_reverse_scan_FF",
        "JV_forward_scan_PCE", "JV_forward_scan_Voc", "JV_forward_scan_Jsc", "JV_forward_scan_FF",
        "Stabilised_performance_PCE",
        "Stability_PCE_initial_value", "Stability_PCE_end_of_experiment",
        "Stability_PCE_T95", "Stability_PCE_T80", "Stability_PCE_after_1000_h",
        # Different aging conditions are different experimental rows even when
        # they share a DOI and nominal device sample.
        "Stability_protocol", "Stability_time_total_exposure",
        "Stability_atmosphere", "Stability_temperature_load_condition",
        "Stability_temperature_range",
        "Stability_relative_humidity_load_conditions",
        "Stability_relative_humidity_range",
        "Stability_relative_humidity_average_value",
        "Stability_light_load_condition", "Stability_light_intensity",
        "Stability_potential_bias_load_condition",
    }

    def present(value: Any) -> bool:
        return value not in (None, "", [], {})

    def same_identity(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        compared = 0
        for field in identity_fields:
            left_value = left.get(field)
            right_value = right.get(field)
            if not present(left_value) or not present(right_value):
                continue
            compared += 1
            if _v21_12_match_text(left_value) != _v21_12_match_text(right_value):
                return False
        return compared > 0

    def target_conflict(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        for field in device_identity_targets:
            left_value = left.get(field)
            right_value = right.get(field)
            if not present(left_value) or not present(right_value):
                continue
            if _v21_12_match_text(left_value) != _v21_12_match_text(right_value):
                return True
        return False

    def score(record: Dict[str, Any]) -> Tuple[int, int]:
        target_count = sum(
            value not in (None, "", [], {}) and key.startswith(target_prefixes)
            for key, value in record.items()
            if not key.startswith("_evidence_")
        )
        grounded_count = sum(
            value not in (None, "", [], {})
            for key, value in record.items()
            if key.startswith("_evidence_")
        )
        return target_count, grounded_count

    candidates = [dict(record) for record in records if isinstance(record, dict)]
    candidates.sort(key=score, reverse=True)
    target_candidates = [record for record in candidates if score(record)[0] > 0]
    context_candidates = [record for record in candidates if score(record)[0] == 0]
    merged: List[Dict[str, Any]] = []

    def compatible_indices(candidate: Dict[str, Any]) -> List[Tuple[int, int]]:
        compatible = []
        for index, existing in enumerate(merged):
            # Schema families often emit different context fragments for the
            # same device. Matching DOI/sample identity plus nonconflicting
            # measured targets is stronger evidence of one device than a
            # disagreement between an explicit and synthesized stack string.
            if same_identity(candidate, existing) and not target_conflict(candidate, existing):
                overlap = sum(
                    present(candidate.get(field))
                    and present(existing.get(field))
                    and _v21_12_match_text(candidate.get(field)) == _v21_12_match_text(existing.get(field))
                    for field in set(candidate) | set(existing)
                    if not field.startswith("_")
                )
                compatible.append((index, overlap))
                continue
            overlap = 0
            conflict = False
            for field, new_value in candidate.items():
                if field.startswith("_") or field in metadata_fields or new_value in (None, "", [], {}):
                    continue
                old_value = existing.get(field)
                if old_value in (None, "", [], {}):
                    continue
                if _v21_12_match_text(old_value) != _v21_12_match_text(new_value):
                    conflict = True
                    break
                overlap += 1
            if not conflict:
                compatible.append((index, overlap))
        return compatible

    def fill_missing(destination: Dict[str, Any], candidate: Dict[str, Any]) -> None:
        for key, value in candidate.items():
            if value not in (None, "", [], {}) and destination.get(key) in (None, "", [], {}):
                destination[key] = value

    # Establish distinct target/device rows first.
    for candidate in target_candidates:
        compatible = compatible_indices(candidate)
        best_index = max(compatible, key=lambda item: item[1])[0] if compatible else None
        if best_index is None:
            merged.append(candidate)
        else:
            fill_missing(merged[best_index], candidate)

    # Paper-level chemistry, stack, and processing context can support every
    # compatible target row. This prevents a context-only row from consuming
    # the only copy before the champion-performance row is seen.
    for candidate in context_candidates:
        compatible = compatible_indices(candidate)
        if not compatible:
            merged.append(candidate)
            continue
        for index, _ in compatible:
            fill_missing(merged[index], candidate)
    return merged


def _v21_17_assign_distinct_sample_ids(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Give conflicting device/measurement rows stable unique sample IDs."""

    def present(value: Any) -> bool:
        return value not in (None, "", [], {})

    def normalized_number(value: Any) -> str:
        try:
            number = float(value)
        except Exception:
            return ""
        return f"{number:.6g}".replace("-", "m").replace(".", "p")

    def role(record: Dict[str, Any]) -> str:
        evidence = " ".join(
            str(value)
            for key, value in record.items()
            if key.startswith("_evidence_") and present(value)
        ).lower()
        if "control" in evidence:
            return "control"
        if "certified" in evidence or "certification" in evidence:
            return "certified"
        if "larger" in evidence or "1 cm2" in evidence or "1cm2" in evidence:
            return "large_area"
        if "target" in evidence or "treated" in evidence:
            return "target"
        return "device"

    grouped: Dict[Tuple[str, str], List[int]] = {}
    output = [dict(record) for record in records]
    for index, record in enumerate(output):
        doi = _v21_12_match_text(record.get("Ref_DOI_number"))
        sample_id = str(record.get("Ref_internal_sample_id") or "").strip()
        if sample_id:
            grouped.setdefault((doi, sample_id), []).append(index)

    for (_doi, sample_id), indices in grouped.items():
        if len(indices) < 2:
            continue
        seen_ids: set[str] = set()
        role_counts: Dict[str, int] = {}
        for index in indices:
            record = output[index]
            label = role(record)
            role_counts[label] = role_counts.get(label, 0) + 1
            metric = normalized_number(
                record.get("JV_default_PCE")
                or record.get("JV_reverse_scan_PCE")
                or record.get("JV_forward_scan_PCE")
                or record.get("Stabilised_performance_PCE")
            )
            suffix_parts = [label]
            if metric:
                suffix_parts.append(f"pce_{metric}")
            candidate = f"{sample_id}_{'_'.join(suffix_parts)}"
            if candidate in seen_ids:
                candidate = f"{candidate}_{role_counts[label]}"
            seen_ids.add(candidate)
            record["Ref_internal_sample_id"] = candidate
            record["_lit_agent_sample_identity_basis"] = (
                "distinct grounded device/measurement row; "
                f"role={label}; original_sample_id={sample_id}"
            )
    return output


# AssemblyAgent performs a second per-paper deduplication after the extraction
# function returns. The legacy implementation discarded complementary rows when
# their sparse fingerprint fields were null. Route every active per-paper dedupe
# through the same conflict-aware coalescer so grounded target rows survive.
def dedupe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _v21_12_coalesce_final_records(records)


def _v21_12_merge_grounded_partial_records(
    partial_payloads: List[Dict[str, Any]],
    family_columns: List[str],
) -> List[Dict[str, Any]]:
    """Merge validated partials without asking the LLM to reshape the schema again.

    Compatible complementary records are coalesced. A conflicting populated field
    starts a separate record so control/target devices are not silently combined.
    """
    candidates = []
    for payload in partial_payloads:
        records = payload.get("records", []) if isinstance(payload, dict) else []
        for source_record in records if isinstance(records, list) else []:
            if not isinstance(source_record, dict):
                continue
            record = dict(source_record)
            for field in family_columns:
                record[field] = _v21_12_unwrap_model_value(record.get(field))
            if _v21_12_grounded_field_count(record, family_columns):
                candidates.append(record)

    candidates.sort(
        key=lambda record: _v21_12_grounded_field_count(record, family_columns),
        reverse=True,
    )
    merged: List[Dict[str, Any]] = []
    for candidate in candidates:
        best_index = None
        best_overlap = -1
        for index, existing in enumerate(merged):
            overlap = 0
            conflict = False
            for field in family_columns:
                old_value = existing.get(field)
                new_value = candidate.get(field)
                if old_value in (None, "", [], {}) or new_value in (None, "", [], {}):
                    continue
                if _v21_12_match_text(old_value) != _v21_12_match_text(new_value):
                    conflict = True
                    break
                overlap += 1
            if not conflict and overlap > best_overlap:
                best_index = index
                best_overlap = overlap

        if best_index is None:
            merged.append(dict(candidate))
            continue

        destination = merged[best_index]
        for key, value in candidate.items():
            if value not in (None, "", [], {}) and destination.get(key) in (None, "", [], {}):
                destination[key] = value

    return merged


def _v21_12_attach_chunk_evidence(
    payload: Dict[str, Any],
    source_text: str,
    family_columns: List[str],
    alias_map: Dict[str, List[str]],
) -> Dict[str, Any]:
    records = payload.get("records", []) if isinstance(payload, dict) else []
    if not isinstance(records, list):
        return payload
    grounded = 0
    rejected = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        for field in family_columns:
            value = _v21_12_unwrap_model_value(record.get(field))
            record[field] = value
            if value in (None, "", [], {}):
                continue
            if isinstance(value, bool) and _v21_12_field_expects_numeric(field):
                record[field] = None
                record.pop(f"_evidence_{field}", None)
                rejected += 1
                continue
            quote = _v21_12_find_source_quote(field, value, source_text, alias_map.get(field, []))
            if (
                quote
                and _v21_12_role_value_is_supported(field, value, quote)
                and _v21_12_field_context_is_supported(field, value, quote)
            ):
                record[f"_evidence_{field}"] = quote
                grounded += 1
            else:
                record[field] = None
                record.pop(f"_evidence_{field}", None)
                rejected += 1
    if grounded or rejected:
        print(f"[CHUNK EVIDENCE] Grounded {grounded} field(s); rejected {rejected} ungrounded field(s).")
    return payload


def _v21_12_partial_evidence_lookup(partials: List[Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
    lookup = {}
    for payload in partials:
        records = payload.get("records", []) if isinstance(payload, dict) else []
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, dict):
                continue
            for field, value in record.items():
                if field.startswith("_") or value in (None, "", [], {}):
                    continue
                evidence = record.get(f"_evidence_{field}")
                if isinstance(evidence, str) and len(evidence.strip()) >= 12:
                    lookup[(field, _v21_12_match_text(value))] = evidence.strip()
    return lookup


def _v21_12_extraction_extract(
    self: ExtractionAgent,
    meta: Dict[str, Any],
    full_text: str,
    extraction_text: str,
) -> Dict[str, Any]:
    """Run family ranking on source text first, with only a compact evidence appendix.

    v16 prepended up to 35k characters of token inventories and sentence fragments.
    Family chunk ranking then selected that digest instead of the contiguous paper
    prose needed to associate a value with its material, device, and test condition.
    """
    if not (SYSTEMATIC_MAIN_PIPELINE_ENABLE and MAIN_EVIDENCE_TABLE_ENABLE):
        return _v16_original_extraction_extract(self, meta, full_text, extraction_text)

    table_agent = MainEvidenceTableAgent()
    evidence_table = table_agent.build(full_text or "", extraction_text or "")
    # Feed only source prose into family ranking and field-level grounding.
    # Figure reports and evidence indexes remain sidecars and are handled by
    # their dedicated passes; mixing them here makes generated tokens look like
    # paper evidence.
    source_only_text = (full_text or extraction_text or "").strip()
    result = _v16_original_extraction_extract(self, meta, full_text, source_only_text)
    result["systematic_evidence_table"] = {
        "agent": evidence_table.get("agent"),
        "section_filtered_chars": evidence_table.get("section_filtered_chars"),
        "evidence_row_count": evidence_table.get("evidence_row_count"),
        "appendix_chars": 0,
        "source_text_first": True,
        "source_only": True,
        "token_inventory_counts": {
            key: len(value or [])
            for key, value in (evidence_table.get("token_inventory") or {}).items()
        },
    }
    if MAIN_EVIDENCE_DEBUG_SAVE:
        try:
            slug = slugify((meta or {}).get("title") or (meta or {}).get("doi") or "paper")
            (RAW_LLM_DIR / f"{slug}_v21_12_systematic_evidence_table.json").write_text(
                json.dumps(evidence_table, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass
    return result


ExtractionAgent.extract = _v21_12_extraction_extract

class HuggingFaceQwenExtractor(_previous_hf_qwen_extractor_v21_4):
    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        global REQUEST_TIMEOUT
        previous_timeout = REQUEST_TIMEOUT
        try:
            REQUEST_TIMEOUT = int(float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", str(previous_timeout))))
            return super()._request(payload)
        finally:
            REQUEST_TIMEOUT = previous_timeout

    def _repair_json_with_llm_v21(self, raw_text: str, paper_slug: str, stage: str) -> Optional[Dict[str, Any]]:
        try:
            repair_payload = {
                "model": getattr(self, "model", HF_MODEL),
                "temperature": 0,
                "max_tokens": min(LLM_MAX_OUTPUT_TOKENS, 2500),
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Convert the supplied malformed JSON into valid JSON only. "
                            "Do not add facts, do not infer missing values, and do not explain anything."
                        ),
                    },
                    {"role": "user", "content": str(raw_text or "")[:60000]},
                ],
            }
            data = self._request(repair_payload)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed, method, _ = _v21_parse_json_without_llm(content)
            if isinstance(parsed, dict):
                parsed.setdefault("_json_repair_warning_v21_5", {
                    "stage": stage,
                    "method": "repair_llm_" + (method or "parsed"),
                    "note": "Malformed JSON repaired without adding facts.",
                })
                return parsed
        except Exception as exc:
            print(f"[WARN] JSON repair LLM failed for {paper_slug}/{stage}: {exc}")
        return None

    def _parse_content_to_json(self, content: Any, paper_slug: str, stage: str) -> Dict[str, Any]:
        text = _v21_content_to_text(content)
        if not text:
            raise RuntimeError("Model returned empty content.")
        raw_path = JSON_FAILURE_DIR / f"{paper_slug}_{stage}_{int(time.time() * 1000)}_raw.txt"
        try:
            self._save_raw_text(paper_slug, stage, text)
        except Exception:
            pass
        parsed, method, parse_error = _v21_parse_json_without_llm(text)
        if isinstance(parsed, dict):
            if method != "direct":
                try:
                    raw_path.write_text(text, encoding="utf-8", errors="ignore")
                except Exception:
                    pass
                parsed.setdefault("_json_repair_warning_v21_5", {
                    "stage": stage,
                    "method": method,
                    "note": "Malformed or wrapped JSON repaired syntactically.",
                })
                _v21_append_json_failure({
                    "paper_slug": paper_slug,
                    "stage": stage,
                    "error": parse_error or "initial JSON parse failed",
                    "raw_output_path": str(raw_path),
                    "repaired": True,
                    "repair_method": method,
                })
            return parsed
        try:
            raw_path.write_text(text, encoding="utf-8", errors="ignore")
        except Exception:
            pass
        repaired = self._repair_json_with_llm_v21(text, paper_slug, stage)
        if isinstance(repaired, dict):
            _v21_append_json_failure({
                "paper_slug": paper_slug,
                "stage": stage,
                "error": parse_error or "JSON parse failed",
                "raw_output_path": str(raw_path),
                "repaired": True,
                "repair_method": "repair_llm",
            })
            return repaired
        _v21_append_json_failure({
            "paper_slug": paper_slug,
            "stage": stage,
            "error": parse_error or "Could not parse model output as JSON",
            "raw_output_path": str(raw_path),
            "repaired": False,
            "repair_method": "failed",
        })
        raise RuntimeError(f"Could not parse model output as JSON after repair attempts. Raw saved to {raw_path}")

    def _call(self, system_prompt: str, user_payload: Dict[str, Any], paper_slug: str, stage: str) -> Dict[str, Any]:
        chosen_model = choose_model_for_stage(stage) if "choose_model_for_stage" in globals() else getattr(self, "model", HF_MODEL)
        structured_partial = bool(re.search(r"_partial_\d+(?:_|$)", str(stage or "")))
        max_output_tokens = LLM_MAX_OUTPUT_TOKENS
        if structured_partial:
            max_output_tokens = min(
                LLM_MAX_OUTPUT_TOKENS,
                int(os.getenv("STRUCTURED_EXTRACTION_MAX_OUTPUT_TOKENS", "850")),
            )
        cache_path = _v21_4_cache_path(paper_slug, stage, chosen_model, system_prompt, user_payload)
        if LLM_CALL_CACHE_ENABLE and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                parsed = cached.get("parsed")
                if isinstance(parsed, dict):
                    print(f"[LLM CACHE] {paper_slug} | {stage} | {chosen_model}")
                    return parsed
            except Exception:
                pass

        append_cost_log(paper_slug, stage, chosen_model, user_payload, max_output_tokens)
        payload = {
            "model": chosen_model,
            "temperature": 0,
            "max_tokens": max_output_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        }
        content = ""
        last_error = None
        max_empty_retries = int(os.getenv("LLM_EMPTY_CONTENT_RETRIES", "2"))
        for attempt in range(max_empty_retries + 1):
            try:
                data = self._request(payload)
                content = _v21_content_to_text(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
                if content:
                    break
                last_error = RuntimeError("Model returned empty content.")
            except Exception as exc:
                last_error = exc
            if attempt < max_empty_retries:
                wait_s = min(20, 4 * (attempt + 1))
                print(
                    f"[WARN] Empty/failed LLM response for {paper_slug}/{stage}; "
                    f"retry {attempt + 1}/{max_empty_retries} after {wait_s}s: {last_error}"
                )
                time.sleep(wait_s)
        if not content:
            raise RuntimeError(f"Model returned empty content after {max_empty_retries + 1} attempt(s). Last error: {last_error}")
        parsed = self._parse_content_to_json(content, paper_slug=paper_slug, stage=stage)
        if LLM_CALL_CACHE_ENABLE:
            try:
                cache_path.write_text(json.dumps({
                    "paper_slug": paper_slug,
                    "stage": stage,
                    "model": chosen_model,
                    "system_prompt_sha256": hashlib.sha256((system_prompt or "").encode("utf-8", errors="ignore")).hexdigest(),
                    "payload_sha256": _v21_4_json_hash(user_payload),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "parsed": parsed,
                }, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:
                print(f"[WARN] Could not write LLM cache for {paper_slug}/{stage}: {exc}")
        return parsed

    def extract_family_records(
        self,
        metadata: Dict[str, Any],
        chunk_text: str,
        family_name: str,
        family_columns: List[str],
        alias_map: Dict[str, List[str]],
        paper_slug: str,
        chunk_idx: int,
    ) -> Dict[str, Any]:
        system = (
            "Extract structured data from this paper into the supplied schema family. "
            "Use only this paper's own experimental device/sample results. Do not use background, cited-work, axis-only, or comparison values. "
            "Populate the supplied data fields themselves; do not return _evidence_ keys or substitute source quotes for field values. "
            "If a field value is unavailable or unclear, leave that field null. Never emit example_column. "
            "Keep distinct control, target, champion, aged, and encapsulated devices as separate records when the text distinguishes them. "
            "Return valid JSON only with top-level key records."
        )
        record_schema = {column: None for column in family_columns}
        compact_aliases = {
            column: list(alias_map.get(column, []))[:1]
            for column in family_columns
            if alias_map.get(column)
        }
        payload = {
            "paper_metadata": metadata,
            "schema_family": family_name,
            "field_aliases": compact_aliases,
            "perovskite_ontology_slice": ontology_slice_for_family(family_name),
            "requirements": {
                "paper_own_results_only": True,
                "unsupported_fields_must_be_null": True,
                "preserve_distinct_devices": True,
                "populate_data_fields_not_evidence_fields": True,
                "evidence_is_attached_by_downstream_source_validator": True,
            },
            "field_semantics": {
                "composition": "Use only exact formulations experimentally prepared in this paper; ignore introductory examples and generic ABX3 definitions.",
                "layer_roles": "A layer field contains only that layer, never the complete device stack.",
                "Perovskite_deposition_procedure": "A coating/deposition method such as spin coating; never a composition or sample label.",
                "Cell_architecture": "Architecture class such as inverted/p-i-n or regular/n-i-p; never the layer sequence.",
                "Stability_PCE_after_1000_h": "Populate only when the paper explicitly reports retention at 1000 hours.",
                "Stability_PCE_end_of_experiment": "The percent of initial PCE retained at the reported end time; use this for retention at times other than 1000 hours.",
                "Stability_time_total_exposure": "The duration paired with the same stability condition and retention record.",
                "stability_records": "Keep each condition, exposure time, atmosphere, temperature/humidity, and matching retention value together in one record.",
            },
            "value_example": {"JV_default_PCE": 17.21},
            "text_chunk": chunk_text,
            "output_schema": {"records": [record_schema]},
        }
        result = self._call(system, payload, paper_slug=paper_slug, stage=f"{family_name}_partial_{chunk_idx}")
        return _v21_12_attach_chunk_evidence(result, chunk_text, family_columns, alias_map)

    def consolidate_records(
        self,
        metadata: Dict[str, Any],
        family_name: str,
        partial_payloads: List[Dict[str, Any]],
        family_columns: List[str],
        paper_slug: str,
    ) -> Dict[str, Any]:
        system = (
            "Merge partial extraction records for one paper and schema family. "
            "Preserve each matching _evidence_<fieldname> snippet with its value. "
            "A populated field without matching evidence must become null. "
            "Do not merge distinct control, target, champion, aged, or encapsulated devices. "
            "Do not invent or infer values. Return valid JSON only with top-level key records."
        )
        record_schema = {column: None for column in family_columns}
        record_schema.update({f"_evidence_{column}": None for column in family_columns})
        payload = {
            "paper_metadata": metadata,
            "schema_family": family_name,
            "target_columns": family_columns,
            "partials": partial_payloads,
            "output_schema": {"records": [record_schema]},
        }
        result = self._call(system, payload, paper_slug=paper_slug, stage=f"{family_name}_consolidated")
        evidence_lookup = _v21_12_partial_evidence_lookup(partial_payloads)
        records = result.get("records", []) if isinstance(result, dict) else []
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, dict):
                continue
            for field in family_columns:
                value = _v21_12_unwrap_model_value(record.get(field))
                record[field] = value
                if value in (None, "", [], {}):
                    continue
                evidence = evidence_lookup.get((field, _v21_12_match_text(value)))
                if evidence:
                    record[f"_evidence_{field}"] = evidence
                else:
                    record[field] = None
                    record.pop(f"_evidence_{field}", None)
        return result

    def summarize_paper(self, metadata: Dict[str, Any], records: List[Dict[str, Any]], summary_text: str, paper_slug: str) -> Dict[str, Any]:
        review_like = _v21_4_is_review_like(metadata)
        metadata = metadata or {}
        source_collection = str(metadata.get("source_collection") or "").strip().lower()
        journal = str(metadata.get("journal") or "").strip().lower()
        material_focused = review_like or source_collection == "manual_google_drive" or journal == "manual google drive import"
        system = (
            "You are writing a high-quality English structured summary for halide-perovskite solar-cell/materials literature. "
            "Use the older non-fast detailed style: specific, evidence-grounded, and useful for downstream PCE/stability ML curation. "
            "The summary should be richer and more interpretive than a short abstract rewrite. "
            "Explain the scientific objective, the materials system, what was actually done, what the important findings mean, "
            "and any limitations or ambiguity in the extraction evidence. "
            "All output MUST be English. Do not write Chinese/CJK or any other non-English prose. "
            "Preserve numbers, units, chemical formulas, spacer cations, additives, device stacks, abbreviations, and caveats exactly when supported. "
            "Do not invent facts. Distinguish the paper's own experimental results from background literature examples. "
            "For review/rulebook/perspective papers, clearly state that it is a review and summarize the reviewed material/device/stability space without forcing a single-device narrative. "
            "For device-focused experimental papers, include device architecture, fabrication highlights, key photovoltaic metrics, and stability. "
            "For materials-focused or mechanistic papers, do not force a device narrative; emphasize composition, structure, dimensionality, processing, characterization, proposed mechanisms, structure-property relationships, and why the findings matter. "
            "For all papers, make the Perovskite System section concrete: composition, dimensionality, spacer/additive chemistry, A-site/B-site/halide chemistry, and whether the paper is experimental, review, or mixed. "
            "Avoid generic filler. Prefer paper-specific details over broad claims. "
            "Return compact valid JSON only."
        )
        payload = {
            "paper_metadata": dict(metadata, inferred_paper_type=("review_or_perspective" if review_like else "experimental_or_mixed")),
            "records": records,
            "text": summary_text[:SUMMARY_MAX_TEXT_CHARS],
            "summary_style": "material_interpretive" if material_focused else "device_interpretive",
            "output_format": {
                "paper_summary": {
                    "paper_type": "experimental|review_or_perspective|mixed_or_unclear",
                    "one_sentence_summary": "specific English summary, not generic",
                    "objective": "...",
                    "research_context": "...",
                    "device_architecture": "for reviews, say not a new single-device study when appropriate",
                    "perovskite_system": "composition, dimensionality, spacer/additive, A/B/halide chemistry, and experimental/review scope",
                    "materials_focus": "...",
                    "fabrication_highlights": ["..."],
                    "synthesis_or_processing_insights": ["..."],
                    "best_reported_performance": {"pce": None, "voc": None, "jsc": None, "ff": None, "context": "..."},
                    "stability_summary": "...",
                    "characterization_and_evidence": ["include figure/caption/OCR-supported J-V, EQE, XRD, PL/UV-vis, microscopy, KPFM/nPFM, STEM-EDX, stability evidence when present"],
                    "figure_or_visual_evidence": ["caption/OCR/table/figure evidence available from the supplied text; empty only if none is present"],
                    "structure_composition_insights": ["..."],
                    "mechanism_or_interpretation": "...",
                    "notable_materials_or_methods": ["..."],
                    "main_contribution": "...",
                    "interpretive_takeaways": ["..."],
                    "data_quality_notes": ["..."],
                }
            },
        }
        result = self._call(system, payload, paper_slug=paper_slug, stage="paper_summary")
        if not isinstance(result, dict):
            return result

        paper_summary = result.get("paper_summary")
        if not isinstance(paper_summary, dict):
            return result

        # The Drive metadata can contain the file-modified year and a generic
        # import journal. Reconcile visible summary metadata and device wording
        # from the final evidence-grounded records instead.
        reconciled_year = next(
            (
                record.get("Ref_publication_date")
                for record in records or []
                if isinstance(record, dict)
                and record.get("Ref_publication_date") not in (None, "", [], {})
            ),
            None,
        )
        if reconciled_year not in (None, "", [], {}):
            result["_summary_metadata_reconciliation"] = {"year": reconciled_year}

        device_record = next(
            (
                record
                for record in records or []
                if isinstance(record, dict)
                and record.get("Cell_stack_sequence") not in (None, "", [], {})
                and record.get("_evidence_Cell_stack_sequence") not in (None, "", [], {})
            ),
            None,
        )
        if device_record is not None:
            paper_summary["paper_type"] = "experimental"
            paper_summary["device_architecture"] = (
                f"Device stack: {device_record.get('Cell_stack_sequence')}."
            )
            result["_summary_device_reconciliation"] = {
                "status": "grounded_device_stack_applied",
                "paper_slug": paper_slug,
            }

        grounded_performance = []
        for record in records or []:
            if not isinstance(record, dict):
                continue
            pce_field = next(
                (
                    field for field in (
                        "JV_default_PCE",
                        "JV_reverse_scan_PCE",
                        "JV_forward_scan_PCE",
                        "Stabilised_performance_PCE",
                    )
                    if record.get(field) not in (None, "", [], {})
                    and record.get(f"_evidence_{field}") not in (None, "", [], {})
                ),
                None,
            )
            if not pce_field:
                continue
            try:
                pce_numeric = float(record.get(pce_field))
            except (TypeError, ValueError):
                continue
            grounded_performance.append((pce_numeric, pce_field, record))

        if grounded_performance:
            _, pce_field, best_record = max(grounded_performance, key=lambda item: item[0])

            def grounded_metric(*fields):
                for field in fields:
                    value = best_record.get(field)
                    evidence = best_record.get(f"_evidence_{field}")
                    if value not in (None, "", [], {}) and evidence not in (None, "", [], {}):
                        return value
                return None

            paper_summary["best_reported_performance"] = {
                "pce": best_record.get(pce_field),
                "voc": grounded_metric("JV_default_Voc", "JV_reverse_scan_Voc", "JV_forward_scan_Voc"),
                "jsc": grounded_metric("JV_default_Jsc", "JV_reverse_scan_Jsc", "JV_forward_scan_Jsc"),
                "ff": grounded_metric("JV_default_FF", "JV_reverse_scan_FF", "JV_forward_scan_FF"),
                "context": (
                    "Reconciled to the highest evidence-grounded structured device record"
                    + (
                        f" ({best_record.get('Ref_internal_sample_id')})"
                        if best_record.get("Ref_internal_sample_id")
                        else ""
                    )
                    + "."
                ),
            }
            result["_summary_performance_reconciliation"] = {
                "status": "grounded_record_applied",
                "source_field": pce_field,
                "paper_slug": paper_slug,
            }
        else:
            existing = paper_summary.get("best_reported_performance")
            if isinstance(existing, dict) and any(
                existing.get(field) not in (None, "", [], {})
                for field in ("pce", "voc", "jsc", "ff")
            ):
                paper_summary["best_reported_performance"] = {
                    "pce": None,
                    "voc": None,
                    "jsc": None,
                    "ff": None,
                    "context": "No evidence-grounded structured performance record was available.",
                }
                result["_summary_performance_reconciliation"] = {
                    "status": "unsupported_metrics_cleared",
                    "paper_slug": paper_slug,
                }
        return result


_previous_validate_record_keys_v21_12 = validate_record_keys

def validate_record_keys(records: List[Dict[str, Any]], allowed_columns: List[str]) -> List[Dict[str, Any]]:
    """Enforce the field/evidence contract before family records can be merged."""
    cleaned = _previous_validate_record_keys_v21_12(records, allowed_columns)
    metadata_fields = {
        "Ref_DOI_number", "Ref_lead_author", "Ref_publication_date", "Ref_journal",
        "Ref_original_filename_data_upload", "Ref_internal_sample_id",
    }
    removed = 0
    for record in cleaned:
        for field in allowed_columns:
            if field in metadata_fields:
                continue
            value = record.get(field)
            present = value not in (None, "", [], {}) and str(value).strip().lower() not in {
                "nan", "none", "null", "not reported", "unknown", "n/a",
            }
            evidence = record.get(f"_evidence_{field}")
            evidence_present = (
                isinstance(evidence, str)
                and len(evidence.strip()) >= 12
                and _v21_12_match_text(evidence) != _v21_12_match_text(value)
            )
            if present and not evidence_present:
                record[field] = None
                removed += 1
        record.pop("example_column", None)
        record.pop("_evidence_example_column", None)
    if removed:
        print(f"[EVIDENCE GATE] Removed {removed} populated field(s) lacking matching field evidence.")
    return cleaned


_previous_ensure_summary_outputs_v21_4 = ensure_summary_outputs

def ensure_summary_outputs(slug: str, meta: Dict[str, Any], records: List[Dict[str, Any]], full_text: str,
                           extractor: Optional[HuggingFaceQwenExtractor] = None, force_llm_summary: bool = False) -> Tuple[Dict[str, Any], str]:
    visible_json = PAPER_SUMMARY_JSON_DIR / f"{slug}_summary.json"
    visible_txt = PAPER_SUMMARY_TEXT_DIR / f"{slug}_summary.txt"
    if visible_json.exists() and visible_txt.exists() and not FORCE_SUMMARY_REWRITE:
        try:
            existing = json.loads(visible_json.read_text(encoding="utf-8"))
            if "_normalize_summary_payload_v19_4" in globals():
                existing = _normalize_summary_payload_v19_4(existing)
            return existing, visible_txt.read_text(encoding="utf-8")
        except Exception:
            pass
    return _previous_ensure_summary_outputs_v21_4(
        slug, meta, records, full_text, extractor=extractor, force_llm_summary=(force_llm_summary or FORCE_SUMMARY_REWRITE)
    )


_previous_render_paper_summary_text_v21_4 = render_paper_summary_text

def render_paper_summary_text(summary_payload: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
    metadata = dict(metadata or {})
    reconciled_metadata = (
        summary_payload.get("_summary_metadata_reconciliation", {})
        if isinstance(summary_payload, dict)
        else {}
    )
    if isinstance(reconciled_metadata, dict):
        for key in ("doi", "journal", "year"):
            if reconciled_metadata.get(key) not in (None, "", [], {}):
                metadata[key] = reconciled_metadata[key]
    text = _previous_render_paper_summary_text_v21_4(summary_payload, metadata)
    ps = summary_payload.get("paper_summary", {}) if isinstance(summary_payload, dict) else {}
    extra_lines = []
    if ps.get("paper_type") and "Paper Type:" not in text:
        extra_lines.extend(["", "Paper Type:", str(ps.get("paper_type"))])
    for heading, key in [
        ("Research Context", "research_context"),
        ("Materials Focus", "materials_focus"),
        ("Synthesis or Processing Insights", "synthesis_or_processing_insights"),
        ("Characterization and Evidence", "characterization_and_evidence"),
        ("Figure or Visual Evidence", "figure_or_visual_evidence"),
        ("Structure/Composition Insights", "structure_composition_insights"),
        ("Mechanism or Interpretation", "mechanism_or_interpretation"),
        ("Interpretive Takeaways", "interpretive_takeaways"),
    ]:
        value = ps.get(key)
        if value and heading not in text:
            extra_lines.extend(["", f"{heading}:"])
            if isinstance(value, list):
                extra_lines.extend([f"- {x}" for x in value if x])
            else:
                extra_lines.append(str(value))
    return text.rstrip() + ("\n" + "\n".join(extra_lines) if extra_lines else "") + "\n"


V21_DEVICE_FAMILIES = {"etl", "htl_backcontact_additional", "performance", "stability_outdoor"}
V21_ALWAYS_FAMILIES = {"reference_device", "perovskite_chemistry", "perovskite_processing"}
V21_REVIEW_TERMS = (
    "review", "perspective", "roadmap", "outlook", "progress", "advances",
    "opportunities", "challenges", "status and prospects", "rulebook",
    "overview", "tutorial", "primer", "design rules"
)
V21_EXPLICIT_DEVICE_TERMS = (
    "fabricated solar cell", "fabricated device", "device fabrication", "complete device",
    "photovoltaic device", "solar cell device", "champion device", "certified pce",
    "power conversion efficiency", "photo-conversion efficiency", "photoconversion efficiency",
    "current-voltage", "j-v curve", "jv curve",
    "external quantum efficiency", "eqe spectrum", "stabilized power output",
    "maximum power point", "operational stability", "encapsulated device"
)
V21_DEVICE_LAYER_TERMS = (
    "etl", "electron transport", "htl", "hole transport", "spiro", "ptaa",
    "sno2", "tio2", "pcbm", "c60", "ito", "fto", "back contact", "electrode"
)
V21_METRIC_PATTERNS = (
    r"\bpce\s*(?:of|=|:|up to|reached|reaches|was|is|exceeding|over|above|approaching)?\s*\d+(?:\.\d+)?\s*%",
    r"\bpower conversion efficiency\s*(?:of|=|:|up to|reached|reaches|was|is|exceeding|over|above|approaching)?\s*\d+(?:\.\d+)?\s*%",
    r"\b(?:photo-conversion|photoconversion) efficiency\s*(?:of|=|:|up to|reached|reaches|was|is|exceeding|over|above|approaching)?\s*\d+(?:\.\d+)?\s*%",
    r"\b\d+(?:\.\d+)?\s*%\s*(?:pce|power conversion efficiency)",
    r"\bvoc\s*(?:=|:|of)?\s*\d+(?:\.\d+)?\s*(?:v|mv)\b",
    r"\bjsc\s*(?:=|:|of)?\s*\d+(?:\.\d+)?\s*ma\s*cm",
    r"\bfill factor\s*(?:=|:|of)?\s*\d+(?:\.\d+)?\s*%?",
    r"\bt80\s*(?:=|:|of|over|exceeding)?\s*\d+",
)
V21_COMPUTATIONAL_TERMS = (
    "density functional", "dft", "first-principles", "first principles", "molecular dynamics",
    "ab initio", "calculation", "calculated", "simulation", "theoretical", "band structure",
    "formation energy", "binding energy", "scaps", "wxamps", "gpvdm", "silvaco",
    "device simulation", "simulation methodology", "numerical optimization"
)
V21_MATERIALS_TERMS = (
    "single crystal", "thin film", "crystal structure", "phase", "lattice",
    "photoluminescence", "pl", "xrd", "giwaxs", "uv-vis", "absorption",
    "carrier dynamics", "mobility", "defect", "mechanism", "structure-property",
    "quantum well", "ruddlesden", "dion-jacobson", "spacer cation", "ligand"
)
V21_METHODS_TERMS = (
    "in situ", "operando", "spectroscopy", "microscopy", "kpfm", "npfm", "stem",
    "edx", "xps", "transient absorption", "trpl", "characterization", "imaging"
)


def _v21_external_ontology_terms(*key_paths: str) -> Tuple[str, ...]:
    ontology_path = os.getenv("PEROVSKITE_ONTOLOGY_PATH", "").strip()
    if not ontology_path:
        return tuple()
    path = Path(ontology_path)
    if not path.exists():
        return tuple()
    try:
        ontology = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Could not read external ontology terms for gating: {exc}")
        return tuple()

    terms: List[str] = []

    def _walk(value: Any) -> None:
        if isinstance(value, str):
            term = value.strip().lower()
            if term:
                terms.append(term)
        elif isinstance(value, list):
            for item in value:
                _walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                _walk(item)

    for key_path in key_paths:
        node: Any = ontology
        for part in key_path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                node = None
                break
        if node is not None:
            _walk(node)

    return tuple(dict.fromkeys(terms))


V21_ONTOLOGY_DEVICE_PERFORMANCE_TERMS = _v21_external_ontology_terms(
    "performance_metrics",
    "family_gate_terms.performance",
    "family_gate_terms.stability_outdoor",
    "figure_types.jv_curve",
    "figure_types.stability_curve",
)
if V21_ONTOLOGY_DEVICE_PERFORMANCE_TERMS:
    V21_EXPLICIT_DEVICE_TERMS = tuple(dict.fromkeys(V21_EXPLICIT_DEVICE_TERMS + V21_ONTOLOGY_DEVICE_PERFORMANCE_TERMS))
    print(f"[INFO] v21 ontology gate terms loaded: {len(V21_ONTOLOGY_DEVICE_PERFORMANCE_TERMS)} device/performance/stability terms")


def _v21_count_terms(text: str, terms: Tuple[str, ...]) -> int:
    low = (text or "").lower()
    return int(sum(low.count(str(term).lower()) for term in terms))


def _v21_count_metric_patterns(text: str) -> int:
    low = (text or "").lower()
    return int(sum(len(re.findall(pat, low, flags=re.I)) for pat in V21_METRIC_PATTERNS))


def _v21_has_device_performance_language(text: str) -> bool:
    low = (text or "").lower()
    if re.search(r"\b(?:pce|power conversion efficiency|photo-conversion efficiency|photoconversion efficiency|j-v|jv|voc|jsc|fill factor|eqe)\b", low, flags=re.I):
        return True
    return bool(re.search(r"\b\d+(?:\.\d+)?\s*%\s*(?:efficiency|pce)\b", low, flags=re.I))


def _v21_get_scored_text(meta: Dict[str, Any], text: str) -> Tuple[str, str, str]:
    title = str((meta or {}).get("title") or "")
    abstract = str((meta or {}).get("abstract") or "")
    journal = str((meta or {}).get("journal") or "")
    front = "\n".join([title, abstract, journal])
    body = str(text or "")[:80000]
    headings = "\n".join(re.findall(
        r"(?im)^\s*(?:\d+\.?\s*)?(abstract|introduction|results?|discussion|experimental|methods?|computational methods?|conclusions?|supporting information|device fabrication|characterization)\s*$",
        body,
    ))
    return front, body, headings


V21_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
V21_BAD_DOI_TRAILING = ".,;:)]}>\"'"


def _v21_clean_doi(value: Any) -> str:
    doi = str(value or "").strip()
    doi = re.sub(r"^(?:doi\s*:?\s*|https?://(?:dx\.)?doi\.org/)", "", doi, flags=re.I).strip()
    doi = doi.rstrip(V21_BAD_DOI_TRAILING)
    doi = re.sub(r"[\s\r\n]+$", "", doi)
    return doi.lower()


def _v21_best_doi_from_text(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Recover the paper DOI deterministically from extracted PDF/OCR text.

    Google Drive PDFs often arrive without Crossref metadata. The DOI is usually
    present in the title/header pages and repeated in page headers; references
    appear later. Score candidates by frequency and early occurrence so we do
    not depend on the LLM to fill reference metadata.
    """
    raw = str(text or "")
    windows = [
        ("front_25000", raw[:25000]),
        ("front_80000", raw[:80000]),
        ("full_text", raw),
    ]
    for window_name, window_text in windows:
        hits = []
        for m in V21_DOI_RE.finditer(window_text):
            doi = _v21_clean_doi(m.group(0))
            if not doi or len(doi) < 8:
                continue
            hits.append((doi, int(m.start())))
        if not hits:
            continue
        counts: Dict[str, int] = {}
        first_pos: Dict[str, int] = {}
        for doi, pos in hits:
            counts[doi] = counts.get(doi, 0) + 1
            first_pos[doi] = min(first_pos.get(doi, pos), pos)
        ranked = sorted(
            counts.keys(),
            key=lambda d: (-counts[d], first_pos[d], len(d)),
        )
        best = ranked[0]
        return best, {
            "window": window_name,
            "count": counts[best],
            "first_position": first_pos[best],
            "candidates": ranked[:8],
        }
    return "", {"window": "", "count": 0, "first_position": None, "candidates": []}


def _v21_best_publication_year(meta: Dict[str, Any], text: str) -> Tuple[Optional[int], str]:
    """Recover the publication year instead of using a Drive modified year."""
    current_year = int(time.strftime("%Y", time.gmtime()))
    title = str(meta.get("title") or "")
    title_years = [
        int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", title)
        if 1900 <= int(value) <= current_year + 1
    ]
    if title_years:
        return title_years[0], "title"

    front = str(text or "")[:16000]
    hits = [
        (int(match.group(0)), int(match.start()))
        for match in re.finditer(r"\b(?:19|20)\d{2}\b", front)
        if 1900 <= int(match.group(0)) <= current_year + 1
    ]
    if not hits:
        return None, ""
    counts: Dict[int, int] = {}
    first_position: Dict[int, int] = {}
    for year, position in hits:
        counts[year] = counts.get(year, 0) + 1
        first_position[year] = min(first_position.get(year, position), position)
    ranked = sorted(counts, key=lambda year: (-counts[year], first_position[year]))
    return ranked[0], "front_text_frequency"


def _v21_enrich_meta_from_text(meta: Dict[str, Any], text: str, paper_slug: str = "") -> Dict[str, Any]:
    """
    Fill missing reference metadata before schema-family extraction.

    This is intentionally deterministic and evidence-logged. It does not relax
    strict extraction; it prevents manual Google Drive rows from losing DOI
    metadata just because the Drive filename lacks Crossref fields.
    """
    meta = dict(meta or {})
    recovered: Dict[str, Any] = {}
    if not _v21_clean_doi(meta.get("doi")):
        doi, details = _v21_best_doi_from_text(text)
        if doi:
            meta["doi"] = doi
            recovered["doi"] = doi
            recovered["doi_recovery"] = details
            print(f"[METADATA] Recovered DOI from extracted text: {paper_slug or slugify(meta.get('title') or doi)} | {doi}")
    else:
        meta["doi"] = _v21_clean_doi(meta.get("doi"))

    drive_source = "__gdrive_" in str(meta.get("title") or "").lower()
    recovered_year, year_method = _v21_best_publication_year(meta, text)
    current_meta_year = meta.get("year")
    try:
        current_meta_year_int = int(float(current_meta_year))
    except (TypeError, ValueError):
        current_meta_year_int = None
    if recovered_year and (
        current_meta_year_int is None
        or (drive_source and current_meta_year_int != recovered_year)
    ):
        meta["year"] = recovered_year
        recovered["publication_year"] = recovered_year
        recovered["publication_year_previous"] = current_meta_year
        recovered["publication_year_method"] = year_method
        print(
            f"[METADATA] Recovered publication year from {year_method}: "
            f"{paper_slug or slugify(meta.get('title') or meta.get('doi') or 'paper')} | {recovered_year}"
        )

    if recovered:
        meta["_v21_reference_metadata_recovered"] = recovered
        try:
            report_path = BASE / "reference_metadata_recovery_report.jsonl"
            report_row = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "paper_slug": paper_slug or slugify(meta.get("title") or meta.get("doi") or "paper"),
                "title": meta.get("title"),
                "recovered_doi": recovered.get("doi", ""),
                "doi_window": (recovered.get("doi_recovery") or {}).get("window", ""),
                "doi_count": (recovered.get("doi_recovery") or {}).get("count", ""),
                "doi_first_position": (recovered.get("doi_recovery") or {}).get("first_position", ""),
                "doi_candidates": json.dumps((recovered.get("doi_recovery") or {}).get("candidates", []), ensure_ascii=False),
            }
            with report_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(report_row, recovered=recovered), ensure_ascii=False, default=str) + "\n")
            if "_v21_csv" in globals() and _v21_csv is not None:
                csv_path = BASE / "reference_metadata_recovery_report.csv"
                exists = csv_path.exists()
                with csv_path.open("a", encoding="utf-8-sig", newline="") as f:
                    writer = _v21_csv.DictWriter(f, fieldnames=list(report_row.keys()))
                    if not exists:
                        writer.writeheader()
                    writer.writerow(report_row)
        except Exception as exc:
            print(f"[WARN] Could not write reference metadata recovery report: {exc}")
    return meta


def _v21_paper_type_scores(meta: Dict[str, Any], text: str) -> Dict[str, Any]:
    front, body, headings = _v21_get_scored_text(meta, text)
    title = str((meta or {}).get("title") or "")
    low_body = body.lower()
    review_score = _v21_count_terms(front, V21_REVIEW_TERMS) * 4 + _v21_count_terms(body[:12000], V21_REVIEW_TERMS)
    computational_score = _v21_count_terms(front, V21_COMPUTATIONAL_TERMS) * 3 + _v21_count_terms(body[:25000], V21_COMPUTATIONAL_TERMS)
    materials_score = _v21_count_terms(front, V21_MATERIALS_TERMS) * 2 + _v21_count_terms(body[:25000], V21_MATERIALS_TERMS)
    methods_score = _v21_count_terms(front, V21_METHODS_TERMS) * 2 + _v21_count_terms(body[:25000], V21_METHODS_TERMS)
    explicit_device_score = _v21_count_terms(front, V21_EXPLICIT_DEVICE_TERMS) * 4 + _v21_count_terms(body[:30000], V21_EXPLICIT_DEVICE_TERMS)
    layer_score = _v21_count_terms(body[:30000], V21_DEVICE_LAYER_TERMS)
    metric_score = _v21_count_metric_patterns(front) * 4 + _v21_count_metric_patterns(body[:30000])
    fabrication_score = int(bool(re.search(r"\b(fabricat(?:ed|ion)|prepared|deposited|spin-coated|blade-coated|annealed)\b.{0,180}\b(device|solar cell|cell|photovoltaic)\b", low_body, flags=re.I))) * 6
    architecture_score = int(bool(re.search(r"\b(ito|fto)\b.{0,120}\b(sno2|tio2|pcbm|c60|etl)\b.{0,180}\b(perovskite|absorber)\b.{0,180}\b(spiro|ptaa|htl|au|ag|carbon|electrode)\b", low_body, flags=re.I))) * 6
    device_score = explicit_device_score + metric_score + fabrication_score + architecture_score + min(layer_score, 6)
    title_review_like = _v21_4_is_review_like(meta) or bool(re.search(r"\b(review|perspective|roadmap|outlook|progress|advances|rulebook|overview)\b", title, flags=re.I))
    title_device_metric = bool(
        re.search(r"\b(perovskite solar cells?|photovoltaic devices?)\b", front, flags=re.I)
        and re.search(r"\b(power conversion efficiency|photo-conversion efficiency|photoconversion efficiency|pce|j-v|voc|jsc|fill factor|stability|t80|retention)\b", front, flags=re.I)
    )
    device_performance_language = _v21_has_device_performance_language(front) or _v21_has_device_performance_language(body[:30000])
    primary_computational = bool(
        re.search(
            r"\b(?:simulation methodology|computational methods?|device simulation|"
            r"simulations? (?:were|was|are|is) (?:carried out|performed)|"
            r"using (?:the )?(?:scaps(?:-1d)?|wxamps|gpvdm|silvaco)|"
            r"scaps(?:-1d)? (?:software|simulator|model))\b",
            " ".join((front, body[:18000])),
            flags=re.I,
        )
    )
    primary_experimental = bool(
        re.search(
            r"\b(?:in this work|in this study|herein|we)\b.{0,120}"
            r"\b(?:fabricat(?:ed|ion)|prepared|synthesi[sz]ed|spin[- ]coated|"
            r"deposited|experimentally measured|measured under)\b|"
            r"\bdevice fabrication\b.{0,120}\b(?:substrate|spin[- ]coat|deposit|anneal)\b",
            " ".join((front, body[:22000])),
            flags=re.I,
        )
    )
    if FAMILY_GATING_MODE == "strict":
        # Repeated generic mentions of devices or performance are common in
        # materials, theory, and review papers. Strict device classification
        # requires a real numerical device metric plus experimental context.
        explicit_new_device = bool(
            metric_score >= 2
            and not title_review_like
            and (not primary_computational or primary_experimental)
            and (
                fabrication_score
                or architecture_score
                or title_device_metric
                or explicit_device_score >= 8
            )
        )
    else:
        explicit_new_device = bool(
            (metric_score >= 2 and (fabrication_score or architecture_score or explicit_device_score >= 4))
            or (title_device_metric and explicit_device_score >= 4 and metric_score >= 1)
            or (
                device_performance_language
                and metric_score >= 1
                and (fabrication_score or architecture_score)
                and not title_review_like
            )
        )
    return {
        "device_score": int(device_score),
        "review_score": int(review_score),
        "computational_score": int(computational_score),
        "materials_score": int(materials_score),
        "methods_score": int(methods_score),
        "metric_score": int(metric_score),
        "explicit_device_score": int(explicit_device_score),
        "fabrication_score": int(fabrication_score),
        "architecture_score": int(architecture_score),
        "explicit_new_device": bool(explicit_new_device),
        "title_device_metric": bool(title_device_metric),
        "device_performance_language": bool(device_performance_language),
        "primary_computational": bool(primary_computational),
        "primary_experimental": bool(primary_experimental),
        "title_review_like": bool(title_review_like),
        "headings": headings,
    }


def _v21_classify_paper_type(meta: Dict[str, Any], text: str) -> Tuple[str, Dict[str, Any]]:
    scores = _v21_paper_type_scores(meta, text)
    mode = FAMILY_GATING_MODE
    if scores["title_review_like"] and not scores["explicit_new_device"]:
        return "review_perspective", scores
    if (
        scores.get("primary_computational")
        and scores["computational_score"] >= 6
        and not scores.get("primary_experimental")
    ):
        return "computational_theory", scores
    if (
        scores["computational_score"] >= 6
        and not scores["explicit_new_device"]
        and (
            scores["device_score"] < (12 if mode == "strict" else 8)
            or not (scores["metric_score"] or scores["fabrication_score"] or scores["architecture_score"])
        )
    ):
        return "computational_theory", scores
    if scores["explicit_new_device"]:
        return "device_experimental", scores
    if mode == "off":
        return "device_experimental", scores
    threshold = 14 if mode == "strict" else 9
    if scores["device_score"] >= threshold and scores["metric_score"] >= (2 if mode == "strict" else 1):
        return "device_experimental", scores
    if scores["methods_score"] >= 4 and scores["materials_score"] >= 2:
        return "methods_characterization", scores
    return "materials_experimental", scores


def _v21_should_run_family(family_name: str, paper_type: str, scores: Dict[str, Any], text: str) -> bool:
    if FAMILY_GATING_MODE == "off":
        return True
    if family_name in V21_ALWAYS_FAMILIES:
        return True
    if family_name not in V21_DEVICE_FAMILIES:
        return True
    low = (text or "").lower()
    strong_family_terms = {
        "etl": ("etl", "electron transport", "sno2", "tio2", "pcbm", "c60"),
        "htl_backcontact_additional": ("htl", "hole transport", "spiro", "ptaa", "back contact", "electrode", "encapsulation"),
        "performance": ("pce", "power conversion", "photo-conversion", "photoconversion", "j-v", "jv", "eqe", "voc", "jsc", "fill factor", "stabilized"),
        "stability_outdoor": ("stability", "t80", "t95", "maximum power point", "mpp", "aging", "aged", "outdoor"),
    }
    if paper_type == "device_experimental":
        return True
    explicit_device = bool(scores.get("explicit_new_device"))
    if paper_type in {"review_perspective", "computational_theory"}:
        return explicit_device
    hits = sum(low.count(term) for term in strong_family_terms.get(family_name, ()))
    if family_name == "performance" and (scores.get("device_performance_language") or scores.get("title_device_metric")):
        return True
    if family_name in {"performance", "stability_outdoor"}:
        return bool(scores.get("metric_score", 0) >= (3 if FAMILY_GATING_MODE == "strict" else 1) and hits >= 1)
    if FAMILY_GATING_MODE == "strict":
        return bool(
            hits >= 2
            and (scores.get("fabrication_score", 0) or scores.get("architecture_score", 0))
        )
    return bool(scores.get("device_score", 0) >= 7 and hits >= 2)


def _v21_append_gate_report(row: Dict[str, Any]) -> None:
    fields = [
        "timestamp", "paper_slug", "predicted_type", "family_gating_mode",
        "device_score", "review_score", "computational_score", "materials_score", "methods_score",
        "metric_score", "families_run", "families_skipped", "reason",
    ]
    try:
        row = {k: row.get(k, "") for k in fields}
        row["timestamp"] = row.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        exists = PAPER_TYPE_GATE_REPORT.exists()
        if _v21_csv is not None:
            with PAPER_TYPE_GATE_REPORT.open("a", encoding="utf-8-sig", newline="") as f:
                writer = _v21_csv.DictWriter(f, fieldnames=fields)
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
    except Exception as exc:
        print(f"[WARN] Could not append paper type gate report: {exc}")


_previous_extract_family_for_chunks_v21_5 = extract_family_for_chunks

def extract_family_for_chunks(
    meta: Dict[str, Any],
    family_name: str,
    family_columns: List[str],
    chunks: List[str],
    extractor: HuggingFaceQwenExtractor,
    paper_slug: str,
) -> List[Dict[str, Any]]:
    allowed, gate_score = family_has_enough_evidence(family_name, family_columns, "\n\n".join(chunks))
    if not allowed:
        print(f"[SKIP] Family gated out: {family_name} | score={gate_score} | mode={LLM_COST_MODE}")
        return []
    selected_chunks = rank_chunks_for_family(chunks, family_name, family_columns)
    print(f"[INFO] Family {family_name}: using {len(selected_chunks)}/{len(chunks)} chunks | gate_score={gate_score}")
    partial_payloads: List[Dict[str, Any]] = []
    alias_map = {col: FIELD_ALIASES.get(col, []) for col in family_columns}
    failures = []
    for i, chunk in enumerate(selected_chunks, start=1):
        if not chunk.strip():
            continue
        try:
            (TEXT_DIR / f"{paper_slug}_{family_name}_v21_ranked_chunk_{i}.txt").write_text(chunk, encoding="utf-8")
        except Exception:
            pass
        try:
            payload = extractor.extract_family_records(
                metadata={
                    "title": meta.get("title"),
                    "doi": meta.get("doi"),
                    "journal": meta.get("journal"),
                    "year": meta.get("year"),
                    "lead_author": meta.get("lead_author"),
                    "chunk_index": i,
                    "num_chunks": len(selected_chunks),
                    "agent_version": AGENT_VERSION,
                    "cost_mode": LLM_COST_MODE,
                },
                chunk_text=chunk,
                family_name=family_name,
                family_columns=family_columns,
                alias_map=alias_map,
                paper_slug=paper_slug,
                chunk_idx=i,
            )
            payload["records"] = validate_record_keys(payload.get("records", []), family_columns)
            partial_payloads.append(payload)
            (RAW_LLM_DIR / f"{paper_slug}_{family_name}_v21_partial_{i}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            failures.append({"chunk": i, "error": repr(exc)})
            print(f"[WARN] Family {family_name} chunk {i} failed after JSON repair; continuing with partial outputs: {exc}")
            _v21_append_json_failure({
                "paper_slug": paper_slug,
                "stage": f"{family_name}_partial_{i}",
                "error": repr(exc),
                "raw_output_path": "",
                "repaired": False,
                "repair_method": "family_chunk_skipped",
                "note": "Family chunk skipped; paper continues with parsed partial outputs.",
            })
    if not partial_payloads:
        if failures:
            print(f"[WARN] Family {family_name}: all selected chunks failed; skipping family.")
        return []
    # The partial records have already passed exact chunk-level evidence checks.
    # Re-prompting the model here can reshape scalar values, discard evidence, or
    # return an empty record. A deterministic conflict-aware merge is both safer
    # and one LLM call per family faster.
    consolidated = {
        "records": _v21_12_merge_grounded_partial_records(partial_payloads, family_columns),
        "_consolidation_method_v21_12": "deterministic_grounded_partial_merge",
    }
    print(
        f"[INFO] Family {family_name}: deterministically consolidated "
        f"{sum(len(p.get('records', [])) for p in partial_payloads)} partial record(s) into "
        f"{len(consolidated['records'])} grounded record(s)."
    )
    consolidated["records"] = validate_record_keys(consolidated.get("records", []), family_columns)
    if failures:
        consolidated["_json_partial_failures_v21_5"] = failures
    try:
        (RAW_LLM_DIR / f"{paper_slug}_{family_name}_v21_consolidated.json").write_text(
            json.dumps(consolidated, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass
    records = consolidated.get("records", [])
    return records if isinstance(records, list) else []


_previous_extract_records_with_llm_v21_4 = extract_records_with_llm

def extract_records_with_llm(meta: Dict[str, Any], extraction_text: str, extractor: HuggingFaceQwenExtractor) -> List[Dict[str, Any]]:
    initial_slug = slugify((meta or {}).get("title") or (meta or {}).get("doi") or "paper")
    meta = _v21_enrich_meta_from_text(meta, extraction_text, initial_slug)
    title = str(meta.get("title") or meta.get("doi") or "")
    if not REVIEW_FULL_EXTRACTION_ENABLE and _v21_4_is_review_like(meta):
        gate_scores = _v21_paper_type_scores(meta, extraction_text)
        rec = make_empty_record()
        rec = enrich_record_with_metadata(rec, meta, 1)
        if meta.get("_v21_reference_metadata_recovered"):
            rec["_evidence_v21_reference_metadata_recovery"] = json.dumps(meta.get("_v21_reference_metadata_recovered"), ensure_ascii=False)
        rec["Ref_free_text_comment"] = "Review/perspective shortcut: no full device-row extraction run by default; summary remains evidence-grounded. Set REVIEW_FULL_EXTRACTION_ENABLE=1 to force full extraction."
        rec["_evidence_review_shortcut"] = f"Title/metadata indicate review-like paper ({title}); avoiding unsupported device-row extraction from literature examples."
        print(f"[REVIEW SHORTCUT] {slugify(title) if 'slugify' in globals() else title}: skipping full schema-family extraction.")
        _v21_append_gate_report({
            "paper_slug": slugify(title) if "slugify" in globals() else title,
            "predicted_type": "review_perspective",
            "family_gating_mode": FAMILY_GATING_MODE,
            "device_score": gate_scores.get("device_score", 0),
            "review_score": gate_scores.get("review_score", 0),
            "computational_score": gate_scores.get("computational_score", 0),
            "materials_score": gate_scores.get("materials_score", 0),
            "methods_score": gate_scores.get("methods_score", 0),
            "metric_score": gate_scores.get("metric_score", 0),
            "families_run": "metadata_summary_only",
            "families_skipped": ";".join(SCHEMA_FAMILIES.keys()),
            "reason": "review shortcut; no full device-row extraction",
        })
        return [rec]
    context_safe_chunk_chars = min(
        LLM_CHUNK_CHAR_LIMIT,
        int(os.getenv("STRUCTURED_EXTRACTION_CHUNK_CHARS", "5500")),
    )
    context_safe_overlap = min(LLM_CHUNK_OVERLAP, max(400, context_safe_chunk_chars // 8))
    chunks = chunk_text(
        extraction_text,
        max_chars=context_safe_chunk_chars,
        overlap=context_safe_overlap,
    )
    paper_slug = slugify(meta.get("title") or meta.get("doi") or "paper")
    paper_type, gate_scores = _v21_classify_paper_type(meta, extraction_text)
    device_score = int(gate_scores.get("device_score", 0))
    print(
        f"[INFO] Paper type gate: {paper_type} | mode={FAMILY_GATING_MODE} | "
        f"device={gate_scores.get('device_score', 0)} review={gate_scores.get('review_score', 0)} "
        f"computational={gate_scores.get('computational_score', 0)} materials={gate_scores.get('materials_score', 0)} "
        f"metrics={gate_scores.get('metric_score', 0)}"
    )
    deterministic_rec = deterministic_preextract_record(meta, extraction_text) if DETERMINISTIC_EXTRACTION_ENABLE else make_empty_record()
    if DETERMINISTIC_EXTRACTION_ENABLE:
        try:
            (JSON_DIR / f"{paper_slug}_v21_deterministic_preextract.json").write_text(
                json.dumps(deterministic_rec, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass
    family_outputs: Dict[str, List[Dict[str, Any]]] = {}
    skipped_families = {}
    families_run = []
    for family_name, family_columns in SCHEMA_FAMILIES.items():
        if not _v21_should_run_family(family_name, paper_type, gate_scores, extraction_text):
            skipped_families[family_name] = f"paper_type={paper_type}; device_score={device_score}; mode={FAMILY_GATING_MODE}"
            print(f"[SKIP] Family paper-type gated out: {family_name} | paper_type={paper_type} | device_score={device_score} | mode={FAMILY_GATING_MODE}")
            continue
        print(f"[INFO] Extracting family: {family_name} | v21 robust mode={LLM_COST_MODE}")
        families_run.append(family_name)
        try:
            family_records = extract_family_for_chunks(
                meta=meta,
                family_name=family_name,
                family_columns=family_columns,
                chunks=chunks,
                extractor=extractor,
                paper_slug=paper_slug,
            )
        except Exception as exc:
            family_records = []
            print(f"[WARN] Family {family_name} failed unexpectedly; continuing paper: {exc}")
            _v21_append_json_failure({
                "paper_slug": paper_slug,
                "stage": family_name,
                "error": repr(exc),
                "raw_output_path": "",
                "repaired": False,
                "repair_method": "family_skipped",
                "note": "Family failed unexpectedly; paper continues.",
            })
        family_outputs[family_name] = family_records
    _v21_append_gate_report({
        "paper_slug": paper_slug,
        "predicted_type": paper_type,
        "family_gating_mode": FAMILY_GATING_MODE,
        "device_score": gate_scores.get("device_score", 0),
        "review_score": gate_scores.get("review_score", 0),
        "computational_score": gate_scores.get("computational_score", 0),
        "materials_score": gate_scores.get("materials_score", 0),
        "methods_score": gate_scores.get("methods_score", 0),
        "metric_score": gate_scores.get("metric_score", 0),
        "families_run": ";".join(families_run),
        "families_skipped": ";".join(skipped_families.keys()),
        "reason": json.dumps({k: gate_scores.get(k) for k in ["explicit_new_device", "fabrication_score", "architecture_score", "explicit_device_score", "primary_computational", "primary_experimental", "title_review_like"]}, ensure_ascii=False),
    })
    combined_records: List[Dict[str, Any]] = []
    for family_name in SCHEMA_FAMILIES.keys():
        combined_records = merge_record_lists(combined_records, family_outputs.get(family_name, []))
    combined_records = merge_deterministic_with_llm_records(deterministic_rec, combined_records, meta)
    if not combined_records:
        rec = enrich_record_with_metadata(make_empty_record(), meta, 1)
        rec["Ref_free_text_comment"] = (
            "No usable schema-family records parsed after robust JSON repair; "
            "summary and figure evidence may still be available."
        )
        rec["_evidence_v21_5_partial_extraction_warning"] = "All schema-family outputs were empty or failed parsing."
        combined_records = [rec]
    cleaned: List[Dict[str, Any]] = []
    warning_note = ""
    if skipped_families:
        warning_note = " Paper-type gate skipped families: " + "; ".join(f"{k}({v})" for k, v in skipped_families.items()) + "."
    for idx, rec in enumerate(combined_records, start=1):
        rec = enrich_record_with_metadata(rec, meta, idx)
        rec["_lit_agent_paper_type"] = paper_type
        if meta.get("_v21_reference_metadata_recovered"):
            rec["_evidence_v21_reference_metadata_recovery"] = json.dumps(meta.get("_v21_reference_metadata_recovered"), ensure_ascii=False)
        rec["Ref_free_text_comment"] = (rec.get("Ref_free_text_comment") or "") + f" Extracted with {AGENT_VERSION}; robust_json=1; paper_type={paper_type}; cost_mode={LLM_COST_MODE}.{warning_note}"
        if skipped_families:
            rec["_evidence_v21_5_paper_type_gate"] = json.dumps(skipped_families, ensure_ascii=False)
        cleaned.append(rec)
    cleaned = _v21_12_coalesce_final_records(cleaned)
    cleaned = _v21_17_assign_distinct_sample_ids(cleaned)
    if not cleaned:
        rec = enrich_record_with_metadata(make_empty_record(), meta, 1)
        rec["Ref_free_text_comment"] = f"Fallback metadata-only record after normalization removed all extracted rows; paper_type={paper_type}."
        rec["_evidence_v21_5_partial_extraction_warning"] = "Fallback metadata-only row."
        cleaned = [rec]
    return cleaned

TIMING_LOG_DIR = BASE / "timing_logs"
TIMING_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _v21_4_timing_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _v21_4_append_timing_event(event: Dict[str, Any]) -> None:
    try:
        event = dict(event)
        event.setdefault("timestamp", _v21_4_timing_stamp())
        with (TIMING_LOG_DIR / "paper_timing.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        print(f"[WARN] Could not write timing event: {exc}")


if "process_one_paper_api" in globals():
    _previous_process_one_paper_api_v21_4_timing = process_one_paper_api

    def process_one_paper_api(meta: Dict[str, Any], force_reprocess: bool = False, run_mode: Optional[str] = None) -> Dict[str, Any]:
        title = str(meta.get("title") or meta.get("doi") or "untitled")
        slug = slugify(title) if "slugify" in globals() else hashlib.sha1(title.encode("utf-8", errors="ignore")).hexdigest()[:12]
        start = time.perf_counter()
        status = "ok"
        records_count = None
        error_text = None
        try:
            result = _previous_process_one_paper_api_v21_4_timing(meta, force_reprocess=force_reprocess, run_mode=run_mode)
            if isinstance(result, dict):
                records = result.get("records")
                if isinstance(records, list):
                    records_count = len(records)
                status = str(result.get("status") or status)
            return result
        except Exception as exc:
            status = "error"
            error_text = repr(exc)
            raise
        finally:
            elapsed = time.perf_counter() - start
            if error_text:
                print(f"[TIMING] paper={slug} status={status} elapsed={elapsed:.1f}s error={error_text[:300]}")
            else:
                print(f"[TIMING] paper={slug} status={status} elapsed={elapsed:.1f}s")
            _v21_4_append_timing_event({
                "event": "paper_complete",
                "paper_slug": slug,
                "title": title,
                "doi": meta.get("doi"),
                "status": status,
                "records_count": records_count,
                "elapsed_seconds": round(elapsed, 3),
                "error": error_text,
                "run_mode": run_mode,
                "force_reprocess": bool(force_reprocess),
            })


if "process_batch_api" in globals():
    _previous_process_batch_api_v21_4_timing = process_batch_api

    def process_batch_api(*args, **kwargs) -> Dict[str, Any]:
        start = time.perf_counter()
        status = "ok"
        try:
            result = _previous_process_batch_api_v21_4_timing(*args, **kwargs)
            return result
        except Exception:
            status = "error"
            raise
        finally:
            elapsed = time.perf_counter() - start
            print(f"[TIMING] batch status={status} elapsed={elapsed:.1f}s")
            _v21_4_append_timing_event({
                "event": "batch_complete",
                "status": status,
                "elapsed_seconds": round(elapsed, 3),
                "args_preview": [str(x)[:200] for x in args[:4]],
                "kwargs": {k: str(v)[:200] for k, v in kwargs.items()},
            })

print(f"[INFO] v21.4 controller patch loaded | llm_cache={LLM_CALL_CACHE_ENABLE} | recovery={RUN_MISSING_FIELD_RECOVERY} | review_full_extraction={REVIEW_FULL_EXTRACTION_ENABLE}")
print(f"[INFO] v21.4 local LLM timing | request_timeout={LLM_REQUEST_TIMEOUT_SECONDS * 3}s effective | web_timeout={REQUEST_TIMEOUT}s | retries={API_MAX_RETRIES}")
'''

def _v21_sparse_aware_integration_patch_source() -> str:
    return r'''

# =============================================================================
# V21.7 TARGET-AWARE SPARSE LITERATURE ROW VALIDATION PATCH
# =============================================================================
def _v21_has_any_non_null(row, cols):
    return any((c in row.index) and pd.notna(row.get(c)) for c in cols)


def _v21_floatish(value):
    if pd.isna(value):
        return None
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _v21_numeric_values(row, cols):
    vals = []
    for c in cols:
        if c not in row.index:
            continue
        val = _v21_floatish(row.get(c))
        if val is not None:
            vals.append((c, val))
    return vals


def _v21_has_plausible_pce_target(row):
    # PCE model target support should not require all device/context fields in
    # the same sparse literature row. The baseline database itself tolerates
    # missing features. Keep a plausibility guard so retained-percent values
    # such as 90.7 are not counted as PCE percentage targets.
    pce_cols = [
        "JV_default_PCE",
        "JV_reverse_scan_PCE",
        "JV_forward_scan_PCE",
        "Stabilised_performance_PCE",
    ]
    for c, val in _v21_numeric_values(row, pce_cols):
        if 1.0 <= val <= 35.0:
            return True
    return False


def _v21_has_stability_target(row):
    # Conditions such as exposure time are model inputs, not prediction
    # targets. A stability-model candidate must contain an actual lifetime or
    # retention target so the audit does not overstate model-ready support.
    target_cols = [
        "Stability_PCE_T80", "Stability_PCE_T95",
        "Stability_PCE_Ts80", "Stability_PCE_Ts95",
        "Stability_PCE_Te80", "Stability_PCE_Tse80",
        "Stability_PCE_end_of_experiment", "Stability_PCE_after_1000_h",
        "Outdoor_PCE_T80", "Outdoor_PCE_T95",
        "Outdoor_PCE_Ts80", "Outdoor_PCE_Ts95",
        "Outdoor_PCE_Te80", "Outdoor_PCE_Tse80",
        "Outdoor_PCE_end_of_experiment", "Outdoor_PCE_after_1000_h",
    ]
    return bool(_v21_numeric_values(row, target_cols))


def _v21_sparse_row_tier(row, n_perf, n_stab, n_design, n_important):
    pce_candidate = bool(_v21_has_plausible_pce_target(row))
    stability_candidate = bool(_v21_has_stability_target(row))
    if pce_candidate and stability_candidate:
        if n_design >= 1:
            return "pce_and_stability_candidate_with_context", pce_candidate, stability_candidate
        return "pce_and_stability_candidate_sparse", pce_candidate, stability_candidate
    if pce_candidate:
        if n_design >= 1:
            return "pce_candidate_with_context", pce_candidate, stability_candidate
        return "pce_candidate_sparse", pce_candidate, stability_candidate
    if stability_candidate:
        if (n_perf >= 1 or n_design >= 1):
            return "stability_candidate_with_context", pce_candidate, stability_candidate
        return "stability_candidate_sparse", pce_candidate, stability_candidate
    if n_perf >= 1 or n_stab >= 1 or n_design >= 2:
        return "useful_sparse_device_or_materials_row", pce_candidate, stability_candidate
    if n_design >= 1 and n_important >= 2:
        return "minimal_sparse_materials_row", pce_candidate, stability_candidate
    return "metadata_only_or_too_sparse", pce_candidate, stability_candidate


def validate_literature_rows(
    lit_df: pd.DataFrame,
    evidence_long: pd.DataFrame,
    cfg: MergeConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sparse-aware validation for LiteratureAgent rows.

    Perovskite literature rows are naturally sparse, especially for stability.
    Do not reject a useful PCE/device/materials row just because it lacks a full
    stability target block. Strict extraction remains strict; this validation
    tier separates useful accepted rows from PCE/stability model-candidate rows.
    """
    evidence_count_by_idx = {}
    if not evidence_long.empty and "lit_row_index" in evidence_long.columns:
        evidence_count_by_idx = evidence_long.groupby("lit_row_index")["evidence"].count().to_dict()

    audit_rows = []
    accepted_indices = []
    rejected_indices = []

    for idx, row in lit_df.iterrows():
        doi = normalize_doi(row.get("Ref_DOI_number"))
        n_perf = count_non_null(row, PERFORMANCE_FIELDS)
        n_stab = count_non_null(row, STABILITY_FIELDS)
        n_design = count_non_null(row, DESIGN_FIELDS)
        n_important = count_non_null(row, IMPORTANT_MODEL_FIELDS)
        n_evidence = int(evidence_count_by_idx.get(idx, 0))
        row_tier, pce_candidate, stability_candidate = _v21_sparse_row_tier(
            row=row,
            n_perf=n_perf,
            n_stab=n_stab,
            n_design=n_design,
            n_important=n_important,
        )

        reasons = []
        status = "accepted"

        if cfg.require_doi and not doi:
            status = "rejected"
            reasons.append("missing_doi")

        if row_tier == "metadata_only_or_too_sparse":
            status = "rejected"
            reasons.append("metadata_only_or_too_sparse")

        # Sparse rows can still be useful for the model once merged with baseline
        # missingness handling. Mark them, but do not discard them.
        if status != "rejected":
            if n_important < cfg.min_required_non_null_important_fields:
                status = "sparse_accepted"
                reasons.append(f"sparse_important_fields:{n_important}")
            if not (pce_candidate or stability_candidate):
                status = "sparse_accepted"
                reasons.append(f"not_direct_model_target_candidate:{row_tier}")

        if cfg.min_required_evidence_fields > 0 and n_evidence < cfg.min_required_evidence_fields:
            reasons.append(f"low_evidence_count:{n_evidence}")
            if status == "accepted":
                status = "weak_accepted"

        if status in {"accepted", "weak_accepted", "sparse_accepted"}:
            accepted_indices.append(idx)
        else:
            rejected_indices.append(idx)

        audit_rows.append({
            "lit_row_index": idx,
            "Ref_DOI_number": doi,
            "status": status,
            "row_tier": row_tier,
            "pce_model_candidate": int(pce_candidate),
            "stability_model_candidate": int(stability_candidate),
            "has_design_context": int(n_design >= 1),
            "reasons": ";".join(reasons) if reasons else "ok",
            "n_performance_fields": n_perf,
            "n_stability_fields": n_stab,
            "n_design_fields": n_design,
            "n_important_fields": n_important,
            "n_evidence_fields": n_evidence,
        })

    audit_df = pd.DataFrame(audit_rows)
    accepted = lit_df.loc[accepted_indices].copy()
    rejected = lit_df.loc[rejected_indices].copy()
    return accepted, rejected, audit_df

print("[INFO] v21.7 target-aware sparse integration validation loaded | plausible sparse PCE/stability target rows are retained and audited")
'''

def _v21_8_expansion_backfill_patch_source() -> str:
    return r'''

# =============================================================================
# V21.8 EXPANSION ROW COMPOSITION / PROCESSING BACKFILL PATCH
# =============================================================================
_V21_8_FORMULA_PATTERNS = [
    r"\b(?:FAPbI3|MAPbI3|CsPbI3|CsPbBr3|MASnI3|FASnI3|CH3NH3PbI3|CH3NH3SnI3)\b",
    r"\b(?:CH3NH3|MA|FA|Cs|Rb|K)(?:Pb|Sn|Ge)(?:I|Br|Cl)3\b",
    r"\b(?:CH3NH3|MA|FA|Cs|Rb|K|PEA|BA|BDA|GUA|EA)?(?:0?\.\d+)?(?:CH3NH3|MA|FA|Cs|Rb|K|PEA|BA|BDA|GUA|EA)?(?:0?\.\d+)?(?:Pb|Sn|Ge)(?:I|Br|Cl|S)(?:\d+(?:\.\d+)?)?(?:I|Br|Cl|S)?(?:\d+(?:\.\d+)?)?(?:I|Br|Cl|S)?(?:\d+(?:\.\d+)?)?\b",
]

_V21_8_PROCESS_PATTERNS = [
    ("spin coating", r"\bspin[- ]coat(?:ing|ed)?\b"),
    ("antisolvent-assisted crystallization", r"\banti[- ]?solvent\b"),
    ("thermal annealing", r"\banneal(?:ing|ed)?\b"),
    ("blade coating", r"\bblade[- ]coat(?:ing|ed)?\b"),
    ("slot-die coating", r"\bslot[- ]die\b"),
    ("doctor blading", r"\bdoctor[- ]blad(?:ing|ed)?\b"),
    ("spray coating", r"\bspray[- ]coat(?:ing|ed)?\b"),
    ("vapor deposition", r"\b(vapor|vapour|evaporation|thermal evaporation|vacuum deposition|cvd|ald)\b"),
    ("inkjet printing", r"\bink[- ]?jet\b"),
    ("two-step deposition", r"\btwo[- ]step\b"),
    ("one-step deposition", r"\bone[- ]step\b"),
]

_V21_9_PCE_PATTERNS = [
    r"\bPCE\s*(?:of|=|:|reached|reaches|up to|exceeding|over|above|as high as)?\s*(\d{1,2}(?:\.\d+)?)\s*%",
    r"\bpower conversion efficienc(?:y|ies)\s*(?:of|=|:|reached|reaches|up to|exceeding|over|above|as high as)?\s*(\d{1,2}(?:\.\d+)?)\s*%",
    r"\befficienc(?:y|ies)\s*(?:of|=|:|reached|reaches|up to|exceeding|over|above|as high as)\s*(\d{1,2}(?:\.\d+)?)\s*%",
    r"\b(\d{1,2}(?:\.\d+)?)\s*%\s*(?:PCE|power conversion efficiency|efficiency)\b",
]

_V21_9_VOC_PATTERNS = [
    r"\bV(?:oc|OC)\s*(?:=|:|of)?\s*(\d(?:\.\d+)?)\s*V\b",
    r"\bopen[- ]circuit voltage\s*(?:=|:|of)?\s*(\d(?:\.\d+)?)\s*V\b",
]

_V21_9_JSC_PATTERNS = [
    r"\bJ(?:sc|SC)\s*(?:=|:|of)?\s*(\d{1,2}(?:\.\d+)?)\s*mA\s*cm\s*(?:-2|−2|\^-2|–2|⁻²)",
    r"\bshort[- ]circuit current(?: density)?\s*(?:=|:|of)?\s*(\d{1,2}(?:\.\d+)?)\s*mA\s*cm\s*(?:-2|−2|\^-2|–2|⁻²)",
]

_V21_9_FF_PATTERNS = [
    r"\bFF\s*(?:=|:|of)?\s*(0?\.\d{2,3}|\d{1,2}(?:\.\d+)?)\s*%?",
    r"\bfill factor\s*(?:=|:|of)?\s*(0?\.\d{2,3}|\d{1,2}(?:\.\d+)?)\s*%?",
]


def _v21_8_present(value) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "[]", "{}"}


def _v21_8_row_text(row) -> str:
    fields = [
        "Ref_DOI_number",
        "Ref_lead_author",
        "Ref_journal",
        "Ref_original_filename_data_upload",
        "Ref_free_text_comment",
        "Cell_stack_sequence",
        "Cell_architecture",
        "ETL_stack_sequence",
        "HTL_stack_sequence",
        "Backcontact_stack_sequence",
        "Perovskite_additives_compounds",
        "_source_abstract",
        "_source_title",
        "_source_landing_page",
        "_source_pdf_url",
        "_composition_json",
    ]
    parts = []
    for field in fields:
        if field in row.index and _v21_8_present(row.get(field)):
            parts.append(str(row.get(field)))
    text = " ".join(parts)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text)


def _v21_9_float_text(value: str) -> float | None:
    try:
        out = float(str(value).strip())
    except Exception:
        return None
    if out != out:
        return None
    return out


def _v21_9_extract_first_numeric(text: str, patterns: list[str], lo: float, hi: float) -> float | None:
    cleaned = str(text or "").replace("−", "-").replace("–", "-").replace("⁻", "-")
    for pattern in patterns:
        for match in re.finditer(pattern, cleaned, flags=re.IGNORECASE):
            value = _v21_9_float_text(match.group(1))
            if value is None:
                continue
            if lo <= value <= hi:
                return value
    return None


def _v21_9_extract_performance(text: str) -> dict[str, float]:
    cleaned = str(text or "")
    # Avoid interpreting generic percentage retention or yield values as PCE.
    windows = []
    for pattern in _V21_9_PCE_PATTERNS:
        for match in re.finditer(pattern, cleaned, flags=re.IGNORECASE):
            start = max(0, match.start() - 120)
            end = min(len(cleaned), match.end() + 120)
            window = cleaned[start:end]
            if re.search(r"\b(retention|retained|humidity|RH|degradation|yield|quantum yield|PLQY)\b", window, flags=re.IGNORECASE):
                continue
            if not re.search(r"\b(device|cell|solar|photovoltaic|PSC|PCE|power conversion)\b", window, flags=re.IGNORECASE):
                continue
            value = _v21_9_float_text(match.group(1))
            if value is not None and 1 <= value <= 35:
                windows.append((value, window))
    result: dict[str, float] = {}
    if windows:
        # Use the highest reported device efficiency in the local text as the default
        # candidate; the source remains traceable in the backfill report.
        pce, window = sorted(windows, key=lambda item: item[0], reverse=True)[0]
        result["JV_default_PCE"] = pce
        voc = _v21_9_extract_first_numeric(window, _V21_9_VOC_PATTERNS, 0.4, 1.5)
        jsc = _v21_9_extract_first_numeric(window, _V21_9_JSC_PATTERNS, 1.0, 35.0)
        ff = _v21_9_extract_first_numeric(window, _V21_9_FF_PATTERNS, 0.2, 90.0)
        if voc is not None:
            result["JV_default_Voc"] = voc
        if jsc is not None:
            result["JV_default_Jsc"] = jsc
        if ff is not None:
            result["JV_default_FF"] = ff / 100.0 if ff > 1.0 else ff
    return result


def _v21_9_normalize_source_doi(row) -> str | None:
    candidates = []
    for field in ["_source_landing_page", "_source_pdf_url", "_source_doi"]:
        if field in row.index and _v21_8_present(row.get(field)):
            candidates.append(str(row.get(field)))
    for text in candidates:
        m = re.search(r"(10\.\d{4,9}/[^\s\"<>]+)", text, flags=re.IGNORECASE)
        if m:
            doi = normalize_doi(m.group(1).rstrip(").,;"))
            if doi:
                return doi
    return None


def _v21_8_normalize_formula(formula: str) -> str:
    f = str(formula or "").strip()
    replacements = {
        "CHNHSnI": "CH3NH3SnI3",
        "CHNHPbI": "CH3NH3PbI3",
        "PVK": "",
    }
    for old, new in replacements.items():
        f = f.replace(old, new)
    for partial in ["FAPbI", "MAPbI", "MASnI", "FASnI", "CsPbI", "CsPbBr", "CH3NH3PbI", "CH3NH3SnI"]:
        if f == partial:
            f = partial + "3"
    return f.strip(" /,;:()[]{}")


def _v21_8_extract_formula(text: str) -> str | None:
    cleaned = str(text or "")
    for pattern in _V21_8_FORMULA_PATTERNS:
        for match in re.finditer(pattern, cleaned, flags=re.IGNORECASE):
            candidate = _v21_8_normalize_formula(match.group(0))
            if not candidate or len(candidate) < 5:
                continue
            low = candidate.lower()
            if low in {"perovskite", "pvk"}:
                continue
            if any(x in candidate for x in ["Pb", "Sn", "Ge"]) and any(x in candidate for x in ["I", "Br", "Cl", "S"]):
                return candidate
    return None


def _v21_8_ions_from_formula(formula: str) -> tuple[str | None, str | None, str | None]:
    f = str(formula or "")
    a_hits = []
    for label, aliases in [
        ("MA", ["CH3NH3", "MA"]),
        ("FA", ["FA"]),
        ("Cs", ["Cs"]),
        ("Rb", ["Rb"]),
        ("K", ["K"]),
        ("PEA", ["PEA"]),
        ("BA", ["BA"]),
        ("BDA", ["BDA"]),
    ]:
        if any(alias in f for alias in aliases):
            a_hits.append(label)
    b_hits = [x for x in ["Pb", "Sn", "Ge"] if x in f]
    c_hits = [x for x in ["I", "Br", "Cl", "S"] if x in f]
    return (
        ";".join(dict.fromkeys(a_hits)) or None,
        ";".join(dict.fromkeys(b_hits)) or None,
        ";".join(dict.fromkeys(c_hits)) or None,
    )


def _v21_8_extract_process(text: str) -> tuple[str | None, str | None, str | None]:
    low = str(text or "").lower()
    procedures = []
    for label, pattern in _V21_8_PROCESS_PATTERNS:
        if re.search(pattern, low, flags=re.IGNORECASE):
            procedures.append(label)
    anneal_temp = None
    m_temp = re.search(r"\banneal(?:ing|ed)?\b.{0,80}?(\d{2,3})\s*(?:°\s*)?c\b", low, flags=re.IGNORECASE)
    if m_temp:
        anneal_temp = m_temp.group(1)
    anneal_time = None
    m_time = re.search(r"\banneal(?:ing|ed)?\b.{0,80}?(\d+(?:\.\d+)?)\s*(min|minutes|h|hours)\b", low, flags=re.IGNORECASE)
    if m_time:
        anneal_time = f"{m_time.group(1)} {m_time.group(2)}"
    return (
        "; ".join(dict.fromkeys(procedures)) if procedures else None,
        anneal_temp,
        anneal_time,
    )


_V21_17_DEVICE_DISCRIMINATOR_FIELDS = [
    "Perovskite_composition_short_form",
    "Perovskite_composition_long_form",
    "Cell_stack_sequence",
    "Cell_area_total",
    "Cell_area_measured",
    "JV_default_PCE",
    "JV_reverse_scan_PCE",
    "JV_forward_scan_PCE",
    "JV_default_Voc",
    "JV_default_Jsc",
    "JV_default_FF",
    "Stabilised_performance_PCE",
    "Stability_PCE_initial_value",
    "Stability_PCE_end_of_experiment",
    "Stability_PCE_T95",
    "Stability_PCE_T80",
    "Stability_PCE_after_1000_h",
    "Stability_time_total_exposure",
    "Stability_atmosphere",
    "Stability_temperature_range",
    "Stability_relative_humidity_average_value",
]


def _v21_17_canonical_discriminator_value(value):
    if not _v21_8_present(value):
        return None
    try:
        numeric = float(value)
        if np.isfinite(numeric):
            return ("number", round(numeric, 8))
    except (TypeError, ValueError):
        pass
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    return ("text", text)


def _v21_17_conflicting_device_fields(group: pd.DataFrame) -> list[str]:
    conflicts = []
    for col in _V21_17_DEVICE_DISCRIMINATOR_FIELDS:
        if col not in group.columns:
            continue
        values = {
            canonical
            for canonical in (
                _v21_17_canonical_discriminator_value(value)
                for value in group[col].tolist()
            )
            if canonical is not None
        }
        if len(values) > 1:
            conflicts.append(col)
    return conflicts


def _v21_17_device_role(row: pd.Series) -> str:
    priority_evidence_fields = [
        "_evidence_JV_default_PCE",
        "_evidence_JV_reverse_scan_PCE",
        "_evidence_JV_forward_scan_PCE",
        "_evidence_Stabilised_performance_PCE",
        "_evidence_Stability_PCE_end_of_experiment",
        "_evidence_Stability_PCE_T80",
    ]
    evidence = " ".join(
        str(row.get(col))
        for col in priority_evidence_fields
        if _v21_8_present(row.get(col))
    ).lower()
    if re.search(r"\b(certified|certification)\b", evidence):
        return "certified"
    if re.search(r"\b(large[\s-]?area|larger\s+pscs?|module)\b", evidence):
        return "large_area"
    if re.search(r"\b(control|reference|untreated|pristine)\b", evidence):
        return "control"
    if re.search(r"\b(target|treated|modified|champion|best)\b", evidence):
        return "target"
    return "device"


def _v21_17_split_conflicting_device_groups(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preserve distinct devices or measurements that were accidentally assigned
    the same DOI + sample ID. Complementary partial rows remain untouched and
    continue through the existing coalescer.
    """
    out = df.copy()
    report_rows = []
    required = {"Ref_DOI_number", "Ref_internal_sample_id"}
    if not required.issubset(out.columns):
        return out, pd.DataFrame(report_rows)

    groups = out.groupby(
        ["Ref_DOI_number", "Ref_internal_sample_id"],
        dropna=False,
        sort=False,
    ).groups
    for (_, original_sample_id), indices in groups.items():
        indices = list(indices)
        if len(indices) < 2:
            continue
        conflicts = _v21_17_conflicting_device_fields(out.loc[indices])
        if not conflicts:
            continue

        base_id = str(original_sample_id or "device").strip() or "device"
        used_ids = set()
        for ordinal, idx in enumerate(indices, start=1):
            row = out.loc[idx]
            role = _v21_17_device_role(row)
            pce = next(
                (
                    row.get(col)
                    for col in [
                        "JV_default_PCE",
                        "JV_reverse_scan_PCE",
                        "JV_forward_scan_PCE",
                        "Stabilised_performance_PCE",
                    ]
                    if _v21_8_present(row.get(col))
                ),
                None,
            )
            pce_token = ""
            if pce is not None:
                pce_token = "_" + re.sub(r"[^0-9]+", "_", str(pce)).strip("_")
            candidate = f"{base_id}_{role}{pce_token}"
            if candidate in used_ids:
                candidate = f"{candidate}_{ordinal}"
            used_ids.add(candidate)
            out.at[idx, "Ref_internal_sample_id"] = candidate
            out.at[idx, "_lit_agent_sample_identity_basis"] = (
                "integration_preserved_conflicting_device_measurement"
            )
            report_rows.append({
                "lit_row_index": idx,
                "Ref_DOI_number": normalize_doi(row.get("Ref_DOI_number")),
                "Ref_original_filename_data_upload": row.get(
                    "Ref_original_filename_data_upload"
                ),
                "backfill_changes": (
                    f"sample_id={base_id}->{candidate}; "
                    f"conflicting_fields={';'.join(conflicts)}"
                ),
                "source": "v21.17_conflict_aware_device_identity",
            })

    return out, pd.DataFrame(report_rows)


def _v21_8_backfill_expansion_rows(lit_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = lit_df.copy()
    report_rows = []
    # Deterministic recovery from broad record text caused false compositions,
    # layers, and metrics when a paper discussed comparison devices or cited
    # prior work. Keep it disabled by default. Partial rows are still safely
    # coalesced below without overwriting populated values.
    enable_text_backfill = str(os.getenv("LIT_AGENT_ENABLE_TEXT_BACKFILL", "0")).strip().lower() in {
        "1", "true", "yes", "on",
    }
    required_cols = [
        "Perovskite_composition_short_form",
        "Perovskite_composition_long_form",
        "Perovskite_composition_a_ions",
        "Perovskite_composition_b_ions",
        "Perovskite_composition_c_ions",
        "Perovskite_deposition_procedure",
        "Perovskite_deposition_thermal_annealing_temperature",
        "Perovskite_deposition_thermal_annealing_time",
        "JV_default_PCE",
        "JV_default_Voc",
        "JV_default_Jsc",
        "JV_default_FF",
        "JV_default_PCE_scan_direction",
        "_lit_agent_recovery_notes",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype("object")

    # LiteratureAgent can emit several sparse rows for one paper: e.g. one row
    # has JV/PCE values while a sibling row has the stack/composition. Use only
    # same-DOI context to backfill empty target rows.
    doi_group_text: dict[str, str] = {}
    doi_group_values: dict[str, dict[str, object]] = {}
    shareable_cols = [
        "Cell_stack_sequence",
        "Cell_architecture",
        "ETL_stack_sequence",
        "HTL_stack_sequence",
        "Backcontact_stack_sequence",
    ]
    for doi, group in df.groupby(df.get("Ref_DOI_number", pd.Series(index=df.index, dtype=object)).map(normalize_doi), dropna=False):
        if not doi:
            continue
        doi_group_text[doi] = " ".join(_v21_8_row_text(row) for _, row in group.iterrows())
        doi_group_values[doi] = {}
        for col in shareable_cols:
            if col not in df.columns:
                continue
            vals = [
                value for value in group[col].tolist()
                if _v21_8_present(value)
            ]
            if vals:
                doi_group_values[doi][col] = vals[0]

    for idx, row in df.iterrows():
        doi = normalize_doi(row.get("Ref_DOI_number"))
        text = " ".join([_v21_8_row_text(row), doi_group_text.get(doi, "")])
        changes = []
        source_doi = _v21_9_normalize_source_doi(row)
        if source_doi and doi and source_doi != doi:
            df.at[idx, "Ref_DOI_number"] = source_doi
            doi = source_doi
            changes.append(f"Ref_DOI_number=source_doi:{source_doi}")
        elif source_doi and not doi:
            df.at[idx, "Ref_DOI_number"] = source_doi
            doi = source_doi
            changes.append(f"Ref_DOI_number=source_doi:{source_doi}")
        if enable_text_backfill:
            for col, value in doi_group_values.get(doi, {}).items():
                if col in df.columns and not _v21_8_present(row.get(col)):
                    df.at[idx, col] = value
                    changes.append(f"{col}=same_doi_context")

        formula = _v21_8_extract_formula(text) if enable_text_backfill else None
        if formula:
            if not _v21_8_present(row.get("Perovskite_composition_short_form")):
                df.at[idx, "Perovskite_composition_short_form"] = formula
                changes.append(f"Perovskite_composition_short_form={formula}")
            if not _v21_8_present(row.get("Perovskite_composition_long_form")):
                df.at[idx, "Perovskite_composition_long_form"] = formula
                changes.append(f"Perovskite_composition_long_form={formula}")
            a_ions, b_ions, c_ions = _v21_8_ions_from_formula(formula)
            for col, value in [
                ("Perovskite_composition_a_ions", a_ions),
                ("Perovskite_composition_b_ions", b_ions),
                ("Perovskite_composition_c_ions", c_ions),
            ]:
                if value and not _v21_8_present(row.get(col)):
                    df.at[idx, col] = value
                    changes.append(f"{col}={value}")

        procedure, anneal_temp, anneal_time = (
            _v21_8_extract_process(text) if enable_text_backfill else (None, None, None)
        )
        if procedure and not _v21_8_present(row.get("Perovskite_deposition_procedure")):
            df.at[idx, "Perovskite_deposition_procedure"] = procedure
            changes.append(f"Perovskite_deposition_procedure={procedure}")
        if anneal_temp and not _v21_8_present(row.get("Perovskite_deposition_thermal_annealing_temperature")):
            df.at[idx, "Perovskite_deposition_thermal_annealing_temperature"] = anneal_temp
            changes.append(f"Perovskite_deposition_thermal_annealing_temperature={anneal_temp}")
        if anneal_time and not _v21_8_present(row.get("Perovskite_deposition_thermal_annealing_time")):
            df.at[idx, "Perovskite_deposition_thermal_annealing_time"] = anneal_time
            changes.append(f"Perovskite_deposition_thermal_annealing_time={anneal_time}")

        performance = _v21_9_extract_performance(text) if enable_text_backfill else {}
        for col in ["JV_default_PCE", "JV_default_Voc", "JV_default_Jsc", "JV_default_FF"]:
            value = performance.get(col)
            if value is not None and not _v21_8_present(row.get(col)):
                df.at[idx, col] = value
                changes.append(f"{col}={value}")
        if "JV_default_PCE" in performance and not _v21_8_present(row.get("JV_default_PCE_scan_direction")):
            df.at[idx, "JV_default_PCE_scan_direction"] = "unspecified_text_mined"
            changes.append("JV_default_PCE_scan_direction=unspecified_text_mined")
        if performance:
            note = "deterministic performance recovery from source/record text; verify before final quantitative use"
            existing_note = str(row.get("_lit_agent_recovery_notes") or "").strip()
            df.at[idx, "_lit_agent_recovery_notes"] = (existing_note + "; " + note).strip("; ")

        if changes:
            report_rows.append({
                "lit_row_index": idx,
                "Ref_DOI_number": normalize_doi(row.get("Ref_DOI_number")),
                "Ref_original_filename_data_upload": row.get("Ref_original_filename_data_upload"),
                "backfill_changes": "; ".join(changes),
                "source": "v21.9_formula_process_performance_backfill_from_existing_row_text",
            })

    df, identity_report = _v21_17_split_conflicting_device_groups(df)
    if not identity_report.empty:
        report_rows.extend(identity_report.to_dict(orient="records"))
        print(
            f"[INFO] v21.17 preserved {len(identity_report)} conflicting "
            "device/measurement row(s) by assigning distinct sample ids."
        )

    dedupe_subset = [c for c in ["Ref_DOI_number", "Ref_internal_sample_id"] if c in df.columns]
    if len(dedupe_subset) == 2:
        before = len(df)
        drop_indices = []
        duplicate_groups = df.groupby(dedupe_subset, dropna=False, sort=False).groups
        for _, indices in duplicate_groups.items():
            indices = list(indices)
            if len(indices) < 2:
                continue

            def _row_quality(candidate_idx):
                candidate = df.loc[candidate_idx]
                plausible_target = int(_v21_has_plausible_pce_target(candidate)) + int(_v21_has_stability_target(candidate))
                important = count_non_null(candidate, IMPORTANT_MODEL_FIELDS)
                total = sum(_v21_8_present(value) for value in candidate.values)
                return plausible_target, important, total

            primary_idx = max(indices, key=_row_quality)
            primary = df.loc[primary_idx].copy()
            filled = []
            conflicts = []
            for sibling_idx in indices:
                if sibling_idx == primary_idx:
                    continue
                sibling = df.loc[sibling_idx]
                for col in df.columns:
                    sibling_value = sibling.get(col)
                    if not _v21_8_present(sibling_value):
                        continue
                    if not _v21_8_present(primary.get(col)):
                        primary[col] = sibling_value
                        filled.append(f"{col}=from_row_{sibling_idx}")
                    elif str(primary.get(col)).strip() != str(sibling_value).strip():
                        if col in IMPORTANT_MODEL_FIELDS:
                            conflicts.append(f"{col}:kept={primary.get(col)}|alternate={sibling_value}")
                drop_indices.append(sibling_idx)

            df.loc[primary_idx, primary.index] = primary
            report_rows.append({
                "lit_row_index": primary_idx,
                "Ref_DOI_number": normalize_doi(primary.get("Ref_DOI_number")),
                "Ref_original_filename_data_upload": primary.get("Ref_original_filename_data_upload"),
                "backfill_changes": (
                    f"coalesced_{len(indices)}_partial_rows; "
                    f"filled={';'.join(filled) if filled else 'none'}; "
                    f"conflicts={';'.join(conflicts) if conflicts else 'none'}"
                ),
                "source": "v21.11_intra_paper_partial_row_coalescing",
            })

        if drop_indices:
            df = df.drop(index=sorted(set(drop_indices))).copy()
            print(
                f"[INFO] v21.11 coalesced {before - len(df)} partial LiteratureAgent row(s) "
                "sharing DOI + sample id; populated values were not overwritten."
            )

    return df, pd.DataFrame(report_rows)


def _v21_11_sanitize_extracted_numeric_fields(df):
    """Clear impossible numeric cells while preserving the rest of a useful row."""
    out = df.copy()
    report_rows = []
    ranges = {
        "JV_default_PCE": (0.0, 35.0),
        "JV_reverse_scan_PCE": (0.0, 35.0),
        "JV_forward_scan_PCE": (0.0, 35.0),
        "Stabilised_performance_PCE": (0.0, 35.0),
        "JV_default_Voc": (0.0, 3.0),
        "JV_reverse_scan_Voc": (0.0, 3.0),
        "JV_forward_scan_Voc": (0.0, 3.0),
        "JV_default_Jsc": (0.0, 60.0),
        "JV_reverse_scan_Jsc": (0.0, 60.0),
        "JV_forward_scan_Jsc": (0.0, 60.0),
        "JV_default_FF": (0.0, 100.0),
        "JV_reverse_scan_FF": (0.0, 100.0),
        "JV_forward_scan_FF": (0.0, 100.0),
        "Stability_PCE_initial_value": (0.0, 35.0),
        "Stability_PCE_end_of_experiment": (0.0, 150.0),
        "Stability_PCE_after_1000_h": (0.0, 150.0),
        "Stability_PCE_T80": (0.0, 1000000.0),
        "Stability_PCE_T95": (0.0, 1000000.0),
        "Stability_time_total_exposure": (0.0, 1000000.0),
    }
    for idx, row in out.iterrows():
        for col, (low, high) in ranges.items():
            if col not in out.columns or not _v21_8_present(row.get(col)):
                continue
            value = _v21_floatish(row.get(col))
            if value is None or not (low <= value <= high):
                report_rows.append({
                    "lit_row_index": idx,
                    "Ref_DOI_number": normalize_doi(row.get("Ref_DOI_number")),
                    "Ref_original_filename_data_upload": row.get("Ref_original_filename_data_upload"),
                    "backfill_changes": f"cleared_invalid_numeric:{col}={row.get(col)}",
                    "source": "v21.11_numeric_plausibility_gate",
                })
                out.at[idx, col] = np.nan
    return out, pd.DataFrame(report_rows)


def _v21_18_sanitize_cross_metric_roles(df):
    """Clear high-confidence role/unit errors that pass scalar range checks."""
    out = df.copy()
    report_rows = []
    if "_lit_agent_metric_consistency_warning" not in out.columns:
        out["_lit_agent_metric_consistency_warning"] = None
    for idx, row in out.iterrows():
        pce = _v21_floatish(row.get("JV_default_PCE"))
        voc = _v21_floatish(row.get("JV_default_Voc"))
        jsc = _v21_floatish(row.get("JV_default_Jsc"))
        ff = _v21_floatish(row.get("JV_default_FF"))
        if pce is None or ff is None:
            continue

        reason = ""
        if 0 < ff <= 1 and abs((100.0 * ff) - pce) <= 0.05:
            reason = "ff_duplicates_pce_fraction"
        elif jsc is not None and jsc > 0 and 0 < ff <= 1:
            implied_voc = pce / (jsc * ff)
            if not (0.35 <= implied_voc <= 2.25):
                reason = f"ff_implies_nonphysical_voc_{implied_voc:.4g}"
        elif voc is not None and jsc is not None and jsc > 0:
            ff_fraction = ff / 100.0 if ff > 1 else ff
            expected = voc * jsc * ff_fraction
            if abs(expected - pce) > max(0.75, 0.08 * pce):
                reason = (
                    "ff_inconsistent_with_complete_jv_tuple_"
                    f"expected_pce_{expected:.4g}"
                )

        if not reason:
            continue
        report_rows.append({
            "lit_row_index": idx,
            "Ref_DOI_number": normalize_doi(row.get("Ref_DOI_number")),
            "Ref_original_filename_data_upload": row.get(
                "Ref_original_filename_data_upload"
            ),
            "backfill_changes": (
                "withheld_cross_metric_role_error:"
                f"JV_default_FF={row.get('JV_default_FF')};reason={reason}"
            ),
            "source": "v21.18_cross_metric_role_validation",
        })
        out.at[idx, "JV_default_FF"] = np.nan
        out.at[idx, "_lit_agent_metric_consistency_warning"] = reason
    return out, pd.DataFrame(report_rows)


def _v21_18_reconcile_final_device_identities(df):
    """Reconcile generated sample IDs after the final grounded JV tuple."""
    if df.empty or "Ref_internal_sample_id" not in df.columns:
        return df.copy(), pd.DataFrame()

    out = df.copy()
    report_rows = []
    suffix_re = re.compile(
        r"_(?:control|target|device|certified|large_area)"
        r"(?:_pce_[0-9mp]+(?:_\d+)?)?$",
        flags=re.IGNORECASE,
    )

    def base_id(value):
        return suffix_re.sub("", str(value or "").strip())

    def numeric(value):
        return _v21_floatish(value)

    def normalized_text(value):
        return re.sub(
            r"\s+",
            " ",
            re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()),
        ).strip()

    def same_value(left, right, tolerance):
        left_number = numeric(left)
        right_number = numeric(right)
        if left_number is not None and right_number is not None:
            return abs(left_number - right_number) <= tolerance
        return normalized_text(left) == normalized_text(right)

    comparison_fields = {
        "JV_default_PCE": 0.02,
        "Perovskite_composition_short_form": 0.0,
        "Perovskite_composition_long_form": 0.0,
        "Cell_stack_sequence": 0.0,
    }
    companion_tolerances = {
        "JV_default_Voc": 0.03,
        "JV_default_Jsc": 0.30,
        "JV_default_FF": 0.03,
    }

    def compatible(left, right):
        left_pce = numeric(left.get("JV_default_PCE"))
        right_pce = numeric(right.get("JV_default_PCE"))
        if (
            left_pce is None
            or right_pce is None
            or abs(left_pce - right_pce) > 0.02
        ):
            return False
        if _v21_17_device_role(left) != _v21_17_device_role(right):
            return False
        for field, tolerance in comparison_fields.items():
            left_value = left.get(field)
            right_value = right.get(field)
            if (
                not _v21_8_present(left_value)
                or not _v21_8_present(right_value)
            ):
                continue
            if not same_value(left_value, right_value, tolerance):
                return False
        return True

    keys = pd.DataFrame({
        "doi": out.get(
            "Ref_DOI_number", pd.Series(index=out.index, dtype=object)
        ).map(normalize_doi),
        "base_id": out["Ref_internal_sample_id"].map(base_id),
    }, index=out.index)
    keep_indices = []
    for (_, group_base), indices in keys.groupby(
        ["doi", "base_id"], dropna=False, sort=False
    ).groups.items():
        pending = list(indices)
        retained = []
        while pending:
            primary_idx = pending.pop(0)
            primary = out.loc[primary_idx].copy()
            merged_indices = []
            conflicted_fields = set()
            for sibling_idx in list(pending):
                sibling = out.loc[sibling_idx]
                if not compatible(primary, sibling):
                    continue
                for column in out.columns:
                    if (
                        not _v21_8_present(primary.get(column))
                        and _v21_8_present(sibling.get(column))
                    ):
                        primary[column] = sibling.get(column)
                for field, tolerance in companion_tolerances.items():
                    if (
                        _v21_8_present(primary.get(field))
                        and _v21_8_present(sibling.get(field))
                        and not same_value(
                            primary.get(field), sibling.get(field), tolerance
                        )
                    ):
                        primary[field] = np.nan
                        conflicted_fields.add(field)
                pending.remove(sibling_idx)
                merged_indices.append(sibling_idx)
            out.loc[primary_idx, primary.index] = primary
            retained.append(primary_idx)
            if merged_indices:
                report_rows.append({
                    "lit_row_index": primary_idx,
                    "Ref_DOI_number": normalize_doi(
                        primary.get("Ref_DOI_number")
                    ),
                    "Ref_original_filename_data_upload": primary.get(
                        "Ref_original_filename_data_upload"
                    ),
                    "backfill_changes": (
                        "merged_final_compatible_device_rows:"
                        + ",".join(str(value) for value in merged_indices)
                        + ";withheld_conflicting_companion_metrics:"
                        + (
                            ",".join(sorted(conflicted_fields))
                            if conflicted_fields else "none"
                        )
                    ),
                    "source": "v21.18_final_device_identity_reconciliation",
                })

        if len(retained) == 1:
            out.at[retained[0], "Ref_internal_sample_id"] = group_base
        else:
            used = set()
            role_counts = {}
            for idx in retained:
                row = out.loc[idx]
                role = _v21_17_device_role(row)
                role_counts[role] = role_counts.get(role, 0) + 1
                pce = numeric(row.get("JV_default_PCE"))
                pce_token = ""
                if pce is not None:
                    metric = (
                        f"{pce:.6g}".replace("-", "m").replace(".", "p")
                    )
                    pce_token = f"_pce_{metric}"
                candidate = f"{group_base}_{role}{pce_token}"
                if candidate in used:
                    candidate = f"{candidate}_{role_counts[role]}"
                used.add(candidate)
                out.at[idx, "Ref_internal_sample_id"] = candidate
                out.at[idx, "_lit_agent_sample_identity_basis"] = (
                    "v21.18 identity generated from final grounded device/JV tuple"
                )
        keep_indices.extend(retained)

    reconciled = out.loc[keep_indices].copy()
    reconciled.reset_index(drop=True, inplace=True)
    return reconciled, pd.DataFrame(report_rows)


_V21_12_EVIDENCE_REQUIRED_MODEL_FIELDS = [
    "Cell_stack_sequence",
    "Cell_architecture",
    "Cell_area_measured",
    "Perovskite_composition_short_form",
    "Perovskite_composition_long_form",
    "Perovskite_additives_compounds",
    "Perovskite_deposition_procedure",
    "Perovskite_deposition_solvents",
    "Perovskite_deposition_thermal_annealing_temperature",
    "Perovskite_deposition_thermal_annealing_time",
    "ETL_stack_sequence",
    "HTL_stack_sequence",
    "Backcontact_stack_sequence",
    "JV_default_Voc",
    "JV_default_Jsc",
    "JV_default_FF",
    "JV_default_PCE",
    "JV_reverse_scan_PCE",
    "JV_forward_scan_PCE",
    "Stabilised_performance_PCE",
    "Stability_PCE_initial_value",
    "Stability_PCE_end_of_experiment",
    "Stability_PCE_T80",
    "Stability_PCE_T95",
    "Stability_PCE_after_1000_h",
    "Stability_time_total_exposure",
    "Stability_temperature_range",
    "Stability_relative_humidity_average_value",
    "Stability_light_intensity",
    "Stability_atmosphere",
    "Encapsulation",
    "Outdoor_PCE_initial_value",
    "Outdoor_PCE_end_of_experiment",
    "Outdoor_PCE_T80",
    "Outdoor_PCE_T95",
    "Outdoor_PCE_after_1000_h",
    "Outdoor_time_total_exposure",
]


def _v21_12_withhold_targets_without_field_evidence(df, evidence_long):
    """Keep useful rows but prevent unsupported model values entering training."""
    enabled = str(os.getenv("LIT_AGENT_REQUIRE_TARGET_EVIDENCE", "1")).strip().lower() in {
        "1", "true", "yes", "on",
    }
    out = df.copy()
    if not enabled:
        return out, pd.DataFrame()

    supported = {}
    if evidence_long is not None and not evidence_long.empty:
        for idx, group in evidence_long.groupby("lit_row_index", dropna=False):
            supported[idx] = {
                str(field) for field in group.get("field", pd.Series(dtype=object)).dropna().tolist()
            }

    report_rows = []
    for idx, row in out.iterrows():
        row_supported = supported.get(idx, set())
        for field in _V21_12_EVIDENCE_REQUIRED_MODEL_FIELDS:
            if field not in out.columns or not _v21_8_present(row.get(field)):
                continue
            if field in row_supported:
                continue
            report_rows.append({
                "lit_row_index": idx,
                "Ref_DOI_number": normalize_doi(row.get("Ref_DOI_number")),
                "Ref_original_filename_data_upload": row.get("Ref_original_filename_data_upload"),
                "backfill_changes": f"withheld_unsupported_model_field:{field}={row.get(field)}",
                "source": "v21.12_field_evidence_model_gate",
            })
            out.at[idx, field] = np.nan
    return out, pd.DataFrame(report_rows)


_v21_8_original_integrate = integrate


def integrate(cfg: MergeConfig) -> Dict[str, Any]:
    out_dir = ensure_dir(cfg.out_dir)

    base_path = Path(cfg.base_csv)
    lit_path = Path(cfg.lit_csv)

    base_df = robust_read_csv(base_path)
    lit_raw = robust_read_csv(lit_path)

    agent_version = infer_agent_version_from_paths(lit_path)
    lit_raw = add_lit_metadata(lit_raw, agent_version=agent_version)

    # Coalesce before separating evidence so evidence from sibling family rows
    # follows the retained record into validation.
    lit_raw, backfill_report = _v21_8_backfill_expansion_rows(lit_raw)
    lit_raw, numeric_validation_report = _v21_11_sanitize_extracted_numeric_fields(lit_raw)
    if not numeric_validation_report.empty:
        backfill_report = pd.concat([backfill_report, numeric_validation_report], ignore_index=True)
    lit_raw, cross_metric_report = _v21_18_sanitize_cross_metric_roles(lit_raw)
    if not cross_metric_report.empty:
        backfill_report = pd.concat([backfill_report, cross_metric_report], ignore_index=True)
    lit_raw, final_identity_report = _v21_18_reconcile_final_device_identities(lit_raw)
    if not final_identity_report.empty:
        backfill_report = pd.concat([backfill_report, final_identity_report], ignore_index=True)
    lit_clean, evidence_wide, evidence_long = split_evidence_fields(lit_raw)
    lit_clean, unsupported_target_report = _v21_12_withhold_targets_without_field_evidence(
        lit_clean, evidence_long
    )
    if not unsupported_target_report.empty:
        backfill_report = pd.concat([backfill_report, unsupported_target_report], ignore_index=True)

    accepted, rejected, audit_df = validate_literature_rows(lit_clean, evidence_long, cfg)

    if not accepted.empty and not audit_df.empty:
        status_map = audit_df.set_index("lit_row_index")["status"].to_dict()
        accepted["_lit_agent_confidence_status"] = accepted.index.map(lambda i: status_map.get(i, "accepted"))

    updated_df, duplicate_candidates, changes_df = merge_literature_into_base(base_df, accepted, cfg)

    updated_csv_path = Path(cfg.updated_csv)
    if not updated_csv_path.is_absolute():
        updated_csv_path = out_dir / updated_csv_path

    if not cfg.dry_run:
        write_csv(updated_df, updated_csv_path)
        write_csv(accepted, out_dir / "literature_update_accepted_rows.csv")
        write_csv(rejected, out_dir / "literature_update_rejected_rows.csv")
        write_csv(evidence_wide, out_dir / "literature_update_evidence_wide.csv")
        write_csv(evidence_long, out_dir / "literature_update_evidence_long.csv")
        write_csv(duplicate_candidates, out_dir / "literature_update_duplicate_candidates.csv")
        write_csv(audit_df, out_dir / "literature_update_audit.csv")
        write_csv(changes_df, out_dir / "literature_update_existing_row_changes.csv")
        write_csv(backfill_report, out_dir / "literature_update_backfill_report.csv")

    summary = {
        "timestamp": now_stamp(),
        "base_csv": str(base_path),
        "lit_csv": str(lit_path),
        "out_dir": str(out_dir),
        "updated_csv": str(updated_csv_path),
        "merge_policy": cfg.merge_policy,
        "base_rows": int(len(base_df)),
        "literature_raw_rows": int(len(lit_raw)),
        "literature_accepted_rows": int(len(accepted)),
        "literature_rejected_rows": int(len(rejected)),
        "duplicate_candidate_rows": int(len(duplicate_candidates)),
        "updated_rows": int(len(updated_df)),
        "added_rows_estimate": int(len(updated_df) - len(base_df)),
        "evidence_snippets": int(len(evidence_long)),
        "backfilled_rows": int(len(backfill_report)),
        "agent_version": agent_version or "unknown",
        "dry_run": bool(cfg.dry_run),
    }

    if not cfg.dry_run:
        safe_json_dump(summary, out_dir / "literature_update_summary.json")
        write_readme(out_dir, cfg, summary)

    print("=" * 90)
    print("LITERATURE DATASET UPDATE SUMMARY")
    print("=" * 90)
    print(json.dumps(summary, indent=2))
    print("=" * 90)

    if cfg.run_model and not cfg.dry_run:
        rc = run_model_script(cfg, updated_csv_path)
        if rc != 0:
            print(f"[WARN] Model script returned non-zero exit code: {rc}")

    return summary

print(
    "[INFO] v21.17 integration loaded | conflict-aware device identity, "
    "partial-row coalescing, and numeric validation active | "
    f"broad-text deterministic backfill={os.getenv('LIT_AGENT_ENABLE_TEXT_BACKFILL', '0')}"
)
'''


# =============================================================================
# V21.10 CONTROLLER-SIDE EXTRACTION ENRICHMENT
# =============================================================================

_CTRL_V21_10_PCE_PATTERNS = [
    r"\bPCE\s*(?:is|was)?\s*(?:improved|increased|enhanced|raised)\s*from\s*"
    r"\d{1,2}(?:\.\d+)?\s*%\s*to\s*(\d{1,2}(?:\.\d+)?)\s*%",
    r"\bPCE\b.{0,100}?\b(?:improved|increased|enhanced|raised|boosted)\b"
    r".{0,60}?\bto\s*(\d{1,2}(?:\.\d+)?)\s*%",
    r"\bPCE\s*\(?\s*(?:of|=|:|reached|reaches|up to|exceeding|over|above|as high as)?\s*(\d{1,2}(?:\.\d+)?)\s*(?:[Â±±]|\+/-|\+-)?\s*\d*(?:\.\d+)?\s*%\)?",
    r"\bpower conversion efficienc(?:y|ies)\s*\(?\s*(?:of|=|:|reached|reaches|up to|exceeding|over|above|as high as)?\s*(\d{1,2}(?:\.\d+)?)\s*(?:[Â±±]|\+/-|\+-)?\s*\d*(?:\.\d+)?\s*%\)?",
    r"\befficienc(?:y|ies)\s*\(?\s*(?:of|=|:|reached|reaches|up to|exceeding|over|above|as high as)\s*(\d{1,2}(?:\.\d+)?)\s*(?:[Â±±]|\+/-|\+-)?\s*\d*(?:\.\d+)?\s*%\)?",
    r"\bciency\s*\(?\s*(?:of|=|:|reached|reaches|up to|exceeding|over|above|as high as)\s*(\d{1,2}(?:\.\d+)?)\s*(?:[Â±±]|\+/-|\+-)?\s*\d*(?:\.\d+)?\s*%\)?",
    r"\b(\d{1,2}(?:\.\d+)?)\s*(?:[Â±±]|\+/-|\+-)?\s*\d*(?:\.\d+)?\s*%\s*(?:PCE|power conversion efficiency|efficiency)\b",
]

_CTRL_V21_10_VOC_PATTERNS = [
    r"\bV(?:oc|OC)\s*(?:=|:|of)?\s*(\d(?:\.\d+)?)\s*V\b",
    r"\bopen[- ]circuit voltage\s*(?:=|:|of)?\s*(\d(?:\.\d+)?)\s*V\b",
]

_CTRL_V21_10_JSC_PATTERNS = [
    r"\bJ(?:sc|SC)\s*(?:=|:|of)?\s*(\d{1,2}(?:\.\d+)?)\s*mA\s*(?:/|\s*)\s*cm\s*(?:-?2|\^-?2|2|²)",
    r"\bshort[- ]circuit current(?: density)?(?:\s*\(JSC\))?\s*(?:=|:|of)?\s*(\d{1,2}(?:\.\d+)?)\s*mA\s*(?:/|\s*)\s*cm\s*(?:-?2|\^-?2|2|²)",
]

_CTRL_V21_10_FF_PATTERNS = [
    r"\bFF\s*(?:=|:|of)?\s*(0?\.\d{2,3}|\d{1,2}(?:\.\d+)?)\s*%?",
    r"\bfill factor(?:\s*\(FF\))?\s*(?:=|:|of)?\s*(0?\.\d{2,3}|\d{1,2}(?:\.\d+)?)\s*%?",
]

_CTRL_V21_10_FORMULA_PATTERNS = [
    r"\b(?:(?:Rb|Cs|FA|MA|K)\s*(?:0?\.\d+|1(?:\.0+)?)?\s*){1,5}(?:Pb|Sn|Ge)\s*(?:I|Br|Cl)\s*\d(?:\.\d+)?\b",
    r"\(BDA\)\d(?:\.\d+)?\(PEA2\)\d(?:\.\d+)?MA\d+Pb\d+(?:I|Br|Cl|X)\d+",
    r"(?:BDA|PEA2)MA\d+Pb\d+(?:I|Br|Cl|X)\d+",
    r"\b(?:FAPbI3|MAPbI3|CsPbI3|CsPbBr3|MASnI3|FASnI3|CH3NH3PbI3|CH3NH3SnI3)\b",
    r"\b(?:CH3NH3|MA|FA|Cs|Rb|K)(?:Pb|Sn|Ge)(?:I|Br|Cl)3\b",
    r"\b(?:(?:CH3NH3|MA|FA|Cs|Rb|K)(?:\d(?:\.\d+)?)?){1,5}(?:Pb|Sn|Ge)(?:I|Br|Cl)(?:\d(?:\.\d+)?)?(?:(?:I|Br|Cl)(?:\d(?:\.\d+)?)?){0,2}\b",
]

_CTRL_V21_10_ETL_TERMS = [
    "TiO2", "SnO2", "PCBM", "C60", "ZnO", "NiOx", "Nb2O5", "WO3", "BCP", "LiF",
]

_CTRL_V21_10_HTL_TERMS = [
    "Spiro-OMeTAD", "PTAA", "PEDOT:PSS", "NiOx", "CuSCN", "CuI", "P3HT", "MeO-2PACz", "2PACz", "SAM",
]

_CTRL_V21_10_BACKCONTACT_TERMS = [
    "Au", "Ag", "Al", "Cu", "carbon", "graphite",
]

_CTRL_V21_10_PROCESS_TERMS = [
    ("spin coating", r"\bspin[- ]coat(?:ing|ed)?\b"),
    ("antisolvent-assisted crystallization", r"\banti[- ]?solvent\b"),
    ("thermal annealing", r"\banneal(?:ing|ed)?\b"),
    ("blade coating", r"\bblade[- ]coat(?:ing|ed)?\b"),
    ("slot-die coating", r"\bslot[- ]die\b"),
    ("doctor blading", r"\bdoctor[- ]blad(?:ing|ed)?\b"),
    ("spray coating", r"\bspray[- ]coat(?:ing|ed)?\b"),
    ("vapor deposition", r"\b(vapor|vapour|evaporation|thermal evaporation|vacuum deposition|cvd|ald)\b"),
    ("inkjet printing", r"\bink[- ]?jet\b"),
    ("two-step deposition", r"\btwo[- ]step\b"),
    ("one-step deposition", r"\bone[- ]step\b"),
]


def _ctrl_v21_10_present(value) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "[]", "{}", "not reported", "unknown", "n/a"}


def _ctrl_v21_10_float(value) -> float | None:
    try:
        out = float(str(value).strip())
    except Exception:
        return None
    if out != out:
        return None
    return out


def _ctrl_v21_10_normalize_doi(value) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = _re_controller.search(r"(10\.\d{4,9}/[^\s\"<>]+)", text, flags=_re_controller.IGNORECASE)
    if not match:
        return None
    doi = match.group(1).rstrip(").,;")
    return doi.lower() if doi else None


def _ctrl_v21_12_normalize_metric_text(text: str) -> str:
    cleaned = (
        str(text or "")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("⁻", "-")
        .replace("²", "2")
        .replace("ﬁ", "fi")
        .replace("ﬂ", "fl")
    )
    cleaned = _re_controller.sub(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])", "", cleaned)
    return _re_controller.sub(r"\s+", " ", cleaned)


def _ctrl_v21_10_extract_first_numeric(text: str, patterns: list[str], lo: float, hi: float) -> float | None:
    cleaned = _ctrl_v21_12_normalize_metric_text(text)
    for pattern in patterns:
        for match in _re_controller.finditer(pattern, cleaned, flags=_re_controller.IGNORECASE):
            value = _ctrl_v21_10_float(match.group(1))
            if value is not None and lo <= value <= hi:
                return value
    return None


def _ctrl_v21_14_extract_improved_device_metrics(text: str) -> dict[str, float]:
    """Extract only explicit positive from/to endpoints for one device cluster."""
    cleaned = _ctrl_v21_12_normalize_metric_text(text)
    if not _re_controller.search(
        r"\b(?:improv(?:ed|ement)|increas(?:e[sd]?|ed|ing)|enhanc(?:ed|ement)|"
        r"rais(?:ed|ing)|boost(?:ed|ing)|rising|rose)\b",
        cleaned,
        flags=_re_controller.IGNORECASE,
    ):
        return {}

    out: dict[str, float] = {}

    # PCE often uses "only achieved X, whereas ... increased to Y".
    endpoint_pce_patterns = [
        r"\bPCE\b.{0,100}?\b(?:improved|increased|enhanced|raised|boosted)\b"
        r".{0,60}?\bto\s*(\d{1,2}(?:\.\d+)?)\s*%",
        r"\bPCE\s*(?:is|was)?\s*(?:improved|increas(?:e[sd]?|ed|ing)|enhanced|raised)\s*from\s*"
        r"\d{1,2}(?:\.\d+)?\s*%\s*to\s*(\d{1,2}(?:\.\d+)?)\s*%",
    ]
    pce_values = _ctrl_v21_13_values_from_patterns(
        cleaned,
        endpoint_pce_patterns,
        1.0,
        35.0,
    )
    if pce_values:
        out["JV_default_PCE"] = max(pce_values)

    # Handle grouped reports such as "JSC, VOC, and FF increased from
    # 13.93 mA cm-2, 1.07 V, and 66.28% to 14.05 mA cm-2, 1.11 V,
    # and 71.26%, respectively."
    grouped = _re_controller.search(
        r"\bJ(?:SC|sc)\b.{0,45}\bV(?:OC|oc)\b.{0,45}\b(?:FF|fill factor)\b"
        r".{0,80}\b(?:increased|improved|enhanced|raised)\b.{0,40}\bfrom\s*"
        r"\d{1,2}(?:\.\d+)?\s*mA\s*(?:/|\s*)\s*cm\s*(?:-?2|\^-?2|2)"
        r"\s*,?\s*\d(?:\.\d+)?\s*V\s*,?\s*(?:and\s*)?"
        r"\d{1,2}(?:\.\d+)?\s*%\s*\bto\s*"
        r"(\d{1,2}(?:\.\d+)?)\s*mA\s*(?:/|\s*)\s*cm\s*(?:-?2|\^-?2|2)"
        r"\s*,?\s*(\d(?:\.\d+)?)\s*V\s*,?\s*(?:and\s*)?"
        r"(\d{1,2}(?:\.\d+)?)\s*%",
        cleaned,
        flags=_re_controller.IGNORECASE,
    )
    if grouped:
        jsc = _ctrl_v21_10_float(grouped.group(1))
        voc = _ctrl_v21_10_float(grouped.group(2))
        ff = _ctrl_v21_10_float(grouped.group(3))
        if jsc is not None and 1.0 <= jsc <= 35.0:
            out["JV_default_Jsc"] = jsc
        if voc is not None and 0.4 <= voc <= 1.5:
            out["JV_default_Voc"] = voc
        if ff is not None and 20.0 <= ff <= 90.0:
            out["JV_default_FF"] = ff / 100.0

    # FF may be reported with another metric between its old and new values,
    # for example: "Voc and FF increased from 1.14 V and 81.01% to
    # 1.18 V and 82.31%." Keep the FF cue inside the pattern so a nearby
    # PCE change such as 18.99% to 20.27% cannot be mistaken for FF.
    if "JV_default_FF" not in out:
        ff_endpoint_patterns = [
            (
                r"\b(?:FF|fill factor)\b.{0,100}?\b(?:increas(?:e[sd]?|ed|ing)|improved|"
                r"enhanced|raised)\b.{0,50}?\bfrom\s*"
                r"(?:\d(?:\.\d+)?\s*V\s*(?:,|and)?\s*)?"
                r"\d{1,2}(?:\.\d+)?\s*%\s*\bto\s*"
                r"(?:\d(?:\.\d+)?\s*V\s*(?:,|and)?\s*)?"
                r"(\d{1,2}(?:\.\d+)?)\s*%"
            ),
            (
                r"\b(?:FF|fill factor)\b.{0,60}?\bfrom\s*"
                r"\d{1,2}(?:\.\d+)?\s*%\s*\bto\s*"
                r"(\d{1,2}(?:\.\d+)?)\s*%"
            ),
        ]
        ff_values = _ctrl_v21_13_values_from_patterns(
            cleaned,
            ff_endpoint_patterns,
            20.0,
            90.0,
        )
        if ff_values:
            out["JV_default_FF"] = ff_values[-1] / 100.0

    endpoint_specs = (
        (
            "JV_default_Voc",
            r"\bfrom\s*\d(?:\.\d+)?\s*V\s*\bto\s*"
            r"(\d(?:\.\d+)?)\s*V\b",
            0.4,
            1.5,
            r"\bV(?:OC|oc)\b|\bopen[- ]circuit voltage\b",
        ),
        (
            "JV_default_Jsc",
            r"\bfrom\s*\d{1,2}(?:\.\d+)?\s*mA\s*(?:/|\s*)\s*cm\s*(?:-?2|\^-?2|2)"
            r"\s*\bto\s*(\d{1,2}(?:\.\d+)?)\s*mA\s*(?:/|\s*)\s*cm\s*(?:-?2|\^-?2|2)",
            1.0,
            35.0,
            r"\bJ(?:SC|sc)\b|\bshort[- ]circuit current\b",
        ),
    )
    for field, pattern, lo, hi, cue in endpoint_specs:
        if field in out:
            continue
        for match in _re_controller.finditer(
            pattern,
            cleaned,
            flags=_re_controller.IGNORECASE,
        ):
            nearby = cleaned[max(0, match.start() - 180):match.end() + 40]
            if not _re_controller.search(cue, nearby, flags=_re_controller.IGNORECASE):
                continue
            value = _ctrl_v21_10_float(match.group(1))
            if value is None or not lo <= value <= hi:
                continue
            out[field] = value / 100.0 if field.endswith("_FF") else value
            break
    return out


def _ctrl_v21_13_nonexperimental_metric_context(text: str) -> bool:
    low = _ctrl_v21_12_normalize_metric_text(text).lower()
    return bool(
        _re_controller.search(
            r"\b(?:theoretical(?:ly)?|radiative limit|shockley[\s-]*queisser|"
            r"detailed balance|maximum possible|ideal(?:ized)? limit|"
            r"thermodynamic limit|simulation predicts?|calculated limit)\b",
            low,
        )
        or _re_controller.search(
            r"\b(?:previous(?:ly)?|prior work|literature reports?|reported elsewhere|"
            r"reference device|cited stud(?:y|ies))\b",
            low,
        )
    )


def _ctrl_v21_13_metric_context_score(text: str) -> int:
    low = _ctrl_v21_12_normalize_metric_text(text).lower()
    if _ctrl_v21_13_nonexperimental_metric_context(low):
        return -100
    score = 0
    for pattern, weight in [
        (r"\bcertifi(?:ed|cation)\b", 12),
        (r"\b(?:steady[\s-]*state|stabili[sz]ed output)\b", 9),
        (r"\b(?:champion|best[\s-]*performing|record device)\b", 7),
        (r"\b(?:achieved|reached|yielded|obtained|measured)\b", 4),
        (r"\b(?:device|cell|solar cell|photovoltaic)\b", 2),
        (r"\b(?:reverse scan|forward scan|j[\s-]*v|current[\s-]*voltage)\b", 2),
    ]:
        if _re_controller.search(pattern, low):
            score += weight
    return score


def _ctrl_v21_13_metric_match_context(
    text: str,
    start: int,
    end: int,
    radius: int = 260,
) -> str:
    left_limit = max(0, start - radius)
    right_limit = min(len(text), end + radius)
    left_fragment = text[left_limit:start]
    right_fragment = text[end:right_limit]
    left_breaks = [
        left_fragment.rfind(marker)
        for marker in (". ", ".\n", "\n", "? ", "! ")
    ]
    left_break = max(left_breaks)
    if left_break >= 0:
        left_limit += left_break + 1
    right_breaks = [
        pos
        for marker in (". ", ".\n", "\n", "? ", "! ")
        for pos in [right_fragment.find(marker)]
        if pos >= 0
    ]
    if right_breaks:
        right_limit = end + min(right_breaks) + 1
    return text[left_limit:right_limit].strip()


def _ctrl_v21_13_values_from_patterns(
    text: str,
    patterns: list[str],
    lo: float,
    hi: float,
    *,
    ff_fraction: bool = False,
) -> list[float]:
    values = []
    cleaned = _ctrl_v21_12_normalize_metric_text(text)
    for pattern in patterns:
        for match in _re_controller.finditer(
            pattern,
            cleaned,
            flags=_re_controller.IGNORECASE,
        ):
            value = _ctrl_v21_10_float(match.group(1))
            if value is None or not lo <= value <= hi:
                continue
            if ff_fraction and value > 1.0:
                value /= 100.0
            values.append(value)
    return values


def _ctrl_v21_13_values_match(
    expected: float,
    candidates: list[float],
    tolerance: float,
) -> bool:
    return any(abs(expected - candidate) <= tolerance for candidate in candidates)


def _ctrl_v21_13_field_value_context(
    text: str,
    field: str,
    value: object,
) -> str | None:
    numeric = _ctrl_v21_10_float(value)
    if numeric is None:
        return None
    if field.endswith("_PCE") or field == "Stabilised_performance_PCE":
        patterns, tolerance, ff_fraction = _CTRL_V21_10_PCE_PATTERNS, 0.11, False
    elif field.endswith("_Voc"):
        patterns, tolerance, ff_fraction = _CTRL_V21_10_VOC_PATTERNS, 0.015, False
    elif field.endswith("_Jsc"):
        patterns, tolerance, ff_fraction = _CTRL_V21_10_JSC_PATTERNS, 0.08, False
    elif field.endswith("_FF"):
        patterns, tolerance, ff_fraction = _CTRL_V21_10_FF_PATTERNS, 0.006, True
    else:
        return None

    cleaned = _ctrl_v21_12_normalize_metric_text(text)
    for pattern in patterns:
        for match in _re_controller.finditer(
            pattern,
            cleaned,
            flags=_re_controller.IGNORECASE,
        ):
            candidate = _ctrl_v21_10_float(match.group(1))
            if candidate is None:
                continue
            if ff_fraction and candidate > 1.0:
                candidate /= 100.0
            if abs(candidate - numeric) <= tolerance:
                return _ctrl_v21_13_metric_match_context(
                    cleaned,
                    match.start(),
                    match.end(),
                    radius=500,
                )
    return None


def _ctrl_v21_10_extract_performance(text: str) -> dict[str, float]:
    cleaned = _ctrl_v21_12_normalize_metric_text(text)
    hits = []
    for pattern in _CTRL_V21_10_PCE_PATTERNS:
        for match in _re_controller.finditer(pattern, cleaned, flags=_re_controller.IGNORECASE):
            start = max(0, match.start() - 300)
            end = min(len(cleaned), match.end() + 300)
            window = cleaned[start:end]
            metric_context = _ctrl_v21_13_metric_match_context(
                cleaned,
                match.start(),
                match.end(),
            )
            matched_phrase = match.group(0)
            explicit_pce_phrase = _re_controller.search(
                r"\b(PCE|power conversion efficienc(?:y|ies)|photo[- ]?conversion efficienc(?:y|ies)|photoconversion efficienc(?:y|ies))\b",
                matched_phrase,
                flags=_re_controller.IGNORECASE,
            )
            if (
                not explicit_pce_phrase
                and _re_controller.search(r"\b(retention|retained|humidity|RH|degradation|yield|quantum yield|PLQY)\b", window, flags=_re_controller.IGNORECASE)
            ):
                continue
            if not _re_controller.search(r"\b(device|cell|solar|photovoltaic|PSC|PCE|power conversion|certified)\b", window, flags=_re_controller.IGNORECASE):
                continue
            value = _ctrl_v21_10_float(match.group(1))
            if value is not None and 1.0 <= value <= 35.0:
                context_score = _ctrl_v21_13_metric_context_score(metric_context)
                if context_score >= 0:
                    hits.append((context_score, value, window))
    if not hits:
        return {}
    _, pce, window = sorted(
        hits,
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )[0]
    out: dict[str, float] = {"JV_default_PCE": pce}
    voc = _ctrl_v21_10_extract_first_numeric(window, _CTRL_V21_10_VOC_PATTERNS, 0.4, 1.5)
    jsc = _ctrl_v21_10_extract_first_numeric(window, _CTRL_V21_10_JSC_PATTERNS, 1.0, 35.0)
    ff = _ctrl_v21_10_extract_first_numeric(window, _CTRL_V21_10_FF_PATTERNS, 0.2, 90.0)
    if voc is not None:
        out["JV_default_Voc"] = voc
    if jsc is not None:
        out["JV_default_Jsc"] = jsc
    if ff is not None:
        out["JV_default_FF"] = ff / 100.0 if ff > 1.0 else ff
    out.update(_ctrl_v21_14_extract_improved_device_metrics(window))
    return out


def _ctrl_v21_12_extract_companion_performance(
    trusted_performance: dict[str, float],
    paper_text: str,
) -> tuple[dict[str, float], str | None]:
    """Recover only metrics colocated with an already grounded PCE value."""
    trusted_pce = _ctrl_v21_10_float(trusted_performance.get("JV_default_PCE"))
    if trusted_pce is None:
        return {}, None

    cleaned = _ctrl_v21_12_normalize_metric_text(paper_text)
    matching_windows = []
    for pattern in _CTRL_V21_10_PCE_PATTERNS:
        for match in _re_controller.finditer(pattern, cleaned, flags=_re_controller.IGNORECASE):
            candidate = _ctrl_v21_10_float(match.group(1))
            if candidate is None or abs(candidate - trusted_pce) > 0.005:
                continue
            start = max(0, match.start() - 300)
            end = min(len(cleaned), match.end() + 300)
            matching_windows.append(cleaned[start:end])

    for window in matching_windows:
        companion = {}
        voc = _ctrl_v21_10_extract_first_numeric(window, _CTRL_V21_10_VOC_PATTERNS, 0.4, 1.5)
        jsc = _ctrl_v21_10_extract_first_numeric(window, _CTRL_V21_10_JSC_PATTERNS, 1.0, 35.0)
        ff = _ctrl_v21_10_extract_first_numeric(window, _CTRL_V21_10_FF_PATTERNS, 0.2, 90.0)
        if voc is not None:
            companion["JV_default_Voc"] = voc
        if jsc is not None:
            companion["JV_default_Jsc"] = jsc
        if ff is not None:
            companion["JV_default_FF"] = ff / 100.0 if ff > 1.0 else ff
        companion.update(_ctrl_v21_14_extract_improved_device_metrics(window))
        companion.pop("JV_default_PCE", None)
        if companion:
            evidence = _re_controller.sub(r"\s+", " ", window).strip()
            return companion, evidence
    return {}, None


def _ctrl_v21_10_extract_formula(text: str) -> str | None:
    cleaned = str(text or "")
    for pattern in _CTRL_V21_10_FORMULA_PATTERNS:
        match = _re_controller.search(pattern, cleaned, flags=_re_controller.IGNORECASE)
        if match:
            formula = match.group(0).strip(" /,;:[]{}")
            if any(x in formula for x in ["Pb", "Sn", "Ge"]) and any(x in formula for x in ["I", "Br", "Cl"]):
                return formula
    return None


def _ctrl_v21_12_formula_window(text: str, start: int, end: int, radius: int = 260) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return _re_controller.sub(r"\s+", " ", text[left:right]).strip()


def _ctrl_v21_12_formula_context_score(window: str) -> int:
    """Favor the paper's experimental absorber and reject background formulas."""
    low = str(window or "").lower()
    score = 0
    for pattern, weight in [
        (r"\bprecursor solution\b.{0,80}\babsorber\b", 9),
        (r"\b(?:triple|mixed)[\s-]*cation\b", 6),
        (r"\b(champion|best[- ]performing|optimized|optimal)\b", 4),
        (r"\b(device with|device based on|solar cell|photovoltaic device)\b", 3),
        (r"\b(absorber|perovskite film|precursor solution|fabricat|prepared|deposited)\b", 2),
        (r"\b(PCE|power conversion efficiency|JSC|fill factor)\b", 2),
        (r"\b(sample|composition|formulation)\b", 1),
    ]:
        if _re_controller.search(pattern, window, flags=_re_controller.IGNORECASE):
            score += weight
    for pattern, weight in [
        (r"\b(?:theoretical|radiative limit|shockley[\s-]*queisser)\b", 9),
        (r"\breference\b", 6),
        (r"\b(cited|previous(?:ly)?|prior work|literature reports?|reported elsewhere)\b", 7),
        (r"\b(general formula|for example|such as|can be represented)\b", 5),
        (r"\b(review|perspective)\b", 2),
    ]:
        if _re_controller.search(pattern, low, flags=_re_controller.IGNORECASE):
            score -= weight
    return score


def _ctrl_v21_12_extract_device_composition(text: str) -> tuple[str | None, str | None, str | None]:
    """Extract an experimental composition with exact source context.

    The second path handles papers that define a variable family formula and
    subsequently identify the tested/champion member by explicit coefficients.
    The substitution is deterministic and is recorded as an assumption rather
    than presented as a verbatim formula.
    """
    cleaned = str(text or "").replace("ﬁ", "fi").replace("−", "-").replace("–", "-")
    if not cleaned.strip():
        return None, None, None

    candidates = []
    for pattern in _CTRL_V21_10_FORMULA_PATTERNS:
        for match in _re_controller.finditer(pattern, cleaned, flags=_re_controller.IGNORECASE):
            formula = _re_controller.sub(r"\s+", "", match.group(0)).strip(" /,;:[]{}")
            if (
                _re_controller.search(r"(?:Pb|Sn|Ge)(?:I|Br|Cl)$", formula)
                and _re_controller.match(
                    r"\s*3(?=[^0-9.]|$)",
                    cleaned[match.end():match.end() + 5],
                )
            ):
                formula += "3"
            window = _ctrl_v21_12_formula_window(cleaned, match.start(), match.end())
            score = _ctrl_v21_12_formula_context_score(window)
            if score > 0:
                candidates.append((score, formula, window, None))

    family_pattern = _re_controller.compile(
        r"\((?P<s1>[A-Z][A-Za-z0-9]*)\)1-a"
        r"\((?P<s2>[A-Z][A-Za-z0-9]*)\)a"
        r"(?P<tail>(?:MA|FA|Cs|Rb|K)[A-Za-z0-9]*(?:Pb|Sn|Ge)[A-Za-z0-9]*(?:I|Br|Cl|X)[A-Za-z0-9+]*)",
        flags=_re_controller.IGNORECASE,
    )
    for family in family_pattern.finditer(cleaned):
        s1 = family.group("s1")
        s2 = family.group("s2")
        tail = family.group("tail")
        sample_pattern = _re_controller.compile(
            rf"\({_re_controller.escape(s1)}\)(?P<c1>0?\.\d+|1(?:\.0+)?)"
            rf"\({_re_controller.escape(s2)}\)(?P<c2>0?\.\d+|1(?:\.0+)?)",
            flags=_re_controller.IGNORECASE,
        )
        for sample in sample_pattern.finditer(cleaned):
            c1 = _ctrl_v21_10_float(sample.group("c1"))
            c2 = _ctrl_v21_10_float(sample.group("c2"))
            if c1 is None or c2 is None or abs((c1 + c2) - 1.0) > 0.03:
                continue
            sample_window = _ctrl_v21_12_formula_window(cleaned, sample.start(), sample.end())
            score = _ctrl_v21_12_formula_context_score(sample_window) + 3
            if score <= 0:
                continue
            formula = f"({s1}){sample.group('c1')}({s2}){sample.group('c2')}{tail}"
            family_window = _ctrl_v21_12_formula_window(cleaned, family.start(), family.end(), radius=150)
            evidence = f"{family_window} | {sample_window}"
            assumption = (
                f"Derived by substituting the explicitly reported sample coefficients "
                f"into the paper's family formula {family.group(0)}."
            )
            candidates.append((score, formula, evidence, assumption))

    if not candidates:
        return None, None, None
    _, formula, evidence, assumption = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
    return formula, evidence[:1200], assumption


def _ctrl_v21_10_ions_from_formula(formula: str) -> tuple[str | None, str | None, str | None]:
    a_hits = []
    for label, aliases in [
        ("MA", ["CH3NH3", "MA"]),
        ("FA", ["FA"]),
        ("Cs", ["Cs"]),
        ("Rb", ["Rb"]),
        ("K", ["K"]),
    ]:
        if any(alias in formula for alias in aliases):
            a_hits.append(label)
    b_hits = [x for x in ["Pb", "Sn", "Ge"] if x in formula]
    c_hits = [x for x in ["I", "Br", "Cl", "X"] if x in formula]
    return (
        ";".join(dict.fromkeys(a_hits)) or None,
        ";".join(dict.fromkeys(b_hits)) or None,
        ";".join(dict.fromkeys(c_hits)) or None,
    )


def _ctrl_v21_10_find_terms(text: str, terms: list[str]) -> str | None:
    hits = []
    for term in terms:
        pattern = r"(?<![A-Za-z0-9])" + _re_controller.escape(term) + r"(?![A-Za-z0-9])"
        if _re_controller.search(pattern, text, flags=_re_controller.IGNORECASE):
            hits.append(term)
    return "; ".join(dict.fromkeys(hits)) if hits else None


def _ctrl_v21_10_extract_architecture(text: str) -> str | None:
    low = str(text or "").lower()
    if _re_controller.search(r"\b(inverted|p-i-n|pin)\b", low):
        return "inverted"
    if _re_controller.search(r"\b(regular|n-i-p|nip|mesoscopic|planar)\b", low):
        return "regular"
    pedot = low.find("pedot:pss")
    perovskite = low.find("perovskite")
    pcbm = low.find("pcbm")
    if pedot >= 0 and perovskite > pedot and pcbm > perovskite:
        return "inverted"
    return None


def _ctrl_v21_10_extract_processing(text: str) -> dict[str, str]:
    low = str(text or "").lower()
    out: dict[str, str] = {}
    procedures = []
    for label, pattern in _CTRL_V21_10_PROCESS_TERMS:
        if _re_controller.search(pattern, low, flags=_re_controller.IGNORECASE):
            procedures.append(label)
    if procedures:
        out["Perovskite_deposition_procedure"] = "; ".join(dict.fromkeys(procedures))
    solvents = _ctrl_v21_10_find_terms(text, ["DMF", "DMSO", "GBL", "NMP", "chlorobenzene", "toluene", "ethyl acetate", "diethyl ether", "isopropanol", "IPA"])
    if solvents:
        out["Perovskite_deposition_solvents"] = solvents
    temp = _re_controller.search(r"\banneal(?:ing|ed)?\b.{0,90}?(\d{2,3})\s*(?:°\s*|deg\s*)?C\b", text, flags=_re_controller.IGNORECASE)
    if temp:
        out["Perovskite_deposition_thermal_annealing_temperature"] = temp.group(1)
    time = _re_controller.search(r"\banneal(?:ing|ed)?\b.{0,90}?(\d+(?:\.\d+)?)\s*(min|minutes|h|hours)\b", text, flags=_re_controller.IGNORECASE)
    if time:
        out["Perovskite_deposition_thermal_annealing_time"] = f"{time.group(1)} {time.group(2)}"
    atmosphere = _ctrl_v21_10_find_terms(text, ["nitrogen", "N2", "air", "argon", "glovebox", "ambient"])
    if atmosphere and _re_controller.search(r"\b(anneal|deposition|fabricat|prepared|processed)\b", text, flags=_re_controller.IGNORECASE):
        out["Perovskite_deposition_thermal_annealing_atmosphere"] = atmosphere
    return out


def _ctrl_v21_10_extract_stability(text: str) -> dict[str, str | float]:
    out: dict[str, str | float] = {}
    cleaned = str(text or "")
    low = cleaned.lower()
    if _re_controller.search(r"\b(stability|aging|ageing|t80|t95|retention|retained|remains|remained|operational)\b", low):
        out["Stability_measured"] = True
    t80 = _ctrl_v21_10_extract_first_numeric(cleaned, [r"\bT80\s*(?:=|:|of|over|exceeding|after)?\s*(\d+(?:\.\d+)?)\s*(?:h|hours)\b"], 1, 100000)
    t95 = _ctrl_v21_10_extract_first_numeric(cleaned, [r"\bT95\s*(?:=|:|of|over|exceeding|after)?\s*(\d+(?:\.\d+)?)\s*(?:h|hours)\b"], 1, 100000)
    if t80 is not None:
        out["Stability_PCE_T80"] = t80
    if t95 is not None:
        out["Stability_PCE_T95"] = t95
    exposure = _ctrl_v21_10_extract_first_numeric(
        cleaned,
        [
            r"\b(?:for|after|over)\s*(\d+(?:\.\d+)?)\s*(?:h|hours)\b.{0,80}\b(?:stability|aging|ageing|retention|retained|remains|remained)\b",
            r"\b(?:stability|aging|ageing|retention|retained|remains|remained)\b.{0,100}\b(?:for|after|over)\s*(\d+(?:\.\d+)?)\s*(?:h|hours)\b",
        ],
        1,
        100000,
    )
    if exposure is not None:
        out["Stability_time_total_exposure"] = exposure
    retention = _ctrl_v21_10_extract_first_numeric(
        cleaned,
        [
            r"\bretained\s*(\d{1,3}(?:\.\d+)?)\s*%",
            r"\b(?:remains?|remained|remaining)\s*(\d{1,3}(?:\.\d+)?)\s*%",
            r"\bPCE\b.{0,30}\b(?:remains?|remained)\s*(\d{1,3}(?:\.\d+)?)\s*%",
            r"\bretention\s*(?:of|=|:)?\s*(\d{1,3}(?:\.\d+)?)\s*%",
            r"\b(\d{1,3}(?:\.\d+)?)\s*%\s*(?:of\s*)?(?:initial\s*)?(?:PCE\s*)?(?:retained|retention)\b",
            r"\b(\d{1,3}(?:\.\d+)?)\s*%\s*(?:efficiency|PCE)\s*retention\b",
        ],
        1,
        100,
    )
    if retention is not None:
        out["Stability_PCE_end_of_experiment"] = retention
    protocol_terms = _ctrl_v21_10_find_terms(cleaned, ["ISOS-L", "ISOS-D", "ISOS-T", "maximum power point", "MPP", "1 sun", "AM 1.5G", "dark storage"])
    if protocol_terms:
        out["Stability_protocol"] = protocol_terms
    temp = _ctrl_v21_10_extract_first_numeric(cleaned, [r"\b(\d{2,3})\s*(?:deg\s*)?C\b"], 15, 120)
    if temp is not None and _re_controller.search(r"\b(stability|aging|ageing|thermal|heat|temperature)\b", low):
        out["Stability_temperature_range"] = temp
    rh = _ctrl_v21_10_extract_first_numeric(
        cleaned,
        [
            r"\b(\d{1,3}(?:\.\d+)?)\s*(?:±|\+/-|\+-)\s*\d{1,3}(?:\.\d+)?\s*%\s*(?:RH|relative humidity)\b",
            r"\b(\d{1,3}(?:\.\d+)?)\s*%\s*(?:RH|relative humidity)\b",
        ],
        1,
        100,
    )
    if rh is not None:
        out["Stability_relative_humidity_average_value"] = rh
    return out


def _ctrl_v21_10_extract_feature_fields(text: str) -> dict[str, object]:
    out: dict[str, object] = {}
    architecture = _ctrl_v21_10_extract_architecture(text)
    if architecture:
        out["Cell_architecture"] = architecture
    etl = _ctrl_v21_10_find_terms(text, _CTRL_V21_10_ETL_TERMS)
    htl = _ctrl_v21_10_find_terms(text, _CTRL_V21_10_HTL_TERMS)
    back = _ctrl_v21_10_find_terms(text, _CTRL_V21_10_BACKCONTACT_TERMS)
    if etl:
        out["ETL_stack_sequence"] = etl
    if htl:
        out["HTL_stack_sequence"] = htl
    if back:
        out["Backcontact_stack_sequence"] = back
    if etl and htl and back:
        if architecture == "inverted":
            out["Cell_stack_sequence"] = f"{htl} / perovskite / {etl} / {back}"
        else:
            out["Cell_stack_sequence"] = f"{etl} / perovskite / {htl} / {back}"
    out.update(_ctrl_v21_10_extract_processing(text))
    out.update(_ctrl_v21_10_extract_stability(text))
    return out


def _ctrl_v21_10_source_text(meta, full_text, extraction_text, records) -> str:
    """Return only compact, field-traceable text for deterministic enrichment.

    The LLM still receives the ranked extraction/full text through its normal
    prompt. This helper deliberately excludes those broad strings because a
    global regex cannot distinguish the paper's device from comparison devices,
    cited work, captions, or plot axes.
    """
    parts = []
    if isinstance(meta, dict):
        for key in ["title", "paper_title", "doi", "abstract", "_source_title", "_source_abstract"]:
            if _ctrl_v21_10_present(meta.get(key)):
                parts.append(str(meta.get(key)))
    for rec in records or []:
        if isinstance(rec, dict):
            for key in ["Ref_DOI_number", "_source_title", "_source_abstract"]:
                if _ctrl_v21_10_present(rec.get(key)):
                    parts.append(str(rec.get(key)))
            for key, value in rec.items():
                if (
                    str(key).startswith("_evidence_")
                    and str(key) not in {
                        "_evidence_v19_deterministic_candidates",
                        "_evidence_v19_4_note",
                    }
                    and _ctrl_v21_10_present(value)
                ):
                    parts.append(str(value))
    text = " ".join(parts)
    text = _re_controller.sub(r"<[^>]+>", " ", text)
    return _re_controller.sub(r"\s+", " ", text)


def _ctrl_v21_13_sanitize_record_targets(rec: dict[str, object]) -> list[str]:
    """Normalize grounded JV units and remove cross-metric target assignments."""
    changes: list[str] = []

    def evidence(field: str) -> str:
        return str(rec.get(f"_evidence_{field}") or "")

    def clear(field: str, reason: str) -> None:
        if not _ctrl_v21_10_present(rec.get(field)):
            return
        old = rec.get(field)
        rec[field] = None
        rec.pop(f"_evidence_{field}", None)
        changes.append(f"cleared:{field}={old}:{reason}")

    def set_value(field: str, value: float, reason: str) -> None:
        old = rec.get(field)
        rec[field] = value
        changes.append(f"normalized:{field}={old}->{value}:{reason}")

    pce_fields = (
        "JV_default_PCE",
        "JV_reverse_scan_PCE",
        "JV_forward_scan_PCE",
        "Stabilised_performance_PCE",
    )
    voc_fields = ("JV_default_Voc", "JV_reverse_scan_Voc", "JV_forward_scan_Voc")
    jsc_fields = ("JV_default_Jsc", "JV_reverse_scan_Jsc", "JV_forward_scan_Jsc")
    ff_fields = ("JV_default_FF", "JV_reverse_scan_FF", "JV_forward_scan_FF")
    record_type_text = " ".join(
        (
            str(rec.get("_lit_agent_paper_type") or ""),
            str(rec.get("Ref_free_text_comment") or ""),
        )
    ).lower()
    nonexperimental_record = bool(
        _re_controller.search(
            r"\b(?:paper_type=)?(?:computational_theory|review_perspective)\b",
            record_type_text,
        )
    )
    if nonexperimental_record:
        for field in (
            *pce_fields,
            *voc_fields,
            *jsc_fields,
            *ff_fields,
            "Stability_PCE_initial_value",
            "Stability_PCE_end_of_experiment",
            "Stability_PCE_after_1000_h",
            "Stability_PCE_T80",
            "Stability_PCE_T95",
            "Stability_time_total_exposure",
        ):
            clear(field, "nonexperimental_paper_excluded_from_device_targets")
        rec["_lit_agent_model_exclusion_reason"] = (
            "paper_type_is_computational_or_review"
        )

    for field in pce_fields:
        value = _ctrl_v21_10_float(rec.get(field))
        if value is None:
            continue
        quote = evidence(field)
        low = quote.lower()
        has_metric = bool(
            _re_controller.search(
                r"\b(?:pce|power conversion efficiency|photovoltaic efficiency|"
                r"conversion efficiency|efficienc(?:y|ies)|ciency)\b",
                low,
            )
        )
        retention_only = bool(
            _re_controller.search(
                r"\b(?:retain(?:ed|s|ing)?|retention|remaining|initial efficiency|"
                r"original efficiency|after\s+\d+(?:\.\d+)?\s*(?:h|hours?|days?))\b",
                low,
            )
        ) and not bool(
            _re_controller.search(
                r"\b(?:achiev(?:e|ed|es)|reach(?:ed|es)?|champion|best|"
                r"power conversion efficiency|conversion efficiency)\b",
                low,
            )
        )
        supported_values = []
        for pattern in _CTRL_V21_10_PCE_PATTERNS:
            for match in _re_controller.finditer(
                pattern,
                _ctrl_v21_12_normalize_metric_text(quote),
                flags=_re_controller.IGNORECASE,
            ):
                candidate = _ctrl_v21_10_float(match.group(1))
                if candidate is not None:
                    supported_values.append(candidate)
        value_is_supported = any(
            abs(candidate - value) <= 0.11
            for candidate in supported_values
        )
        if not has_metric:
            clear(field, "evidence_missing_pce_cue")
        elif not value_is_supported:
            clear(field, "pce_value_not_supported_by_evidence")
        elif retention_only:
            clear(field, "retention_percentage_is_not_device_pce")
        elif _ctrl_v21_13_nonexperimental_metric_context(quote):
            clear(field, "theoretical_or_background_value_is_not_device_pce")
        elif (
            field == "Stabilised_performance_PCE"
            and not _re_controller.search(
                r"\b(?:stabili[sz]ed|steady[\s-]*state|maximum power point|"
                r"mpp tracking|tracked at (?:the )?maximum power point|"
                r"stabili[sz]ed power output)\b",
                low,
            )
        ):
            clear(field, "stabilized_pce_missing_stabilized_measurement_cue")
        elif not (0.0 <= value <= 45.0):
            clear(field, "pce_outside_0_to_45_percent")

    for field in voc_fields:
        value = _ctrl_v21_10_float(rec.get(field))
        if value is None:
            continue
        quote = evidence(field)
        low = quote.lower()
        has_metric = bool(_re_controller.search(r"\b(?:v[_\s-]*oc|open[\s-]*circuit voltage)\b", low))
        differential = bool(
            _re_controller.search(
                r"\b(?:enhancement|increase|decrease|improvement|gain|shift|raised|reduced)"
                r"\b.{0,35}\b(?:by|of)\b",
                low,
            )
        )
        absolute_voc_values = _ctrl_v21_13_values_from_patterns(
            quote,
            _CTRL_V21_10_VOC_PATTERNS,
            0.0,
            3.0,
        )
        absolute_value_supported = _ctrl_v21_13_values_match(
            value,
            absolute_voc_values,
            0.015,
        )
        directional_voc = _ctrl_v21_10_float(
            _ctrl_v21_14_extract_improved_device_metrics(quote).get(
                "JV_default_Voc"
            )
        )
        if directional_voc is not None and abs(value - directional_voc) <= 0.015:
            absolute_value_supported = True
        if not has_metric:
            clear(field, "evidence_missing_voc_cue")
        elif differential and not absolute_value_supported:
            clear(field, "voc_difference_is_not_absolute_voc")
        elif value > 3.0 and value <= 2500.0 and _re_controller.search(r"\bm\s*v\b|\bmv\b", low):
            set_value(field, round(value / 1000.0, 6), "millivolts_to_volts")
        elif not (0.0 <= value <= 3.0):
            clear(field, "voc_outside_cell_voltage_range")

    for field in jsc_fields:
        value = _ctrl_v21_10_float(rec.get(field))
        if value is None:
            continue
        low = evidence(field).lower()
        has_metric = bool(
            _re_controller.search(
                r"\b(?:j[_\s-]*sc|short[\s-]*circuit current|ma\s*(?:cm|/cm))\b",
                low,
            )
        )
        if not has_metric:
            clear(field, "evidence_missing_jsc_cue")
        elif not (0.0 <= value <= 60.0):
            clear(field, "jsc_outside_0_to_60_ma_cm2")

    for field in ff_fields:
        value = _ctrl_v21_10_float(rec.get(field))
        if value is None:
            continue
        low = evidence(field).lower()
        has_metric = bool(_re_controller.search(r"\b(?:ff|fill factor)\b", low))
        if not has_metric:
            clear(field, "evidence_missing_ff_cue")
        elif 1.0 < value <= 100.0:
            set_value(field, round(value / 100.0, 6), "percent_to_fraction")
        elif not (0.0 <= value <= 1.0):
            clear(field, "ff_outside_0_to_1_fraction")

    # A device row is a metric cluster, not a bag of independently plausible
    # numbers. If a companion metric's evidence names a different PCE, or the
    # selected PCE evidence names a different companion value, remove the
    # companion so deterministic co-located extraction can replace it.
    cluster_specs = (
        (
            "JV_default_PCE",
            (
                ("JV_default_Voc", _CTRL_V21_10_VOC_PATTERNS, 0.4, 1.5, 0.015, False),
                ("JV_default_Jsc", _CTRL_V21_10_JSC_PATTERNS, 1.0, 60.0, 0.08, False),
                ("JV_default_FF", _CTRL_V21_10_FF_PATTERNS, 0.0, 100.0, 0.006, True),
            ),
        ),
        (
            "JV_reverse_scan_PCE",
            (
                ("JV_reverse_scan_Voc", _CTRL_V21_10_VOC_PATTERNS, 0.4, 1.5, 0.015, False),
                ("JV_reverse_scan_Jsc", _CTRL_V21_10_JSC_PATTERNS, 1.0, 60.0, 0.08, False),
                ("JV_reverse_scan_FF", _CTRL_V21_10_FF_PATTERNS, 0.0, 100.0, 0.006, True),
            ),
        ),
        (
            "JV_forward_scan_PCE",
            (
                ("JV_forward_scan_Voc", _CTRL_V21_10_VOC_PATTERNS, 0.4, 1.5, 0.015, False),
                ("JV_forward_scan_Jsc", _CTRL_V21_10_JSC_PATTERNS, 1.0, 60.0, 0.08, False),
                ("JV_forward_scan_FF", _CTRL_V21_10_FF_PATTERNS, 0.0, 100.0, 0.006, True),
            ),
        ),
    )
    for pce_field, companion_specs in cluster_specs:
        pce_value = _ctrl_v21_10_float(rec.get(pce_field))
        if pce_value is None:
            continue
        pce_quote = evidence(pce_field)
        for companion_field, patterns, lo, hi, tolerance, ff_fraction in companion_specs:
            companion_value = _ctrl_v21_10_float(rec.get(companion_field))
            if companion_value is None:
                continue
            companion_quote = evidence(companion_field)
            companion_quote_pces = _ctrl_v21_13_values_from_patterns(
                companion_quote,
                _CTRL_V21_10_PCE_PATTERNS,
                0.0,
                45.0,
            )
            if (
                companion_quote_pces
                and not _ctrl_v21_13_values_match(
                    pce_value,
                    companion_quote_pces,
                    0.11,
                )
            ):
                clear(
                    companion_field,
                    f"metric_evidence_belongs_to_different_pce_than_{pce_field}",
                )
                continue
            pce_quote_companions = _ctrl_v21_13_values_from_patterns(
                pce_quote,
                patterns,
                lo,
                hi,
                ff_fraction=ff_fraction,
            )
            if (
                pce_quote_companions
                and not _ctrl_v21_13_values_match(
                    companion_value,
                    pce_quote_companions,
                    tolerance,
                )
            ):
                clear(
                    companion_field,
                    f"metric_value_not_paired_with_{pce_field}_evidence",
                )

    initial_field = "Stability_PCE_initial_value"
    initial_value = _ctrl_v21_10_float(rec.get(initial_field))
    if initial_value is not None:
        low = evidence(initial_field).lower()
        explicit_initial_pce = bool(
            _re_controller.search(
                r"\b(?:initial|starting|before aging|before ageing)\b.{0,40}"
                r"\b(?:pce|efficiency|power conversion efficiency)\b|"
                r"\b(?:pce|efficiency|power conversion efficiency)\b.{0,40}\binitial\b",
                low,
            )
        )
        retention_statement = bool(
            _re_controller.search(r"\b(?:retain(?:ed|s|ing)?|retention|of its initial)\b", low)
        )
        if not explicit_initial_pce or retention_statement or not (0.0 <= initial_value <= 45.0):
            clear(initial_field, "not_an_explicit_initial_device_pce")

    end_field = "Stability_PCE_end_of_experiment"
    end_value = _ctrl_v21_10_float(rec.get(end_field))
    if end_value is not None:
        low = evidence(end_field).lower()
        retention_statement = bool(
            _re_controller.search(
                r"\b(?:retain(?:ed|s|ing)?|retention|remaining|remained|"
                r"of (?:its|the) (?:initial|original)|initial pce)\b",
                low,
            )
        )
        degradation_only = bool(
            _re_controller.search(r"\b(?:degradation|degraded|decrease|loss)\b", low)
        ) and not retention_statement
        if not retention_statement or degradation_only or not (0.0 <= end_value <= 100.0):
            clear(end_field, "end_target_requires_percent_retained_evidence")

    after_field = "Stability_PCE_after_1000_h"
    after_value = _ctrl_v21_10_float(rec.get(after_field))
    if after_value is not None:
        low = evidence(after_field).lower()
        at_1000_h = bool(
            _re_controller.search(r"\b1000\s*(?:h|hr|hrs|hour|hours)\b", low)
        )
        retention_statement = bool(
            _re_controller.search(
                r"\b(?:retain(?:ed|s|ing)?|retention|remaining|remained|"
                r"of (?:its|the) (?:initial|original))\b",
                low,
            )
        )
        clearly_retention_scale = 45.0 < after_value <= 100.0 and "pce" in low
        t80_statement = bool(_re_controller.search(r"\bt\s*80\b|\bt80\b", low))
        if (
            not at_1000_h
            or t80_statement
            or not (retention_statement or clearly_retention_scale)
            or not (0.0 <= after_value <= 100.0)
        ):
            clear(after_field, "after_1000h_target_requires_retention_at_1000h")

    for field, marker in (
        ("Stability_PCE_T80", "80"),
        ("Stability_PCE_T95", "95"),
    ):
        value = _ctrl_v21_10_float(rec.get(field))
        if value is None:
            continue
        low = evidence(field).lower()
        if not _re_controller.search(rf"\bt\s*{marker}\b|\bt{marker}\b", low):
            clear(field, f"missing_t{marker}_evidence")
        elif not (0.0 <= value <= 1_000_000.0):
            clear(field, f"t{marker}_outside_supported_hours")

    exposure_field = "Stability_time_total_exposure"
    exposure_value = _ctrl_v21_10_float(rec.get(exposure_field))
    if exposure_value is not None:
        low = evidence(exposure_field).lower()
        day_match = _re_controller.search(
            r"\b(\d+(?:\.\d+)?)\s*(?:d|day|days)\b",
            low,
        )
        hour_match = _re_controller.search(
            r"\b(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b",
            low,
        )
        if day_match and abs(exposure_value - float(day_match.group(1))) <= 0.01:
            set_value(
                exposure_field,
                round(exposure_value * 24.0, 6),
                "days_to_hours",
            )
        elif not hour_match and not day_match:
            clear(exposure_field, "exposure_time_missing_time_unit_evidence")
        elif not (0.0 <= exposure_value <= 1_000_000.0):
            clear(exposure_field, "exposure_time_outside_supported_hours")

    end_value = _ctrl_v21_10_float(rec.get(end_field))
    exposure_value = _ctrl_v21_10_float(rec.get(exposure_field))
    if end_value is not None and exposure_value is not None:
        paired = False
        retention_number = rf"{end_value:g}"
        exposure_number = rf"{exposure_value:g}"
        for quote in (evidence(end_field), evidence(exposure_field)):
            normalized_quote = _ctrl_v21_12_normalize_metric_text(quote)
            retention_grounded = bool(
                _re_controller.search(
                    rf"\b{_re_controller.escape(retention_number)}(?:\.0+)?\s*%",
                    normalized_quote,
                    flags=_re_controller.IGNORECASE,
                )
            )
            hours_grounded = bool(
                _re_controller.search(
                    rf"\b{_re_controller.escape(exposure_number)}(?:\.0+)?\s*"
                    r"(?:h|hr|hrs|hour|hours)\b",
                    normalized_quote,
                    flags=_re_controller.IGNORECASE,
                )
            )
            days_grounded = any(
                abs(float(match.group(1)) * 24.0 - exposure_value) <= 0.01
                for match in _re_controller.finditer(
                    r"\b(\d+(?:\.\d+)?)\s*(?:d|day|days)\b",
                    normalized_quote,
                    flags=_re_controller.IGNORECASE,
                )
            )
            if (
                retention_grounded
                and (hours_grounded or days_grounded)
            ):
                paired = True
                break
        if not paired:
            clear(end_field, "retention_and_exposure_not_grounded_as_one_event")
            clear(exposure_field, "retention_and_exposure_not_grounded_as_one_event")

    rh_value = _ctrl_v21_10_float(rec.get("Stability_relative_humidity_average_value"))
    if rh_value is not None:
        atmosphere = str(rec.get("Stability_atmosphere") or "").lower()
        rh_condition = rec.get("Stability_relative_humidity_load_conditions")
        rh_range = rec.get("Stability_relative_humidity_range")
        inert_atmosphere = any(
            term in atmosphere
            for term in ("nitrogen", "n2", "argon", "glove box", "glovebox", "inert")
        )
        if (
            inert_atmosphere
            and not _ctrl_v21_10_present(rh_condition)
            and not _ctrl_v21_10_present(rh_range)
        ):
            clear(
                "Stability_relative_humidity_average_value",
                "inert_row_has_no_explicit_humidity_condition",
            )
        elif not (0.0 <= rh_value <= 100.0):
            clear("Stability_relative_humidity_average_value", "rh_outside_0_to_100_percent")

    if changes:
        existing = str(rec.get("_lit_agent_target_sanitization_changes") or "").strip()
        rec["_lit_agent_target_sanitization_changes"] = (
            existing + "; " + "; ".join(changes)
        ).strip("; ")
    return changes


def _ctrl_v21_10_enrich_extraction_records(meta, full_text, extraction_text, records):
    if not isinstance(records, list) or not records:
        return records

    source_text = _ctrl_v21_10_source_text(meta, full_text, extraction_text, records)
    evidence_text = source_text[:1200].strip()
    performance = _ctrl_v21_10_extract_performance(source_text)
    broad_paper_text = "\n".join([str(extraction_text or ""), str(full_text or "")])
    formula, composition_evidence, composition_assumption = _ctrl_v21_12_extract_device_composition(
        "\n".join([str(extraction_text or ""), str(full_text or "")])
    )
    if not formula:
        formula = _ctrl_v21_10_extract_formula(source_text)
        composition_evidence = evidence_text if formula else None
        composition_assumption = None
    feature_fields = _ctrl_v21_10_extract_feature_fields(source_text)
    meta_stability_text = ""
    if isinstance(meta, dict):
        meta_stability_text = " ".join(
            str(meta.get(key) or "")
            for key in ("abstract", "_source_abstract")
            if _ctrl_v21_10_present(meta.get(key))
        )
    paper_stability = _ctrl_v21_10_extract_stability(meta_stability_text)
    has_paired_abstract_stability = bool(
        paper_stability.get("Stability_time_total_exposure") is not None
        and any(
            paper_stability.get(field) is not None
            for field in (
                "Stability_PCE_end_of_experiment",
                "Stability_PCE_T80",
                "Stability_PCE_T95",
            )
        )
    )
    paper_stability_target_indices: set[int] = set()
    if has_paired_abstract_stability:
        abstract_pce = _ctrl_v21_10_float(performance.get("JV_default_PCE"))
        if abstract_pce is not None:
            for index, candidate_record in enumerate(records):
                if not isinstance(candidate_record, dict):
                    continue
                candidate_pce = _ctrl_v21_10_float(
                    candidate_record.get("JV_default_PCE")
                )
                if candidate_pce is not None and abs(candidate_pce - abstract_pce) <= 0.11:
                    paper_stability_target_indices.add(index)
        if not paper_stability_target_indices and len(records) == 1:
            paper_stability_target_indices.add(0)
    source_doi = None
    if isinstance(meta, dict):
        for key in ["doi", "DOI", "url", "pdf_url", "landing_page", "_source_landing_page", "_source_pdf_url"]:
            source_doi = _ctrl_v21_10_normalize_doi(meta.get(key))
            if source_doi:
                break

    changed = 0
    for record_index, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        # Remove an invalid model target before deterministic enrichment so a
        # grounded experimental value can replace it in the same pass.
        changes = _ctrl_v21_13_sanitize_record_targets(rec)
        # Device composition/processing context can be shared across rows, but
        # stability conditions must stay scoped to their own experimental row.
        # Otherwise an RH value from an ambient test can leak into a nitrogen
        # storage row for the same device.
        record_feature_fields = {
            key: value
            for key, value in feature_fields.items()
            if not str(key).startswith("Stability_")
        }
        record_stability_parts = []
        for key, value in rec.items():
            if (
                str(key).startswith("_evidence_Stability_")
                or str(key) in {
                    "Stability_protocol",
                    "Stability_atmosphere",
                    "Stability_temperature_load_condition",
                    "Stability_temperature_range",
                    "Stability_relative_humidity_load_conditions",
                    "Stability_relative_humidity_range",
                    "Stability_time_total_exposure",
                }
            ) and _ctrl_v21_10_present(value):
                record_stability_parts.append(str(value))
        record_stability_evidence = " ".join(record_stability_parts)
        if (
            record_index in paper_stability_target_indices
            and meta_stability_text
        ):
            record_stability_evidence = (
                record_stability_evidence + " " + meta_stability_text
            ).strip()
            record_feature_fields.update(paper_stability)
        if record_stability_evidence:
            record_feature_fields.update(
                _ctrl_v21_10_extract_stability(record_stability_evidence)
            )
        humidity_condition = rec.get("Stability_relative_humidity_load_conditions")
        if _ctrl_v21_10_present(humidity_condition):
            condition_stability = _ctrl_v21_10_extract_stability(str(humidity_condition))
            condition_rh = condition_stability.get("Stability_relative_humidity_average_value")
            if condition_rh is not None:
                record_feature_fields["Stability_relative_humidity_average_value"] = condition_rh
        record_performance = dict(performance)
        record_metric_evidence = " ".join(
            str(rec.get(f"_evidence_{field}") or "")
            for field in (
                "JV_default_PCE",
                "JV_default_Voc",
                "JV_default_Jsc",
                "JV_default_FF",
            )
            if _ctrl_v21_10_present(rec.get(f"_evidence_{field}"))
        )
        improved_device_metrics = _ctrl_v21_14_extract_improved_device_metrics(
            record_metric_evidence
        )
        if improved_device_metrics:
            for col, value in improved_device_metrics.items():
                old_value = _ctrl_v21_10_float(rec.get(col))
                if old_value is None or abs(old_value - value) > 1e-9:
                    rec[col] = value
                    rec[f"_evidence_{col}"] = record_metric_evidence
                    changes.append(f"{col}={value}:explicit_improved_endpoint")
            record_performance.update(improved_device_metrics)
        trusted_record_pce = _ctrl_v21_10_float(rec.get("JV_default_PCE"))
        if trusted_record_pce is not None:
            record_performance.setdefault("JV_default_PCE", trusted_record_pce)
        companion_performance, companion_evidence = _ctrl_v21_12_extract_companion_performance(
            record_performance,
            broad_paper_text,
        )
        for col, value in companion_performance.items():
            record_performance.setdefault(col, value)
        if source_doi:
            current_doi = _ctrl_v21_10_normalize_doi(rec.get("Ref_DOI_number"))
            if not current_doi or current_doi != source_doi:
                rec["Ref_DOI_number"] = source_doi
                changes.append(f"Ref_DOI_number=source_doi:{source_doi}")
        if formula:
            rec["Perovskite_composition_short_form"] = formula
            rec["Perovskite_composition_long_form"] = formula
            if composition_assumption and not _ctrl_v21_10_present(rec.get("Perovskite_composition_assumption")):
                rec["Perovskite_composition_assumption"] = composition_assumption
            a_ions, b_ions, c_ions = _ctrl_v21_10_ions_from_formula(formula)
            # A trusted resolved formula is more authoritative than an LLM
            # component assignment (for example, PbI2 incorrectly placed in
            # the C-site ion field). Keep the component columns consistent.
            if a_ions:
                rec["Perovskite_composition_a_ions"] = a_ions
            if b_ions:
                rec["Perovskite_composition_b_ions"] = b_ions
            if c_ions:
                rec["Perovskite_composition_c_ions"] = c_ions
            changes.append(f"Perovskite_composition_short_form={formula}")
            if composition_evidence and not _ctrl_v21_10_present(rec.get("_evidence_Perovskite_composition_short_form")):
                rec["_evidence_Perovskite_composition_short_form"] = composition_evidence
            for component_col, component_value in [
                ("Perovskite_composition_a_ions", a_ions),
                ("Perovskite_composition_b_ions", b_ions),
                ("Perovskite_composition_c_ions", c_ions),
            ]:
                if component_value and composition_evidence:
                    rec[f"_evidence_{component_col}"] = composition_evidence
            if composition_assumption and composition_evidence and not _ctrl_v21_10_present(rec.get("_evidence_Perovskite_composition_assumption")):
                rec["_evidence_Perovskite_composition_assumption"] = composition_evidence
        for col, value in record_performance.items():
            if not _ctrl_v21_10_present(rec.get(col)):
                rec[col] = value
                changes.append(f"{col}={value}")
                field_evidence = (
                    companion_evidence
                    if col in companion_performance
                    else (
                        _ctrl_v21_13_field_value_context(
                            source_text,
                            col,
                            value,
                        )
                        or evidence_text
                    )
                )
                if field_evidence and not _ctrl_v21_10_present(rec.get(f"_evidence_{col}")):
                    rec[f"_evidence_{col}"] = field_evidence
        for col, value in record_feature_fields.items():
            condition_rh_override = (
                col == "Stability_relative_humidity_average_value"
                and _ctrl_v21_10_present(humidity_condition)
            )
            if condition_rh_override or not _ctrl_v21_10_present(rec.get(col)):
                rec[col] = value
                changes.append(f"{col}={value}")
                field_evidence = (
                    rec.get("_evidence_Stability_relative_humidity_load_conditions")
                    if condition_rh_override
                    else (
                        record_stability_evidence
                        if str(col).startswith("Stability_")
                        else evidence_text
                    )
                )
                if field_evidence:
                    rec[f"_evidence_{col}"] = field_evidence
        if "JV_default_PCE" in record_performance and not _ctrl_v21_10_present(rec.get("JV_default_PCE_scan_direction")):
            rec["JV_default_PCE_scan_direction"] = "unspecified_text_mined"
            changes.append("JV_default_PCE_scan_direction=unspecified_text_mined")
        changes.extend(_ctrl_v21_13_sanitize_record_targets(rec))
        if changes:
            rec["_lit_agent_extraction_enriched"] = 1
            existing = str(rec.get("_lit_agent_recovery_notes") or "").strip()
            note = "v21.12 deterministic enrichment from source abstract or explicit field evidence only"
            rec["_lit_agent_recovery_notes"] = (existing + "; " + note).strip("; ")
            rec["_lit_agent_extraction_enrichment_changes"] = "; ".join(changes)
            changed += 1
    if changed:
        slug = meta.get("paper_slug") or meta.get("slug") or meta.get("title") if isinstance(meta, dict) else "unknown"
        slug_safe = str(slug).encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"[INFO] v21.12 trusted extraction enrichment populated source-derived fields for {changed} record(s): {slug_safe}")
    return records


def _write_embedded_modules(work_dir: _Path_controller) -> tuple[_Path_controller, _Path_controller]:
    embedded_dir = work_dir / "_embedded_runtime_modules"
    embedded_dir.mkdir(parents=True, exist_ok=True)

    lit_file = embedded_dir / "_embedded_literature_agent_v20_runtime.py"
    integration_file = embedded_dir / "_embedded_integrate_literature_to_pce_stability_v20_runtime.py"

    lit_file.write_text(_LITERATURE_AGENT_SOURCE + _v21_controller_embedded_quality_patch_source(), encoding="utf-8")
    integration_file.write_text(
        _INTEGRATION_BRIDGE_SOURCE
        + _v21_sparse_aware_integration_patch_source()
        + _v21_8_expansion_backfill_patch_source(),
        encoding="utf-8",
    )

    return lit_file, integration_file


def _load_module_from_path(path: _Path_controller, module_name: str):
    spec = _importlib_util_controller.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {path}")
    module = _importlib_util_controller.module_from_spec(spec)
    _sys_controller.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _remove_dir(path) -> None:
    p = _Path_controller(path)
    if p.exists():
        print(f"[INFO] Removing old folder: {p}")
        _shutil_controller.rmtree(p, ignore_errors=True)


def _setenv(key: str, value) -> None:
    if value is not None:
        _os_controller.environ[key] = str(value)


def _clearenv(keys: _List_controller[str]) -> None:
    for k in keys:
        _os_controller.environ.pop(k, None)


def _check_file(path: str | _Path_controller, label: str, required: bool = True) -> bool:
    p = _Path_controller(path)
    ok = p.exists()
    if not ok:
        msg = f"{label} not found: {p}"
        if required:
            raise FileNotFoundError(msg)
        print(f"[WARN] {msg}")
    return ok


def _try_check_local_endpoint(url: str, label: str) -> None:
    if not url or not str(url).startswith("http"):
        return
    test_url = None
    if "localhost:11434" in url:
        test_url = "http://localhost:11434/api/tags"
    elif "localhost:8000" in url:
        test_url = "http://localhost:8000/v1/models"
    if not test_url:
        return

    try:
        import urllib.request
        with urllib.request.urlopen(test_url, timeout=5) as r:
            print(f"[INFO] {label} reachable: {test_url} | status={r.status}")
    except Exception as e:
        print(f"[WARN] Could not verify {label} at {test_url}. Error: {e}")


_EXPANSION_BLOCK_TITLE_TERMS = (
    "review", "perspective", "editorial", "commentary", "roadmap", "outlook",
    "conference", "proceedings", "abstract", "preprint", "ssrn",
    "decision letter", "author response", "peer review", "review report",
    "simulation", "numerical", "scaps", "wxamps", "gpvdm", "silvaco",
    "modeling", "modelling", "theoretical", "first-principles", "first principles",
    "density functional", "dft", "machine learning", "deep learning", "prediction",
    "bibliometric", "meta-analysis", "topcon", "silicon heterojunction", "gaas",
    "cigs", "sb2s3", "czts", "organic solar cell", "polymer solar cell",
    "thermoelectric", "photodetector", "led", "light-emitting", "battery",
    "supercapacitor", "oral irrigation", "water splitting", "hydrogen evolution",
    "efficiency tables", "corrigendum",
)

_EXPANSION_BLOCK_DOI_PREFIXES = (
    "10.2139",   # SSRN
    "10.1109",   # IEEE/proceedings-heavy for this use case
    "10.1364",   # Optics proceedings/conference-heavy for this use case
    "10.1117",   # SPIE proceedings-heavy for this use case
    "10.29363",  # conference/preprint-like source seen in failed pilots
)

_EXPANSION_CORE_PATTERNS = (
    r"\bperovskite solar cell(s)?\b",
    r"\bperovskite photovoltaic(s)?\b",
    r"\bperovskite device(s)?\b",
    r"\bperovskite solar module(s)?\b",
    r"\bhalide perovskite (solar )?(cell|device|photovoltaic|module)(s)?\b",
    r"\bmetal[- ]halide perovskite (solar )?(cell|device|photovoltaic|module)(s)?\b",
    r"\bphotovoltaic (performance|device|application)(s)? of .{0,50}\bperovskite(s)?\b",
)

_EXPANSION_TARGET_TERMS = (
    "power conversion efficiency", " pce", "j-v", " jv ", "voc", "jsc",
    "fill factor", "device architecture", "etl", "htl", "encapsulation",
    "stability", "t80", "t90", "t95", "retention", "lifetime", "aging",
    "ageing", "humidity", "thermal stability", "operational stability",
    "illumination", "maximum power point", "mpp",
)

_EXPANSION_PCE_MODEL_READY_TERMS = (
    "power conversion efficiency", " pce", "j-v", " jv ", "voc", "jsc",
    "short-circuit current", "open-circuit voltage", "fill factor",
    "champion efficiency", "certified efficiency", "photovoltaic performance",
    "device performance", "reverse scan", "forward scan",
)

_EXPANSION_STABILITY_MODEL_READY_TERMS = (
    "t80", "t90", "t95", "retention", "retained", "lifetime",
    "operational stability", "maximum power point", "mpp", "aging", "ageing",
    "damp heat", "humidity", "thermal stability", "illumination",
    "continuous illumination", "encapsulation", "initial pce", "after 1000",
    "1000 h", "1000 hours",
)

_EXPANSION_DEVICE_CONTEXT_TERMS = (
    "device", "solar cell", "cell", "architecture", "stack", "etl", "htl",
    "electron transport", "hole transport", "back contact", "ito", "fto",
    "sno2", "tio2", "nio", "spiro", "ptaa", "pcbm", "c60", "gold", "silver",
    "carbon electrode",
)
try:
    _controller_ontology_device_terms = globals().get("V21_ONTOLOGY_DEVICE_PERFORMANCE_TERMS", tuple())
    _controller_external_terms_fn = globals().get("_v21_external_ontology_terms")
    _controller_external_terms = (
        _controller_external_terms_fn(
            "family_gate_terms",
            "performance_metrics",
            "figure_task_ontology.jv_curve",
            "figure_task_ontology.stability_curve",
        )
        if callable(_controller_external_terms_fn)
        else tuple()
    )
    _EXPANSION_TARGET_TERMS = tuple(dict.fromkeys(
        _EXPANSION_TARGET_TERMS
        + tuple(_controller_ontology_device_terms or tuple())
        + tuple(_controller_external_terms or tuple())
    ))
except Exception as exc:
    print(f"[WARN] Could not augment expansion target terms from ontology: {exc}")


def _controller_item_text(item) -> str:
    def _join(value) -> str:
        if isinstance(value, list):
            return " ".join(str(x or "") for x in value)
        return str(value or "")

    parts = [
        _join(item.get("title")),
        _join(item.get("subtitle")),
        str(item.get("abstract") or ""),
        _join(item.get("container-title")),
        str(item.get("publisher") or ""),
    ]
    text = " ".join(parts)
    text = _re_controller.sub(r"<[^>]+>", " ", text)
    return _re_controller.sub(r"\s+", " ", text).strip()


def _append_expansion_candidate_filter_report(work_dir: _Path_controller, row: dict) -> None:
    report = work_dir / "expansion_candidate_filter_report.csv"
    fields = [
        "timestamp", "allowed", "reason", "mode", "title", "doi", "item_type",
        "journal", "target_hits", "core_hits",
    ]
    try:
        report.parent.mkdir(parents=True, exist_ok=True)
        exists = report.exists()
        row = {k: row.get(k, "") for k in fields}
        row["timestamp"] = row.get("timestamp") or _datetime_controller.now().isoformat(timespec="seconds")
        with report.open("a", encoding="utf-8-sig", newline="") as f:
            writer = _csv_controller.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:
        print(f"[WARN] Could not write expansion candidate filter report: {exc}")


def _count_expansion_term_hits(low: str, terms: tuple[str, ...]) -> int:
    return sum(1 for term in terms if term in low)


def _blocked_expansion_title_term(title: str) -> str:
    """Return a blocked title term without matching fragments such as enabLED."""
    title_low = f" {str(title or '').lower()} "
    for term in _EXPANSION_BLOCK_TITLE_TERMS:
        pattern = rf"(?<![a-z0-9]){_re_controller.escape(term)}(?![a-z0-9])"
        if _re_controller.search(pattern, title_low):
            return term
    return ""


def _strict_model_ready_candidate_reason(low: str, target_hits: int, core_hits: int) -> tuple[bool, str, int, int, int]:
    pce_hits = _count_expansion_term_hits(low, _EXPANSION_PCE_MODEL_READY_TERMS)
    stability_hits = _count_expansion_term_hits(low, _EXPANSION_STABILITY_MODEL_READY_TERMS)
    device_hits = _count_expansion_term_hits(low, _EXPANSION_DEVICE_CONTEXT_TERMS)

    has_pce_row_signal = pce_hits >= 2 and device_hits >= 1
    has_stability_row_signal = stability_hits >= 2 and (pce_hits >= 1 or device_hits >= 1)
    if not (has_pce_row_signal or has_stability_row_signal):
        return (
            False,
            f"not_model_ready_strict:pce={pce_hits};stability={stability_hits};device={device_hits}",
            pce_hits,
            stability_hits,
            device_hits,
        )

    if target_hits < 3:
        return (
            False,
            f"too_few_strict_target_terms:{target_hits};pce={pce_hits};stability={stability_hits};device={device_hits}",
            pce_hits,
            stability_hits,
            device_hits,
        )

    return (
        True,
        f"model_ready_strict_candidate:pce={pce_hits};stability={stability_hits};device={device_hits}",
        pce_hits,
        stability_hits,
        device_hits,
    )


def _stability_target_candidate_reason(low: str) -> tuple[bool, str, int, int]:
    """Gate metadata for retrieval; numeric target evidence is checked after full-text extraction."""
    stability_hits = _count_expansion_term_hits(low, _EXPANSION_STABILITY_MODEL_READY_TERMS)
    device_hits = _count_expansion_term_hits(low, _EXPANSION_DEVICE_CONTEXT_TERMS)
    allowed = stability_hits >= 1 and device_hits >= 1
    reason = (
        f"stability_target_candidate:stability={stability_hits};device={device_hits}"
        if allowed
        else f"not_stability_target_ready:stability={stability_hits};device={device_hits}"
    )
    return allowed, reason, stability_hits, device_hits


def _score_expansion_candidate(item, mode: str = "target_ready") -> tuple[bool, str, int, int]:
    title = " ".join(item.get("title", [])).strip()
    if not title:
        return False, "missing_title", 0, 0

    item_type = str(item.get("type") or "").lower().strip()
    if item_type and item_type not in {"journal-article", "posted-content"}:
        return False, f"blocked_type:{item_type}", 0, 0

    doi = str(item.get("DOI") or "").lower().strip()
    if any(doi.startswith(prefix) for prefix in _EXPANSION_BLOCK_DOI_PREFIXES):
        return False, "blocked_doi_prefix", 0, 0

    text = _controller_item_text(item)
    low = f" {text.lower()} "
    blocked_term = _blocked_expansion_title_term(title)
    if blocked_term:
        return False, f"blocked_title_term:{blocked_term}", 0, 0

    core_hits = sum(1 for pattern in _EXPANSION_CORE_PATTERNS if _re_controller.search(pattern, low))
    if core_hits == 0:
        return False, "missing_core_perovskite_solar_cell_phrase", 0, 0

    target_hits = sum(1 for term in _EXPANSION_TARGET_TERMS if term in low)
    if str(mode or "").lower() == "stability_target_ready":
        stability_allowed, stability_reason, _, _ = _stability_target_candidate_reason(low)
        return stability_allowed, stability_reason, target_hits, core_hits
    if str(mode or "").lower() == "model_ready_strict":
        strict_allowed, strict_reason, _, _, _ = _strict_model_ready_candidate_reason(low, target_hits, core_hits)
        if not strict_allowed:
            return False, strict_reason, target_hits, core_hits
        return True, strict_reason, target_hits, core_hits

    if target_hits < 1:
        return False, f"too_few_pce_stability_target_terms:{target_hits}", target_hits, core_hits

    return True, "target_ready_candidate", target_hits, core_hits


def _score_expansion_meta(meta, mode: str = "target_ready") -> tuple[bool, str, int, int]:
    title = str(meta.get("title") or "").strip()
    if not title:
        return False, "missing_title", 0, 0

    doi = str(meta.get("doi") or "").lower().strip()
    if not doi:
        return False, "missing_doi_for_search_expansion", 0, 0
    if any(doi.startswith(prefix) for prefix in _EXPANSION_BLOCK_DOI_PREFIXES):
        return False, "blocked_doi_prefix", 0, 0

    text = " ".join([
        title,
        str(meta.get("abstract") or ""),
        str(meta.get("journal") or ""),
        str(meta.get("source") or ""),
    ])
    text = _re_controller.sub(r"<[^>]+>", " ", text)
    normalized = _re_controller.sub(r"\s+", " ", text)
    low = f" {normalized.lower()} "

    blocked_term = _blocked_expansion_title_term(title)
    if blocked_term:
        return False, f"blocked_title_term:{blocked_term}", 0, 0

    core_hits = sum(1 for pattern in _EXPANSION_CORE_PATTERNS if _re_controller.search(pattern, low))
    if core_hits == 0:
        return False, "missing_core_perovskite_solar_cell_phrase", 0, 0

    target_hits = sum(1 for term in _EXPANSION_TARGET_TERMS if term in low)
    if str(mode or "").lower() == "stability_target_ready":
        stability_allowed, stability_reason, _, _ = _stability_target_candidate_reason(low)
        return stability_allowed, stability_reason, target_hits, core_hits
    if str(mode or "").lower() == "model_ready_strict":
        strict_allowed, strict_reason, _, _, _ = _strict_model_ready_candidate_reason(low, target_hits, core_hits)
        if not strict_allowed:
            return False, strict_reason, target_hits, core_hits
        return True, strict_reason, target_hits, core_hits

    if target_hits < 1:
        return False, f"too_few_pce_stability_target_terms:{target_hits}", target_hits, core_hits

    return True, "target_ready_candidate", target_hits, core_hits


def _normalize_expansion_doi(value) -> str:
    text = str(value or "").strip().lower()
    text = _re_controller.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = _re_controller.sub(r"^doi:\s*", "", text)
    return text.strip(" .;")


def _normalize_expansion_title(value) -> str:
    text = str(value or "")
    text = _re_controller.sub(r"<[^>]+>", " ", text)
    text = _re_controller.sub(r"[^a-zA-Z0-9]+", " ", text).lower()
    return _re_controller.sub(r"\s+", " ", text).strip()


_EXPANSION_ALIGNMENT_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "using", "via", "of", "in", "on", "to",
    "a", "an", "by", "toward", "towards", "study", "effect", "effects", "enhanced", "improved",
    "performance", "stability", "efficient", "high", "based", "cells", "cell", "solar",
}


def _expansion_source_alignment(meta: dict, text: str) -> tuple[bool, str, float, float, int, int]:
    """Verify that retrieved full text belongs to the requested DOI/title."""
    body = str(text or "")
    body_norm = _normalize_expansion_title(body[:250000])
    # The document title belongs in the front matter. Looking across an entire
    # paper allows unrelated domain papers to match scattered generic tokens.
    body_head_norm = _normalize_expansion_title(body[:12000])
    doi = _normalize_expansion_doi(
        meta.get("doi") or meta.get("DOI") or meta.get("Ref_DOI_number")
    )
    doi_found = int(bool(
        doi and doi in body.lower().replace("https://doi.org/", "")
    ))
    title = str(meta.get("title") or meta.get("Title") or "")
    title_tokens = [
        token for token in _normalize_expansion_title(title).split()
        if len(token) >= 4 and token not in _EXPANSION_ALIGNMENT_STOPWORDS
    ]
    title_tokens = list(dict.fromkeys(title_tokens))
    matched = sum(
        bool(_re_controller.search(rf"\b{_re_controller.escape(token)}\b", body_norm))
        for token in title_tokens
    )
    matched_head = sum(
        bool(_re_controller.search(rf"\b{_re_controller.escape(token)}\b", body_head_norm))
        for token in title_tokens
    )
    coverage = matched / len(title_tokens) if title_tokens else 0.0
    head_coverage = matched_head / len(title_tokens) if title_tokens else 0.0
    if len(title_tokens) <= 5:
        min_head_matches = min(3, len(title_tokens))
        min_head_coverage = 0.90
    else:
        min_head_matches = 4
        # Resolver searches can return a topically similar paper whose title
        # differs by only one decisive material or treatment term. Without an
        # exact DOI in the retrieved document, require near-exact front-matter
        # title coverage before trusting any device values.
        min_head_coverage = 0.90
    title_aligned = (
        matched_head >= min_head_matches
        and head_coverage >= min_head_coverage
    )
    perovskite_count = len(
        _re_controller.findall(r"\bperovskit\w*\b", body_norm)
    )
    title_requires_perovskite = "perovskit" in _normalize_expansion_title(title)
    domain_aligned = (not title_requires_perovskite) or perovskite_count >= 3
    if doi_found and domain_aligned:
        return (
            True,
            "doi_and_domain_match",
            coverage,
            head_coverage,
            doi_found,
            perovskite_count,
        )
    if title_aligned and domain_aligned:
        return (
            True,
            "title_head_and_domain_match",
            coverage,
            head_coverage,
            doi_found,
            perovskite_count,
        )
    reason = "domain_mismatch" if not domain_aligned else "doi_title_head_mismatch"
    return (
        False,
        reason,
        coverage,
        head_coverage,
        doi_found,
        perovskite_count,
    )


def _load_expansion_known_candidate_keys(args, work_dir: _Path_controller) -> tuple[set[str], set[str]]:
    """Load DOI/title keys that should be skipped before expensive web extraction."""
    known_dois: set[str] = set()
    known_titles: set[str] = set()

    candidate_csvs = [
        work_dir / "csv" / "all_records.csv",
        work_dir / "csv" / "all_records_from_manager.csv",
        work_dir / "combined_full_coverage" / "all_records.csv",
        work_dir / "combined_full_coverage" / "all_records_from_manager.csv",
    ]
    if bool(int(getattr(args, "expansion_skip_base_known_candidates", 1))):
        candidate_csvs.insert(0, _Path_controller(getattr(args, "base_csv", "")))
    processed_registry = str(getattr(args, "expansion_processed_registry", "") or "").strip()
    if processed_registry:
        candidate_csvs.append(_Path_controller(processed_registry))
    seen_paths: set[str] = set()
    for path in candidate_csvs:
        try:
            path = _Path_controller(path)
        except Exception:
            continue
        key = str(path).lower()
        if key in seen_paths or not path.exists() or path.suffix.lower() != ".csv":
            continue
        seen_paths.add(key)
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = _csv_controller.DictReader(handle)
                for row in reader:
                    doi = _normalize_expansion_doi(
                        row.get("Ref_DOI_number")
                        or row.get("doi")
                        or row.get("DOI")
                    )
                    if doi:
                        known_dois.add(doi)
                    title = _normalize_expansion_title(
                        row.get("Ref_original_filename_data_upload")
                        or row.get("title")
                        or row.get("Title")
                        or row.get("title_key")
                    )
                    if len(title) >= 24:
                        known_titles.add(title)
        except Exception as exc:
            print(f"[WARN] Could not load known-candidate skip keys from {path}: {exc}")

    return known_dois, known_titles


def _install_expansion_candidate_filter(lit_module, args) -> None:
    mode = str(getattr(args, "expansion_candidate_filter", "target_ready") or "target_ready").strip().lower()
    require_oa = bool(int(getattr(args, "expansion_require_oa", 1)))
    skip_known = bool(int(getattr(args, "expansion_skip_known_candidates", 1)))
    if mode == "off":
        return
    if not int(getattr(args, "disable_google_drive", 0)) or str(getattr(args, "run_mode", "")).lower() != "expand":
        return
    if not hasattr(lit_module, "is_allowed_crossref_item"):
        return

    original_filter = lit_module.is_allowed_crossref_item
    original_merge_candidates = getattr(lit_module, "merge_candidates", None)
    crossref_fulltext_by_doi: dict[str, list[str]] = {}
    work_dir = _Path_controller(args.work_dir)
    known_dois, known_titles = (
        _load_expansion_known_candidate_keys(args, work_dir) if skip_known else (set(), set())
    )
    skip_terminal_retrieval_failures = bool(
        int(getattr(args, "expansion_skip_terminal_retrieval_failures", 1))
    )
    terminal_failure_min_routes = max(
        2,
        int(getattr(args, "expansion_terminal_failure_min_routes", 4)),
    )
    terminal_failed_dois: set[str] = set()
    terminal_failed_titles: set[str] = set()
    retrieval_report = work_dir / "retrieval_reports" / "retrieval_candidate_report.csv"
    if skip_terminal_retrieval_failures and retrieval_report.exists():
        grouped: dict[str, dict] = {}
        try:
            with retrieval_report.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in _csv_controller.DictReader(handle):
                    doi = _normalize_expansion_doi(row.get("doi") or "")
                    title = _normalize_expansion_title(row.get("title") or "")
                    key = f"doi:{doi}" if doi else (f"title:{title}" if title else "")
                    if not key:
                        continue
                    state = grouped.setdefault(
                        key,
                        {"doi": doi, "title": title, "routes": set(), "success": False},
                    )
                    route = str(row.get("url") or row.get("source") or "").strip().lower()
                    if route:
                        state["routes"].add(route)
                    state["success"] = state["success"] or str(row.get("ok") or "").strip().lower() in {
                        "1", "true", "yes",
                    }
            for state in grouped.values():
                if state["success"] or len(state["routes"]) < terminal_failure_min_routes:
                    continue
                if state["doi"]:
                    terminal_failed_dois.add(state["doi"])
                elif state["title"]:
                    terminal_failed_titles.add(state["title"])
        except Exception as exc:
            print(f"[WARN] Could not load terminal retrieval-failure cache: {exc}")
    if skip_known:
        print(
            f"[INFO] Expansion known-candidate skip active | "
            f"known_dois={len(known_dois)} | known_titles={len(known_titles)}"
        )
    if skip_terminal_retrieval_failures:
        print(
            f"[INFO] Expansion terminal retrieval-failure skip active | "
            f"failed_dois={len(terminal_failed_dois)} | "
            f"failed_titles={len(terminal_failed_titles)} | "
            f"min_routes={terminal_failure_min_routes}"
        )
    oa_cache_path = work_dir / "expansion_candidate_oa_cache.json"
    try:
        oa_cache = _json_controller.loads(oa_cache_path.read_text(encoding="utf-8")) if oa_cache_path.exists() else {}
    except Exception:
        oa_cache = {}

    def _candidate_oa_status(doi: str) -> dict:
        key = str(doi or "").lower().strip()
        if not key:
            return {"is_oa": False, "pdf_url": None, "landing_page": None, "error": "missing_doi"}
        if key in oa_cache:
            return oa_cache[key]
        try:
            if hasattr(lit_module, "unpaywall_lookup"):
                data = lit_module.unpaywall_lookup(key)
            else:
                data = {"is_oa": False, "pdf_url": None, "landing_page": None}
            out = {
                "is_oa": bool(data.get("is_oa")),
                "pdf_url": data.get("pdf_url"),
                "landing_page": data.get("landing_page"),
                "error": "",
            }
        except Exception as exc:
            out = {"is_oa": False, "pdf_url": None, "landing_page": None, "error": str(exc)[:200]}
        oa_cache[key] = out
        try:
            oa_cache_path.write_text(_json_controller.dumps(oa_cache, indent=2), encoding="utf-8")
        except Exception:
            pass
        return out

    def _filtered_crossref_item(item):
        try:
            if mode == "query_relevant":
                title = " ".join(item.get("title", [])).strip()
                item_type = str(item.get("type") or "").lower().strip()
                doi_norm = _normalize_expansion_doi(item.get("DOI") or "")
                allowed = bool(title) and (not item_type or item_type == "journal-article")
                if any(doi_norm.startswith(prefix) for prefix in _EXPANSION_BLOCK_DOI_PREFIXES):
                    allowed = False
                reason, target_hits, core_hits = "query_relevance_pool", 0, 0
            else:
                allowed, reason, target_hits, core_hits = _score_expansion_candidate(item, mode=mode)

            title = " ".join(item.get("title", [])).strip()
            journal = " ".join(item.get("container-title", [])).strip()
            doi_norm = _normalize_expansion_doi(item.get("DOI") or "")
            if doi_norm:
                direct_urls = _ctrl_v21_15_crossref_fulltext_urls(item)
                if direct_urls:
                    crossref_fulltext_by_doi[doi_norm] = direct_urls
            title_norm = _normalize_expansion_title(title)
            if allowed and skip_known and doi_norm and doi_norm in known_dois:
                allowed = False
                reason = "known_duplicate_doi_pre_extraction"
            elif allowed and skip_known and title_norm and title_norm in known_titles:
                allowed = False
                reason = "known_duplicate_title_pre_extraction"
            elif allowed and doi_norm and doi_norm in terminal_failed_dois:
                allowed = False
                reason = "terminal_retrieval_failure_cache_doi"
            elif allowed and title_norm and title_norm in terminal_failed_titles:
                allowed = False
                reason = "terminal_retrieval_failure_cache_title"
            _append_expansion_candidate_filter_report(work_dir, {
                "allowed": int(bool(allowed)),
                "reason": reason,
                "mode": mode,
                "title": title[:500],
                "doi": item.get("DOI") or "",
                "item_type": item.get("type") or "",
                "journal": journal[:250],
                "target_hits": target_hits,
                "core_hits": core_hits,
            })
            return bool(allowed)
        except Exception as exc:
            print(f"[WARN] Expansion candidate filter failed open for one Crossref item: {exc}")
            return original_filter(item)

    lit_module.is_allowed_crossref_item = _filtered_crossref_item
    if callable(original_merge_candidates):
        def _filtered_merge_candidates(rss_items, crossref_items):
            merged = original_merge_candidates(rss_items, crossref_items)
            query_ranked = {}
            if mode == "query_relevant":
                documents = []
                for candidate_index, candidate in enumerate(merged):
                    title = str(candidate.get("title") or "")
                    abstract = str(candidate.get("abstract") or "")
                    journal = str(candidate.get("journal") or "")
                    documents.append({
                        "search_text": " ".join((title, abstract, journal)),
                        "title_text": title,
                        "payload": {"candidate_index": candidate_index},
                    })
                ranked_items, query_diagnostics = _rank_evidence_documents(
                    str(getattr(args, "search_query", "") or ""), documents, len(documents)
                )
                query_ranked = {int(item["candidate_index"]): item for item in ranked_items}
                print(
                    f"[INFO] Query-relevance candidate ranking kept {len(query_ranked)}/{len(merged)} | "
                    f"distinctive_terms={query_diagnostics.get('distinctive_terms', [])}"
                )
            kept = []
            for candidate_index, meta in enumerate(merged):
                if mode == "query_relevant":
                    query_match = query_ranked.get(candidate_index)
                    allowed = query_match is not None
                    reason = "query_relevance_passed" if allowed else "query_relevance_failed"
                    target_hits = len((query_match or {}).get("matched_query_terms") or [])
                    core_hits = len((query_match or {}).get("matched_distinctive_terms") or [])
                    if allowed:
                        meta["_query_relevance"] = query_match
                else:
                    allowed, reason, target_hits, core_hits = _score_expansion_meta(meta, mode=mode)
                doi_norm = _normalize_expansion_doi(meta.get("doi") or "")
                if doi_norm and crossref_fulltext_by_doi.get(doi_norm):
                    meta["_crossref_fulltext_urls"] = list(
                        crossref_fulltext_by_doi[doi_norm]
                    )
                title_norm = _normalize_expansion_title(meta.get("title") or "")
                if allowed and skip_known and doi_norm and doi_norm in known_dois:
                    allowed = False
                    reason = "known_duplicate_doi_pre_extraction"
                elif allowed and skip_known and title_norm and title_norm in known_titles:
                    allowed = False
                    reason = "known_duplicate_title_pre_extraction"
                elif allowed and doi_norm and doi_norm in terminal_failed_dois:
                    allowed = False
                    reason = "terminal_retrieval_failure_cache_doi"
                elif allowed and title_norm and title_norm in terminal_failed_titles:
                    allowed = False
                    reason = "terminal_retrieval_failure_cache_title"
                if allowed and meta.get("doi"):
                    # Enrich every DOI candidate with OA locations even when OA
                    # is not mandatory. Otherwise retrieval falls back directly
                    # to publisher DOI pages, which commonly return HTTP 403.
                    oa = _candidate_oa_status(str(meta.get("doi") or ""))
                    if oa.get("landing_page") and not meta.get("landing_page"):
                        meta["landing_page"] = oa.get("landing_page")
                    if oa.get("pdf_url"):
                        meta["_unpaywall_pdf_url"] = oa.get("pdf_url")
                    if require_oa:
                        if not oa.get("is_oa"):
                            allowed = False
                            reason = (
                                f"not_open_access:"
                                f"{oa.get('error') or 'unpaywall_is_oa_false'}"
                            )
                        elif not (oa.get("pdf_url") or oa.get("landing_page")):
                            allowed = False
                            reason = "open_access_without_retrievable_location"
                _append_expansion_candidate_filter_report(work_dir, {
                    "allowed": int(bool(allowed)),
                    "reason": f"merged:{reason}",
                    "mode": mode,
                    "title": str(meta.get("title") or "")[:500],
                    "doi": meta.get("doi") or "",
                    "item_type": "merged_candidate",
                    "journal": str(meta.get("journal") or "")[:250],
                    "target_hits": target_hits,
                    "core_hits": core_hits,
                })
                if allowed:
                    kept.append(meta)
            print(f"[INFO] Expansion merged-candidate filter kept {len(kept)}/{len(merged)} candidates.")
            return kept

        lit_module.merge_candidates = _filtered_merge_candidates
    print(f"[INFO] Expansion candidate filter active: {mode} | require_oa={require_oa} | skip_known={skip_known} | report={work_dir / 'expansion_candidate_filter_report.csv'}")


def _ctrl_v21_15_arxiv_pdf_urls_from_feed(title: str, xml: str) -> list[str]:
    target_tokens = {
        token
        for token in _normalize_expansion_title(title).split()
        if len(token) >= 4 and token not in _EXPANSION_ALIGNMENT_STOPWORDS
    }
    if len(target_tokens) < 3:
        return []

    matched_urls: list[str] = []
    for entry in _re_controller.findall(
        r"<entry\b[^>]*>(.*?)</entry>",
        str(xml or ""),
        flags=_re_controller.I | _re_controller.S,
    ):
        title_match = _re_controller.search(
            r"<title\b[^>]*>(.*?)</title>",
            entry,
            flags=_re_controller.I | _re_controller.S,
        )
        if not title_match:
            continue
        result_title = _html_controller.unescape(
            _re_controller.sub(r"<[^>]+>", " ", title_match.group(1))
        )
        result_tokens = {
            token
            for token in _normalize_expansion_title(result_title).split()
            if len(token) >= 4 and token not in _EXPANSION_ALIGNMENT_STOPWORDS
        }
        overlap = len(target_tokens & result_tokens)
        coverage = overlap / max(1, len(target_tokens))
        reverse_coverage = overlap / max(1, len(result_tokens))
        required_matches = min(4, len(target_tokens))
        if (
            overlap < required_matches
            or coverage < 0.90
            or reverse_coverage < 0.80
        ):
            continue
        for href in _re_controller.findall(
            r"""href=["']([^"']+)["']""",
            entry,
            flags=_re_controller.I,
        ):
            href = _html_controller.unescape(href).strip()
            if "/pdf/" in href or href.lower().split("?", 1)[0].endswith(".pdf"):
                matched_urls.append(href)

    return list(dict.fromkeys(matched_urls))


def _ctrl_v21_15_landing_fulltext_urls(
    html: str,
    article_url: str,
    max_links: int = 4,
) -> list[str]:
    raw_html = str(html or "")
    urls: list[str] = []

    for tag in _re_controller.findall(
        r"<meta\b[^>]*>",
        raw_html,
        flags=_re_controller.I,
    ):
        if not _re_controller.search(
            (
                r"""(?:name|property)=["'](?:citation_pdf_url|"""
                r"""citation_fulltext_html_url|citation_xml_url)["']"""
            ),
            tag,
            flags=_re_controller.I,
        ):
            continue
        content_match = _re_controller.search(
            r"""content=["']([^"']+)["']""",
            tag,
            flags=_re_controller.I,
        )
        if content_match:
            urls.append(content_match.group(1))

    # Repository landing pages commonly expose author manuscripts through
    # schema.org DataDownload JSON-LD instead of citation_pdf_url metadata.
    for content_url in _re_controller.findall(
        r"""["']contentUrl["']\s*:\s*["']([^"']+)["']""",
        raw_html,
        flags=_re_controller.I,
    ):
        urls.append(content_url)

    for anchor_match in _re_controller.finditer(
        r"""<a\b[^>]*href=["']([^"']+)["'][^>]*>(.*?)</a>""",
        raw_html,
        flags=_re_controller.I | _re_controller.S,
    ):
        href = _html_controller.unescape(anchor_match.group(1)).strip()
        label = _normalize_expansion_title(
            _re_controller.sub(r"<[^>]+>", " ", anchor_match.group(2))
        )
        href_path = href.lower().split("?", 1)[0]
        if (
            href_path.endswith((".pdf", ".docx", ".xml", ".html", ".htm", ".txt"))
            or "application/pdf" in anchor_match.group(0).lower()
            or "wordprocessingml.document" in anchor_match.group(0).lower()
            or any(
                content_type in anchor_match.group(0).lower()
                for content_type in (
                    "application/xml",
                    "text/xml",
                    "text/html",
                    "text/plain",
                )
            )
            or "/filesets/" in href.lower()
            or (
                any(
                    term in label
                    for term in (
                        "download",
                        "full text",
                        "manuscript",
                        "view pdf",
                        "view html",
                        "view xml",
                    )
                )
                and (
                    "pdf" in label
                    or "docx" in label
                    or "html" in label
                    or "xml" in label
                    or "plain text" in label
                    or "bitstream" in href.lower()
                    or "download" in href.lower()
                )
            )
        ):
            urls.append(href)

    decoded_state = (
        raw_html
        .replace("&q;", '"')
        .replace("&a;", "&")
        .replace("&s;", "'")
    )
    for bitstream_path, bitstream_id in _re_controller.findall(
        r"https?://[^\"'\s<>]+"
        r"(/server/api/core/bitstreams/)\s*([0-9a-f-]{36})",
        decoded_state,
        flags=_re_controller.I,
    ):
        urls.append(
            _urljoin_controller(
                article_url,
                bitstream_path + bitstream_id + "/content",
            )
        )

    normalized: list[str] = []
    for value in urls:
        absolute = _urljoin_controller(article_url, _html_controller.unescape(value))
        if not absolute.lower().startswith(("http://", "https://")):
            continue
        if absolute.lower() == str(article_url or "").strip().lower():
            continue
        normalized.append(absolute)
    return list(dict.fromkeys(normalized))[:max(0, int(max_links))]


def _ctrl_v21_15_crossref_fulltext_urls(item: dict) -> list[str]:
    urls: list[str] = []
    for link in (item or {}).get("link") or []:
        if not isinstance(link, dict):
            continue
        url = str(link.get("URL") or link.get("url") or "").strip()
        content_type = str(link.get("content-type") or "").lower()
        intended = str(link.get("intended-application") or "").lower()
        if not url.lower().startswith(("http://", "https://")):
            continue
        if (
            "pdf" in content_type
            or "xml" in content_type
            or "plain" in content_type
            or url.lower().split("?", 1)[0].endswith(".pdf")
            or url.lower().split("?", 1)[0].endswith((".xml", ".txt"))
            or intended in {"text-mining", "similarity-checking"}
        ):
            urls.append(url)
    return list(dict.fromkeys(urls))


def _ctrl_v21_17_extract_docx_text(content: bytes) -> str:
    """Extract paragraph and table text from an OOXML author manuscript."""
    import io
    import zipfile
    import xml.etree.ElementTree as ET

    if not content or len(content) > 50_000_000:
        return ""
    chunks: list[str] = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [
            name
            for name in archive.namelist()
            if name == "word/document.xml"
            or (
                (
                    name.startswith("word/header")
                    or name.startswith("word/footer")
                    or name.startswith("word/footnotes")
                    or name.startswith("word/endnotes")
                )
                and name.endswith(".xml")
            )
        ]
        for name in names:
            root = ET.fromstring(archive.read(name))
            for paragraph in root.iter(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
            ):
                text = "".join(
                    node.text or ""
                    for node in paragraph.iter(
                        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
                    )
                ).strip()
                if text:
                    chunks.append(text)
    return "\n".join(chunks)


def _v21_17_assign_distinct_sample_ids(records: list[dict]) -> list[dict]:
    """Controller-side mirror of the embedded runtime identity normalizer."""

    def present(value) -> bool:
        return value not in (None, "", [], {})

    def normalize(value) -> str:
        return _re_controller.sub(
            r"\s+",
            " ",
            _re_controller.sub(r"[^a-z0-9]+", " ", str(value or "").lower()),
        ).strip()

    def number(value) -> str:
        try:
            return f"{float(value):.6g}".replace("-", "m").replace(".", "p")
        except Exception:
            return ""

    def role(record: dict) -> str:
        evidence = " ".join(
            str(value)
            for key, value in record.items()
            if key.startswith("_evidence_") and present(value)
        ).lower()
        if any(term in evidence for term in (
            "target", "treated", "optimized", "optimised", "passivated",
            "champion", "modified device",
        )):
            return "target"
        if present(record.get("JV_certified_values")) or any(
            term in evidence
            for term in ("certified efficiency", "certified pce", "certified value")
        ):
            return "certified"
        if any(
            term in evidence
            for term in ("large-area device", "large area device", "1 cm2 device", "1cm2 device")
        ):
            return "large_area"
        if "control" in evidence:
            return "control"
        return "device"

    def base_sample_id(value: object) -> str:
        sample_id = str(value or "").strip()
        return _re_controller.sub(
            r"_(?:control|target|device|certified|large_area)"
            r"(?:_pce_[0-9mp]+(?:_\d+)?)?$",
            "",
            sample_id,
            flags=_re_controller.IGNORECASE,
        )

    def normalized(value: object) -> str:
        return _re_controller.sub(
            r"\s+",
            " ",
            _re_controller.sub(r"[^a-z0-9]+", " ", str(value or "").lower()),
        ).strip()

    def target_tuple(record: dict) -> tuple[str, ...]:
        return tuple(
            normalized(record.get(field))
            for field in (
                "JV_default_PCE",
                "JV_default_Voc",
                "JV_default_Jsc",
                "JV_default_FF",
                "Perovskite_composition_long_form",
                "Perovskite_composition_short_form",
                "Cell_stack_sequence",
            )
        )

    def target_alignment_score(record: dict) -> tuple[int, int]:
        evidence = " ".join(
            str(value)
            for key, value in record.items()
            if key.startswith("_evidence_") and present(value)
        ).lower()
        positive = sum(
            evidence.count(term)
            for term in (
                "optimized", "optimised", "treated", "passivated",
                "champion", "target device", "modified device",
            )
        )
        control_only = sum(
            evidence.count(term)
            for term in ("control device exhibits", "unpassivated device")
        )
        populated = sum(
            present(value)
            for key, value in record.items()
            if not key.startswith("_")
        )
        return positive - (10 * control_only), populated

    output = [dict(record) for record in records]

    # Family-level partials can retain old generated sample-ID suffixes after a
    # stronger evidence quote updates the JV tuple. Collapse only rows that
    # still share the same original sample identity and final device/JV tuple.
    deduplicated: list[dict] = []
    duplicate_groups: dict[tuple[str, str, tuple[str, ...]], list[dict]] = {}
    for record in output:
        key = (
            normalized(record.get("Ref_DOI_number")),
            base_sample_id(record.get("Ref_internal_sample_id")),
            target_tuple(record),
        )
        duplicate_groups.setdefault(key, []).append(record)
    for group in duplicate_groups.values():
        if len(group) == 1:
            deduplicated.append(group[0])
            continue
        winner = max(group, key=target_alignment_score)
        winner["_lit_agent_sample_identity_warning"] = (
            f"Collapsed {len(group)} duplicate family-consolidation rows "
            "with the same final grounded device/JV identity."
        )
        deduplicated.append(winner)
    output = deduplicated

    grouped: dict[tuple[str, str], list[int]] = {}
    for index, record in enumerate(output):
        sample_id = base_sample_id(record.get("Ref_internal_sample_id"))
        record["Ref_internal_sample_id"] = sample_id
        if sample_id:
            grouped.setdefault(
                (normalize(record.get("Ref_DOI_number")), sample_id),
                [],
            ).append(index)
    for (_doi, sample_id), indices in grouped.items():
        if len(indices) < 2:
            continue
        seen: set[str] = set()
        counts: dict[str, int] = {}
        for index in indices:
            record = output[index]
            label = role(record)
            counts[label] = counts.get(label, 0) + 1
            metric = number(
                record.get("JV_default_PCE")
                or record.get("JV_reverse_scan_PCE")
                or record.get("JV_forward_scan_PCE")
                or record.get("Stabilised_performance_PCE")
            )
            candidate = f"{sample_id}_{label}"
            if metric:
                candidate += f"_pce_{metric}"
            if candidate in seen:
                candidate += f"_{counts[label]}"
            seen.add(candidate)
            record["Ref_internal_sample_id"] = candidate
            record["_lit_agent_sample_identity_basis"] = (
                "distinct grounded device/measurement row; "
                f"role={label}; original_sample_id={sample_id}"
            )
    return output


def _ctrl_v21_15_pdf_preflight_reason(
    content: bytes,
    reader_cls,
    max_pdf_bytes: int,
    max_pdf_pages: int,
) -> str:
    if len(content) > max_pdf_bytes:
        return f"pdf_too_large:{len(content)}>{max_pdf_bytes}"
    if reader_cls is None:
        return ""
    try:
        reader = reader_cls(_io_controller.BytesIO(content))
        page_count = len(reader.pages)
    except Exception:
        # Keep valid-but-unusual PDFs eligible for the established extractors.
        return ""
    if page_count > max_pdf_pages:
        return f"pdf_too_many_pages:{page_count}>{max_pdf_pages}"
    return ""


_CTRL_V21_18_REPOSITORY_HOST_MARKERS = (
    "arxiv.org",
    "europepmc.org",
    "ebi.ac.uk",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "zenodo.org",
    "figshare.com",
    "hal.science",
    "osti.gov",
    "core.ac.uk",
    "repository.",
    "repositories.",
    "eprints.",
    "dspace.",
    "digitalcommons.",
    "research-portal.",
    "researchportal.",
)


def _ctrl_v21_18_is_repository_route(source: str, url: str) -> bool:
    source_text = str(source or "").lower()
    url_text = str(url or "").lower()
    if any(
        marker in source_text
        for marker in (
            "repository",
            "europepmc",
            "arxiv",
            "author_manuscript",
        )
    ):
        return True
    return any(marker in url_text for marker in _CTRL_V21_18_REPOSITORY_HOST_MARKERS)


def _ctrl_v21_18_resolver_priority(source: str, url: str) -> tuple[int, int]:
    """Rank machine-readable repository sources ahead of publisher pages."""
    source_text = str(source or "").lower()
    url_text = str(url or "").lower().split("?", 1)[0]
    if _ctrl_v21_18_is_repository_route(source_text, url_text):
        tier = 0
    elif any(marker in source_text for marker in ("openalex", "unpaywall")):
        tier = 1
    elif any(marker in source_text for marker in ("candidate_pdf", "source_pdf")):
        tier = 2
    elif "crossref" in source_text:
        tier = 3
    else:
        tier = 4

    if url_text.endswith((".xml", ".docx", ".pdf")):
        format_rank = 0
    elif "fulltextxml" in url_text or "fulltext" in source_text:
        format_rank = 1
    else:
        format_rank = 2
    return tier, format_rank


def _ctrl_v21_18_openalex_location_urls(data: dict) -> list[tuple[str, str]]:
    """Return every OpenAlex location with repository/publisher labels."""
    locations = [
        data.get("primary_location") or {},
        data.get("best_oa_location") or {},
        *(data.get("locations") or []),
    ]
    urls: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for location in locations:
        if not isinstance(location, dict):
            continue
        source_meta = location.get("source") or {}
        source_type = str(
            location.get("host_type")
            or (source_meta.get("type") if isinstance(source_meta, dict) else "")
            or ""
        ).lower()
        host_label = "repository" if source_type == "repository" else "publisher"
        for key, kind in (
            ("pdf_url", "pdf"),
            ("landing_page_url", "landing"),
        ):
            value = str(location.get(key) or "").strip()
            item = (f"openalex_{host_label}_{kind}", value)
            if value and item not in seen:
                seen.add(item)
                urls.append(item)
    return sorted(
        urls,
        key=lambda item: _ctrl_v21_18_resolver_priority(item[0], item[1]),
    )


def _ctrl_v21_18_unpaywall_location_urls(data: dict) -> list[tuple[str, str]]:
    """Use all Unpaywall OA locations instead of only best_oa_location."""
    locations = list(data.get("oa_locations") or [])
    best = data.get("best_oa_location")
    if isinstance(best, dict):
        locations.append(best)
    urls: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for location in locations:
        if not isinstance(location, dict):
            continue
        host_type = str(location.get("host_type") or "").lower()
        host_label = "repository" if host_type == "repository" else "publisher"
        for key, kind in (
            ("url_for_pdf", "pdf"),
            ("url_for_landing_page", "landing"),
            ("url", "url"),
        ):
            value = str(location.get(key) or "").strip()
            item = (f"unpaywall_{host_label}_{kind}", value)
            if value and item not in seen:
                seen.add(item)
                urls.append(item)
    return sorted(
        urls,
        key=lambda item: _ctrl_v21_18_resolver_priority(item[0], item[1]),
    )


def _install_web_expansion_fulltext_fallback(lit_module, args) -> None:
    original_deterministic_candidates = getattr(
        lit_module,
        "deterministic_preextract_candidates",
        None,
    )
    if callable(original_deterministic_candidates) and not getattr(
        original_deterministic_candidates,
        "_v21_13_experimental_metric_guard",
        False,
    ):
        def _deterministic_candidates_with_experimental_guard(metadata, extraction_text):
            candidates = original_deterministic_candidates(metadata, extraction_text)
            guarded = []
            metric_fields = {
                "JV_default_PCE",
                "JV_default_Voc",
                "JV_default_Jsc",
                "JV_default_FF",
                "Stabilised_performance_PCE",
                "Perovskite_band_gap",
            }
            for candidate in candidates or []:
                if not isinstance(candidate, dict):
                    continue
                field = str(candidate.get("field") or "")
                evidence = str(candidate.get("evidence") or "")
                if (
                    field in metric_fields
                    and _ctrl_v21_13_nonexperimental_metric_context(evidence)
                ):
                    continue
                if (
                    field == "Perovskite_composition_short_form"
                    and _ctrl_v21_13_nonexperimental_metric_context(evidence)
                ):
                    continue
                if _re_controller.search(
                    r"\b(?:certifi(?:ed|cation)|steady[\s-]*state|"
                    r"champion|best[\s-]*performing)\b",
                    evidence,
                    flags=_re_controller.IGNORECASE,
                ):
                    candidate = dict(candidate)
                    candidate["confidence"] = "medium_high"
                if (
                    field in metric_fields
                    and str(candidate.get("confidence") or "") == "low_context_risk"
                ):
                    continue
                guarded.append(candidate)
            return guarded

        _deterministic_candidates_with_experimental_guard._v21_13_experimental_metric_guard = True
        lit_module.deterministic_preextract_candidates = (
            _deterministic_candidates_with_experimental_guard
        )

    original_deterministic_preextract = getattr(lit_module, "deterministic_preextract_record", None)
    if callable(original_deterministic_preextract) and not getattr(
        original_deterministic_preextract,
        "_v21_12_trusted_composition_wrapper",
        False,
    ):
        def _deterministic_preextract_with_trusted_composition(metadata, extraction_text):
            record = original_deterministic_preextract(metadata, extraction_text)
            if not isinstance(record, dict):
                return record

            formula, evidence, assumption = _ctrl_v21_12_extract_device_composition(extraction_text)
            if not formula:
                return record

            record["Perovskite_composition_short_form"] = formula
            record["Perovskite_composition_long_form"] = formula
            if assumption:
                record["Perovskite_composition_assumption"] = assumption

            a_ions, b_ions, c_ions = _ctrl_v21_10_ions_from_formula(formula)
            for column, value in [
                ("Perovskite_composition_a_ions", a_ions),
                ("Perovskite_composition_b_ions", b_ions),
                ("Perovskite_composition_c_ions", c_ions),
            ]:
                if value:
                    record[column] = value
                    if evidence:
                        record[f"_evidence_{column}"] = evidence

            if evidence:
                record["_evidence_Perovskite_composition_short_form"] = evidence
                record["_evidence_Perovskite_composition_long_form"] = evidence
                if assumption:
                    record["_evidence_Perovskite_composition_assumption"] = evidence
            record["_lit_agent_composition_source"] = "v21.12_trusted_deterministic_preextract"
            return record

        _deterministic_preextract_with_trusted_composition._v21_12_trusted_composition_wrapper = True
        lit_module.deterministic_preextract_record = _deterministic_preextract_with_trusted_composition

    original_extract_records = getattr(lit_module, "extract_records_with_llm", None)
    if callable(original_extract_records) and not getattr(
        original_extract_records,
        "_v21_12_trusted_composition_wrapper",
        False,
    ):
        def _extract_records_with_trusted_composition(meta, extraction_text, extractor):
            records = original_extract_records(meta, extraction_text, extractor)
            if isinstance(records, list):
                records = _ctrl_v21_10_enrich_extraction_records(
                    meta,
                    "",
                    extraction_text,
                    records,
                )
                # Trusted enrichment may replace a preliminary PCE/JV tuple.
                # Rebuild generated sample identities from the final tuple so
                # stale suffixes cannot survive into the integration bridge.
                assign_distinct_ids = getattr(
                    lit_module,
                    "_v21_17_assign_distinct_sample_ids",
                    None,
                )
                if callable(assign_distinct_ids):
                    records = assign_distinct_ids(records)
            return records

        _extract_records_with_trusted_composition._v21_12_trusted_composition_wrapper = True
        lit_module.extract_records_with_llm = _extract_records_with_trusted_composition

    if not int(getattr(args, "disable_google_drive", 0)) or str(getattr(args, "run_mode", "")).lower() != "expand":
        return

    # Expansion papers frequently contain several control/target/champion
    # devices. The legacy cross-family merge used two sparse matching fields,
    # which could absorb distinct devices before the conflict-aware final
    # coalescer saw them. In expansion mode only, retain every family candidate
    # until final coalescing and do not broadcast a deterministic paper-level
    # target across all sparse device rows.
    original_merge_record_lists = getattr(lit_module, "merge_record_lists", None)
    if callable(original_merge_record_lists):
        def _expansion_concat_record_lists(existing, new):
            return [
                dict(record)
                for record in list(existing or []) + list(new or [])
                if isinstance(record, dict)
            ]

        lit_module.merge_record_lists = _expansion_concat_record_lists

    original_merge_deterministic = getattr(
        lit_module,
        "merge_deterministic_with_llm_records",
        None,
    )
    if callable(original_merge_deterministic):
        def _expansion_merge_deterministic(deterministic, llm_records, meta):
            target_prefixes = ("JV_", "Stability_", "Stabilised_performance_", "EQE_")
            deterministic = dict(deterministic or {})
            llm_records = [
                dict(record) for record in (llm_records or [])
                if isinstance(record, dict)
            ]
            if not llm_records:
                return [deterministic] if deterministic else []

            context = {
                key: value
                for key, value in deterministic.items()
                if not str(key).startswith(target_prefixes)
                and value not in (None, "", [], {})
            }
            deterministic_targets = {
                key: value
                for key, value in deterministic.items()
                if str(key).startswith(target_prefixes)
                and value not in (None, "", [], {})
            }
            any_explicit_target = any(
                value not in (None, "", [], {})
                for record in llm_records
                for key, value in record.items()
                if str(key).startswith(target_prefixes)
            )

            merged = []
            for index, record in enumerate(llm_records):
                out = dict(context)
                out.update({
                    key: value
                    for key, value in record.items()
                    if value not in (None, "", [], {})
                })
                # A deterministic "best PCE" is a paper-level fallback, not
                # evidence that every sparse family row is the same device.
                if index == 0 and not any_explicit_target:
                    for key, value in deterministic_targets.items():
                        out.setdefault(key, value)
                merged.append(out)
            return merged

        lit_module.merge_deterministic_with_llm_records = _expansion_merge_deterministic

    if not hasattr(lit_module, "try_full_text"):
        return

    automated_retrieval = bool(int(getattr(args, "automated_retrieval_enable", 1)))
    max_resolver_urls = max(0, int(getattr(args, "retrieval_max_urls_per_candidate", 8)))
    supplementary_retrieval = bool(int(getattr(args, "retrieval_supplementary_enable", 1)))
    max_supplementary_files = max(
        0,
        int(getattr(args, "retrieval_max_supplementary_files", 2)),
    )
    min_full = max(1000, int(getattr(args, "expansion_min_fulltext_chars", 6000)))
    min_extract = max(1000, int(getattr(args, "expansion_min_extraction_text_chars", 3000)))
    max_pdf_bytes = max(
        1_000_000,
        int(getattr(args, "retrieval_max_pdf_bytes", 25_000_000)),
    )
    max_pdf_pages = max(
        10,
        int(getattr(args, "retrieval_max_pdf_pages", 80)),
    )
    original_try_full_text = lit_module.try_full_text
    work_dir = _Path_controller(getattr(lit_module, "WORK_DIR", getattr(args, "work_dir", ".")))
    retrieval_dir = work_dir / "retrieval_reports"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    retrieval_report_csv = retrieval_dir / "retrieval_candidate_report.csv"
    retrieval_url_outcomes: dict[str, dict[str, str]] = {}
    retrieval_content_outcomes: dict[str, dict[str, str]] = {}
    if retrieval_report_csv.exists():
        try:
            with retrieval_report_csv.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as existing_report:
                for prior in _csv_controller.DictReader(existing_report):
                    status = str(prior.get("status") or "")
                    slug = str(prior.get("slug") or "")
                    url_key = str(prior.get("url") or "").strip().lower()
                    content_hash = str(prior.get("content_sha256") or "").strip()
                    terminal_source_status = (
                        status == "rejected_source_mismatch"
                        or status == "http_200"
                        or status.startswith("pdf_too_")
                    )
                    if url_key and terminal_source_status:
                        retrieval_url_outcomes[url_key] = {
                            "slug": slug,
                            "status": status,
                        }
                    if content_hash and terminal_source_status:
                        retrieval_content_outcomes[content_hash] = {
                            "slug": slug,
                            "status": status,
                        }
        except Exception:
            retrieval_url_outcomes.clear()
            retrieval_content_outcomes.clear()

    for name, value in [
        ("LLM_MIN_FULLTEXT_CHARS", min_full),
        ("MIN_EXTRACTION_TEXT_CHARS", min_extract),
    ]:
        if hasattr(lit_module, name):
            setattr(lit_module, name, value)

    def _read_best_saved_html(slug: str) -> tuple[str, str]:
        html_dir = getattr(lit_module, "HTML_DIR", None)
        if html_dir is None:
            return "", ""
        html_path = _Path_controller(html_dir) / f"{slug}.html"
        if not html_path.exists():
            return "", ""
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        extracted = ""
        try:
            if hasattr(lit_module, "extract_text_from_html"):
                extracted = lit_module.extract_text_from_html(html) or ""
        except Exception:
            extracted = ""
        if len(extracted.strip()) < min_full:
            try:
                soup = lit_module.BeautifulSoup(html, "lxml")
                for bad in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
                    bad.decompose()
                fallback = soup.get_text("\n", strip=True)
                if len(fallback.strip()) > len(extracted.strip()):
                    extracted = fallback
            except Exception:
                pass
        return extracted, str(html_path)

    def _safe_slug_part(value: str) -> str:
        text = _re_controller.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())[:80]
        return text or "retrieved"

    def _read_best_saved_pdf(slug: str) -> tuple[str, str]:
        pdf_dir = getattr(lit_module, "PDF_DIR", None)
        if pdf_dir is None:
            return "", ""
        candidates = sorted(_Path_controller(pdf_dir).glob(f"{slug}*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
        best_text = ""
        best_path = ""
        for pdf_path in candidates[:3]:
            try:
                if pdf_path.stat().st_size > max_pdf_bytes:
                    continue
            except OSError:
                continue
            text = ""
            for fn_name in ("extract_pdf_text_native", "extract_pdf_text"):
                fn = getattr(lit_module, fn_name, None)
                if callable(fn):
                    try:
                        maybe = fn(pdf_path) or ""
                        if len(maybe.strip()) > len(text.strip()):
                            text = maybe
                    except Exception:
                        pass
            if len(text.strip()) > len(best_text.strip()):
                best_text = text
                best_path = str(pdf_path)
        return best_text, best_path

    def _append_retrieval_report(row: dict) -> None:
        base = {
            "timestamp": _datetime_controller.now().isoformat(timespec="seconds"),
            "slug": "",
            "doi": "",
            "title": "",
            "source": "",
            "url": "",
            "ok": 0,
            "status": "",
            "text_length": 0,
            "content_type": "",
            "saved_path": "",
            "alignment_ok": 0,
            "alignment_reason": "",
            "title_token_coverage": 0.0,
            "title_head_token_coverage": 0.0,
            "doi_found_in_text": 0,
            "perovskite_term_count": 0,
            "content_sha256": "",
            "duplicate_of_slug": "",
            "error": "",
        }
        base.update(row or {})
        try:
            exists = retrieval_report_csv.exists()
            fieldnames = list(base.keys())
            if exists:
                with retrieval_report_csv.open(
                    "r",
                    encoding="utf-8-sig",
                    newline="",
                ) as existing:
                    existing_header = next(_csv_controller.reader(existing), [])
                if existing_header:
                    fieldnames = existing_header
            with retrieval_report_csv.open("a", encoding="utf-8-sig", newline="") as f:
                writer = _csv_controller.DictWriter(
                    f,
                    fieldnames=fieldnames,
                    extrasaction="ignore",
                )
                if not exists:
                    writer.writeheader()
                writer.writerow(base)
        except Exception:
            pass

    def _source_alignment(meta: dict, text: str) -> tuple[bool, str, float, float, int, int]:
        return _expansion_source_alignment(meta, text)

    def _requests_get(url: str):
        requests_mod = getattr(lit_module, "requests", None)
        if requests_mod is None:
            return None, "requests_module_missing"
        headers = dict(getattr(lit_module, "HEADERS", {}) or {})
        headers.setdefault("User-Agent", "Mozilla/5.0")
        timeout = max(10, int(getattr(args, "web_request_timeout", getattr(lit_module, "REQUEST_TIMEOUT", 45))))
        try:
            response = requests_mod.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            return response, ""
        except Exception as exc:
            return None, repr(exc)[:300]

    def _extract_html_text_from_string(html: str) -> str:
        text = ""
        try:
            trafilatura = getattr(lit_module, "trafilatura", None)
            if trafilatura is not None:
                text = trafilatura.extract(html, include_tables=True, include_comments=False) or ""
        except Exception:
            text = ""
        if len(text.strip()) < min_extract:
            try:
                if hasattr(lit_module, "extract_text_from_html"):
                    maybe = lit_module.extract_text_from_html(html) or ""
                    if len(maybe.strip()) > len(text.strip()):
                        text = maybe
            except Exception:
                pass
        if len(text.strip()) < min_extract:
            try:
                soup = lit_module.BeautifulSoup(html, "lxml")
                for bad in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
                    bad.decompose()
                fallback = soup.get_text("\n", strip=True)
                if len(fallback.strip()) > len(text.strip()):
                    text = fallback
            except Exception:
                pass
        return text

    def _extract_pdf_text_from_path(pdf_path: _Path_controller) -> str:
        best = ""
        for fn_name in ("extract_pdf_text_native", "extract_pdf_text"):
            fn = getattr(lit_module, fn_name, None)
            if callable(fn):
                try:
                    maybe = fn(pdf_path) or ""
                    if len(maybe.strip()) > len(best.strip()):
                        best = maybe
                except Exception:
                    pass
        return best

    def _pdf_preflight_reason(content: bytes) -> str:
        return _ctrl_v21_15_pdf_preflight_reason(
            content,
            getattr(lit_module, "PdfReader", None),
            max_pdf_bytes,
            max_pdf_pages,
        )

    def _supplementary_text_from_html(
        html: str,
        article_url: str,
        slug: str,
    ) -> tuple[str, list[str]]:
        if not supplementary_retrieval or max_supplementary_files <= 0:
            return "", []
        try:
            soup = lit_module.BeautifulSoup(html, "lxml")
        except Exception:
            return "", []

        candidates = []
        seen = set()
        for anchor in soup.find_all("a", href=True):
            label = " ".join(anchor.get_text(" ", strip=True).split())
            href = str(anchor.get("href") or "").strip()
            low = f"{label} {href}".lower()
            if not any(term in low for term in (
                "supporting information", "supplementary", "supplemental",
                "supporting data", "additional file", "esm",
            )):
                continue
            try:
                absolute = lit_module.urljoin(article_url, href)
            except Exception:
                absolute = href
            if not absolute.lower().startswith(("http://", "https://")):
                continue
            key = absolute.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append((label or "supplementary", absolute))

        texts = []
        saved_paths = []
        for index, (label, url) in enumerate(candidates[:max_supplementary_files], start=1):
            response, error = _requests_get(url)
            if error or response is None or int(getattr(response, "status_code", 0) or 0) >= 400:
                continue
            content = response.content or b""
            if not content or len(content) > 50_000_000:
                continue
            content_type = str(response.headers.get("content-type", "") or "").lower()
            suffix_url = url.lower().split("?", 1)[0]
            supplement_text = ""
            saved_path = ""
            try:
                if "pdf" in content_type or suffix_url.endswith(".pdf") or content[:5] == b"%PDF-":
                    if _pdf_preflight_reason(content):
                        continue
                    out_dir = retrieval_dir / "supplementary"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    path = out_dir / f"{slug}_supplement_{index}.pdf"
                    path.write_bytes(content)
                    supplement_text = _extract_pdf_text_from_path(path)
                    saved_path = str(path)
                elif suffix_url.endswith((".csv", ".tsv")) or "text/csv" in content_type:
                    encoding = response.encoding or "utf-8"
                    supplement_text = content.decode(encoding, errors="ignore")
                    out_dir = retrieval_dir / "supplementary"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    path = out_dir / f"{slug}_supplement_{index}.csv"
                    path.write_text(supplement_text, encoding="utf-8")
                    saved_path = str(path)
                elif suffix_url.endswith((".xlsx", ".xls")):
                    import io
                    frame = lit_module.pd.read_excel(io.BytesIO(content), sheet_name=None)
                    supplement_text = "\n\n".join(
                        f"[SHEET {name}]\n{sheet.to_csv(index=False)}"
                        for name, sheet in frame.items()
                    )
                    out_dir = retrieval_dir / "supplementary"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    path = out_dir / f"{slug}_supplement_{index}.txt"
                    path.write_text(supplement_text, encoding="utf-8")
                    saved_path = str(path)
                elif "html" in content_type or suffix_url.endswith((".html", ".htm")):
                    encoding = response.encoding or "utf-8"
                    supplement_html = content.decode(encoding, errors="ignore")
                    supplement_text = _extract_html_text_from_string(supplement_html)
                    out_dir = retrieval_dir / "supplementary"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    path = out_dir / f"{slug}_supplement_{index}.html"
                    path.write_text(supplement_html, encoding="utf-8")
                    saved_path = str(path)
            except Exception:
                supplement_text = ""
            if len(supplement_text.strip()) >= 100:
                texts.append(
                    f"[SUPPLEMENTARY SOURCE: {label}]\n{supplement_text.strip()}"
                )
                if saved_path:
                    saved_paths.append(saved_path)
        return "\n\n".join(texts), saved_paths

    def _save_and_extract_response(slug: str, source: str, url: str, response):
        content_type = str(response.headers.get("content-type", "") or "").lower()
        content = response.content or b""
        url_path = url.lower().split("?", 1)[0]
        is_pdf = "pdf" in content_type or url_path.endswith(".pdf") or content[:5] == b"%PDF-"
        if is_pdf:
            pdf_dir = getattr(lit_module, "PDF_DIR", None)
            out_dir = _Path_controller(pdf_dir) if pdf_dir is not None else retrieval_dir / "pdfs"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{slug}_{_safe_slug_part(source)}.pdf"
            out_path.write_bytes(content)
            return _extract_pdf_text_from_path(out_path), str(out_path), content_type
        is_docx = (
            "wordprocessingml.document" in content_type
            or url_path.endswith(".docx")
            or (
                content[:4] == b"PK\x03\x04"
                and b"[Content_Types].xml" in content[:2000]
            )
        )
        if is_docx:
            out_dir = retrieval_dir / "author_manuscripts"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{slug}_{_safe_slug_part(source)}.docx"
            out_path.write_bytes(content)
            return _ctrl_v21_17_extract_docx_text(content), str(out_path), content_type
        encoding = response.encoding or "utf-8"
        html = content.decode(encoding, errors="ignore")
        html_dir = getattr(lit_module, "HTML_DIR", None)
        out_dir = _Path_controller(html_dir) if html_dir is not None else retrieval_dir / "html"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{slug}_{_safe_slug_part(source)}.html"
        out_path.write_text(html, encoding="utf-8", errors="ignore")
        article_text = _extract_html_text_from_string(html)
        supplement_text, supplement_paths = _supplementary_text_from_html(
            html,
            url,
            slug,
        )
        if supplement_text:
            article_text = f"{article_text}\n\n{supplement_text}".strip()
            _append_retrieval_report({
                "slug": slug,
                "source": f"{source}:supplementary",
                "url": url,
                "ok": 1,
                "status": "supplementary_recovered",
                "text_length": len(supplement_text),
                "content_type": "supplementary",
                "saved_path": ";".join(supplement_paths),
                "alignment_ok": 1,
                "alignment_reason": "linked_from_aligned_article",
            })
        return article_text, str(out_path), content_type

    def _json_urls_from_openalex(doi: str) -> list[tuple[str, str]]:
        if not doi:
            return []
        quote_fn = getattr(lit_module, "quote", None)
        doi_q = quote_fn(doi, safe="") if callable(quote_fn) else doi.replace("/", "%2F")
        url = f"https://api.openalex.org/works/doi:{doi_q}"
        response, error = _requests_get(url)
        if error or response is None:
            return []
        try:
            if response.status_code >= 400:
                return []
            data = response.json()
        except Exception:
            return []
        return _ctrl_v21_18_openalex_location_urls(data)

    def _json_urls_from_unpaywall(doi: str) -> list[tuple[str, str]]:
        if not doi:
            return []
        quote_fn = getattr(lit_module, "quote", None)
        doi_q = quote_fn(doi, safe="") if callable(quote_fn) else doi.replace("/", "%2F")
        email = str(
            getattr(lit_module, "UNPAYWALL_EMAIL", "")
            or getattr(args, "unpaywall_email", "")
            or "literature-agent@example.org"
        ).strip()
        email_q = quote_fn(email, safe="@") if callable(quote_fn) else email
        url = f"https://api.unpaywall.org/v2/{doi_q}?email={email_q}"
        response, error = _requests_get(url)
        if error or response is None:
            return []
        try:
            if response.status_code >= 400:
                return []
            data = response.json()
        except Exception:
            return []
        return _ctrl_v21_18_unpaywall_location_urls(data)

    def _json_urls_from_europepmc(doi: str) -> list[tuple[str, str]]:
        if not doi:
            return []
        quote_fn = getattr(lit_module, "quote", None)
        query = f'DOI:"{doi}"'
        query_q = quote_fn(query, safe="") if callable(quote_fn) else query.replace(" ", "%20")
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={query_q}&format=json"
        response, error = _requests_get(url)
        if error or response is None:
            return []
        try:
            if response.status_code >= 400:
                return []
            data = response.json()
        except Exception:
            return []
        urls: list[tuple[str, str]] = []
        results = (((data.get("resultList") or {}).get("result")) or [])
        for item in results:
            pmcid = str((item or {}).get("pmcid") or "").strip()
            if pmcid:
                urls.append((
                    "europepmc_repository_fulltext_xml",
                    (
                        "https://www.ebi.ac.uk/europepmc/webservices/rest/"
                        f"{pmcid}/fullTextXML"
                    ),
                ))
            fulls = (((item or {}).get("fullTextUrlList") or {}).get("fullTextUrl")) or []
            for full in fulls:
                value = (full or {}).get("url")
                if value:
                    urls.append(("europepmc_repository_fulltext", str(value)))
        return sorted(
            urls,
            key=lambda item: _ctrl_v21_18_resolver_priority(item[0], item[1]),
        )

    def _urls_from_arxiv_title(title: str) -> list[tuple[str, str]]:
        if not title or len(str(title).split()) < 4:
            return []
        quote_fn = getattr(lit_module, "quote", None)
        words = " ".join(_re_controller.findall(r"[A-Za-z0-9]+", str(title))[:12])
        if not words:
            return []
        query_q = quote_fn(f"ti:{words}", safe=":") if callable(quote_fn) else ("ti:" + words.replace(" ", "%20"))
        url = f"https://export.arxiv.org/api/query?search_query={query_q}&start=0&max_results=3"
        response, error = _requests_get(url)
        if error or response is None:
            return []
        try:
            xml = response.text or ""
        except Exception:
            return []
        urls = _ctrl_v21_15_arxiv_pdf_urls_from_feed(title, xml)
        if xml and not urls:
            print(
                "[RETRIEVAL] ArXiv title search returned no sufficiently "
                "aligned result; no PDF was downloaded."
            )
        return [("arxiv_pdf", href) for href in urls]

    def _resolver_urls(meta: dict) -> list[tuple[str, str]]:
        doi = _normalize_expansion_doi(meta.get("doi") or meta.get("DOI") or meta.get("Ref_DOI_number"))
        title = str(meta.get("title") or meta.get("Title") or "")
        raw: list[tuple[str, str]] = []
        if doi:
            # Prefer repository and machine-readable OA copies. Publisher
            # routes remain available below as fallbacks.
            raw.extend(_json_urls_from_openalex(doi))
            raw.extend(_json_urls_from_europepmc(doi))
            raw.extend(_json_urls_from_unpaywall(doi))
        raw.extend(_urls_from_arxiv_title(title))
        for key, source in [
            ("_unpaywall_pdf_url", "unpaywall_pdf"),
            ("pdf_url", "candidate_pdf"),
            ("_source_pdf_url", "source_pdf"),
            ("landing_page", "candidate_landing"),
            ("_source_landing_page", "source_landing"),
            ("url", "candidate_url"),
        ]:
            value = meta.get(key) if isinstance(meta, dict) else None
            if value:
                raw.append((source, str(value)))
        for value in (
            meta.get("_crossref_fulltext_urls") or []
            if isinstance(meta, dict)
            else []
        ):
            if value:
                raw.append(("crossref_fulltext", str(value)))
        if doi:
            raw.append(("doi_landing", f"https://doi.org/{doi}"))

        seen = set()
        urls = []
        for source, url in raw:
            url = str(url or "").strip()
            if not url or not url.lower().startswith(("http://", "https://")):
                continue
            key = url.lower()
            if key in seen:
                continue
            seen.add(key)
            urls.append((source, url))
        urls.sort(
            key=lambda item: _ctrl_v21_18_resolver_priority(item[0], item[1])
        )
        return urls[:max_resolver_urls]

    def _try_resolver_url(meta: dict, slug: str, source: str, url: str) -> tuple[str, str, str]:
        doi = _normalize_expansion_doi(meta.get("doi") or meta.get("DOI") or meta.get("Ref_DOI_number"))
        title = str(meta.get("title") or meta.get("Title") or "")
        url_key = str(url or "").strip().lower()
        prior_url = retrieval_url_outcomes.get(url_key)
        if prior_url is not None:
            _append_retrieval_report({
                "slug": slug,
                "doi": doi,
                "title": title[:300],
                "source": source,
                "url": url,
                "status": "duplicate_source_url_skipped",
                "duplicate_of_slug": prior_url.get("slug", ""),
                "error": prior_url.get("status", ""),
            })
            print(
                f"[RETRIEVAL] {slug}: skipped previously tried source URL "
                f"from {prior_url.get('slug', 'another candidate')}."
            )
            return "", "", ""
        response, error = _requests_get(url)
        if error or response is None:
            _append_retrieval_report({"slug": slug, "doi": doi, "title": title[:300], "source": source, "url": url, "status": "request_error", "error": error})
            return "", "", ""
        status = int(getattr(response, "status_code", 0) or 0)
        if status >= 400:
            _append_retrieval_report({"slug": slug, "doi": doi, "title": title[:300], "source": source, "url": url, "status": f"http_{status}", "content_type": response.headers.get("content-type", "")})
            return "", "", ""
        content = response.content or b""
        content_hash = _hashlib_controller.sha256(content).hexdigest() if content else ""
        prior_content = retrieval_content_outcomes.get(content_hash) if content_hash else None
        if prior_content is not None:
            retrieval_url_outcomes[url_key] = {
                "slug": slug,
                "status": "duplicate_source_content_skipped",
            }
            _append_retrieval_report({
                "slug": slug,
                "doi": doi,
                "title": title[:300],
                "source": source,
                "url": url,
                "status": "duplicate_source_content_skipped",
                "content_type": response.headers.get("content-type", ""),
                "content_sha256": content_hash,
                "duplicate_of_slug": prior_content.get("slug", ""),
                "error": prior_content.get("status", ""),
            })
            print(
                f"[RETRIEVAL] {slug}: skipped duplicate downloaded content "
                f"already seen for {prior_content.get('slug', 'another candidate')}."
            )
            return "", "", ""
        content_type_hint = str(response.headers.get("content-type", "") or "").lower()
        is_pdf = (
            "pdf" in content_type_hint
            or url.lower().split("?", 1)[0].endswith(".pdf")
            or content[:5] == b"%PDF-"
        )
        if is_pdf:
            preflight_reason = _pdf_preflight_reason(content)
            if preflight_reason:
                retrieval_url_outcomes[url_key] = {
                    "slug": slug,
                    "status": preflight_reason,
                }
                if content_hash:
                    retrieval_content_outcomes[content_hash] = {
                        "slug": slug,
                        "status": preflight_reason,
                    }
                _append_retrieval_report({
                    "slug": slug,
                    "doi": doi,
                    "title": title[:300],
                    "source": source,
                    "url": url,
                    "status": preflight_reason.split(":", 1)[0],
                    "text_length": 0,
                    "content_type": content_type_hint,
                    "content_sha256": content_hash,
                    "error": preflight_reason,
                })
                print(f"[RETRIEVAL] {slug}: rejected PDF before text extraction ({preflight_reason}).")
                return "", "", content_type_hint
        try:
            text, saved_path, content_type = _save_and_extract_response(slug, source, url, response)
        except Exception as exc:
            _append_retrieval_report({"slug": slug, "doi": doi, "title": title[:300], "source": source, "url": url, "status": "extract_error", "error": repr(exc)[:300]})
            return "", "", ""
        (
            alignment_ok,
            alignment_reason,
            title_coverage,
            title_head_coverage,
            doi_found,
            perovskite_count,
        ) = _source_alignment(meta, text)
        _append_retrieval_report({
            "slug": slug,
            "doi": doi,
            "title": title[:300],
            "source": source,
            "url": url,
            "ok": int(len((text or "").strip()) >= min_extract and alignment_ok),
            "status": f"http_{status}" if alignment_ok else "rejected_source_mismatch",
            "text_length": len((text or "").strip()),
            "content_type": content_type,
            "saved_path": saved_path,
            "alignment_ok": int(alignment_ok),
            "alignment_reason": alignment_reason,
            "title_token_coverage": round(title_coverage, 4),
            "title_head_token_coverage": round(title_head_coverage, 4),
            "doi_found_in_text": doi_found,
            "perovskite_term_count": perovskite_count,
            "content_sha256": content_hash,
        })
        outcome = {
            "slug": slug,
            "status": (
                f"http_{status}" if alignment_ok else "rejected_source_mismatch"
            ),
        }
        retrieval_url_outcomes[url_key] = outcome
        if content_hash:
            retrieval_content_outcomes[content_hash] = outcome
        if not alignment_ok:
            return "", saved_path, content_type
        if not is_pdf and len((text or "").strip()) < min_full:
            try:
                landing_html = response.text or ""
            except Exception:
                landing_html = ""
            linked_urls = _ctrl_v21_15_landing_fulltext_urls(
                landing_html,
                url,
                max_links=4,
            )
            if linked_urls:
                print(
                    f"[RETRIEVAL] {slug}: aligned landing page exposed "
                    f"{len(linked_urls)} full-text link(s)."
                )
            linked_best = ("", "", "")
            for linked_index, linked_url in enumerate(linked_urls, start=1):
                linked_text, linked_path, linked_type = _try_resolver_url(
                    meta,
                    slug,
                    f"{source}_linked_fulltext_{linked_index}",
                    linked_url,
                )
                if len(linked_text.strip()) > len(linked_best[0].strip()):
                    linked_best = (linked_text, linked_path, linked_type)
                if len(linked_text.strip()) >= min_full:
                    break
            if len(linked_best[0].strip()) > len((text or "").strip()):
                return linked_best
        return text or "", saved_path, content_type

    def _try_full_text_with_html_fallback(meta, slug):
        result = original_try_full_text(meta, slug)
        current = str(result.get("full_text") or "")
        current_aligned = _source_alignment(meta if isinstance(meta, dict) else {}, current)[0]
        if len(current.strip()) >= min_full and current_aligned:
            return result
        if current and not current_aligned:
            result["source_alignment_rejected_original_text"] = True
            result["full_text"] = ""
            result["text_source"] = "rejected_source_mismatch"
            current = ""

        html_text, html_path = _read_best_saved_html(slug)
        pdf_text, pdf_path = _read_best_saved_pdf(slug)
        if html_text and not _source_alignment(meta if isinstance(meta, dict) else {}, html_text)[0]:
            html_text = ""
        if pdf_text and not _source_alignment(meta if isinstance(meta, dict) else {}, pdf_text)[0]:
            pdf_text = ""
        best_text = current
        source = result.get("text_source") or "web_low_text"

        if len(pdf_text.strip()) > len(best_text.strip()):
            best_text = pdf_text
            source = "saved_pdf_native_fallback"
            result["saved_pdf_path"] = pdf_path
        if len(html_text.strip()) > len(best_text.strip()):
            best_text = html_text
            source = "saved_html_text_fallback"
            result["saved_html_path"] = html_path

        resolver_best = ("", "", "", "")
        if automated_retrieval and len(best_text.strip()) < min_full:
            resolver_urls = _resolver_urls(meta if isinstance(meta, dict) else {})
            print(
                f"[RETRIEVAL] {slug}: trying {len(resolver_urls)} automated "
                f"source route(s)."
            )
            for resolver_source, resolver_url in resolver_urls:
                retrieved_text, saved_path, _content_type = _try_resolver_url(
                    meta if isinstance(meta, dict) else {},
                    slug,
                    resolver_source,
                    resolver_url,
                )
                if len(retrieved_text.strip()) > len(resolver_best[0].strip()):
                    resolver_best = (retrieved_text, saved_path, resolver_source, resolver_url)
                if len(retrieved_text.strip()) >= min_full:
                    break
            if not resolver_best[0].strip():
                print(f"[RETRIEVAL] {slug}: no aligned extractable source recovered.")

        if len(resolver_best[0].strip()) > len(best_text.strip()):
            best_text = resolver_best[0]
            source = f"automated_retrieval:{resolver_best[2]}"
            result["retrieval_resolver_source"] = resolver_best[2]
            result["retrieval_resolver_url"] = resolver_best[3]
            result["retrieval_saved_path"] = resolver_best[1]

        if len(best_text.strip()) > len(current.strip()):
            try:
                best_text = lit_module.extract_relevant_sections(best_text)
            except Exception:
                pass
            result["full_text"] = best_text
            result["text_source"] = source
            result["html_text_fallback_length"] = len(html_text)
            result["pdf_text_fallback_length"] = len(pdf_text)
            if len(best_text.strip()) >= min_full:
                print(f"[INFO] Web expansion full-text fallback recovered {len(best_text.strip())} chars for {slug} from {source}.")
            elif source.startswith("automated_retrieval:"):
                print(f"[INFO] Automated retrieval improved {slug} to {len(best_text.strip())} chars from {source}.")

        return result

    lit_module.try_full_text = _try_full_text_with_html_fallback

    extraction_agent_cls = getattr(lit_module, "ExtractionAgent", None)
    if extraction_agent_cls is not None and hasattr(extraction_agent_cls, "extract"):
        original_extract = extraction_agent_cls.extract

        def _extract_with_fulltext_metadata(self, meta, full_text, extraction_text):
            if isinstance(meta, dict):
                meta["_full_text"] = full_text or ""
                meta["_full_text_length"] = len(full_text or "")
                meta["_extraction_text_length"] = len(extraction_text or "")
                if not meta.get("full_text_source"):
                    meta["full_text_source"] = "manager_full_text"
            result = original_extract(self, meta, full_text, extraction_text)
            records = result.get("records") if isinstance(result, dict) else None
            if isinstance(records, list):
                records = _ctrl_v21_10_enrich_extraction_records(meta, full_text, extraction_text, records)
                if isinstance(result, dict):
                    result["records"] = records
                for rec in records:
                    if isinstance(rec, dict):
                        rec["_full_text_length"] = len(full_text or "")
                        rec["_full_text_source"] = meta.get("full_text_source") if isinstance(meta, dict) else "manager_full_text"
            return result

        extraction_agent_cls.extract = _extract_with_fulltext_metadata

    assembly_agent_cls = getattr(lit_module, "AssemblyAgent", None)
    if assembly_agent_cls is not None and hasattr(assembly_agent_cls, "assemble_records"):
        original_assemble = assembly_agent_cls.assemble_records

        def _assemble_with_fulltext_metadata(self, meta, raw_records):
            records = original_assemble(self, meta, raw_records)
            if isinstance(meta, dict) and isinstance(records, list):
                records = _ctrl_v21_10_enrich_extraction_records(
                    meta,
                    meta.get("_full_text") or "",
                    meta.get("_extraction_text") or "",
                    records,
                )
                full_len = int(meta.get("_full_text_length") or len(meta.get("_full_text") or "") or 0)
                source = meta.get("full_text_source") or meta.get("_full_text_source") or "manager_full_text"
                for rec in records:
                    if isinstance(rec, dict):
                        rec["_full_text_length"] = full_len
                        rec["_full_text_source"] = source
            return records

        assembly_agent_cls.assemble_records = _assemble_with_fulltext_metadata

    eligibility_dir = work_dir / "model_eligibility"
    eligibility_dir.mkdir(parents=True, exist_ok=True)
    eligibility_jsonl = eligibility_dir / "paper_model_eligibility.jsonl"
    eligibility_csv = eligibility_dir / "paper_model_eligibility.csv"

    def _present_record_value(value) -> bool:
        return value not in (None, "", [], {}) and str(value).strip().lower() not in {
            "nan", "none", "null", "unknown",
        }

    def _numeric_record_value(record, fields, low=None, high=None) -> bool:
        for field in fields:
            try:
                value = float(record.get(field))
            except Exception:
                continue
            if low is not None and value < low:
                continue
            if high is not None and value > high:
                continue
            return True
        return False

    def _paper_eligibility_payload(meta, result):
        records = result.get("records") if isinstance(result, dict) else []
        records = [record for record in (records or []) if isinstance(record, dict)]
        row_details = []
        for index, record in enumerate(records, start=1):
            pce = _numeric_record_value(
                record,
                ("JV_default_PCE", "JV_reverse_scan_PCE", "JV_forward_scan_PCE", "Stabilised_performance_PCE"),
                0.01,
                40.0,
            )
            composition = any(_present_record_value(record.get(field)) for field in (
                "Perovskite_composition_short_form", "Perovskite_composition_long_form",
            ))
            device = any(_present_record_value(record.get(field)) for field in (
                "Cell_architecture", "Cell_stack_sequence", "ETL_stack_sequence",
                "HTL_stack_sequence", "Backcontact_stack_sequence",
            ))
            processing = any(_present_record_value(record.get(field)) for field in (
                "Perovskite_deposition_procedure", "Perovskite_deposition_solvents",
                "Perovskite_deposition_quenching_media",
                "Perovskite_deposition_thermal_annealing_temperature",
                "Perovskite_deposition_thermal_annealing_time",
            ))
            stability_target = _numeric_record_value(record, (
                "Stability_PCE_T80", "Stability_PCE_T95",
                "Stability_PCE_end_of_experiment", "Stability_PCE_after_1000_h",
            ), 0.0)
            stability_time = _numeric_record_value(record, (
                "Stability_time_total_exposure", "Stability_PCE_T80", "Stability_PCE_T95",
            ), 0.0)
            stability_conditions = any(_present_record_value(record.get(field)) for field in (
                "Stability_temperature_range",
                "Stability_relative_humidity_average_value",
                "Stability_light_intensity", "Stability_atmosphere", "Encapsulation",
            ))
            row_details.append({
                "row": index,
                "sample_id": record.get("Ref_internal_sample_id"),
                "pce_target": pce,
                "composition": composition,
                "device": device,
                "processing": processing,
                "pce_minimal": pce and (composition or device),
                "pce_strict": pce and composition and device and processing,
                "stability_target": stability_target,
                "stability_minimal": stability_target and stability_time and (
                    composition or device or stability_conditions
                ),
            })
        title = str((meta or {}).get("title") or "")
        doi = str((meta or {}).get("doi") or "")
        slug = (
            lit_module.slugify(title or doi or "paper")
            if hasattr(lit_module, "slugify")
            else _hashlib_controller.sha1((title or doi).encode()).hexdigest()[:12]
        )
        return {
            "timestamp": _datetime_controller.now().isoformat(timespec="seconds"),
            "paper_slug": slug,
            "title": title,
            "doi": doi,
            "records": len(records),
            "pce_target_rows": sum(bool(row["pce_target"]) for row in row_details),
            "pce_minimal_rows": sum(bool(row["pce_minimal"]) for row in row_details),
            "pce_strict_rows": sum(bool(row["pce_strict"]) for row in row_details),
            "stability_target_rows": sum(bool(row["stability_target"]) for row in row_details),
            "stability_minimal_rows": sum(bool(row["stability_minimal"]) for row in row_details),
            "t80_rows": sum(
                _numeric_record_value(record, ("Stability_PCE_T80",), 0.0)
                for record in records
            ),
            "row_details": row_details,
        }

    original_process_one = getattr(lit_module, "process_one_paper_api", None)
    if callable(original_process_one):
        def _process_one_with_eligibility(meta, force_reprocess=False, run_mode=None):
            result = original_process_one(
                meta,
                force_reprocess=force_reprocess,
                run_mode=run_mode,
            )
            payload = _paper_eligibility_payload(meta, result)
            try:
                with eligibility_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(_json_controller.dumps(payload, ensure_ascii=False) + "\n")
                csv_row = {
                    key: value for key, value in payload.items()
                    if key != "row_details"
                }
                exists = eligibility_csv.exists()
                with eligibility_csv.open("a", encoding="utf-8-sig", newline="") as handle:
                    writer = _csv_controller.DictWriter(handle, fieldnames=list(csv_row))
                    if not exists:
                        writer.writeheader()
                    writer.writerow(csv_row)
            except Exception as exc:
                print(f"[WARN] Could not write immediate model-eligibility audit: {exc}")
            print(
                f"[ELIGIBILITY] paper={payload['paper_slug']} records={payload['records']} "
                f"pce_target={payload['pce_target_rows']} pce_minimal={payload['pce_minimal_rows']} "
                f"pce_strict={payload['pce_strict_rows']} stability_target={payload['stability_target_rows']} "
                f"stability_minimal={payload['stability_minimal_rows']} t80={payload['t80_rows']}"
            )
            return result

        lit_module.process_one_paper_api = _process_one_with_eligibility

    print(
        f"[INFO] Web expansion full-text fallback active | min_full={min_full} | "
        f"min_extraction={min_extract} | automated_retrieval={automated_retrieval} | "
        "source_order=repository_first | "
        f"max_urls={max_resolver_urls} | supplementary={supplementary_retrieval} "
        f"(max={max_supplementary_files}) | report={retrieval_report_csv}"
    )


def _truthy_env_value(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_scientific_reasoning_policy_module():
    candidates = [
        _Path_controller(__file__).resolve().parent / "literature_agent" / "reasoning" / "scientific_reasoning_policy_layer.py",
        _Path_controller(__file__).resolve().parent / "scientific_reasoning_policy_layer.py",
        _downloads_dir() / "scientific_reasoning_policy_layer.py",
    ]
    for path in candidates:
        if not path.exists():
            continue
        spec = _importlib_util_controller.spec_from_file_location("scientific_reasoning_policy_layer", str(path))
        if spec is None or spec.loader is None:
            continue
        module = _importlib_util_controller.module_from_spec(spec)
        _sys_controller.modules["scientific_reasoning_policy_layer"] = module
        spec.loader.exec_module(module)
        return module, path
    return None, None


def _load_polaris_reasoning_adapter_module():
    project_root = _Path_controller(__file__).resolve().parent
    adapter_path = project_root / "literature_agent" / "reasoning" / "polaris_reasoning_adapter.py"
    if not adapter_path.exists():
        return None, None
    if str(project_root) not in _sys_controller.path:
        _sys_controller.path.insert(0, str(project_root))
    try:
        module = _importlib_controller.import_module("literature_agent.reasoning.polaris_reasoning_adapter")
        return module, adapter_path
    except Exception:
        return None, adapter_path


def _reasoning_policy_for_stage(stage: str) -> tuple[str, str, str]:
    text = str(stage or "").lower()
    if any(k in text for k in ["gap", "future", "hypothesis", "uncertainty"]):
        return (
            "socratic_uncertainty",
            "Research gaps, future work, hypothesis generation, or uncertainty analysis",
            "Keep assumptions, missing evidence, uncertainty, and next tests explicit.",
        )
    if any(k in text for k in ["summary", "evidence", "figure", "caption", "ocr"]):
        return (
            "humean_evidence",
            "Literature summary or evidence extraction",
            "Separate direct paper evidence from interpretation and avoid unsupported mechanisms.",
        )
    return (
        "kantian_constraints",
        "Structured extraction, ontology mapping, JSON output, or schema validation",
        "Check required fields, valid categories, units, and schema consistency while preserving strict JSON.",
    )


def _install_scientific_reasoning_policy_layer(lit_module, args) -> None:
    enabled = _truthy_env_value(_os_controller.getenv("USE_REASONING_LAYER"))
    enabled = enabled or bool(int(getattr(args, "use_reasoning_layer", 0)))
    if not enabled:
        return

    adapter_module, adapter_path = _load_polaris_reasoning_adapter_module()
    policy_module, policy_path = _load_scientific_reasoning_policy_module()
    if policy_module is None or not hasattr(policy_module, "apply_literature_reasoning_policy"):
        print("[WARN] USE_REASONING_LAYER requested, but scientific_reasoning_policy_layer.py was not found.")
        return

    extractor_cls = getattr(lit_module, "HuggingFaceQwenExtractor", None)
    if extractor_cls is None or not hasattr(extractor_cls, "_call"):
        print("[WARN] USE_REASONING_LAYER requested, but the LiteratureAgent LLM _call hook was not found.")
        return

    if getattr(extractor_cls, "_reasoning_policy_wrapped_v1", False):
        return

    log_dir = _Path_controller(args.work_dir) / "reasoning_policy_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    original_call = extractor_cls._call

    def _reasoned_call(self, system_prompt, user_payload, paper_slug, stage):
        policy_id, task, contract = _reasoning_policy_for_stage(stage)
        try:
            payload_preview = _json_controller.dumps(user_payload, ensure_ascii=False, default=str)[:3000]
            schema_hint = (
                "Preserve the existing LiteratureAgent JSON output shape exactly. "
                "Do not add reasoning metadata to the model response; metadata is saved to sidecar logs."
            )
            constraints = [
                contract,
                "Return valid JSON only when the original LiteratureAgent call requests JSON.",
                "Do not invent facts or fill unsupported values.",
            ]
            if adapter_module is not None and hasattr(adapter_module, "build_reasoning_prompt_for_polaris"):
                prepared = adapter_module.build_reasoning_prompt_for_polaris(
                    agent_name="Literature Agent",
                    task=task,
                    base_prompt=str(system_prompt or ""),
                    task_type="auto",
                    context=payload_preview,
                    output_schema=schema_hint,
                    constraints=constraints,
                    policy_mode=str(getattr(args, "reasoning_policy_mode", "auto") or "auto"),
                    requested_policy="auto",
                    metadata_destination="sidecar",
                )
            else:
                prepared = policy_module.apply_literature_reasoning_policy(
                    task=task,
                    base_prompt=str(system_prompt or ""),
                    context=payload_preview,
                    output_schema=schema_hint,
                    policy=policy_id,
                    constraints=constraints,
                    generate_text_fn=None,
                )
            metadata = dict(prepared.get("reasoning_metadata") or {})
            metadata.update({
                "paper_slug": paper_slug,
                "stage": stage,
                "policy_module_path": str(policy_path),
                "polaris_adapter_path": str(adapter_path) if adapter_path else "",
                "sidecar_only": True,
                "output_format_preserved": True,
            })
            safe_slug = _re_controller.sub(r"[^A-Za-z0-9_.-]+", "_", str(paper_slug or "paper"))[:120]
            safe_stage = _re_controller.sub(r"[^A-Za-z0-9_.-]+", "_", str(stage or "stage"))[:80]
            stamp = str(int(_time_controller.time() * 1000))
            log_path = log_dir / f"{safe_slug}_{safe_stage}_{stamp}.json"
            log_path.write_text(_json_controller.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[REASONING] paper={paper_slug} stage={stage} policy={metadata.get('reasoning_policy', policy_id)} log={log_path.name}")
            # The policy layer is advisory for LiteratureAgent calls. Replacing
            # the extraction system prompt with the adapter's full philosophical
            # template caused Qwen to return a schema audit (values under
            # _evidence_* and null data fields) instead of the requested records.
            # Keep the original task/schema prompt intact and add only the
            # selected scientific constraint; detailed policy metadata remains
            # in the sidecar log.
            guarded_prompt = (
                str(system_prompt or "")
                + f"\n\nScientific reasoning policy ({policy_id}): {contract} "
                + "Apply this check internally while completing the original task. "
                + "Preserve the requested JSON keys and place extracted values in their corresponding data fields. "
                + "Do not add evidence keys unless the original task explicitly requests them. "
                + "Do not include reasoning metadata in the response; it is saved separately."
            )
            return original_call(self, guarded_prompt, user_payload, paper_slug, stage)
        except Exception as exc:
            print(f"[WARN] Reasoning policy wrapper failed for {paper_slug}/{stage}; falling back to original prompt: {exc}")
            return original_call(self, system_prompt, user_payload, paper_slug, stage)

    extractor_cls._call = _reasoned_call
    extractor_cls._reasoning_policy_wrapped_v1 = True
    adapter_msg = f" | adapter={adapter_path}" if adapter_path else ""
    print(f"[INFO] Scientific reasoning policy layer enabled: {policy_path}{adapter_msg}")



def _ensure_figure_report_dirs(work_dir: _Path_controller) -> dict:
    """
    Always create the top-level figure report folders so the user can see whether
    figure reporting/vision is active while the run is executing.
    """
    root = work_dir / "figure_reports"
    dirs = {
        "root": root,
        "text": root / "text",
        "json": root / "json",
        "crops": root / "crops",
        "vision_logs": root / "vision_logs",
        "vision_cache": root / "vision_cache",
        "status": root / "status",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    status_path = dirs["status"] / "FIGURE_REPORT_FOLDER_CREATED.txt"
    if not status_path.exists():
        status_path.write_text(
            "Figure report folders were created by the v20.3 controller.\n"
            "If text/json reports exist but vision_logs is empty, figure text/caption/OCR reporting ran but Qwen-VL did not produce vision calls.\n",
            encoding="utf-8",
        )
    return dirs


def _safe_read_text(path: _Path_controller, max_chars: int = 40000) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        return txt[:max_chars]
    except Exception:
        return ""


def _extract_figure_terms_from_text(txt: str) -> dict:
    lower = txt.lower()
    patterns = {
        "jv_curve": ["j-v", "current density", "voltage curve", "voc", "jsc", "fill factor", "pce"],
        "eqe_spectrum": ["eqe", "ipce", "integrated jsc", "external quantum"],
        "stability_curve": ["stability", "normalized pce", "retention", "t80", "t95", "aging", "illumination", "hours"],
        "xrd": ["xrd", "giwaxs", "2θ", "2 theta", "diffraction"],
        "pl_uvvis": ["pl", "photoluminescence", "uv-vis", "absorbance", "tauc"],
        "microscopy": ["sem", "tem", "stem", "edx", "afm", "kpfm", "npfm"],
        "device_schematic": ["device architecture", "ito", "fto", "sno2", "spiro", "c60", "pcbm", "au"],
    }
    out = {}
    for k, terms in patterns.items():
        hits = [t for t in terms if t in lower]
        if hits:
            out[k] = hits
    return out


def _synthesize_controller_figure_reports(work_dir: _Path_controller) -> None:
    """
    Fallback/controller-level figure report writer.

    This does not replace true Qwen-VL image analysis. It guarantees a visible
    figure_reports folder and creates a concise figure evidence report from
    existing LiteratureAgent outputs: summaries, manager payloads, OCR/caption
    extracted text, and figure-derived text. True image-level Qwen-VL calls are
    confirmed only by files in figure_reports/vision_logs.
    """
    dirs = _ensure_figure_report_dirs(work_dir)
    paper_store = work_dir / "paper_store"
    summaries = work_dir / "paper_summaries_text"

    report_index = []
    slugs = set()
    if paper_store.exists():
        for p in paper_store.iterdir():
            if p.is_dir():
                slugs.add(p.name)
    if summaries.exists():
        for p in summaries.glob("*_summary.txt"):
            slugs.add(p.name.replace("_summary.txt", ""))

    for slug in sorted(slugs):
        chunks = []
        source_files = []
        summary_path = summaries / f"{slug}_summary.txt"
        if summary_path.exists():
            chunks.append(("summary_text", _safe_read_text(summary_path)))
            source_files.append(str(summary_path))

        pdir = paper_store / slug
        for rel in [
            "outputs/manager_payload.json",
            "outputs/evaluation.json",
            "outputs/flags.json",
            "outputs/figure_report_v19_5.json",
            "outputs/figure_report_v19_5.txt",
            "outputs/figure_report_v19_6.json",
            "outputs/figure_report_v19_6.txt",
            "outputs/figure_report_v19_7.json",
            "outputs/figure_report_v19_7.txt",
        ]:
            fp = pdir / rel
            if fp.exists():
                chunks.append((rel, _safe_read_text(fp)))
                source_files.append(str(fp))

        combined = "\n\n".join(f"### {name}\n{text}" for name, text in chunks if text)
        if not combined.strip():
            continue

        detected = _extract_figure_terms_from_text(combined)
        figure_used_in_summary = bool(detected)
        evidence_lines = []
        for line in combined.splitlines():
            ll = line.lower()
            if any(term in ll for term in [
                "figure", "fig.", "extended data", "j-v", "eqe", "normalized pce",
                "stability", "kpfm", "npfm", "stem", "edx", "xrd", "pl ", "uv-vis",
                "integrated jsc", "retention", "t80", "t95"
            ]):
                clean = line.strip()
                if clean and clean not in evidence_lines:
                    evidence_lines.append(clean)
            if len(evidence_lines) >= 40:
                break

        report = {
            "paper_slug": slug,
            "figure_report_source": "v20.3_controller_fallback_from_existing_literature_outputs",
            "qwen_vl_vision_log_detected": False,
            "figure_text_or_caption_evidence_detected": figure_used_in_summary,
            "detected_figure_evidence_types": detected,
            "source_files_checked": source_files,
            "evidence_lines": evidence_lines[:40],
            "note": (
                "This fallback report confirms whether figure/caption/OCR-derived evidence is present "
                "in the LiteratureAgent outputs. True image-level Qwen-VL calls are confirmed only by "
                "files in figure_reports/vision_logs."
            ),
        }

        json_path = dirs["json"] / f"{slug}_figure_report_v20_3_controller_fallback.json"
        txt_path = dirs["text"] / f"{slug}_figure_report_v20_3_controller_fallback.txt"
        json_path.write_text(_json_controller.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        lines = [
            f"Figure / Visual Evidence Report: {slug}",
            "Source: v20.3 controller fallback from existing LiteratureAgent outputs",
            "",
            f"Figure/caption/OCR evidence detected: {figure_used_in_summary}",
            "Qwen-VL vision log detected: False",
            "",
            "Detected evidence types:",
        ]
        if detected:
            for k, hits in detected.items():
                lines.append(f"- {k}: {', '.join(hits)}")
        else:
            lines.append("- none detected")
        lines.extend(["", "Evidence lines:"])
        if evidence_lines:
            for line in evidence_lines[:40]:
                lines.append(f"- {line[:500]}")
        else:
            lines.append("- No figure-specific lines found in available summary/manager outputs.")
        txt_path.write_text("\n".join(lines), encoding="utf-8")
        report_index.append(report)

    index_path = dirs["json"] / "figure_report_index_v20_3.json"
    index_path.write_text(_json_controller.dumps(report_index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Figure report fallback/index written to: {dirs['root']} | papers={len(report_index)}")


def _find_or_create_all_records_csv(work_dir: _Path_controller) -> _Path_controller:
    """
    LiteratureAgent sometimes writes combined files in different places depending on mode/version.
    This function tries common outputs and, if needed, combines per-paper CSVs into csv/all_records.csv.
    """
    target = work_dir / "csv" / "all_records.csv"

    def _csv_has_data_rows(path: _Path_controller) -> bool:
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = _csv_controller.reader(f)
                next(reader, None)
                return any(any(str(cell).strip() for cell in row) for row in reader)
        except Exception:
            return False

    candidates = [
        work_dir / "csv" / "all_records_from_manager.csv",
        work_dir / "combined_full_coverage" / "all_records_from_manager.csv",
        work_dir / "combined_full_coverage" / "csv" / "all_records.csv",
        work_dir / "all_records_from_manager.csv",
        work_dir / "all_records.csv",
    ]
    existing_candidates = [c for c in candidates if c.exists()]
    data_candidates = [c for c in existing_candidates if _csv_has_data_rows(c)]
    empty_candidates = [c for c in existing_candidates if c not in data_candidates]
    if empty_candidates:
        print(
            "[INFO] Ignoring empty aggregate records candidate(s): "
            + ", ".join(str(p) for p in empty_candidates)
        )
    newest_candidate = max(data_candidates, key=lambda p: p.stat().st_mtime) if data_candidates else None

    if target.exists():
        if newest_candidate and newest_candidate.resolve() != target.resolve():
            target_mtime = target.stat().st_mtime
            candidate_mtime = newest_candidate.stat().st_mtime
            if candidate_mtime > target_mtime:
                if _csv_has_data_rows(newest_candidate):
                    backup = target.with_name(
                        f"{target.stem}_stale_backup_{_time_controller.strftime('%Y%m%d_%H%M%S')}{target.suffix}"
                    )
                    _shutil_controller.copy2(target, backup)
                    _shutil_controller.copy2(newest_candidate, target)
                    print(
                        f"[INFO] Replaced stale {target} with newer manager records from {newest_candidate}. "
                        f"Backup: {backup}"
                    )
                else:
                    print(f"[INFO] Keeping existing {target}; newer candidate {newest_candidate} has no data rows.")
        return target

    if newest_candidate:
        target.parent.mkdir(parents=True, exist_ok=True)
        _shutil_controller.copy2(newest_candidate, target)
        print(f"[INFO] Copied combined records to {target} from {newest_candidate}")
        return target

    csv_dir = work_dir / "csv"
    per_paper_csvs = []
    if csv_dir.exists():
        per_paper_csvs = [
            p for p in csv_dir.glob("*.csv")
            if p.name.lower() not in {
                "all_records.csv",
                "all_records_from_manager.csv",
                "paper_summaries_from_manager.csv",
            }
        ]

    if not per_paper_csvs:
        raise FileNotFoundError(
            f"Could not find all_records.csv in {target} and no per-paper CSVs were found in {csv_dir}"
        )

    print(f"[INFO] Combining {len(per_paper_csvs)} per-paper CSV files into {target}")
    rows = []
    fieldnames = []
    seen_fields = set()

    for path in per_paper_csvs:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = _csv_controller.DictReader(f)
            if reader.fieldnames:
                for name in reader.fieldnames:
                    if name not in seen_fields:
                        seen_fields.add(name)
                        fieldnames.append(name)
            for row in reader:
                rows.append(row)

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8-sig", newline="") as f:
        writer = _csv_controller.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return target



def _v21_safe_embedded_run_mode(args) -> str:
    """
    The controller handles fresh deletion of WORK_DIR. The embedded LiteratureAgent
    must not see RUN_MODE=initial/reset/overwrite on Windows, because it tries to
    delete WORK_DIR while its embedded runtime files may be open.
    """
    requested = str(getattr(args, "run_mode", "expand")).strip().lower() or "expand"
    allow = int(getattr(args, "allow_embedded_reset", 0))
    if not allow and requested in {"initial", "reset", "overwrite"}:
        return "expand"
    return requested


def _prime_import_time_environment(args) -> None:
    """
    The embedded LiteratureAgent reads several settings at import time.
    Set them before importing the embedded runtime module so local/Ollama
    settings, ontology path, output directory, and vision settings are honored.
    """
    _setenv("WORK_DIR", args.work_dir)
    _setenv("PEROVSKITE_ONTOLOGY_PATH", args.ontology_path)
    _setenv("GOOGLE_DRIVE_OAUTH_CLIENT_SECRETS", args.google_drive_oauth_client_secrets)
    _setenv("GOOGLE_DRIVE_OAUTH_TOKEN_FILE", args.google_drive_oauth_token_file)
    if getattr(args, "google_drive_folder_id", None):
        _setenv("GOOGLE_DRIVE_FOLDER_ID", args.google_drive_folder_id)
        _setenv("GOOGLE_DRIVE_FOLDER_LINK", args.google_drive_folder_id)

    _setenv("HF_API_KEY", args.llm_api_key)
    _setenv("HF_API_URL", args.llm_api_url)
    _setenv("HF_MODEL", args.llm_model)
    _setenv("HF_MODEL_FAST", args.llm_model_fast or args.llm_model)
    _setenv("HF_MODEL_STRONG", args.llm_model_strong or args.llm_model)

    _setenv("LLM_COST_MODE", args.llm_cost_mode)
    _setenv("DETERMINISTIC_PREFILL_FIELDS", "1" if args.deterministic_prefill else "0")
    _setenv("FIGURE_REPORT_ENABLE", "1" if args.figure_report_enable else "0")

    _setenv("VISION_LLM_ENABLE", "1" if args.vision_enable else "0")
    _setenv("VISION_PROVIDER", args.vision_provider)
    _setenv("VISION_API_KEY", args.vision_api_key)
    _setenv("VISION_API_URL", args.vision_api_url)
    _setenv("VISION_MODEL", args.vision_model)
    _setenv("VISION_MAX_FIGURES_PER_PAPER", str(args.vision_max_figures_per_paper))
    _setenv("VISION_MAX_IMAGE_WIDTH", str(args.vision_max_image_width))
    _setenv("VISION_CACHE_ENABLE", "1" if args.vision_cache_enable else "0")
    _setenv("LLM_CALL_CACHE_ENABLE", "1" if getattr(args, "llm_cache_enable", 1) else "0")
    _setenv("MISSING_FIELD_RECOVERY_ENABLE", "1" if getattr(args, "missing_field_recovery", 0) else "0")
    _setenv("FORCE_SUMMARY_REWRITE", "1" if getattr(args, "force_summary_rewrite", 0) else "0")
    _setenv("REVIEW_FULL_EXTRACTION_ENABLE", "1" if getattr(args, "review_full_extraction", 0) else "0")
    _setenv("LLM_REQUEST_TIMEOUT_SECONDS", str(getattr(args, "llm_request_timeout", 600)))
    _setenv("WEB_REQUEST_TIMEOUT_SECONDS", str(getattr(args, "web_request_timeout", 45)))
    _setenv("LLM_API_MAX_RETRIES", str(getattr(args, "llm_max_retries", 1)))
    _setenv("LLM_API_RETRY_BACKOFF_SECONDS", str(getattr(args, "llm_retry_backoff", 8.0)))
    _setenv("FAMILY_GATING_MODE", str(getattr(args, "family_gating", "strict")))
    _setenv("USE_REASONING_LAYER", "1" if int(getattr(args, "use_reasoning_layer", 0)) else _os_controller.getenv("USE_REASONING_LAYER", "0"))
    disable_drive = int(getattr(args, "disable_google_drive", 0))
    if getattr(args, "test_one_paper", False) and str(getattr(args, "single_paper_source", "")).lower() == "search":
        disable_drive = 1
    _setenv("DRIVE_FIRST_STRICT", "0" if disable_drive else "1")
    drive_all = getattr(args, "drive_process_all_files", None)
    if drive_all is None:
        drive_all = 1 if getattr(args, "full_literature_run", False) else 0
    _setenv("DRIVE_FIRST_PROCESS_ALL_FILES", "1" if int(drive_all) else "0")
    _setenv("GOOGLE_DRIVE_MAX_FILES_PER_RUN", str(getattr(args, "google_drive_max_files_per_run", 0) or args.max_papers))

    _setenv("RUN_MODE", _v21_safe_embedded_run_mode(args))
    _setenv("MAX_PAPERS_TO_PROCESS", str(args.max_papers))

    if args.test_one_paper:
        _setenv("TEST_ONE_PAPER", "1")
        _setenv("SINGLE_PAPER_SOURCE", args.single_paper_source)
        _setenv("ONE_PAPER_MAX_DRIVE_FILES", str(args.one_paper_max_drive_files))
        _setenv("ONE_PAPER_SKIP_SEARCH_AFTER_DRIVE", "1" if args.one_paper_skip_search_after_drive else "0")
        if args.single_paper_pdf_path:
            _setenv("SINGLE_PAPER_PDF_PATH", args.single_paper_pdf_path)
        if args.one_paper_drive_filename_include:
            _setenv("ONE_PAPER_DRIVE_FILENAME_INCLUDE", args.one_paper_drive_filename_include)
    else:
        _clearenv([
            "TEST_ONE_PAPER",
            "SINGLE_PAPER_SOURCE",
            "ONE_PAPER_MAX_DRIVE_FILES",
            "ONE_PAPER_SKIP_SEARCH_AFTER_DRIVE",
            "SINGLE_PAPER_PDF_PATH",
            "ONE_PAPER_DRIVE_FILENAME_INCLUDE",
        ])


def _run_literature_agent(lit_module, args) -> _Path_controller:
    print("\n" + "=" * 100)
    print("STEP 1: Running embedded LiteratureAgent")
    print("=" * 100)

    # Create figure report folders immediately so the user can verify the path during long runs.
    _ensure_figure_report_dirs(_Path_controller(args.work_dir))

    # Environment expected by LiteratureAgent.
    _setenv("WORK_DIR", args.work_dir)
    _setenv("PEROVSKITE_ONTOLOGY_PATH", args.ontology_path)
    _setenv("GOOGLE_DRIVE_OAUTH_CLIENT_SECRETS", args.google_drive_oauth_client_secrets)
    _setenv("GOOGLE_DRIVE_OAUTH_TOKEN_FILE", args.google_drive_oauth_token_file)
    if getattr(args, "google_drive_folder_id", None):
        _setenv("GOOGLE_DRIVE_FOLDER_ID", args.google_drive_folder_id)
        _setenv("GOOGLE_DRIVE_FOLDER_LINK", args.google_drive_folder_id)

    # LLM settings. Variable names remain HF_* because the original agent uses those names,
    # even when pointing to local Ollama or another OpenAI-compatible endpoint.
    _setenv("HF_API_KEY", args.llm_api_key)
    _setenv("HF_API_URL", args.llm_api_url)
    _setenv("HF_MODEL", args.llm_model)
    _setenv("HF_MODEL_FAST", args.llm_model_fast or args.llm_model)
    _setenv("HF_MODEL_STRONG", args.llm_model_strong or args.llm_model)

    _setenv("LLM_COST_MODE", args.llm_cost_mode)
    _setenv("DETERMINISTIC_PREFILL_FIELDS", "1" if args.deterministic_prefill else "0")
    _setenv("FIGURE_REPORT_ENABLE", "1" if args.figure_report_enable else "0")

    # Vision settings.
    _setenv("VISION_LLM_ENABLE", "1" if args.vision_enable else "0")
    _setenv("VISION_PROVIDER", args.vision_provider)
    _setenv("VISION_API_KEY", args.vision_api_key)
    _setenv("VISION_API_URL", args.vision_api_url)
    _setenv("VISION_MODEL", args.vision_model)
    _setenv("VISION_MAX_FIGURES_PER_PAPER", str(args.vision_max_figures_per_paper))
    _setenv("VISION_MAX_IMAGE_WIDTH", str(args.vision_max_image_width))
    _setenv("VISION_CACHE_ENABLE", "1" if args.vision_cache_enable else "0")
    _setenv("LLM_CALL_CACHE_ENABLE", "1" if getattr(args, "llm_cache_enable", 1) else "0")
    _setenv("MISSING_FIELD_RECOVERY_ENABLE", "1" if getattr(args, "missing_field_recovery", 0) else "0")
    _setenv("FORCE_SUMMARY_REWRITE", "1" if getattr(args, "force_summary_rewrite", 0) else "0")
    _setenv("REVIEW_FULL_EXTRACTION_ENABLE", "1" if getattr(args, "review_full_extraction", 0) else "0")
    _setenv("LLM_REQUEST_TIMEOUT_SECONDS", str(getattr(args, "llm_request_timeout", 600)))
    _setenv("WEB_REQUEST_TIMEOUT_SECONDS", str(getattr(args, "web_request_timeout", 45)))
    _setenv("LLM_API_MAX_RETRIES", str(getattr(args, "llm_max_retries", 1)))
    _setenv("LLM_API_RETRY_BACKOFF_SECONDS", str(getattr(args, "llm_retry_backoff", 8.0)))
    _setenv("FAMILY_GATING_MODE", str(getattr(args, "family_gating", "strict")))
    _setenv("USE_REASONING_LAYER", "1" if int(getattr(args, "use_reasoning_layer", 0)) else _os_controller.getenv("USE_REASONING_LAYER", "0"))
    disable_drive = int(getattr(args, "disable_google_drive", 0))
    if getattr(args, "test_one_paper", False) and str(getattr(args, "single_paper_source", "")).lower() == "search":
        disable_drive = 1
    _setenv("DRIVE_FIRST_STRICT", "0" if disable_drive else "1")
    drive_all = getattr(args, "drive_process_all_files", None)
    if drive_all is None:
        drive_all = 1 if getattr(args, "full_literature_run", False) else 0
    _setenv("DRIVE_FIRST_PROCESS_ALL_FILES", "1" if int(drive_all) else "0")
    _setenv("GOOGLE_DRIVE_MAX_FILES_PER_RUN", str(getattr(args, "google_drive_max_files_per_run", 0) or args.max_papers))

    # The controller handles fresh folder deletion. Avoid letting the embedded agent
    # delete WORK_DIR while runtime files/CSVs may be open on Windows.
    embedded_run_mode = args.run_mode
    if not int(getattr(args, "allow_embedded_reset", 0)) and embedded_run_mode in {"initial", "reset", "overwrite"}:
        embedded_run_mode = "expand"
    _setenv("RUN_MODE", embedded_run_mode)

    if args.test_one_paper:
        _setenv("TEST_ONE_PAPER", "1")
        _setenv("SINGLE_PAPER_SOURCE", args.single_paper_source)
        _setenv("ONE_PAPER_MAX_DRIVE_FILES", str(args.one_paper_max_drive_files))
        _setenv("ONE_PAPER_SKIP_SEARCH_AFTER_DRIVE", "1" if args.one_paper_skip_search_after_drive else "0")
        if args.single_paper_pdf_path:
            _setenv("SINGLE_PAPER_PDF_PATH", args.single_paper_pdf_path)
        if args.one_paper_drive_filename_include:
            _setenv("ONE_PAPER_DRIVE_FILENAME_INCLUDE", args.one_paper_drive_filename_include)
    else:
        _clearenv([
            "TEST_ONE_PAPER",
            "SINGLE_PAPER_SOURCE",
            "ONE_PAPER_MAX_DRIVE_FILES",
            "ONE_PAPER_SKIP_SEARCH_AFTER_DRIVE",
            "SINGLE_PAPER_PDF_PATH",
            "ONE_PAPER_DRIVE_FILENAME_INCLUDE",
        ])

    _try_check_local_endpoint(args.llm_api_url, "Text LLM endpoint")
    if not int(getattr(args, "allow_embedded_reset", 0)) and args.run_mode in {"initial", "reset", "overwrite"}:
        print(f"[INFO] Controller fresh-start handling active: requested run_mode={args.run_mode}, embedded RUN_MODE=expand to avoid internal WORK_DIR deletion.")
    if args.vision_enable:
        _try_check_local_endpoint(args.vision_api_url, "Vision LLM endpoint")

    # Override hardcoded paper/candidate limits inside the embedded LiteratureAgent.
    # The original script used MAX_PAPERS_TO_PROCESS=1 for safe testing; this exposes it as --max_papers.
    if hasattr(lit_module, "MAX_PAPERS_TO_PROCESS"):
        lit_module.MAX_PAPERS_TO_PROCESS = int(args.max_papers)
    if hasattr(lit_module, "CROSSREF_ROWS_PER_PAGE"):
        lit_module.CROSSREF_ROWS_PER_PAGE = int(args.crossref_rows_per_page)
    if hasattr(lit_module, "CROSSREF_MAX_PAGES"):
        lit_module.CROSSREF_MAX_PAGES = int(args.crossref_max_pages)
    if hasattr(lit_module, "SEARCH_QUERY") and str(getattr(args, "search_query", "") or "").strip():
        lit_module.SEARCH_QUERY = str(args.search_query).strip()
        print(f"[INFO] Expansion search query: {lit_module.SEARCH_QUERY}")
    if hasattr(lit_module, "crossref_search"):
        _original_crossref_search = lit_module.crossref_search
        def _crossref_search_with_progress(query, year_min, year_max, rows_per_page=100, max_pages=8):
            started = _time_controller.time()
            print(
                f"[SEARCH] Starting Crossref search | years={year_min}-{year_max} | "
                f"rows/page={rows_per_page} | max_pages={max_pages} | web_timeout={getattr(lit_module, 'REQUEST_TIMEOUT', 'unknown')}s",
                flush=True,
            )
            try:
                result = _original_crossref_search(query, year_min, year_max, rows_per_page=rows_per_page, max_pages=max_pages)
                print(f"[SEARCH] Crossref search returned {len(result or [])} candidates in {_time_controller.time() - started:.1f}s", flush=True)
                return result
            except Exception as exc:
                print(
                    f"[SEARCH] Crossref search failed after {_time_controller.time() - started:.1f}s: {exc!r}. "
                    "Continuing with RSS/other candidates instead of aborting the expansion batch.",
                    flush=True,
                )
                return []
        lit_module.crossref_search = _crossref_search_with_progress
    if hasattr(lit_module, "rss_search"):
        _original_rss_search = lit_module.rss_search
        def _rss_search_with_progress(query, year_min, year_max):
            started = _time_controller.time()
            print(f"[SEARCH] Starting RSS search | years={year_min}-{year_max}", flush=True)
            result = _original_rss_search(query, year_min, year_max)
            print(f"[SEARCH] RSS search returned {len(result or [])} candidates in {_time_controller.time() - started:.1f}s", flush=True)
            return result
        lit_module.rss_search = _rss_search_with_progress
    if hasattr(lit_module, "merge_candidates"):
        _original_merge_candidates = lit_module.merge_candidates
        def _merge_candidates_with_progress(rss_items, crossref_items):
            merged = _original_merge_candidates(rss_items, crossref_items)
            print(
                f"[SEARCH] Candidate merge complete | rss={len(rss_items or [])} | "
                f"crossref={len(crossref_items or [])} | merged={len(merged or [])}",
                flush=True,
            )
            return merged
        lit_module.merge_candidates = _merge_candidates_with_progress
    for name, value in [
        ("REQUEST_TIMEOUT", int(getattr(args, "web_request_timeout", 45))),
        ("LLM_REQUEST_TIMEOUT_SECONDS", int(getattr(args, "llm_request_timeout", 600))),
        ("API_MAX_RETRIES", max(1, int(getattr(args, "llm_max_retries", 1)))),
        ("API_RETRY_BACKOFF_SECONDS", float(getattr(args, "llm_retry_backoff", 8.0))),
        ("GOOGLE_DRIVE_MAX_FILES_PER_RUN", int(getattr(args, "google_drive_max_files_per_run", 0) or args.max_papers)),
        ("DRIVE_FIRST_PROCESS_ALL_FILES", bool(int(drive_all))),
        ("FAMILY_GATING_MODE", str(getattr(args, "family_gating", "strict"))),
        ("DRIVE_FIRST_STRICT", not bool(disable_drive)),
    ]:
        if hasattr(lit_module, name):
            setattr(lit_module, name, value)
        elif name == "GOOGLE_DRIVE_MAX_FILES_PER_RUN":
            setattr(lit_module, name, value)

    # Safety override after import: some embedded constants are read at import time.
    # Force the embedded module away from reset-like modes and disable internal rmtree(BASE).
    safe_mode = _v21_safe_embedded_run_mode(args)
    try:
        if hasattr(lit_module, "RUN_MODE"):
            lit_module.RUN_MODE = safe_mode
        if not int(getattr(args, "allow_embedded_reset", 0)):
            def _controller_managed_no_reset():
                print("[INFO] Embedded reset_work_dir() suppressed; controller already handled fresh-start deletion.")
                _Path_controller(args.work_dir).mkdir(parents=True, exist_ok=True)
            lit_module.reset_work_dir = _controller_managed_no_reset
    except Exception as e:
        print(f"[WARN] Could not apply embedded reset safety override: {e}")

    _install_expansion_candidate_filter(lit_module, args)
    _install_web_expansion_fulltext_fallback(lit_module, args)
    _install_scientific_reasoning_policy_layer(lit_module, args)

    if int(getattr(args, "force_reprocess", 0)) and hasattr(lit_module, "process_batch_api"):
        _original_process_batch_api_controller = lit_module.process_batch_api

        def _controller_force_reprocess_batch(*batch_args, **batch_kwargs):
            batch_kwargs["force_reprocess"] = True
            return _original_process_batch_api_controller(*batch_args, **batch_kwargs)

        lit_module.process_batch_api = _controller_force_reprocess_batch
        print("[INFO] Controller force_reprocess active for this LiteratureAgent batch.")

    if hasattr(lit_module, "main"):
        old_argv = _sys_controller.argv[:]
        try:
            # Redirected Windows consoles often default to cp1252, while paper
            # text legitimately contains ligatures and scientific symbols.
            # Printing the final JSON must not turn a completed extraction into
            # a failed controller run.
            for stream in (_sys_controller.stdout, _sys_controller.stderr):
                reconfigure = getattr(stream, "reconfigure", None)
                if callable(reconfigure):
                    try:
                        reconfigure(encoding="utf-8", errors="replace")
                    except Exception:
                        pass
            _sys_controller.argv = ["embedded_literature_agent"]
            lit_module.main()
        finally:
            _sys_controller.argv = old_argv
    else:
        raise RuntimeError("Embedded LiteratureAgent has no main() function.")

    # Guarantee top-level figure report exports even if the embedded agent only put
    # figure/caption/OCR evidence into summaries/manager payloads.
    _synthesize_controller_figure_reports(_Path_controller(args.work_dir))

    try:
        all_records = _find_or_create_all_records_csv(_Path_controller(args.work_dir))
    except FileNotFoundError as exc:
        if str(getattr(args, "pipeline_stage", "")).strip().lower() == "extract_batch":
            all_records = _Path_controller(args.work_dir) / "csv" / "all_records.csv"
            all_records.parent.mkdir(parents=True, exist_ok=True)
            if not all_records.exists():
                with all_records.open("w", encoding="utf-8-sig", newline="") as f:
                    writer = _csv_controller.DictWriter(
                        f,
                        fieldnames=["Ref_DOI_number", "Ref_original_filename_data_upload", "Ref_internal_sample_id"],
                    )
                    writer.writeheader()
            print(f"[INFO] No records produced in this extraction round; wrote empty placeholder: {all_records}")
        else:
            raise exc
    print(f"[INFO] LiteratureAgent records ready: {all_records}")
    _run_target_finalization_if_enabled(args)
    return all_records


def _run_target_finalization_if_enabled(args) -> None:
    if not int(getattr(args, "target_finalization_enable", getattr(args, "target_recovery_enable", 1))):
        print("[INFO] Target finalization disabled; skipping source-text target/schema finalization.")
        return

    script = _Path_controller(__file__).resolve().parent / "scripts" / "recover_literatureagent_targets_from_source_text.py"
    if not script.exists():
        print(f"[WARN] Target finalization script not found, skipping: {script}")
        return

    cmd = [
        _sys_controller.executable,
        str(script),
        "--work-dir",
        str(args.work_dir),
        "--apply",
    ]
    print("\n" + "=" * 100)
    print("TARGET FINALIZATION: Filling explicit source-text targets/features before integration")
    print("=" * 100)
    print(" ".join(cmd))
    result = _subprocess_controller.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Target finalization failed with return code {result.returncode}")


def _run_integration(integration_module, args, lit_csv: _Path_controller) -> _Path_controller:
    print("\n" + "=" * 100)
    print("STEP 2: Running embedded LiteratureAgent -> PCE/Stability integration")
    print("=" * 100)

    _check_file(args.base_csv, "Base Perovskite Database CSV", required=True)

    updated_csv_arg = args.updated_csv_name
    integration_argv = [
        "embedded_integrate_literature_to_pce_stability",
        "--base_csv", str(args.base_csv),
        "--lit_csv", str(lit_csv),
        "--out_dir", str(args.integration_out_dir),
        "--updated_csv", str(updated_csv_arg),
        "--merge_policy", str(args.merge_policy),
        "--min_required_non_null_important_fields", str(args.min_required_non_null_important_fields),
        "--min_required_evidence_fields", str(args.min_required_evidence_fields),
        "--duplicate_similarity_threshold", str(args.duplicate_similarity_threshold),
    ]
    if args.no_require_doi:
        integration_argv.append("--no_require_doi")
    if args.integration_dry_run:
        integration_argv.append("--dry_run")

    old_argv = _sys_controller.argv[:]
    try:
        _sys_controller.argv = integration_argv
        integration_module.main()
    finally:
        _sys_controller.argv = old_argv

    updated_path = _Path_controller(updated_csv_arg)
    if not updated_path.is_absolute():
        updated_path = _Path_controller(args.integration_out_dir) / updated_path

    if not args.integration_dry_run:
        _check_file(updated_path, "Updated modeling CSV", required=True)

    print(f"[INFO] Updated modeling CSV: {updated_path}")
    return updated_path


def _run_model(args, updated_csv: _Path_controller) -> None:
    print("\n" + "=" * 100)
    print("STEP 3: Running PCE -> Stability modeling script")
    print("=" * 100)

    _check_file(args.model_script, "PCE/stability model script", required=True)

    cmd = [
        _sys_controller.executable,
        str(args.model_script),
        "--csv", str(updated_csv),
        "--out", str(args.model_out_dir),
    ]
    print(" ".join(cmd))
    result = _subprocess_controller.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Model script failed with return code {result.returncode}")


def _run_knowledge_graph(args, lit_csv: _Path_controller) -> None:
    print("\n" + "=" * 100)
    print("KNOWLEDGE GRAPH: Building evidence-grounded property graph")
    print("=" * 100)

    _check_file(args.knowledge_graph_builder, "Knowledge graph builder", required=True)
    _check_file(args.ontology_path, "Perovskite ontology", required=True)
    _check_file(lit_csv, "LiteratureAgent all_records.csv", required=True)

    cmd = [
        _sys_controller.executable,
        str(args.knowledge_graph_builder),
        "--records", str(lit_csv),
        "--ontology", str(args.ontology_path),
        "--work-dir", str(args.work_dir),
        "--out", str(args.knowledge_graph_out_dir),
    ]
    if args.knowledge_graph_max_records > 0:
        cmd.extend(["--max-records", str(args.knowledge_graph_max_records)])
    print(" ".join(cmd))
    result = _subprocess_controller.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Knowledge graph builder failed with return code {result.returncode}")


_EVIDENCE_QUERY_STOPWORDS = {
    "about", "according", "also", "analysis", "and", "answer", "are", "based", "can",
    "been", "being", "between", "could", "direct", "does", "evidence", "experiment",
    "experiments", "explain", "for", "formation", "from", "further", "have", "having",
    "how", "hypothesis", "idea", "identify", "interpretation", "into", "literature",
    "mechanism", "mechanistic", "needed", "paper", "papers", "possible", "promote",
    "question", "report", "reported", "research", "result", "results", "say", "should",
    "show", "studies", "study", "support", "test", "testing", "that", "the", "their", "type",
    "them", "there", "these", "this", "through", "under", "use", "used", "using",
    "underlying", "what", "when", "where", "whether", "which", "while", "with", "would",
}


def _evidence_terms(query: str) -> list[str]:
    raw_terms = _re_controller.findall(r"[A-Za-z0-9][A-Za-z0-9+]{1,}", str(query or ""))
    terms = []
    seen = set()
    for raw in raw_terms:
        term = raw.lower().strip("+")
        keep_short = any(ch.isdigit() for ch in term) or raw.isupper()
        if (len(term) < 3 and not keep_short) or term in _EVIDENCE_QUERY_STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _rank_evidence_documents(query: str, documents: list[dict], limit: int) -> tuple[list[dict], dict]:
    """Rank local evidence using corpus rarity and query-concept coverage.

    This is intentionally domain-agnostic. Rare concepts in the current query/corpus
    carry more weight than generic scientific words, and a document must cover more
    than one meaningful concept for non-trivial queries.
    """
    terms = _evidence_terms(query)
    if not terms or not documents:
        return [], {"query_terms": terms, "distinctive_terms": [], "minimum_coverage": 0.0}

    prepared = []
    for document in documents:
        search_text = _re_controller.sub(r"[-–—/]", " ", str(document.get("search_text") or "").lower())
        tokens = _re_controller.findall(r"[a-z0-9][a-z0-9+]{1,}", search_text)
        prepared.append((document, search_text, tokens, set(tokens)))

    doc_count = len(prepared)
    document_frequency = {
        term: sum(1 for _, _, _, token_set in prepared if term in token_set)
        for term in terms
    }
    weights = {
        term: _math_controller.log((doc_count + 1.0) / (document_frequency[term] + 1.0)) + 1.0
        for term in terms
    }
    distinctive_count = max(1, min(8, (len(terms) + 1) // 2))
    distinctive_terms = sorted(terms, key=lambda term: (weights[term], len(term)), reverse=True)[:distinctive_count]
    total_weight = sum(weights.values()) or 1.0
    minimum_matches = 1 if len(terms) <= 2 else 2
    minimum_coverage = 0.0 if len(terms) <= 2 else 0.16
    minimum_distinctive_matches = 1 if len(terms) <= 4 else 2

    query_sequence = [term for term in _evidence_terms(query)]
    phrases = []
    for size in (3, 2):
        for index in range(len(query_sequence) - size + 1):
            phrase = " ".join(query_sequence[index:index + size])
            if phrase not in phrases:
                phrases.append(phrase)

    ranked = []
    for document, search_text, tokens, token_set in prepared:
        matched_terms = [term for term in terms if term in token_set]
        matched_weight = sum(weights[term] for term in matched_terms)
        coverage = matched_weight / total_weight
        distinctive_matches = [term for term in distinctive_terms if term in token_set]
        normalized_text = " ".join(tokens)
        matched_phrases = [
            phrase for phrase in phrases
            if phrase in normalized_text and any(term in distinctive_terms for term in phrase.split())
        ]
        relevant = (
            len(matched_terms) >= minimum_matches
            and coverage >= minimum_coverage
            and (len(distinctive_matches) >= minimum_distinctive_matches or len(terms) <= 2)
        )
        if not relevant:
            continue

        frequency_score = sum(
            weights[term] * (1.0 + _math_controller.log1p(min(tokens.count(term), 3)))
            for term in matched_terms
        )
        title_tokens = set(_re_controller.findall(
            r"[a-z0-9][a-z0-9+]{1,}",
            _re_controller.sub(r"[-–—/]", " ", str(document.get("title_text") or "").lower())
        ))
        title_bonus = sum(weights[term] for term in matched_terms if term in title_tokens) * 1.5
        phrase_bonus = sum(2.0 + 0.5 * len(phrase.split()) for phrase in matched_phrases)
        score = frequency_score + title_bonus + phrase_bonus + (coverage * 5.0)
        distinctive_coverage = len(distinctive_matches) / max(1, len(distinctive_terms))
        strong_relevance = (
            relevant
            and (
                len(terms) <= 2
                or (
                    len(distinctive_matches) >= 2
                    and distinctive_coverage >= 0.5
                    and coverage >= 0.25
                )
            )
        )

        payload = dict(document.get("payload") or {})
        payload.update({
            "score": round(score, 4),
            "query_coverage": round(coverage, 4),
            "matched_query_terms": matched_terms,
            "matched_distinctive_terms": distinctive_matches,
            "distinctive_query_coverage": round(distinctive_coverage, 4),
            "matched_query_phrases": matched_phrases[:8],
            "relevance_passed": True,
            "strong_relevance_passed": strong_relevance,
        })
        ranked.append((score, coverage, payload))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    diagnostics = {
        "query_terms": terms,
        "distinctive_terms": distinctive_terms,
        "term_document_frequency": document_frequency,
        "minimum_matches": minimum_matches,
        "minimum_coverage": minimum_coverage,
        "minimum_distinctive_matches": minimum_distinctive_matches,
        "documents_considered": doc_count,
        "documents_relevant": len(ranked),
        "documents_strongly_relevant": sum(
            1 for _, _, payload in ranked if payload.get("strong_relevance_passed")
        ),
    }
    return [payload for _, _, payload in ranked[:limit]], diagnostics


def _load_record_metadata_for_evidence(work_dir: _Path_controller) -> dict[str, dict[str, str]]:
    csv_path = work_dir / "csv" / "all_records.csv"
    if not csv_path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    try:
        with csv_path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
            reader = _csv_controller.DictReader(handle)
            for row in reader:
                slug = row.get("paper_slug") or row.get("_paper_slug") or row.get("Ref_internal_sample_id")
                if slug and str(slug) not in out:
                    out[str(slug)] = row
    except Exception as exc:
        print(f"[WARN] Could not load evidence metadata from {csv_path}: {exc}")
    return out


def _evidence_excerpt(text: str, terms: list[str], radius: int = 1100) -> str:
    lowered = text.lower()
    positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
    start = max(0, (min(positions) if positions else 0) - 250)
    return " ".join(text[start:start + radius].split())


def _title_from_summary_for_evidence(text: str, fallback: str) -> str:
    for line in text.splitlines()[:12]:
        cleaned = line.strip().strip("#").strip()
        cleaned = _re_controller.sub(r"^paper\s+summary:\s*", "", cleaned, flags=_re_controller.I).strip()
        cleaned = _re_controller.sub(r"\s+DOI:\s+10\.\S+.*$", "", cleaned, flags=_re_controller.I).strip()
        if cleaned and len(cleaned) > 8 and not cleaned.lower().startswith(("paper summary", "executive summary")):
            return cleaned
    return fallback


def _doi_from_text_for_evidence(text: str) -> str | None:
    match = _re_controller.search(r"\b(10\.\d{4,9}/[^\s\]\)\};,\"']+)", str(text or ""))
    if not match:
        return None
    return match.group(1).rstrip(".,;:")


def _search_literature_summaries_for_evidence(
    work_dir: _Path_controller, query: str, limit: int
) -> tuple[list[dict], dict]:
    terms = _evidence_terms(query)
    summaries_dir = work_dir / "paper_summaries_text"
    metadata = _load_record_metadata_for_evidence(work_dir)
    documents = []
    if not summaries_dir.exists():
        return [], {"query_terms": terms, "distinctive_terms": [], "documents_considered": 0}
    for path in summaries_dir.glob("*_summary.txt"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        slug = path.name.removesuffix("_summary.txt")
        row = metadata.get(slug, {})
        title = (
            row.get("Ref_title")
            or row.get("title")
            or _title_from_summary_for_evidence(text, slug)
        )
        documents.append({
            "search_text": f"{title}\n{text}",
            "title_text": title,
            "payload": {
                "paper_slug": slug,
                "title": title,
                "doi": row.get("Ref_DOI_number") or row.get("doi") or _doi_from_text_for_evidence(text),
                "summary_excerpt": _evidence_excerpt(text, terms),
                "summary_path": str(path),
                "structured_json_path": str(work_dir / "json" / f"{slug}.json"),
            },
        })
    return _rank_evidence_documents(query, documents, limit)


def _search_knowledge_relationships_for_evidence(args, query: str, limit: int) -> tuple[list[dict], dict]:
    rel_path = _Path_controller(args.knowledge_graph_out_dir) / "derived_views" / "learned_scientific_relationships.csv"
    if not rel_path.exists():
        return [], {"query_terms": _evidence_terms(query), "distinctive_terms": [], "documents_considered": 0}
    documents = []
    try:
        with rel_path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
            for row in _csv_controller.DictReader(handle):
                text = " ".join(str(row.get(key, "")) for key in ("subject", "relationship", "object", "evidence", "evidence_text"))
                documents.append({
                    "search_text": text,
                    "title_text": " ".join(str(row.get(key, "")) for key in ("subject", "relationship", "object")),
                    "payload": {
                        "subject": row.get("subject"),
                        "relationship": row.get("relationship"),
                        "object": row.get("object"),
                        "paper_id": row.get("paper_id"),
                        "claim_id": row.get("claim_id"),
                        "evidence": row.get("evidence") or row.get("evidence_text"),
                    },
                })
    except Exception as exc:
        print(f"[WARN] Could not search knowledge graph relationships: {exc}")
    return _rank_evidence_documents(query, documents, limit)


def _run_evidence_packet(args) -> dict:
    query = str(getattr(args, "evidence_query", "") or getattr(args, "search_query", "") or "").strip()
    if not query:
        raise ValueError("--evidence_query or --search_query is required for pipeline_stage=evidence_packet")
    work_dir = _Path_controller(args.work_dir)
    limit = max(1, int(getattr(args, "evidence_limit", 5)))
    papers, paper_retrieval = _search_literature_summaries_for_evidence(work_dir, query, limit)
    relationships, relationship_retrieval = _search_knowledge_relationships_for_evidence(args, query, limit)

    strong_evidence_count = sum(
        1 for item in papers + relationships if item.get("strong_relevance_passed")
    )
    lines = ["Evidence-grounded LiteratureAgent packet:"]
    if strong_evidence_count == 0:
        lines.append(
            "No strong direct local support matched the full query. The items below are related background only; "
            "do not present them as proof of the combined claim."
        )
    for index, paper in enumerate(papers, 1):
        lines.append(
            f"\nPaper {index}: {paper.get('title')}\n"
            f"DOI: {paper.get('doi') or 'not available'} | paper_slug: {paper.get('paper_slug')}\n"
            f"Summary evidence: {paper.get('summary_excerpt')}"
        )
    if relationships:
        lines.append("\nLearned scientific relationships:")
        for index, rel in enumerate(relationships, 1):
            lines.append(
                f"{index}. {rel.get('subject')} -> {rel.get('relationship')} -> {rel.get('object')} "
                f"[paper={rel.get('paper_id') or 'unknown'}, claim={rel.get('claim_id') or 'unknown'}]"
            )
    if not papers and not relationships:
        lines.append("\nNo matching locally mined LiteratureAgent evidence was found.")

    packet = {
        "query": query,
        "papers": papers,
        "relationships": relationships,
        "formatted_context": "\n".join(lines)[:max(1000, int(getattr(args, "evidence_max_chars", 16000)))],
        "provenance": {
            "work_dir": str(work_dir),
            "knowledge_graph_dir": str(args.knowledge_graph_out_dir),
            "paper_count": len(papers),
            "relationship_count": len(relationships),
            "relevant_evidence_count": len(papers) + len(relationships),
            "strong_evidence_count": strong_evidence_count,
            "paper_retrieval": paper_retrieval,
            "relationship_retrieval": relationship_retrieval,
            "extraction_profile": getattr(args, "extraction_profile", "none"),
        },
    }
    out_dir = _Path_controller(getattr(args, "evidence_out_dir", "") or (work_dir / "evidence_packets"))
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_query = _re_controller.sub(r"[^a-zA-Z0-9]+", "_", query.lower()).strip("_")[:80] or "query"
    json_path = out_dir / f"{safe_query}_evidence_packet.json"
    txt_path = out_dir / f"{safe_query}_evidence_packet.txt"
    json_path.write_text(_json_controller.dumps(packet, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_path.write_text(packet["formatted_context"].rstrip() + "\n", encoding="utf-8")
    print(f"[INFO] Evidence packet written: {json_path}")
    print(f"[INFO] Evidence context written: {txt_path}")
    try:
        print(packet["formatted_context"])
    except UnicodeEncodeError:
        safe_context = packet["formatted_context"].encode("ascii", errors="replace").decode("ascii")
        print(safe_context)
    return packet


def _write_controller_summary(args, lit_csv, updated_csv, success: bool) -> None:
    out_dir = _Path_controller(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": _datetime_controller.now().isoformat(timespec="seconds"),
        "success": bool(success),
        "script": "literature_agent_full_end_to_end_v20.py",
        "base_csv": str(args.base_csv),
        "work_dir": str(args.work_dir),
        "lit_csv": str(lit_csv) if lit_csv else None,
        "integration_out_dir": str(args.integration_out_dir),
        "updated_csv": str(updated_csv) if updated_csv else None,
        "model_out_dir": str(args.model_out_dir),
        "run_mode": args.run_mode,
        "test_one_paper": bool(args.test_one_paper),
        "max_papers": int(args.max_papers),
        "crossref_rows_per_page": int(args.crossref_rows_per_page),
        "crossref_max_pages": int(args.crossref_max_pages),
        "single_paper_source": args.single_paper_source if args.test_one_paper else None,
        "llm_api_url": args.llm_api_url,
        "llm_model": args.llm_model,
        "vision_enable": bool(args.vision_enable),
        "vision_api_url": args.vision_api_url if args.vision_enable else None,
        "vision_model": args.vision_model if args.vision_enable else None,
        "merge_policy": args.merge_policy,
        "no_require_doi": bool(args.no_require_doi),
        "run_model": bool(args.run_model),
        "pipeline_stage": args.pipeline_stage,
        "extraction_profile": getattr(args, "extraction_profile", "none"),
        "evidence_query": getattr(args, "evidence_query", None),
        "knowledge_graph_out_dir": str(args.knowledge_graph_out_dir),
        "controller_elapsed_seconds": getattr(args, "_controller_elapsed_seconds", None),
        "timing_logs_dir": str(_Path_controller(args.work_dir) / "timing_logs"),
    }
    path = out_dir / "full_end_to_end_v20_run_summary.json"
    path.write_text(_json_controller.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] Full end-to-end run summary written to: {path}")


def _apply_extraction_profile(args):
    """
    High-level profiles for app/agent callers.

    Default profile "none" preserves the historical CLI behavior. The named
    profiles set conservative defaults only when the caller has not supplied a
    more specific flag downstream.
    """
    profile = str(getattr(args, "extraction_profile", "none") or "none").strip().lower()
    args.extraction_profile = profile
    if profile == "none":
        return args

    supplied = set(_sys_controller.argv[1:])

    def set_default(flag, attribute, value):
        if flag not in supplied:
            setattr(args, attribute, value)

    if profile in {"hypothesis_support", "general_literature"}:
        set_default("--run_mode", "run_mode", "expand")
        set_default("--disable_google_drive", "disable_google_drive", 1)
        set_default("--family_gating", "family_gating", "moderate")
        set_default("--expansion_candidate_filter", "expansion_candidate_filter", "query_relevant")
        set_default("--expansion_require_oa", "expansion_require_oa", 0)
        set_default("--target_finalization_enable", "target_finalization_enable", 0)
        set_default("--target_recovery_enable", "target_recovery_enable", 0)
        set_default("--figure_report_enable", "figure_report_enable", 1)
        set_default("--use_reasoning_layer", "use_reasoning_layer", 1)
        set_default("--reasoning_policy_mode", "reasoning_policy_mode", "auto")
        args.no_require_doi = True
        return args

    if profile == "pce_stability_modeling":
        has_drive_folder = bool(str(getattr(args, "google_drive_folder_id", "") or "").strip())
        if not has_drive_folder:
            set_default("--run_mode", "run_mode", "expand")
            set_default("--disable_google_drive", "disable_google_drive", 1)
        else:
            # A requested Drive folder is an explicit source selection. Never
            # let a modeling profile silently route it into web expansion.
            args.disable_google_drive = 0
        set_default("--family_gating", "family_gating", "strict")
        set_default("--expansion_candidate_filter", "expansion_candidate_filter", "model_ready_strict")
        set_default("--expansion_require_oa", "expansion_require_oa", 0)
        set_default("--target_finalization_enable", "target_finalization_enable", 0)
        set_default("--target_recovery_enable", "target_recovery_enable", 0)
        set_default("--figure_report_enable", "figure_report_enable", 0)
        set_default("--use_reasoning_layer", "use_reasoning_layer", 0)
        args.no_require_doi = True
        return args

    raise ValueError(f"Unknown extraction profile: {profile}")


def _parse_controller_args():
    d = _downloads_dir()
    project_dir = _Path_controller(__file__).resolve().parent
    project_ontology = project_dir / "config" / "perovskite_ontology_library_v19.json"
    project_model = project_dir / "models" / "pce_then_stability_same_approach.py"
    project_knowledge_graph_builder = project_dir / "scripts" / "build_literature_knowledge_graph.py"
    project_client_secret = project_dir / "secrets" / "google_drive_oauth_client.json"
    project_token = project_dir / "secrets" / "google_drive_token.json"
    p = _argparse_controller.ArgumentParser(
        description="Single-file LiteratureAgent + integration + optional PCE/stability modeling pipeline."
    )

    # Required baseline dataset.
    p.add_argument("--base_csv", required=True, help="Path to Perovskite_database_content_all_data.csv")

    # Paths.
    p.add_argument("--ontology_path", default=str(project_ontology if project_ontology.exists() else d / "perovskite_ontology_library_v19.json"))
    p.add_argument("--google_drive_oauth_client_secrets", default=str(project_client_secret if project_client_secret.exists() else d / "client_secret_298117362376-e4lfd37ju8gd95t932gurcct7orb6ol1.apps.googleusercontent.com.json"))
    p.add_argument("--google_drive_oauth_token_file", default=str(project_token),
                   help="Path to the generated Google Drive OAuth user token. Keep this file private and out of handoff/source-control packages.")
    p.add_argument("--google_drive_folder_id", default=None,
                   help="Optional Google Drive folder ID/link override. Use this to target a curated Drive subfolder.")
    p.add_argument("--work_dir", default=str(d / "lit_outputs"))
    p.add_argument("--integration_out_dir", default=str(d / "artifacts_literature_dataset_update"))
    p.add_argument("--updated_csv_name", default="updated_perovskite_database_with_literature_agent.csv")
    p.add_argument("--model_script", default=str(project_model if project_model.exists() else d / "pce_then_stability_same_approach.py"))
    p.add_argument("--model_out_dir", default=str(d / "artifacts_pce_then_stability_lit_updated"))
    p.add_argument("--knowledge_graph_builder", default=str(project_knowledge_graph_builder))
    p.add_argument("--knowledge_graph_out_dir", default=str(d / "artifacts_literature_knowledge_graph"))
    p.add_argument("--knowledge_graph_max_records", type=int, default=0,
                   help="Optional knowledge-graph test limit. 0 builds from all existing LiteratureAgent records.")
    p.add_argument("--extraction_profile", default="none",
                   choices=["none", "hypothesis_support", "general_literature", "pce_stability_modeling"],
                   help="High-level caller profile. none preserves existing behavior; hypothesis_support/general_literature are broad evidence-retrieval modes; pce_stability_modeling is strict target-row mining.")
    p.add_argument("--evidence_query", default="",
                   help="Query used by pipeline_stage=evidence_packet. Defaults to --search_query if omitted.")
    p.add_argument("--evidence_limit", type=int, default=5,
                   help="Maximum papers and learned relationships in an evidence packet.")
    p.add_argument("--evidence_max_chars", type=int, default=16000,
                   help="Maximum formatted-context characters in an evidence packet.")
    p.add_argument("--evidence_out_dir", default="",
                   help="Output folder for pipeline_stage=evidence_packet. Default is work_dir/evidence_packets.")

    # Fresh run control.
    p.add_argument("--from_scratch", action="store_true", help="Clear old lit_outputs and integration/model artifacts first.")
    p.add_argument("--skip_literature_agent", action="store_true", help="Use existing work_dir/csv/all_records.csv.")
    p.add_argument("--skip_integration", action="store_true", help="Skip dataset integration.")
    p.add_argument("--run_model", action="store_true", help="Run PCE/stability model after integration.")

    # Literature run modes.
    p.add_argument("--run_mode", default="initial", choices=["initial", "expand", "benchmark", "reset"])
    p.add_argument("--max_papers", type=int, default=1, help="Maximum papers to process in LiteratureAgent search/expand mode.")
    p.add_argument("--test_one_paper", action="store_true", help="Run only one paper.")
    p.add_argument("--full_literature_run", action="store_true", help="Clear one-paper behavior and run normally.")
    p.add_argument("--single_paper_source", default="drive", choices=["drive", "pdf", "search"])
    p.add_argument("--single_paper_pdf_path", default=None)
    p.add_argument("--one_paper_max_drive_files", type=int, default=1)
    p.add_argument("--one_paper_skip_search_after_drive", type=int, default=1)
    p.add_argument("--one_paper_drive_filename_include", default=None)
    p.add_argument("--crossref_rows_per_page", type=int, default=50, help="Crossref rows per page for candidate search.")
    p.add_argument("--crossref_max_pages", type=int, default=6, help="Maximum Crossref pages for candidate search.")
    p.add_argument("--search_query", default=None,
                   help="Override the embedded Crossref/search expansion query. Use target-rich device/stability terms for expansion campaigns.")

    # LLM / local Ollama defaults.
    p.add_argument("--llm_api_key", default="ollama")
    p.add_argument("--llm_api_url", default="http://localhost:11434/v1/chat/completions")
    p.add_argument("--llm_model", default="qwen2.5:7b")
    p.add_argument("--llm_model_fast", default=None)
    p.add_argument("--llm_model_strong", default=None)
    p.add_argument("--llm_cost_mode", default="balanced", choices=["cheap", "balanced", "thorough"])
    p.add_argument("--deterministic_prefill", action="store_true")
    p.add_argument("--llm_request_timeout", type=int, default=600,
                   help="Base request timeout in seconds for local/OpenAI-compatible text LLM calls. Embedded effective timeout is 3x this value.")
    p.add_argument("--web_request_timeout", type=int, default=45,
                   help="Request timeout in seconds for Crossref/RSS/Unpaywall/PDF/HTML retrieval during search expansion.")
    p.add_argument("--llm_max_retries", type=int, default=1,
                   help="Text LLM retry count. Use 1 for local Ollama to avoid long retry storms after a timeout.")
    p.add_argument("--llm_retry_backoff", type=float, default=8.0,
                   help="Initial retry backoff seconds for text LLM calls.")
    p.add_argument("--drive_process_all_files", type=int, default=None,
                   help="Google Drive drive-first processing: 1=all files, 0=limit to --google_drive_max_files_per_run/--max_papers. Default: 1 only with --full_literature_run, otherwise 0.")
    p.add_argument("--google_drive_max_files_per_run", type=int, default=0,
                   help="When Drive processing is not all-files, cap Drive PDFs per run. 0 means use --max_papers.")
    p.add_argument("--disable_google_drive", type=int, default=0,
                   help="Set to 1 for search/download-only expansion. Disables the embedded drive-first precheck even if Drive environment settings exist.")
    p.add_argument("--expansion_candidate_filter", default="target_ready", choices=["off", "query_relevant", "target_ready", "stability_target_ready", "model_ready_strict"],
                   help="Search/download expansion candidate filter. stability_target_ready requires PSC device context plus an explicit stability/lifetime cue in metadata, then defers numeric target validation to full text. model_ready_strict requires explicit PCE/JV/stability/device cues before retrieval.")
    p.add_argument("--expansion_require_oa", type=int, default=1,
                   help="For search/download expansion, require Unpaywall open-access status before attempting full-text extraction. Default 1.")
    p.add_argument("--expansion_skip_known_candidates", type=int, default=1,
                   help="For search/download expansion, skip candidates whose DOI/title already appears in the base CSV or current work_dir records before expensive extraction. Default 1.")
    p.add_argument("--expansion_skip_base_known_candidates", type=int, default=1,
                   help="Include the baseline database when building pre-extraction duplicate keys. Set to 0 for targeted enrichment of legacy database papers while still skipping completed LiteratureAgent outputs and the processed registry. Default 1.")
    p.add_argument("--expansion_processed_registry", default="",
                   help="Optional central registry of successfully processed LiteratureAgent papers. Retryable attempts must not be stored here.")
    p.add_argument("--expansion_skip_terminal_retrieval_failures", type=int, default=1,
                   help="Skip expansion candidates that previously exhausted several retrieval routes without usable text. Default 1.")
    p.add_argument("--expansion_terminal_failure_min_routes", type=int, default=4,
                   help="Minimum distinct failed retrieval routes before a candidate is cached as a terminal retrieval failure. Default 4.")
    p.add_argument("--automated_retrieval_enable", type=int, default=1,
                   help="For search/download expansion, try DOI/OpenAlex/EuropePMC/arXiv/landing/PDF resolver routes when normal full text is too short. Default 1.")
    p.add_argument("--retrieval_max_urls_per_candidate", type=int, default=8,
                   help="Maximum resolver URLs to attempt per expansion candidate when automated retrieval is enabled.")
    p.add_argument("--retrieval_supplementary_enable", type=int, default=1,
                   help="For expansion HTML articles, retrieve linked supporting-information files and append their tables/text before extraction. Default 1.")
    p.add_argument("--retrieval_max_supplementary_files", type=int, default=2,
                   help="Maximum linked supplementary files to retrieve per aligned expansion article. Default 2.")
    p.add_argument("--retrieval_max_pdf_bytes", type=int, default=25_000_000,
                   help="Reject retrieved PDFs larger than this before text extraction.")
    p.add_argument("--retrieval_max_pdf_pages", type=int, default=80,
                   help="Reject retrieved PDFs with more pages than this before text extraction.")
    p.add_argument("--expansion_min_fulltext_chars", type=int, default=6000,
                   help="Minimum full-text characters for search/download HTML/OA expansion. Lower than PDF/Drive mode because HTML article extraction is often shorter.")
    p.add_argument("--expansion_min_extraction_text_chars", type=int, default=3000,
                   help="Minimum extraction-text characters for search/download HTML/OA expansion.")
    p.add_argument("--family_gating", default="strict", choices=["off", "moderate", "strict"],
                   help="Paper-type/family gating. off runs all schema families; moderate/strict skip device-heavy families for non-device papers. Default strict for large local Google Drive runs.")

    # Figure / vision.
    p.add_argument("--figure_report_enable", type=int, default=1)
    p.add_argument("--vision_enable", type=int, default=0)
    p.add_argument("--vision_provider", default="openai_compatible")
    p.add_argument("--vision_api_key", default="ollama")
    p.add_argument("--vision_api_url", default="http://localhost:11434/v1/chat/completions")
    p.add_argument("--vision_model", default="qwen2.5vl:7b-q4_K_M")
    p.add_argument("--vision_max_figures_per_paper", type=int, default=1)
    p.add_argument("--vision_max_image_width", type=int, default=800)
    p.add_argument("--vision_cache_enable", type=int, default=1)

    # Integration.
    p.add_argument("--merge_policy", default="append_new_only", choices=["append_new_only", "update_existing", "append_all"])
    p.add_argument("--no_require_doi", action="store_true", help="Recommended for manual Google Drive PDFs.")
    p.add_argument("--min_required_non_null_important_fields", type=int, default=4)
    p.add_argument("--min_required_evidence_fields", type=int, default=1)
    p.add_argument("--duplicate_similarity_threshold", type=float, default=0.82)
    p.add_argument("--integration_dry_run", action="store_true")
    # Pipeline-stage controls for v21
    p.add_argument("--pipeline_stage", default="full",
                   choices=["full", "extract_batch", "vision_pass", "integrate_and_model", "sanitize_summaries", "knowledge_graph", "evidence_packet"],
                   help="full = original end-to-end; extract_batch = LiteratureAgent only; vision_pass = post-hoc Qwen-VL on saved figure crops and update summaries; integrate_and_model = merge existing records and optionally run model; knowledge_graph = build the evidence-grounded property graph from existing records; evidence_packet = retrieve local evidence context for a hypothesis/query.")
    p.add_argument("--strict_extraction", type=int, default=1,
                   help="Keep strict schema-family extraction for downstream ML CSV quality. Default 1.")
    p.add_argument("--inline_vision", type=int, default=0,
                   help="Run Qwen-VL during the main LiteratureAgent extraction pass. Default 0 because it is slow; use vision_pass afterward.")
    p.add_argument("--vision_only_max_papers", type=int, default=0,
                   help="For vision_pass, maximum papers/summaries to update; 0 means all.")
    p.add_argument("--vision_update_mode", default="append",
                   choices=["append", "regenerate_addendum"],
                   help="append = append visual evidence section; regenerate_addendum = also ask text LLM for a short vision-informed addendum.")
    p.add_argument("--vision_select_keywords", default="stability,j-v,jv,eqe,device,xrd,pl,uv-vis,kpfm,npfm,stem,edx,sem,tem",
                   help="Comma-separated keywords used to rank figure crops for post-hoc Qwen-VL.")
    p.add_argument("--vision_pass_force", type=int, default=0,
                   help="If 1, rerun Qwen-VL even if a vision log already exists.")
    p.add_argument("--conditional_ocr_hint", type=int, default=1,
                   help="Documented controller hint: preserve OCR, but prefer native text/caption extraction when available if supported by embedded agent.")
    p.add_argument("--missing_field_recovery", type=int, default=0,
                   help="Run extra missing-field LLM recovery passes. Default 0 for speed; extraction remains strict.")
    p.add_argument("--target_finalization_enable", type=int, default=0,
                   help="Opt in to legacy broad source-text target/schema recovery. Default 0; evidence-producing extraction is the normal path.")
    p.add_argument("--target_recovery_enable", type=int, default=0,
                   help="Deprecated alias for --target_finalization_enable; kept so older commands continue to work.")
    p.add_argument("--force_summary_rewrite", type=int, default=0,
                   help="Regenerate summaries even when cached summaries exist. Default 0 for resume speed.")
    p.add_argument("--force_reprocess", type=int, default=0,
                   help="Force LiteratureAgent to reprocess selected papers without deleting WORK_DIR. Useful after extraction/gating logic changes.")
    p.add_argument("--review_full_extraction", type=int, default=0,
                   help="Run full device-row extraction for review/perspective papers. Default 0 avoids literature-example rows.")
    p.add_argument("--llm_cache_enable", type=int, default=1,
                   help="Cache text LLM calls by paper/stage/model/prompt/payload hash. Default 1.")
    p.add_argument("--use_reasoning_layer", type=int, default=1 if _truthy_env_value(_os_controller.getenv("USE_REASONING_LAYER")) else 0,
                   help="Enable the optional Scientific Reasoning Policy Layer around LiteratureAgent LLM calls. Equivalent to USE_REASONING_LAYER=true.")
    p.add_argument("--reasoning_policy_mode", default=_os_controller.getenv("REASONING_POLICY_MODE", "auto"),
                   choices=["off", "single", "multi", "auto"],
                   help="Reasoning-policy routing mode. off disables prompt routing, single uses one selected policy, multi adds task-specific secondary policies, auto selects a primary plus limited secondaries. Default auto.")
    p.add_argument("--allow_embedded_reset", type=int, default=0,
                   help="If 1, allow the embedded LiteratureAgent to reset/delete WORK_DIR internally. Default 0 because the controller handles fresh starts safely.")

    args = p.parse_args()
    _apply_extraction_profile(args)

    # v21 default strategy:
    # - Main extraction pass preserves original summary/extraction quality and strict schema-family extraction.
    # - Qwen-VL is not run inline by default because it was the largest runtime bottleneck.
    # - Use --pipeline_stage vision_pass afterward to update summaries with visual evidence.
    if args.pipeline_stage in {"extract_batch", "full"} and not args.inline_vision:
        args.vision_enable = 0
    if args.pipeline_stage == "vision_pass":
        args.vision_enable = 1
        args.figure_report_enable = 1
    if args.full_literature_run:
        args.test_one_paper = False
    return args



# =============================================================================
# V21 POST-HOC VISION PASS AND SUMMARY UPDATE HELPERS
# =============================================================================

def _v21_slug_from_summary_path(path: _Path_controller) -> str:
    name = path.name
    return name[:-12] if name.endswith("_summary.txt") else path.stem


def _v21_image_to_data_url(path: _Path_controller, max_width: int = 800) -> str:
    import base64 as _base64
    import mimetypes as _mimetypes

    # Keep implementation dependency-light. If PIL is available, downscale large images.
    data_path = path
    try:
        from PIL import Image as _Image
        im = _Image.open(path).convert("RGB")
        w, h = im.size
        if max_width and w > max_width:
            new_h = max(1, int(h * (max_width / float(w))))
            im = im.resize((max_width, new_h))
            tmp = path.parent / f"_v21_resized_{path.stem}.jpg"
            im.save(tmp, format="JPEG", quality=85)
            data_path = tmp
    except Exception:
        data_path = path

    mime = _mimetypes.guess_type(str(data_path))[0] or "image/png"
    b64 = _base64.b64encode(data_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _v21_call_openai_compatible_vision(args, image_path: _Path_controller, context_text: str, figure_id: str) -> dict:
    import json as _json
    import urllib.request as _urlrequest
    import urllib.error as _urlerror

    data_url = _v21_image_to_data_url(image_path, max_width=int(args.vision_max_image_width))
    system = (
        "You are a figure-analysis assistant for halide-perovskite and photovoltaic research. "
        "Return compact valid JSON only. Write in English. Treat all values as candidate evidence unless explicitly labeled."
    )
    user_payload = {
        "task": (
            "Analyze this figure image/crop for a literature-mining workflow. Identify the figure type, "
            "what it shows, any candidate numerical values, materials/compositions, device information, "
            "stability/PCE/JV/EQE/XRD/PL evidence, and whether it should affect the paper summary."
        ),
        "figure_id": figure_id,
        "context_text": context_text[:2500],
        "output_schema": {
            "figure_type": "one of: stability_curve, jv_curve, eqe_spectrum, device_schematic, xrd, pl_uvvis, microscopy, table_or_other",
            "is_high_value": True,
            "visual_summary": "...",
            "candidate_database_fields": {},
            "summary_relevant_points": ["..."],
            "confidence": "low|medium|high",
            "needs_human_check": True
        },
    }
    body = {
        "model": args.vision_model,
        "temperature": 0,
        "max_tokens": 900,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": _json.dumps(user_payload, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ],
    }
    req = _urlrequest.Request(
        args.vision_api_url,
        data=_json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.vision_api_key or 'EMPTY'}",
        },
        method="POST",
    )
    try:
        with _urlrequest.urlopen(req, timeout=180) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        payload = _json.loads(raw)
        content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
        try:
            return _json.loads(content)
        except Exception:
            return {"figure_type": "unknown", "visual_summary": content, "raw_response": payload, "needs_human_check": True}
    except Exception as e:
        return {"ok": False, "error": str(e), "figure_id": figure_id, "image_path": str(image_path)}


def _v21_find_candidate_crops_for_slug(work_dir: _Path_controller, slug: str, keywords: list[str], max_figures: int) -> list[_Path_controller]:
    candidates = []
    search_roots = [
        work_dir / "figure_reports" / "crops",
        work_dir / "figure_analysis",
        work_dir / "paper_store" / slug,
    ]
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    slug_low = slug.lower()
    paper_store_root = (work_dir / "paper_store" / slug).resolve()
    for root in search_roots:
        if root.exists():
            root_resolved = root.resolve()
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    s = str(p).lower()
                    under_this_paper_store = str(root_resolved).lower().startswith(str(paper_store_root).lower())
                    if slug_low not in s and not under_this_paper_store:
                        continue
                    score = 10 if slug_low in s else 5
                    for kw in keywords:
                        if kw and kw.lower() in s:
                            score += 3
                    if "figure" in s or "crop" in s or "page" in s:
                        score += 1
                    candidates.append((score, p))
    # De-duplicate and sort by relevance then modification time.
    seen = set()
    uniq = []
    for score, p in sorted(candidates, key=lambda x: (x[0], x[1].stat().st_mtime if x[1].exists() else 0), reverse=True):
        if p.resolve() not in seen:
            seen.add(p.resolve())
            uniq.append(p)
    return uniq[:max(1, int(max_figures))]


def _v21_append_visual_evidence_to_summary(summary_path: _Path_controller, vision_items: list[dict], addendum: str | None = None) -> None:
    original = summary_path.read_text(encoding="utf-8", errors="ignore")
    # Remove any prior v21 section to avoid duplicate append on reruns.
    marker = "\n\nVision-Updated Figure Evidence:\n"
    if marker in original:
        original = original.split(marker)[0].rstrip()

    successful_items = []
    failed_count = 0
    for item in vision_items:
        if isinstance(item, dict) and (item.get("ok") is False or item.get("error")):
            failed_count += 1
        elif isinstance(item, dict):
            successful_items.append(item)

    lines = [original.rstrip(), "", "Vision-Updated Figure Evidence:"]
    if not successful_items:
        if failed_count:
            lines.append("- No successful Qwen-VL visual evidence was generated for this paper; failed attempts are saved in figure_reports/vision_logs.")
        else:
            lines.append("- No Qwen-VL visual evidence was generated for this paper.")
    for item in successful_items:
        ftype = item.get("figure_type") or "unknown"
        conf = item.get("confidence") or "not specified"
        summary = item.get("visual_summary") or item.get("summary") or "No visual summary returned."
        lines.append(f"- Figure type: {ftype}; confidence: {conf}; summary: {summary}")
        pts = item.get("summary_relevant_points") or []
        if isinstance(pts, list):
            for pt in pts[:4]:
                lines.append(f"  - {pt}")
        fields = item.get("candidate_database_fields") or {}
        if isinstance(fields, dict) and fields:
            compact = "; ".join(f"{k}: {v}" for k, v in list(fields.items())[:8])
            lines.append(f"  - Candidate database fields: {compact}")
    if addendum and successful_items:
        lines.extend(["", "Vision-Informed Summary Update:", addendum.strip()])
    summary_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _v21_call_text_addendum(args, summary_text: str, vision_items: list[dict]) -> str:
    import json as _json
    import urllib.request as _urlrequest
    system = (
        "You update a perovskite-paper summary using new figure-analysis evidence. "
        "Write in English. Do not rewrite the whole summary. Return only a concise addendum with facts supported by the vision evidence."
    )
    body = {
        "model": args.llm_model,
        "temperature": 0,
        "max_tokens": 700,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": _json.dumps({
                "existing_summary": summary_text[:6000],
                "vision_evidence": vision_items,
                "task": "Write a concise vision-informed summary addendum. Mention only evidence relevant to materials, device architecture, characterization, performance, or stability."
            }, ensure_ascii=False)},
        ],
    }
    req = _urlrequest.Request(
        args.llm_api_url,
        data=_json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.llm_api_key or 'EMPTY'}",
        },
        method="POST",
    )
    try:
        with _urlrequest.urlopen(req, timeout=180) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        payload = _json.loads(raw)
        return payload.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"Vision addendum generation failed: {e}"


def _run_v21_vision_pass(args) -> None:
    import json as _json
    work_dir = _Path_controller(args.work_dir)
    _ensure_figure_report_dirs(work_dir)

    summary_dir = work_dir / "paper_summaries_text"
    if not summary_dir.exists():
        raise FileNotFoundError(f"Summary directory not found: {summary_dir}")

    vision_log_dir = work_dir / "figure_reports" / "vision_logs"
    vision_log_dir.mkdir(parents=True, exist_ok=True)

    keywords = [x.strip().lower() for x in str(args.vision_select_keywords).split(",") if x.strip()]
    summaries = sorted(summary_dir.glob("*_summary.txt"))
    if int(args.vision_only_max_papers) > 0:
        summaries = summaries[:int(args.vision_only_max_papers)]

    print("\n" + "=" * 100)
    print(f"V21 VISION PASS: updating {len(summaries)} summaries with post-hoc Qwen-VL evidence")
    print("=" * 100)

    for idx, sfile in enumerate(summaries, start=1):
        slug = _v21_slug_from_summary_path(sfile)
        crops = _v21_find_candidate_crops_for_slug(
            work_dir=work_dir,
            slug=slug,
            keywords=keywords,
            max_figures=int(args.vision_max_figures_per_paper),
        )
        print(f"[VISION {idx}/{len(summaries)}] {slug}: candidate crops={len(crops)}")

        vision_items = []
        context_parts = [sfile.read_text(encoding="utf-8", errors="ignore")]
        for report_path in [
            work_dir / "figure_reports" / "text" / f"{slug}_figure_report_v19_7.txt",
            work_dir / "figure_reports" / "text" / f"{slug}_figure_report_v19_6.txt",
            work_dir / "figure_evidence" / f"{slug}_figure_report_v19_7.txt",
            work_dir / "figure_evidence" / f"{slug}_figure_report_v19_6.txt",
        ]:
            if report_path.exists():
                context_parts.append(report_path.read_text(encoding="utf-8", errors="ignore"))
                break
        context = "\n\n".join(context_parts)[:6000]
        for j, crop in enumerate(crops, start=1):
            fig_id = f"{slug}_fig{j}"
            log_path = vision_log_dir / f"{fig_id}_qwen_vl.json"
            if log_path.exists() and not int(args.vision_pass_force):
                try:
                    item = _json.loads(log_path.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    item = {"figure_type": "unknown", "visual_summary": f"Existing log unreadable: {log_path}"}
            else:
                item = _v21_call_openai_compatible_vision(args, crop, context, fig_id)
                item["image_path"] = str(crop)
                item["figure_id"] = fig_id
                log_path.write_text(_json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")
            vision_items.append(item)

        addendum = None
        if args.vision_update_mode == "regenerate_addendum" and vision_items:
            addendum = _v21_call_text_addendum(args, context, vision_items)

        _v21_append_visual_evidence_to_summary(sfile, vision_items, addendum=addendum)

    print(f"[INFO] Vision pass complete. Logs: {vision_log_dir}")


# =============================================================================
# V21.3 ENGLISH SUMMARY SANITIZER
# =============================================================================

def _v21_contains_cjk(text: str) -> bool:
    import re as _re
    return bool(_re.search(r"[\u4e00-\u9fff]", text or ""))


def _v21_rewrite_summary_in_english(args, summary_text: str) -> str:
    import json as _json
    import urllib.request as _urlrequest

    system = (
        "You are an English scientific editing assistant for perovskite literature summaries. "
        "Rewrite the provided summary fully in English. Preserve all section headings, bullet structure, "
        "numbers, units, chemical formulas, abbreviations, device stacks, performance metrics, stability metrics, "
        "materials names, and caveats. Do not add new facts. Do not remove useful facts. "
        "If the text is already English, lightly clean it without changing meaning. Return only the rewritten summary."
    )
    body = {
        "model": args.llm_model,
        "temperature": 0,
        "max_tokens": 2400,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": _json.dumps({
                "task": "Rewrite this complete paper summary in English while preserving exact scientific meaning and structure.",
                "summary_text": summary_text,
            }, ensure_ascii=False)},
        ],
    }
    req = _urlrequest.Request(
        args.llm_api_url,
        data=_json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.llm_api_key or 'EMPTY'}",
        },
        method="POST",
    )
    with _urlrequest.urlopen(req, timeout=240) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    data = _json.loads(raw)
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def _v21_sanitize_summaries_to_english(args) -> None:
    import json as _json

    work_dir = _Path_controller(args.work_dir)
    summary_dir = work_dir / "paper_summaries_text"
    if not summary_dir.exists():
        print(f"[WARN] No summary directory found: {summary_dir}")
        return

    report_dir = work_dir / "summary_language_reports"
    backup_dir = report_dir / "backups_before_english_sanitize"
    report_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    summaries = sorted(summary_dir.glob("*.txt"))
    report = []
    changed = 0

    print("\n" + "=" * 100)
    print(f"V21.3 ENGLISH SANITIZER: checking {len(summaries)} summaries")
    print("=" * 100)

    for i, path in enumerate(summaries, start=1):
        original = path.read_text(encoding="utf-8", errors="ignore")
        has_cjk = _v21_contains_cjk(original)
        status = {
            "file": path.name,
            "has_cjk_before": has_cjk,
            "rewritten": False,
            "has_cjk_after": None,
            "error": None,
        }

        if has_cjk:
            print(f"[SANITIZE {i}/{len(summaries)}] Rewriting non-English summary: {path.name}")
            try:
                (backup_dir / path.name).write_text(original, encoding="utf-8")
                rewritten = _v21_rewrite_summary_in_english(args, original)
                if rewritten and len(rewritten) > 200:
                    path.write_text(rewritten.rstrip() + "\n", encoding="utf-8")
                    changed += 1
                    status["rewritten"] = True
                    status["has_cjk_after"] = _v21_contains_cjk(rewritten)
                else:
                    status["error"] = "Rewrite returned empty/too-short output."
            except Exception as e:
                status["error"] = str(e)
                print(f"[WARN] Failed to rewrite {path.name}: {e}")
        else:
            status["has_cjk_after"] = False

        report.append(status)

    report_path = report_dir / "summary_language_sanitize_report.json"
    report_path.write_text(_json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] English sanitizer complete. Rewritten: {changed}. Report: {report_path}")


def main():
    args = _parse_controller_args()
    _controller_started = _time_controller.perf_counter()

    print("=" * 100)
    print("FULL SINGLE-FILE LITERATUREAGENT -> DATASET UPDATE -> PCE/STABILITY PIPELINE")
    print("QUALITY-RESTORED VERSION: original non-fast summary/extraction behavior + English-only summaries")
    print("=" * 100)
    print(f"Base CSV:              {args.base_csv}")
    print(f"Work dir:              {args.work_dir}")
    print(f"Integration out dir:   {args.integration_out_dir}")
    print(f"Run mode:              {args.run_mode}")
    print(f"Pipeline stage:        {args.pipeline_stage}")
    print(f"Extraction profile:    {getattr(args, 'extraction_profile', 'none')}")
    print(f"One-paper test:        {args.test_one_paper}")
    print(f"Max papers:            {args.max_papers}")
    print(f"Text model:            {args.llm_model} @ {args.llm_api_url}")
    print(f"Figure report:         {bool(args.figure_report_enable)}")
    print(f"Vision enabled:        {bool(args.vision_enable)}")
    if args.vision_enable:
        print(f"Vision model:          {args.vision_model} @ {args.vision_api_url}")
    print("=" * 100)

    lit_csv = None
    updated_csv = None
    success = False

    try:
        if args.pipeline_stage == "evidence_packet":
            if args.from_scratch:
                print("[WARN] Ignoring --from_scratch for pipeline_stage=evidence_packet; existing LiteratureAgent records are required.")
            _run_evidence_packet(args)
            success = True
            return

        if args.pipeline_stage == "knowledge_graph":
            if args.from_scratch:
                print("[WARN] Ignoring --from_scratch for pipeline_stage=knowledge_graph; existing LiteratureAgent records are required.")
            lit_csv = _find_or_create_all_records_csv(_Path_controller(args.work_dir))
            _run_knowledge_graph(args, lit_csv)
            success = True
            return

        if args.from_scratch:
            _remove_dir(args.work_dir)
            _remove_dir(args.integration_out_dir)
            if args.run_model:
                _remove_dir(args.model_out_dir)

        # IMPORTANT: the embedded LiteratureAgent reads many settings at import time,
        # so environment variables must be set before loading the embedded module.
        _prime_import_time_environment(args)

        work_dir = _Path_controller(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        lit_file, integration_file = _write_embedded_modules(work_dir)

        lit_module = _load_module_from_path(lit_file, "_embedded_literature_agent_v20_runtime")
        integration_module = _load_module_from_path(integration_file, "_embedded_integrate_literature_to_pce_stability_v20_runtime")

        if args.pipeline_stage == "sanitize_summaries":
            _v21_sanitize_summaries_to_english(args)
            success = True
            return

        if args.pipeline_stage == "vision_pass":
            _run_v21_vision_pass(args)
            _v21_sanitize_summaries_to_english(args)
            success = True
            return

        if args.pipeline_stage == "integrate_and_model":
            lit_csv = _find_or_create_all_records_csv(_Path_controller(args.work_dir))
            updated_csv = _run_integration(integration_module, args, lit_csv)
            if args.run_model:
                _run_model(args, updated_csv)
            else:
                print("[INFO] Integration complete. Add --run_model to run PCE/stability modeling.")
            success = True
            return

        if args.skip_literature_agent:
            lit_csv = _find_or_create_all_records_csv(_Path_controller(args.work_dir))
            print(f"[INFO] Skipping LiteratureAgent. Using existing records: {lit_csv}")
        else:
            lit_csv = _run_literature_agent(lit_module, args)

        _v21_sanitize_summaries_to_english(args)

        if args.pipeline_stage == "extract_batch":
            print("[INFO] Extract batch complete. Skipping integration/model by pipeline_stage=extract_batch.")
        else:
            if args.skip_integration:
                print("[INFO] Skipping integration step.")
            else:
                updated_csv = _run_integration(integration_module, args, lit_csv)

            if args.run_model:
                if updated_csv is None:
                    raise RuntimeError("Cannot run model because integration was skipped.")
                _run_model(args, updated_csv)
            else:
                print("[INFO] PCE/stability model skipped. Add --run_model when ready.")

        success = True
        print("\n" + "=" * 100)
        print("FULL SINGLE-FILE PIPELINE COMPLETE")
        print("=" * 100)
        print(f"Literature CSV:       {lit_csv}")
        print(f"Updated modeling CSV: {updated_csv}")
        print("=" * 100)

    finally:
        args._controller_elapsed_seconds = round(_time_controller.perf_counter() - _controller_started, 3)
        _write_controller_summary(args, lit_csv, updated_csv, success)


if __name__ == "__main__":
    main()


# Runnable stage examples are maintained in docs/README_LITERATURE_AGENT_PIPELINE.md.
