IE_VALIDATION_DIR = cfg(config["validation"]["ie_maps_dir"])
IE_REFERENCE_MAP_DIR = cfg(config["validation"]["ie_reference_map_dir"])
IE_BASE_MAP_DIR = cfg(config["validation"]["ie_base_map_dir"])
IE_ALTERNA_MAP_DIR = cfg(config["validation"].get("ie_alterna_map_dir", ""))

rule validate_ie_base_vs_reference:
    input:
        base_maps=expand(IE_BASE_MAP_DIR + "/eicoastal_{region}.tif", region=REGIONS),
        reference_maps=expand(IE_REFERENCE_MAP_DIR + "/{region}.tif", region=REGIONS),
    output:
        IE_VALIDATION_DIR + "/ie_base_vs_ei_qnint_summary.csv"
    params:
        base_dir=IE_BASE_MAP_DIR,
        reference_dir=IE_REFERENCE_MAP_DIR,
        normalization_range=config["bayes"].get("normalization_range", "1,5"),
    shell:
        "python {SCRIPTS_DIR}/validation/0_ie_maps/compare_ie_maps_to_reference.py "
        "--candidate-dir {params.base_dir} "
        "--candidate-pattern \"eicoastal_{{region}}.tif\" "
        "--reference-dir {params.reference_dir} "
        "--reference-pattern \"{{region}}.tif\" "
        "--reference-normalize-from {params.normalization_range} "
        "--output {output}"


rule validate_ie_base_alterna_vs_reference:
    input:
        base_maps=expand(IE_BASE_MAP_DIR + "/eicoastal_{region}.tif", region=REGIONS),
        alterna_maps=expand(IE_ALTERNA_MAP_DIR + "/eicoastal_{region}.tif", region=REGIONS),
        reference_maps=expand(IE_REFERENCE_MAP_DIR + "/{region}.tif", region=REGIONS),
    output:
        IE_VALIDATION_DIR + "/ie_base_alterna_vs_ei_qnint_summary.csv"
    params:
        base_dir=IE_BASE_MAP_DIR,
        alterna_dir=IE_ALTERNA_MAP_DIR,
        reference_dir=IE_REFERENCE_MAP_DIR,
        normalization_range=config["bayes"].get("normalization_range", "1,5"),
    shell:
        "python {SCRIPTS_DIR}/validation/0_ie_maps/compare_ie_maps_to_reference.py "
        "--candidate-dir {params.base_dir} "
        "--candidate-pattern \"eicoastal_{{region}}.tif\" "
        "--baseline-dir {params.alterna_dir} "
        "--baseline-pattern \"eicoastal_{{region}}.tif\" "
        "--reference-dir {params.reference_dir} "
        "--reference-pattern \"{{region}}.tif\" "
        "--reference-normalize-from {params.normalization_range} "
        "--output {output}"