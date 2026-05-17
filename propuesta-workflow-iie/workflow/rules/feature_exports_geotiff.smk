FEATURE_EXPORT_VARS = [
    "tasa_erosion",
    "corales",
    "tipo_costa",
    "zvh",
    "velocidad_del_viento",
    "estructuras_costeras",
    "spp_invasoras",
    "pasto_marino",
    "batimetria",
    "madmex_uso_suelo",
    "manglares",
    "movimiento_dunas",
    "condicion_dunas",
]

rule export_feature_geotiff:
    input:
        feature_tables=lambda wc: expand(
            f"{FEATURES_DIR}/{wc.feature}/{{region}}.parquet",
            region=REGIONS,
        ),
        ref_grids=expand(REFERENCE_DIR + "/{region}/ref_grid.tif", region=REGIONS),
    output:
        directory(FEATURE_GEOTIFFS_DIR + "/{feature}")
    params:
        feature_dir=lambda wc: f"{FEATURES_DIR}/{wc.feature}",
        ref_grid_dir=REFERENCE_DIR,
    shell:
        "python ../scripts/features/16_wf_export_feature_geotiff.py "
        "--feature-dir {params.feature_dir} "
        "--ref-grid-dir {params.ref_grid_dir} "
        "--output-dir {output}"
