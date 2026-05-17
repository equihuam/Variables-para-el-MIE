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
    "condicion_dunas"
]

rule export_feature_geopackage:
    input:
        lambda wc: expand(f"{FEATURES_DIR}/{wc.feature}/{{region}}.parquet", region=REGIONS)
    output:
        f"{VECTOR_EXPORTS_DIR}" + "/{feature}.gpkg"
    params:
        feature_dir=lambda wc: f"{FEATURES_DIR}/{wc.feature}"
    shell:
        "python ../scripts/features/16_wf_export_feature_geopackage.py \
              --feature-dir {params.feature_dir} \
              --output {output}"