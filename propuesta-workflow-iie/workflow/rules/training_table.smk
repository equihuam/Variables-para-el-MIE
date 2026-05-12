rule assemble_training_table:
    input:
        expand(f"{FEATURES_DIR}/tasa_erosion/{{region}}.parquet", region=REGIONS),
        expand(f"{FEATURES_DIR}/corales/{{region}}.parquet", region=REGIONS),
        expand(f"{FEATURES_DIR}/tipo_costa/{{region}}.parquet", region=REGIONS),
        expand(f"{FEATURES_DIR}/zvh/{{region}}.parquet", region=REGIONS),
        expand(f"{FEATURES_DIR}/velocidad_del_viento/{{region}}.parquet", region=REGIONS),
        expand(f"{FEATURES_DIR}/estructuras_costeras/{{region}}.parquet", region=REGIONS),
        expand(f"{FEATURES_DIR}/spp_invasoras/{{region}}.parquet", region=REGIONS),
        expand(f"{FEATURES_DIR}/pasto_marino/{{region}}.parquet", region=REGIONS),
        expand(f"{FEATURES_DIR}/batimetria/{{region}}.parquet", region=REGIONS),
        expand(f"{FEATURES_DIR}/madmex_uso_suelo/{{region}}.parquet", region=REGIONS),
    output:
        f"{TRAINING_DIR}/master_features.parquet"
    params:
        variables=",".join(["tasa_erosion", "corales", "tipo_costa", "zvh", "velocidad_del_viento",
                            "estructuras_costeras", "spp_invasoras", "pasto_marino", "batimetria",
                            "madmex_uso_suelo"]),
        regions=",".join(REGIONS)
    shell:
        "C:/QGis_env/python.exe ../scripts/features/14_wf_create_data_table.py \
              --features-dir {FEATURES_DIR} \
              --variables {params.variables} \
              --regions {params.regions} \
              --output {output}"
