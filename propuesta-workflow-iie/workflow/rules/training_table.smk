rule assemble_training_table:
    input:
        erosion=expand(f"{FEATURES_DIR}/tasa_erosion/{{region}}.parquet", region=REGIONS),
        corales=expand(f"{FEATURES_DIR}/corales/{{region}}.parquet", region=REGIONS),
    output:
        f"{TRAINING_DIR}/master_features.parquet"
    shell:
        "C:/QGis_env/python.exe ../scripts/features/14_wf_create_data_table.py --erosion-dir {FEATURES_DIR}/tasa_erosion --corales-dir {FEATURES_DIR}/corales --output {output}"
