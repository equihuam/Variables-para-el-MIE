rule create_ie_rasters:
    input:
        training_table=cfg(config["bayes"]["bn_input_csv"]),
        predictions=cfg(config["bayes"]["ie_predictions_csv"]),
        ref_grids=expand(f"{REFERENCE_DIR}/{{region}}/ref_grid.tif", region=REGIONS),
    output:
        expand(f"{FINAL_MAPS_DIR}/eicoastal_{{region}}.tif", region=REGIONS),
        f"{TRAINING_DIR}/master_features_with_ie.parquet",
    params:
        prediction_column_arg=(
            "" if config["bayes"].get("ie_prediction_column") in [None, "", "null"]
            else f"--prediction-column {config['bayes']['ie_prediction_column']}"
        ),
        normalization_range=config["bayes"].get("normalization_range", "1.5,5.5"),
    shell:
        r"""
        python ../scripts/features/18_wf_create_ie_raster.py \
          --training-table {input.training_table} \
          --predictions {input.predictions} \
          --ref-grid-dir {REFERENCE_DIR} \
          --output-dir {FINAL_MAPS_DIR} \
          --output-table {TRAINING_DIR}/master_features_with_ie.parquet \
          --normalize-from {params.normalization_range} \
          {params.prediction_column_arg}
        """