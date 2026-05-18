rule create_ie_rasters:
    input:
        training_table=cfg(config["bayes"]["bn_input_csv"]),
        predictions=cfg(config["bayes"]["ie_predictions_csv"]),
        ref_grids=expand(REFERENCE_DIR + "/{region}/ref_grid.tif", region=REGIONS),
    output:
        maps=expand(cfg(config["results"]["final_maps_dir"]) + "/eicoastal_{region}.tif", region=REGIONS),
        output_table=cfg(config["results"]["training_dir"]) + "/master_features_with_ie.parquet",
    params:
        ref_grid_dir=REFERENCE_DIR,
        output_dir=cfg(config["results"]["final_maps_dir"]),
        normalization_range=config["bayes"].get("normalization_range", "1,5"),
        prediction_column=config["bayes"].get("ie_prediction_column", None),
    shell:
        "python {SCRIPTS_DIR}/features/18_wf_create_ie_raster.py "
        "--training-table {input.training_table} "
        "--predictions {input.predictions} "
        "--ref-grid-dir {params.ref_grid_dir} "
        "--output-dir {params.output_dir} "
        "--output-table {output.output_table} "
        "--normalize-from {params.normalization_range}"