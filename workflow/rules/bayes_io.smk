rule prepare_bn_table:
    input:
        f"{TRAINING_DIR}/master_features.parquet"
    output:
        parquet=cfg(config["bayes"]["bn_input_parquet"]),
        csv=cfg(config["bayes"]["bn_input_csv"]),
    shell:
        """
        python {SCRIPTS_DIR}/features/15_wf_prepare_bn_table.py \
          --input {input} \
          --output-parquet {output.parquet} \
          --output-csv {output.csv}
        """