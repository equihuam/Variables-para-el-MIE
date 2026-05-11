rule feature_tasa_erosion:
    input:
        erosion_table=lambda wc: cfg(config["inputs"]["erosion_table"]),
        ref_grid=lambda wc: f"{REFERENCE_DIR}/{wc.region}/ref_grid.tif",
    output:
        FEATURES_DIR + "/tasa_erosion/{region}.parquet"
    shell:
        """
        python ../scripts/features/1_wf_features_tasa_erosion.py \
          --erosion-table {input.erosion_table} \
          --ref-grid {input.ref_grid} \
          --region-id {wildcards.region} \
          --output {output}
        """

rule feature_corales:
    input:
        corals_shp=lambda wc: cfg(config["inputs"]["corals_shp"]),
        ref_grid=lambda wc: f"{REFERENCE_DIR}/{wc.region}/ref_grid.tif",
    output:
        FEATURES_DIR + "/corales/{region}.parquet"
    shell:
        """
        python ../scripts/features/4_wf_corales_global.py \
          --corals-shp {input.corals_shp} \
          --ref-grid {input.ref_grid} \
          --region-id {wildcards.region} \
          --output {output}
        """