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

rule feature_tipo_costa:
    input:
        coast_types_shp=lambda wc: cfg(config["inputs"]["coast_types_shp"]),
        ref_grid=REFERENCE_DIR + "/{region}/ref_grid.tif",
    output:
        FEATURES_DIR + "/tipo_costa/{region}.parquet"
    shell:
        "python ../scripts/features/11_wf_tipo_costa.py \
            --coast-types-shp {input.coast_types_shp} \
            --ref-grid {input.ref_grid} \
            --region-id {wildcards.region} \
            --output {output}"

rule feature_zvh:
    input:
        mangroves_shp=lambda wc: cfg(config["inputs"]["mangroves_shp"]),
        zvh_raster=lambda wc: cfg(config["inputs"]["zvh_raster"]),
        ref_grid=REFERENCE_DIR + "/{region}/ref_grid.tif",
    output:
        FEATURES_DIR + "/zvh/{region}.parquet"
    shell:
        "python ../scripts/features/13_wf_z_v_holdridge.py \
            --mangroves-shp {input.mangroves_shp} \
            --zvh-raster {input.zvh_raster} \
            --ref-grid {input.ref_grid} \
            --region-id {wildcards.region} \
            --output {output}"

rule feature_wind_speed:
    input:
        estructuras_shp=lambda wc: cfg(config["inputs"]["estructuras_shp"]),
        wind_nc=lambda wc: cfg(config["inputs"]["wind_nc"]),
        ref_grid=REFERENCE_DIR + "/{region}/ref_grid.tif",
    output:
        FEATURES_DIR + "/velocidad_del_viento/{region}.parquet"
    shell:
        "python ../scripts/features/12_wf_wind_speed.py \
            --structures-shp {input.estructuras_shp} \
            --wind-nc {input.wind_nc} --ref-grid {input.ref_grid} \
            --region-id {wildcards.region} --output {output}"

rule feature_estructuras_costeras:
    input:
        estructuras_shp=lambda wc: cfg(config["inputs"]["estructuras_shp"]),
        ref_grid=REFERENCE_DIR + "/{region}/ref_grid.tif",
    output:
        FEATURES_DIR + "/estructuras_costeras/{region}.parquet"
    shell:
        "python ../scripts/features/2_wf_estructuras_costeras.py \
            --structures-shp {input.estructuras_shp} \
            --ref-grid {input.ref_grid} \
            --region-id {wildcards.region} \
            --output {output}"

rule feature_spp_invasoras:
    input:
        species_points_csv=lambda wc: cfg(config["inputs"]["species_points_csv"]),
        ref_grid=REFERENCE_DIR + "/{region}/ref_grid.tif",
    output:
        FEATURES_DIR + "/spp_invasoras/{region}.parquet"
    shell:
        "python ../scripts/features/3_wf_spp_invasoras.py \
            --species-points-csv {input.species_points_csv} \
            --ref-grid {input.ref_grid} \
            --region-id {wildcards.region} \
            --output {output}"

rule feature_pasto_marino:
    input:
        pasto_marino_shp=lambda wc: cfg(config["inputs"]["pasto_marino_shp"]),
        ref_grid=REFERENCE_DIR + "/{region}/ref_grid.tif",
    output:
        FEATURES_DIR + "/pasto_marino/{region}.parquet"
    shell:
        "python ../scripts/features/5_wf_pasto_marino.py \
            --pasto-marino-shp {input.pasto_marino_shp} \
            --ref-grid {input.ref_grid} \
            --region-id {wildcards.region} \
            --output {output}"