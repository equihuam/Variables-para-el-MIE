rule feature_tasa_erosion:
    input:
        erosion_table=lambda wc: cfg(config["inputs"]["erosion_table"]),
        ref_grid=lambda wc: f"{REFERENCE_DIR}/{wc.region}/ref_grid.tif",
    output:
        FEATURES_DIR + "/tasa_erosion/{region}.parquet"
    shell:
        "python {SCRIPTS_DIR}/features/1_wf_features_tasa_erosion.py"
        "  --erosion-table {input.erosion_table} "
        "  --ref-grid {input.ref_grid} "
        "  --region-id {wildcards.region} "
        "  --output {output}"

rule corales_global_stats:
    input:
        corals_shp=lambda wc: cfg(config["inputs"]["corals_shp"]),
        ref_grids=expand(REFERENCE_DIR + "/{region}/ref_grid.tif", region=REGIONS),
    output:
        FEATURES_DIR + "/corales/_global_stats.csv"
    shell:
        "python {SCRIPTS_DIR}/features/4_wf_corales_global_stats.py "
        "--corals-shp {input.corals_shp} "
        "--ref-grids {input.ref_grids} "
        "--output {output}"

rule feature_corales:
    input:
        corals_shp=lambda wc: cfg(config["inputs"]["corals_shp"]),
        ref_grid=REFERENCE_DIR + "/{region}/ref_grid.tif",
        corals_global_stats=FEATURES_DIR + "/corales/_global_stats.csv",
    output:
        FEATURES_DIR + "/corales/{region}.parquet"
    shell:
        "python {SCRIPTS_DIR}/features/4_wf_corales_global.py "
        "--corals-shp {input.corals_shp} "
        "--ref-grid {input.ref_grid} "
        "--region-id {wildcards.region} "
        "--corals-global-stats {input.corals_global_stats} "
        "--sentinel-mode global "
        "--output {output}"

rule feature_tipo_costa:
    input:
        coast_types_shp=lambda wc: cfg(config["inputs"]["coast_types_shp"]),
        ref_grid=REFERENCE_DIR + "/{region}/ref_grid.tif",
    output:
        FEATURES_DIR + "/tipo_costa/{region}.parquet"
    shell:
        "python {SCRIPTS_DIR}/features/11_wf_tipo_costa.py \
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
        "python {SCRIPTS_DIR}/features/13_wf_z_v_holdridge.py \
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
        "python {SCRIPTS_DIR}/features/12_wf_wind_speed.py \
            --structures-shp {input.estructuras_shp} \
            --wind-nc {input.wind_nc} \
            --ref-grid {input.ref_grid} \
            --region-id {wildcards.region} \
            --output {output}"

rule feature_estructuras_costeras:
    input:
        estructuras_shp=lambda wc: cfg(config["inputs"]["estructuras_shp"]),
        ref_grid=REFERENCE_DIR + "/{region}/ref_grid.tif",
    output:
        FEATURES_DIR + "/estructuras_costeras/{region}.parquet"
    shell:
        "python {SCRIPTS_DIR}/features/2_wf_estructuras_costeras.py \
            --structures-shp {input.estructuras_shp} \
            --ref-grid {input.ref_grid} \
            --region-id {wildcards.region} \
            --output {output}"

rule spp_invasoras_global_stats:
    input:
        species_points_csv=lambda wc: cfg(config["inputs"]["species_points_csv"]),
        ref_grids=expand(REFERENCE_DIR + "/{region}/ref_grid.tif", region=REGIONS),
    output:
        FEATURES_DIR + "/spp_invasoras/_global_stats.csv"
    shell:
        "python {SCRIPTS_DIR}/features/3_wf_spp_invasoras_global_stats.py "
        "--species-points-csv {input.species_points_csv} "
        "--ref-grids {input.ref_grids} "
        "--output {output}"

rule feature_spp_invasoras:
    input:
        species_points_csv=lambda wc: cfg(config["inputs"]["species_points_csv"]),
        ref_grid=REFERENCE_DIR + "/{region}/ref_grid.tif",
        normalization_stats=FEATURES_DIR + "/spp_invasoras/_global_stats.csv",
    output:
        FEATURES_DIR + "/spp_invasoras/{region}.parquet"
    shell:
        "python {SCRIPTS_DIR}/features/3_wf_spp_invasoras.py "
        "--species-points-csv {input.species_points_csv} "
        "--ref-grid {input.ref_grid} "
        "--region-id {wildcards.region} "
        "--normalization-stats {input.normalization_stats} "
        "--output {output}"

rule pasto_marino_global_stats:
    input:
        pasto_marino_shp=lambda wc: cfg(config["inputs"]["pasto_marino_shp"]),
        ref_grids=expand(REFERENCE_DIR + "/{region}/ref_grid.tif", region=REGIONS),
    output:
        FEATURES_DIR + "/pasto_marino/_global_stats.csv"
    shell:
        "python {SCRIPTS_DIR}/features/5_wf_pasto_marino_global_stats.py "
        "--pasto-marino-shp {input.pasto_marino_shp} "
        "--ref-grids {input.ref_grids} "
        "--output {output}"

rule feature_pasto_marino:
    input:
        pasto_marino_shp=lambda wc: cfg(config["inputs"]["pasto_marino_shp"]),
        ref_grid=REFERENCE_DIR + "/{region}/ref_grid.tif",
        pasto_global_stats=FEATURES_DIR + "/pasto_marino/_global_stats.csv",
    output:
        FEATURES_DIR + "/pasto_marino/{region}.parquet"
    shell:
        "python {SCRIPTS_DIR}/features/5_wf_pasto_marino.py "
        "--pasto-marino-shp {input.pasto_marino_shp} "
        "--ref-grid {input.ref_grid} "
        "--region-id {wildcards.region} "
        "--pasto-global-stats {input.pasto_global_stats} "
        "--sentinel-mode global "
        "--output {output}"

rule feature_batimetria:
    input:
        batimetria_raster=lambda wc: cfg(config["inputs"]["batimetria_raster"]),
        ref_grid=REFERENCE_DIR + "/{region}/ref_grid.tif",
    output:
        FEATURES_DIR + "/batimetria/{region}.parquet"
    shell:
        "python {SCRIPTS_DIR}/features/6_wf_batimetria_caracteristica.py \
            --batimetria-raster {input.batimetria_raster} \
            --ref-grid {input.ref_grid} \
            --region-id {wildcards.region} \
            --output {output}"

rule feature_madmex_uso_suelo:
    input:
        madmex_raster=lambda wc: cfg(config["inputs"]["madmex_raster"]),
        base_table=FEATURES_DIR + "/tasa_erosion/{region}.parquet",
    output:
        FEATURES_DIR + "/madmex_uso_suelo/{region}.parquet"
    shell:
        "python {SCRIPTS_DIR}/features/7_wf_madmex_uso_suelo_3.py \
            --madmex-raster {input.madmex_raster} \
            --base-table {input.base_table} \
            --output {output}"

rule feature_manglares:
    input:
        mangroves_shp=lambda wc: cfg(config["inputs"]["mangroves_shp"]),
        ref_grid=REFERENCE_DIR + "/{region}/ref_grid.tif",
    output:
        FEATURES_DIR + "/manglares/{region}.parquet"
    shell:
        "python {SCRIPTS_DIR}/features/8_wf_manglares.py "
        "--mangroves-shp {input.mangroves_shp} "
        "--ref-grid {input.ref_grid} "
        "--region-id {wildcards.region} "
        "--output {output}"

rule feature_movimiento_dunas:
    input:
        dunes_shp=lambda wc: cfg(config["inputs"]["dunes_other_shp"]),
        base_table=FEATURES_DIR + "/tasa_erosion/{region}.parquet",
    output:
        FEATURES_DIR + "/movimiento_dunas/{region}.parquet"
    shell:
        "python {SCRIPTS_DIR}/features/10_wf_movimiento_dunas.py \
            --dunes-shp {input.dunes_shp} \
            --base-table {input.base_table} \
            --output {output}"

rule feature_condicion_dunas:
    input:
        dunes_shp=lambda wc: cfg(config["inputs"]["dunes_other_shp"]),
        base_table=FEATURES_DIR + "/tasa_erosion/{region}.parquet",
    output:
        FEATURES_DIR + "/condicion_dunas/{region}.parquet"
    shell:
        "python {SCRIPTS_DIR}/features/9_wf_condicion_dunas.py "
        "--dunes-shp {input.dunes_shp} "
        "--base-table {input.base_table} "
        "--output {output}"