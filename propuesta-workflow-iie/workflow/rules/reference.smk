rule reference_grid:
    input:
        dunes_inegi=lambda wc: cfg(config["inputs"]["dunes_inegi_raster"]),
        dunes_other=lambda wc: cfg(config["inputs"]["dunes_other_shp"]),
        coastal_regions=lambda wc: cfg(config["inputs"]["coastal_regions_shp"]),
    output:
        f"{REFERENCE_DIR}" + "/{region}/ref_grid.tif"
    shell:
        """
        python ../scripts/features/1_wf_create_reference_grid.py \
          --dunes-inegi {input.dunes_inegi} \
          --dunes-other {input.dunes_other} \
          --coastal-regions {input.coastal_regions} \
          --region-id {wildcards.region} \
          --output {output}
        """