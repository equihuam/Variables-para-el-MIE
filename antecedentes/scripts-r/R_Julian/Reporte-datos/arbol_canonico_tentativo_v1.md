# Árbol canónico tentativo

```text
proyecto/
├── data_raw/
│   ├── vectores_base/
│   ├── rasters_base/
│   ├── tablas_base/
│   └── externos_modelado/
├── data_intermediate/
│   ├── grids/
│   │   └── ref_grid_50m_regional/
│   ├── features/
│   │   ├── por_script/
│   │   └── consolidadas/
│   ├── training_tables/
│   └── model_inputs/
├── outputs/
│   ├── maps/
│   │   ├── ei/
│   │   └── otras_salidas_raster/
│   ├── tables/
│   └── reports/
├── scripts/
│   ├── prep/
│   ├── features/
│   ├── training/
│   ├── inference/
│   └── utils/
└── metadata/
    ├── manifests/
    ├── dictionaries/
    └── workflow/
```

## Alias actuales que conviene validar
- `data/06_DunasCost250116_malla_ref_50m` → familia troncal de grillas/regiones.
- `data/7_ref_grid_50m` y `data/8_ref_grid_50m` → probable misma familia operativa con inconsistencia de nombre.
- `data_features` → features derivadas consumidas por etapas posteriores.
- `data_training_tables` → tablas de entrenamiento/modelado.
- `BN_maps/.../R` → salidas raster/mapas.
