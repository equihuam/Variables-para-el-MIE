# Compara concordancia de coordenadas de centros de celdas con redondeo

library(dplyr)
library(readr)

py_grid <- read_csv(
  "C:/wf-ie-data/validar/tasa_erosion/tasa_erosion_region_1_grid.csv",
  show_col_types = FALSE
)

r_grid <- read_csv(
  "C:/wf-ie-data/validar/tasa_erosion/tasa_erosion_region_1_grid_R.csv",
  show_col_types = FALSE
)

# Crear llaves redondeadas
py_key <- py_grid %>%
  mutate(
    x_key = round(x, 8),
    y_key = round(y, 8),
    source_py = TRUE
  )

r_key <- r_grid %>%
  mutate(
    x_key = round(x, 8),
    y_key = round(y, 8),
    source_r = TRUE
  )

# Comparación por coordenadas redondeadas
xy_join <- full_join(
  r_key,
  py_key,
  by = c("regionid", "x_key", "y_key"),
  suffix = c("_R", "_Py")
) %>%
  mutate(
    match_status = case_when(
      !is.na(source_r) & !is.na(source_py) ~ "both",
      !is.na(source_r) & is.na(source_py)  ~ "only_R",
      is.na(source_r)  & !is.na(source_py) ~ "only_Python",
      TRUE ~ "unexpected"
    )
  )

# Resumen principal
summary_xy <- xy_join %>%
  count(match_status)

print(summary_xy)

# Guardar comparación completa
write_csv(
  xy_join,
  "C:/wf-ie-data/validar/tasa_erosion/tasa_erosion_region_1_xy_compare.csv"
)

# Guardar sólo discrepancias
xy_diff <- xy_join %>%
  filter(match_status != "both")

write_csv(
  xy_diff,
  "C:/wf-ie-data/validar/tasa_erosion/tasa_erosion_region_1_xy_diff.csv"
)

print(head(xy_diff, 20))