# Compara valores de celdas equivalentes

library(dplyr)
library(readr)
library(arrow)

py_grid <- read_csv(
  "C:/wf-ie-data/validar/tasa_erosion/tasa_erosion_region_1_grid.csv",
  show_col_types = FALSE
)

r_grid <- read_csv(
  "C:/wf-ie-data/validar/tasa_erosion/tasa_erosion_region_1_grid_R.csv",
  show_col_types = FALSE
)

py_out <- read_parquet(
  "C:/wf-ie-data/validar/tasa_erosion/region_1a.parquet"
)

# Si el grid_R_valid todavía no tiene erosion, usar la tabla R final o agregarla antes.
# Suponiendo que r_grid ya incluye erosion:
py_key <- py_out %>%
  mutate(
    x_key = round(x, 8),
    y_key = round(y, 8)
  )

r_key <- r_grid %>%
  mutate(
    x_key = round(x, 8),
    y_key = round(y, 8)
  )

erosion_compare <- inner_join(
  r_key,
  py_key,
  by = c("regionid", "x_key", "y_key"),
  suffix = c("_R", "_Py")
) %>%
  mutate(
    erosion_diff = erosion_Py - erosion_R,
    abs_erosion_diff = abs(erosion_diff)
  )

summary_stats <- erosion_compare %>%
  summarise(
    n = n(),
    erosion_R_min = min(erosion_R, na.rm = TRUE),
    erosion_R_max = max(erosion_R, na.rm = TRUE),
    erosion_Py_min = min(erosion_Py, na.rm = TRUE),
    erosion_Py_max = max(erosion_Py, na.rm = TRUE),
    mae = mean(abs_erosion_diff, na.rm = TRUE),
    rmse = sqrt(mean(erosion_diff^2, na.rm = TRUE)),
    max_abs_diff = max(abs_erosion_diff, na.rm = TRUE),
    cor = cor(erosion_R, erosion_Py, use = "complete.obs")
  )

print(summary_stats)

write_csv(
  erosion_compare,
  "C:/wf-ie-data/validar/tasa_erosion/tasa_erosion_region_1_erosion_compare.csv"
)

write_csv(
  summary_stats,
  "C:/wf-ie-data/validar/tasa_erosion/tasa_erosion_region_1_erosion_summary.csv"
)