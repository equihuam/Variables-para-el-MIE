# compara valores de estructuras costeras

library(dplyr)
library(readr)
library(arrow)
library(purrr)

py_out <- read_parquet(
  "C:/wf-ie-data/validar/estructuras_costeras/region_1_v1A.parquet"
)

r_out <- read_csv(
  "C:/wf-ie-data/validar/estructuras_costeras/region_1_R.csv",
  show_col_types = FALSE
)

vars <- c("escollera", "espigon", "muro", "rompeolas", "puerto")

py_key <- py_out %>%
  mutate(
    x_key = round(x, 8),
    y_key = round(y, 8)
  )

r_key <- r_out %>%
  mutate(
    x_key = round(x, 8),
    y_key = round(y, 8)
  )

joined <- inner_join(
  r_key,
  py_key,
  by = c("regionid", "x_key", "y_key"),
  suffix = c("_R", "_Py")
)

grid_summary <- tibble(
  n_R = nrow(r_key),
  n_Py = nrow(py_key),
  n_common = nrow(joined)
)

print(grid_summary)

summary_values <- map_dfr(vars, function(v) {
  r_col <- paste0(v, "_R")
  py_col <- paste0(v, "_Py")
  
  diff <- joined[[py_col]] - joined[[r_col]]
  abs_diff <- abs(diff)
  
  tibble(
    variable = v,
    n = sum(complete.cases(joined[[r_col]], joined[[py_col]])),
    R_min = min(joined[[r_col]], na.rm = TRUE),
    R_max = max(joined[[r_col]], na.rm = TRUE),
    Py_min = min(joined[[py_col]], na.rm = TRUE),
    Py_max = max(joined[[py_col]], na.rm = TRUE),
    mae = mean(abs_diff, na.rm = TRUE),
    rmse = sqrt(mean(diff^2, na.rm = TRUE)),
    max_abs_diff = max(abs_diff, na.rm = TRUE),
    cor = cor(joined[[r_col]], joined[[py_col]], use = "complete.obs")
  )
})

print(summary_values)

write_csv(
  summary_values,
  "C:/wf-ie-data/validar/estructuras_costeras/estructuras_region_1_values_summary.csv"
)