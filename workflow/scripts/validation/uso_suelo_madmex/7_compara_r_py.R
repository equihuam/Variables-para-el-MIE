# compara R y Python. Lo hace por regionid + pixid, 
# porque Python conserva la malla de la tabla base

library(dplyr)
library(readr)
library(arrow)
library(purrr)

py_out <- read_parquet(
  "C:/wf-ie-data/validar/madmex_uso_suelo/region_1_v1.parquet"
)

r_out <- read_csv(
  "C:/wf-ie-data/validar/madmex_uso_suelo/region_1_R.csv",
  show_col_types = FALSE
)

joined <- inner_join(
  r_out,
  py_out,
  by = c("regionid", "pixid"),
  suffix = c("_R", "_Py")
)

vars <- c("d_grassland", "d_agriculture", "d_urban")

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

print(tibble(
  n_R = nrow(r_out),
  n_Py = nrow(py_out),
  n_common = nrow(joined)
))

print(summary_values)

write_csv(
  summary_values,
  "C:/wf-ie-data/validar/madmex_uso_suelo/madmex_region_1_values_summary.csv"
)