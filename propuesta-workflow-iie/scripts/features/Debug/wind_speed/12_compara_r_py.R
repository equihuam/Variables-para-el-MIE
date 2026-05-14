# Compara r contra Python

library(dplyr)
library(readr)
library(arrow)

py_out <- read_parquet(
  "C:/wf-ie-data/validar/wind_speed/region_1_v1.parquet"
)

r_out <- read_csv(
  "C:/wf-ie-data/validar/wind_speed/region_1_R.csv",
  show_col_types = FALSE
)

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

print(tibble(
  n_R = nrow(r_key),
  n_Py = nrow(py_key),
  n_common = nrow(joined)
))

diff <- joined$windspeed_Py - joined$windspeed_R
abs_diff <- abs(diff)

summary_values <- tibble(
  variable = "windspeed",
  n = sum(complete.cases(joined$windspeed_R, joined$windspeed_Py)),
  R_min = min(joined$windspeed_R, na.rm = TRUE),
  R_max = max(joined$windspeed_R, na.rm = TRUE),
  Py_min = min(joined$windspeed_Py, na.rm = TRUE),
  Py_max = max(joined$windspeed_Py, na.rm = TRUE),
  mae = mean(abs_diff, na.rm = TRUE),
  rmse = sqrt(mean(diff^2, na.rm = TRUE)),
  max_abs_diff = max(abs_diff, na.rm = TRUE),
  cor = cor(joined$windspeed_R, joined$windspeed_Py, use = "complete.obs")
)

print(summary_values)

write_csv(
  summary_values,
  "C:/wf-ie-data/validar/wind_speed/wind_region_1_values_summary.csv"
)


