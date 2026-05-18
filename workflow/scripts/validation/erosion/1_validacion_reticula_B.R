# Compara distancia entre coordenadas de centros de celdas 

library(FNN)

xy_diff <- read_csv(
  "C:/wf-ie-data/validar/tasa_erosion/tasa_erosion_region_1_xy_diff.csv",
  show_col_types = FALSE
)

only_r <- xy_diff %>%
  filter(match_status == "only_R") %>%
  select(regionid, pixid_R, x_R, y_R)

only_py <- xy_diff %>%
  filter(match_status == "only_Python") %>%
  select(regionid, pixid_Py, x_Py, y_Py)

nn <- get.knnx(
  data = as.matrix(only_py[, c("x_Py", "y_Py")]),
  query = as.matrix(only_r[, c("x_R", "y_R")]),
  k = 1
)

nearest_diff <- only_r %>%
  mutate(
    nearest_py_index = nn$nn.index[, 1],
    nearest_distance_deg = nn$nn.dist[, 1],
    nearest_pixid_Py = only_py$pixid_Py[nearest_py_index],
    nearest_x_Py = only_py$x_Py[nearest_py_index],
    nearest_y_Py = only_py$y_Py[nearest_py_index],
    dx = x_R - nearest_x_Py,
    dy = y_R - nearest_y_Py
  )

summary(nearest_diff$nearest_distance_deg)

write_csv(
  nearest_diff,
  "C:/wf-ie-data/validar/tasa_erosion/tasa_erosion_region_1_xy_nearest_diff.csv"
)