library(terra)
library(dplyr)
library(readr)
library(kknn)

ref_grid <- "C:/wf-ie-data/results/reference/region_1/ref_grid.tif"
coast_shp <- "C:/wf-ie-data/varsIni/tipo_de_costa/TipoCosta.SHP"

costas <- vect(coast_shp)

region_1 <- rast(ref_grid)
region_1_pr <- project(region_1, y = crs(costas), method = "near")

region_points <- as.data.frame(region_1_pr, xy = TRUE, na.rm = TRUE)
region_points$pixid <- seq_len(nrow(region_points))
region_points$regionid <- "region_1"

costas_rast <- rasterize(costas, region_1_pr, field = "TipoCosta")
costas_table <- as.data.frame(costas_rast, xy = TRUE, na.rm = TRUE)

costas_table$TipoCosta <- as.factor(costas_table$TipoCosta)

modelkknn <- kknn(
  TipoCosta ~ x + y,
  costas_table,
  region_points,
  distance = 2,
  k = 1,
  kernel = "rectangular"
)

out_R <- region_points %>%
  mutate(
    tipo_costa = as.character(modelkknn$fitted.values)
  ) %>%
  select(regionid, pixid, x, y, tipo_costa)

dir.create(
  "C:/wf-ie-data/validar/tipo_costa",
  recursive = TRUE,
  showWarnings = FALSE
)

write_csv(
  out_R,
  "C:/wf-ie-data/validar/tipo_costa/region_1_R.csv"
)

write_csv(
  costas_table,
  "C:/wf-ie-data/validar/tipo_costa/tipo_costa_region_1_points_R.csv"
)

cat("OK -> C:/wf-ie-data/validar/tipo_costa/region_1_R.csv\n")
cat("filas:", nrow(out_R), "\n")
print(table(out_R$tipo_costa))