library(terra)
library(dplyr)
library(readr)
library(kknn)

ref_grid <- "C:/wf-ie-data/results/reference/region_1/ref_grid.tif"
zvh_file <- "C:/wf-ie-data/varsIni/zonas_de_vida/zvh_mx3gw.tif"
manglares_shp <- "C:/wf-ie-data/varsIni/manglares/cm-conabio.shp"

manglares <- vect(manglares_shp)

zvh <- rast(zvh_file)
zvh <- project(zvh, y = crs(manglares), method = "near")

region_1 <- rast(ref_grid)
region_1_pr <- project(region_1, y = crs(manglares), method = "near")

region_points <- as.data.frame(region_1_pr, xy = TRUE, na.rm = TRUE)
region_points$pixid <- seq_len(nrow(region_points))
region_points$regionid <- "region_1"

zvh_crop <- crop(zvh, region_1_pr)
zvh_points <- as.data.frame(zvh_crop, xy = TRUE, na.rm = TRUE)

# La tercera columna puede llamarse layer o tener otro nombre.
names(zvh_points)[3] <- "layer"

cat("filas region_points:", nrow(region_points), "\n")
cat("filas zvh_points:", nrow(zvh_points), "\n")
print(table(zvh_points$layer))

zvh_points$layer <- as.factor(zvh_points$layer)

modelkknn <- kknn(
  layer ~ x + y,
  train = zvh_points,
  test = region_points,
  distance = 2,
  k = 1,
  kernel = "rectangular"
)

out_R <- region_points %>%
  mutate(
    zvh = as.character(modelkknn$fitted.values)
  ) %>%
  select(regionid, pixid, x, y, zvh)

dir.create(
  "C:/wf-ie-data/validar/zvh",
  recursive = TRUE,
  showWarnings = FALSE
)

write_csv(
  out_R,
  "C:/wf-ie-data/validar/zvh/region_1_R.csv"
)

write_csv(
  zvh_points,
  "C:/wf-ie-data/validar/zvh/zvh_region_1_points_R.csv"
)

cat("OK -> C:/wf-ie-data/validar/zvh/region_1_R.csv\n")
cat("filas:", nrow(out_R), "\n")
print(table(out_R$zvh))