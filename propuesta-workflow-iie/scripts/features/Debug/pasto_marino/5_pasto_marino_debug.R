library(terra)
library(dplyr)
library(readr)
library(kknn)

ref_grid <- "C:/wf-ie-data/results/reference/region_1/ref_grid.tif"
pasto_shp <- "C:/wf-ie-data/varsIni/pastos_marinos/seagrasses-pol-simar.shp"

grass <- vect(pasto_shp)

region_1 <- rast(ref_grid)
region_1_pr <- project(region_1, y = crs(grass), method = "near")

region_points <- as.data.frame(region_1_pr, xy = TRUE, na.rm = TRUE)
region_points$pixid <- seq_len(nrow(region_points))
region_points$regionid <- "region_1"
region_points$grass <- 999

grass_rast <- rasterize(grass, region_1_pr)

grass_points <- as.data.frame(grass_rast, xy = TRUE, na.rm = TRUE)

cat("validos region reproyectada:\n")
print(global(region_1_pr, fun = function(x) sum(!is.na(x))))

cat("celdas pasto rasterizadas:\n")
print(global(grass_rast, fun = function(x) sum(!is.na(x))))

cat("filas grass_points:\n")
print(nrow(grass_points))

if (nrow(grass_points) > 0) {
  grass_points$part <- 1
  
  if (nrow(grass_points) == 1) {
    grass_points <- grass_points[c(1, 1), ]
  }
  
  modelkknn <- kknn(
    part ~ x + y,
    train = grass_points,
    test = region_points,
    distance = 2,
    k = 1,
    kernel = "rectangular"
  )
  
  region_points$grass <- as.numeric(modelkknn$D)
}

out_R <- region_points %>%
  transmute(
    regionid,
    pixid,
    x,
    y,
    d_pastosmarinos = grass
  )

dir.create(
  "C:/wf-ie-data/validar/pasto_marino",
  recursive = TRUE,
  showWarnings = FALSE
)

write_csv(
  out_R,
  "C:/wf-ie-data/validar/pasto_marino/region_1_R.csv"
)

write_csv(
  grass_points,
  "C:/wf-ie-data/validar/pasto_marino/pasto_region_1_points_R.csv"
)

cat("OK -> C:/wf-ie-data/validar/pasto_marino/region_1_R.csv\n")
cat("filas:", nrow(out_R), "\n")