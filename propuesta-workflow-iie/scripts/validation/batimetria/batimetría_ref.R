library(terra)
library(dplyr)
library(readr)
library(kknn)

ref_grid <- "../../../../../../../../wf-ie-data/results/reference/region_1/ref_grid.tif"
bat_file <- "../../../../../../../../wf-ie-data/varsIni/batimetria/01_GEBCO2020_SIMAR.tif"

bat <- rast(bat_file)

region_1 <- rast(ref_grid)
region_1_pr <- project(region_1, y = crs(bat), method = "near")

region_points <- as.data.frame(region_1_pr, xy = TRUE, na.rm = TRUE)

bat_crop <- crop(bat, region_1_pr)

bat_points <- as.data.frame(bat_crop, xy = TRUE, na.rm = TRUE)
names(bat_points)[3] <- "bat"

cat("validos region reproyectada:\n")
print(global(region_1_pr, fun = function(x) sum(!is.na(x))))

cat("puntos entrenamiento batimetría:\n")
print(nrow(bat_points))

batkknn <- kknn(
  bat ~ x + y,
  train = bat_points,
  test = region_points,
  distance = 2,
  k = 7,
  kernel = "optimal"
)

out_R <- region_points %>%
  mutate(
    pixid = seq_len(n()),
    regionid = "region_1",
    bati_char = as.numeric(batkknn$fitted.values)
  ) %>%
  select(regionid, pixid, x, y, bati_char)

dir.create(
  "C:/wf-ie-data/validar/batimetria",
  recursive = TRUE,
  showWarnings = FALSE
)

write_csv(
  out_R,
  "C:/wf-ie-data/validar/batimetria/region_1_R.csv"
)

write_csv(
  bat_points,
  "C:/wf-ie-data/validar/batimetria/batimetria_region_1_bat_points_R.csv"
)

cat("OK -> C:/wf-ie-data/validar/batimetria/region_1_R.csv\n")
cat("filas region:", nrow(out_R), "\n")
cat("bat_points:", nrow(bat_points), "\n")