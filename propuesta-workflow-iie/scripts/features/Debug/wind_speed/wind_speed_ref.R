library(terra)
library(dplyr)
library(readr)

wspeed <- rast("C:/wf-ie-data/varsIni/velocidad_del_viento/wind-speed_monthly-mean_era5_1979-2018_v1.0.nc")
wspeed_mean <- app(wspeed, mean)

struct <- vect("C:/wf-ie-data/varsIni/estructuras/estructuras_final_unido_.shp")

wspeed_reproj <- project(wspeed_mean, y = crs(struct), method = "near")
wspeed_reproj <- crop(wspeed_reproj, struct)

region_1 <- rast("C:/wf-ie-data/results/reference/region_1/ref_grid.tif")
region_1_pr <- project(region_1, y = crs(struct), method = "near")

region_points <- as.data.frame(region_1_pr, xy = TRUE, na.rm = TRUE)
region_points$pixid <- seq_len(nrow(region_points))
region_points$regionid <- "region_1"

xy <- as.matrix(region_points[, c("x", "y")])

extracted <- terra::extract(wspeed_reproj, xy)

print(names(extracted))
print(head(extracted))
print(summary(extracted))

value_col <- names(extracted)[1]

out_R <- region_points %>%
  mutate(
    windspeed = extracted[[value_col]]
  ) %>%
  select(regionid, pixid, x, y, windspeed)

write_csv(
  out_R,
  "C:/wf-ie-data/validar/wind_speed/region_1_R.csv"
)

cat("OK -> C:/wf-ie-data/validar/wind_speed/region_1_R.csv\n")
cat("filas:", nrow(out_R), "\n")
cat("NA windspeed:", sum(is.na(out_R$windspeed)), "\n")
cat("min:", min(out_R$windspeed, na.rm = TRUE), "\n")
cat("max:", max(out_R$windspeed, na.rm = TRUE), "\n")
cat("mean:", mean(out_R$windspeed, na.rm = TRUE), "\n")