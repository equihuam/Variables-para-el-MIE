# El script R tiene una aparente errata:
# en los bloques de agricultura y urbano asigna las distancias a region_points$grassland,
# no a agriculture ni urban. Para la versión Python conviene reproducir
# la intención funcional: generar d_grassland, d_agriculture y d_urban
# consistentemente y por separado.


library(terra)
library(dplyr)
library(readr)
library(arrow)
library(kknn)

madmex_file <- "C:/wf-ie-data/varsIni/madmex/madmex_landsat_2017_31.tif"
base_file <- "C:/wf-ie-data/results/features/tasa_erosion/region_1.parquet"

madmex <- rast(madmex_file)

base <- read_parquet(base_file) %>%
  select(regionid, pixid, x, y)

# Transformar puntos base de EPSG:4326 al CRS MADMEX
base_vect <- vect(
  base,
  geom = c("x", "y"),
  crs = "EPSG:4326",
  keepgeom = TRUE
)

base_madmex <- project(base_vect, crs(madmex))
base_xy <- crds(base_madmex)

region_points <- base %>%
  mutate(
    x_madmex = base_xy[, 1],
    y_madmex = base_xy[, 2]
  )

# Recortar MADMEX a extensión de puntos
ext_region <- ext(
  min(region_points$x_madmex),
  max(region_points$x_madmex),
  min(region_points$y_madmex),
  max(region_points$y_madmex)
)

madmex_crop <- crop(madmex, ext_region)

madmex_points <- as.data.frame(madmex_crop, xy = TRUE, na.rm = TRUE)
names(madmex_points)[3] <- "layer"

madmex_points <- madmex_points %>%
  filter(layer %in% c(27, 28, 29))

calc_dist <- function(class_code) {
  class_points <- madmex_points %>%
    filter(layer == class_code) %>%
    mutate(part = seq_len(n()))

  if (nrow(class_points) == 0) {
    return(rep(9999, nrow(region_points)))
  }

  model <- kknn(
    part ~ x_madmex + y_madmex,
    train = class_points %>% rename(x_madmex = x, y_madmex = y),
    test = region_points,
    distance = 2,
    k = 1,
    kernel = "rectangular"
  )

  as.numeric(model$D)
}

out_R <- base %>%
  mutate(
    d_grassland = calc_dist(27),
    d_agriculture = calc_dist(28),
    d_urban = calc_dist(29)
  )

dir.create(
  "C:/wf-ie-data/validar/madmex_uso_suelo",
  recursive = TRUE,
  showWarnings = FALSE
)

write_csv(
  out_R,
  "C:/wf-ie-data/validar/madmex_uso_suelo/region_1_R.csv"
)

write_csv(
  madmex_points,
  "C:/wf-ie-data/validar/madmex_uso_suelo/madmex_region_1_points_R.csv"
)

cat("OK -> C:/wf-ie-data/validar/madmex_uso_suelo/region_1_R.csv\n")
cat("filas:", nrow(out_R), "\n")
cat("madmex_points:", nrow(madmex_points), "\n")