library(readr)
library(dplyr)
library(here)
library(terra)
library(kknn)

#load data
variables_ie <- read_csv(here("data", "tablas_finales", "final_data_v7_batimetria.csv"))

# Eliminar la columna previa de batimetría
variables_ie <- variables_ie %>%
  select(-any_of("batimetria_aligned"))

############
# Raster de referencia (para CRS/extent)
r_ref <- rast(here("data", "regiones_unidas", "reg_unidas.tif"))
crs_ref <- crs(r_ref)


# Batimetría AQUI CAMBIO EL NOMBRE POR EL DE BATIMETRIA GEBCO
bathy <- rast(
  here("data", "batimetria_gebco", "GEBCO_compressed.tif")
)

# Alinear CRS 
if (!same.crs(bathy, r_ref)) {
  bathy <- project(bathy, r_ref, method = "bilinear")
}
bathy <- crop(bathy, ext(r_ref))


# 1) Extraer batimetría a puntos (x,y)
pts <- vect(variables_ie[, c("x", "y")], geom = c("x", "y"), crs = crs_ref)

variables_ie$batimetria_gebco <- terra::extract(bathy, pts)[, 2]


# 2) Rellenar NA con pixel más cercano (kNN, k=1)
nas <- which(is.na(variables_ie$batimetria_gebco))

if (length(nas) > 0) {
  train_bathy <- variables_ie %>%
    filter(!is.na(batimetria_gebco)) %>%
    select(x, y, batimetria_gebco)
  
  predict_bathy <- variables_ie %>%
    slice(nas) %>%
    select(x, y)
  
  bathy.kknn <- kknn(
    batimetria_gebco ~ .,
    train_bathy,
    predict_bathy,
    k = 1,
    distance = 2,
    kernel = "optimal"
  )
  
  # Meter los resultados del modelo en la tabla original
  variables_ie$batimetria_gebco[nas] <- bathy.kknn$fitted.values
}


# (Opcional) guardar
write_csv(variables_ie, here("data", "tablas_finales", "final_data_v8_batimetria.csv"), na = "")
