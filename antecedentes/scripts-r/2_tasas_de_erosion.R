library(here)
library(terra)
library(readr)
library(kknn)

# Load data.
variables_ie <- read_csv(here("data", "tablas_finales", "final_data_v3_completa.csv"))

# NAs en erosion.
nas <- is.na(variables_ie$tasaserosion2_aligned)

# Remove huracanes.
variables_ie <- variables_ie[, -(6:10)]

# Entrenamiento para interpolar tasas de erosion.
train_erosion <- variables_ie[!is.na(variables_ie$tasaserosion2_aligned), c("x", "y", "tasaserosion2_aligned")]

predict_erosion <- variables_ie[is.na(variables_ie$tasaserosion2_aligned), c("x", "y", "tasaserosion2_aligned")]

# Si en el modelo quieres rellenar una categoria
# e.g. ZVH o tipo de costa
# tienes que antes declarar la variable factor
# e.g. variables_ie$tipo_costa <- as.factor(variables_ie$tipo_costa)
erosion.kknn <- kknn(tasaserosion2_aligned~., 
                     train_erosion,
                     predict_erosion,
                     k = 1,
                     distance = 2,
                     kernel = "optimal")

# Meter los resultados del modelo en la tabla original
variables_ie$tasaserosion2_aligned[nas] <- erosion.kknn$fitted.values 

r_ref <- rast(here("data", "regiones_unidas", "reg_unidas.tif"))
crs_ref <- crs(r_ref)

r_template <- rast(
  ext(r_ref),
  resolution = res(r_ref),
  crs = crs_ref
)

v <- vect(variables_ie[, c("x", "y", "tasaserosion2_aligned")], geom = c("x", "y"), crs = r_ref)

r_pred <- rasterize(
  v,
  r_template,
  field = "tasaserosion2_aligned",
  fun = "mean"
)

plot(r_pred)

writeRaster(r_pred, here("data", "prueba_erosion_3.tif"), overwrite=TRUE)

write_csv(variables_ie, here("data", "tablas_finales",
                             "final_data_v2.csv"))
