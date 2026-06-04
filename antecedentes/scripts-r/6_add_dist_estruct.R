library(readr)
library(dplyr)
library(here)
library(terra)

# Cargar datos
variables_ie <- read_csv(
  here("data", "tablas_finales", "final_data_v8_batimetria.csv")
)

df <- variables_ie

# Eliminar la columna anterior

df <- df %>% 
  select(-estruct_dist_dunas_aligned)

# Raster de referencia (CRS, extent y resolución base)
r_ref <- rast(here("data", "regiones_unidas", "reg_unidas.tif"))
pts   <- vect(df, geom = c("x", "y"), crs = crs(r_ref))

# Listar rasters de estructuras

estruct_files <- list.files(
  here("data", "estructuras_separadas", "dist_estruct_compress"),
  pattern = "\\.tif$",
  full.names = TRUE,
  recursive = TRUE
)

print(estruct_files)
print(length(estruct_files))

if (length(estruct_files) == 0) {
  stop("No se encontraron archivos .tif en data/estructuras_separadas/dist_estruct_compress")
}
# Extraer valores de cada raster y crear una columna nueva

for (f in estruct_files) {
  
  r <- rast(f)
  
  # Verificar CRS; 
  if (!same.crs(r, r_ref)) {
    r <- project(r, r_ref, method = "near")
  }
  
  # Extraer valores en los puntos x,y
  vals <- extract(r, pts)[, 2]
  
  # Nombre de columna a partir del nombre del archivo
  colname <- tools::file_path_sans_ext(basename(f))
  
  # Agregar columna al dataframe
  df[[colname]] <- vals
  
  # Revisión rápida
  print(table(is.na(vals)))
}

# Guardar csv
write_csv(
  df,
  here("data", "tablas_finales", "final_data_v9_3.csv")
)