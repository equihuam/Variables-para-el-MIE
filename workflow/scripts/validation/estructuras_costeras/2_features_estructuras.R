pacman::p_load(terra, dplyr, readr, kknn)

# Insumos
ref_grid <- "c:/wf-ie-data/results/reference/region_1/ref_grid.tif"
structures_shp <- "c:/wf-ie-data/varsIni/estructuras/estructuras_final_unido_.shp"

# Cargar estructuras
struct <- vect(structures_shp)

# Normalización igual a la del script R/Python
struct$Tipo <- dplyr::recode(
  struct$Tipo,
  "Escollera2" = "Escollera",
  "Espigób" = "Espigón",
  "espigón" = "Espigón",
  "Espigón de M" = "Espigón",
  "Muelle" = "Puerto",
  "Rompeolas2" = "Rompeolas",
  .default = struct$Tipo
)

# Cargar y reproyectar ref_grid al CRS de estructuras
region_1 <- rast(ref_grid)
region_1_pr <- project(region_1, crs(struct), method = "near")

# Muy importante: quitar NA
region_points <- as.data.frame(region_1_pr, xy = TRUE, na.rm = TRUE)

region_points$pixid <- seq_len(nrow(region_points))
region_points$regionid <- "region_1"

# Tipos y nombres canónicos
tipos <- c(
  "Escollera" = "escollera",
  "Espigón" = "espigon",
  "Muro" = "muro",
  "Rompeolas" = "rompeolas",
  "Puerto" = "puerto"
)

# Calcular distancia por tipo replicando el script R
for (tipo_original in names(tipos)) {
  tipo_canonico <- tipos[[tipo_original]]

  struct_tipo <- struct[struct$Tipo == tipo_original, ]
  struct_coords <- as.data.frame(geom(struct_tipo))

  # kknn necesita una respuesta dummy
  struct_coords$part <- seq_len(nrow(struct_coords))

  modelkknn <- kknn(
    part ~ x + y,
    train = struct_coords,
    test = region_points,
    distance = 2,
    k = 1,
    kernel = "rectangular"
  )

  region_points[[tipo_canonico]] <- as.numeric(modelkknn$D)
}

# Ordenar columnas como salida Python
out_R <- region_points %>%
  select(
    regionid,
    pixid,
    x,
    y,
    escollera,
    espigon,
    muro,
    rompeolas,
    puerto
  )

dir.create(
  "C:/wf-ie-data/validar/estructuras_costeras",
  recursive = TRUE,
  showWarnings = FALSE
)

write_csv(
  out_R,
  "C:/wf-ie-data/validar/estructuras_costeras/region_1_R.csv"
)

cat("OK -> C:/wf-ie-data/validar/estructuras_costeras/region_1_R.csv\n")
cat("filas:", nrow(out_R), "\n")