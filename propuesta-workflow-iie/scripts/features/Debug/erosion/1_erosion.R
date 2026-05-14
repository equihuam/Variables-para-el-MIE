library("terra")
library("sf")
library("kknn")

# Load tasa de erosion
tasa_ero <- read.csv("C:/wf-ie-data/varsIni/erosion/Tasas_erosionMEX_Actualizado2018.txt",
                     sep = ",", header = TRUE)

names(tasa_ero)[2] <- "x"
names(tasa_ero)[3] <- "y"

tasa_ero_spat <- st_as_sf(tasa_ero, coords = c("x", "y"))

st_crs(tasa_ero_spat) <- 4326

# List coastal refetence grids.
c_list <- list.files("C:/wf-ie-data/results/reference",
                     pattern = "\\.tif$",
                     full.names = TRUE,
                     recursive = TRUE)

df_list <- list()
counter <- 0
for (region in c_list){

  print(region)

  region_ <- rast(region)

  region_ <- project(region_, y  = crs(tasa_ero_spat), method = "near")

  region_id <- strsplit(region, split = "/")[[1]][4]

  region_points <- as.data.frame(region_, xy = TRUE)

  erokknn <- kknn(Tasa~x+y, tasa_ero, region_points, distance = 2, k=3,
                  kernel = "optimal")

  predictions <- erokknn$fitted.values

  counter = counter+1
  region_points$pixid <- 1:nrow(region_points)
  region_points$regionid <- region_id
  region_points$erosion <- predictions

  df_list[[counter]] <- region_points
}

full_df <- dplyr::bind_rows(df_list)

saveRDS(full_df,"C:/wf-ie-data/validar/tasa_erosion/1_tasa_erosion.rds")

# Sección de diagnóstico

dir.create("C:/wf-ie-data/validar/tasa_erosion/", recursive = TRUE, showWarnings = FALSE)

region_points_debug <- region_points
region_points_debug$pixid <- seq_len(nrow(region_points_debug))
region_points_debug$regionid <- "region_1"

write.csv(
  region_points_debug,
  "C:/wf-ie-data/validar/tasa_erosion/tasa_erosion_region_1_grid_R.csv",
  row.names = FALSE
)

metadata_R <- data.frame(
  regionid = "region_1",
  nrow_points = nrow(region_points),
  x_min = min(region_points$x, na.rm = TRUE),
  x_max = max(region_points$x, na.rm = TRUE),
  y_min = min(region_points$y, na.rm = TRUE),
  y_max = max(region_points$y, na.rm = TRUE),
  n_na_ref = sum(is.na(region_points[[3]]))
)

write.csv(
  metadata_R,
  "C:/wf-ie-data/validar/tasa_erosion/tasa_erosion_region_1_metadata_R.csv",
  row.names = FALSE
)
