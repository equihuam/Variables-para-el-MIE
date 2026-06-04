library(terra)
library(dplyr)
library(here)
library(readr)
library(kknn)

# si quieres poner las paths como variables pon todo al principio

# Tifs.
base_dir <- here()

tifs <- list.files(here("data", "alineadas_regunidas"), pattern="\\.tif$", full.names=TRUE, ignore.case=TRUE)

# Mallas.
mallas <- list.files(here("data", "malla_reg_unidas"), 
                     pattern = "\\.tif$", full.names = TRUE,recursive = TRUE)

# PROCESAR REGIÓN POR REGIÓN (GUARDA CSV).
final_data_list <- list()

for (m in seq_along(mallas)) {

  malla_rast <- rast(mallas[m])
  
  tifs_malla_m_list <- list()
  xy_df <- NULL
  
  for (t in seq_along(tifs)) {
    tif_rast <- rast(tifs[t])
    tifs
    # Crop to malla.
    tif_rast_cropped <- crop(tif_rast, malla_rast)
    
    # Stack variable with 
    tif_rast_cropped <- c(tif_rast_cropped, malla_rast)
    
    # To data frame
    df <- as.data.frame(tif_rast_cropped, xy = TRUE)
    df <- df[!is.na(df$OID_1), ]
    
    # guardar x,y solo una vez (para esa región)
    if (is.null(xy_df)) {
      xy_df <- df[, c("x", "y")]
      malla_id <- df$OID_1[1]
    }
    
    # guardar variable (columna 3 = variable)
    tifs_malla_m_list[[t]] <- df[, 3]
  }
  
  names(tifs_malla_m_list) <- tools::file_path_sans_ext(basename(tifs))
  
  tif_df <- cbind(xy_df, as.data.frame(tifs_malla_m_list))
  tif_df$mallaid <- malla_id
  
  final_data_list[[m]] <- tif_df
  gc()
}


# Bind zones into single data frame.
final_data <- dplyr::bind_rows(final_data_list)

# Save to disk.
write_csv(final_data, "c:/Users/Octavio/Dropbox/variables_ie_c/tablas_finales/final_data_v3.csv")

