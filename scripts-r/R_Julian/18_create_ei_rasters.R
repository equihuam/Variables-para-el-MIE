#library("terra")
#library("data.table")
#library("dplyr")
#library("raster")   ------ version terra ya lo incluye
pacman::p_load(dplyr, data.table, terra)


dropbox_dir <- "C:/Users/equih/1 Nubes/Dropbox/ei-coastal"
dat_csv <- paste0(dropbox_dir,"/data/cei_final_train_v1ask.csv")

# Estos datos están en coordenadas geográficas
dat <- fread(dat_csv)
dat <- dat[,1:3]
dat$idx <- 1:nrow(dat)

regiones <- unique(dat$regionId)
head(dat)

# Lee datos ei salida Netica
datos_ei_file <- paste0(dropbox_dir, "/BN-results/EII-data/cei_final_ie_expected_port_5_equal_2026.csv")
eipred <- fread(datos_ei_file, data.table = FALSE)
min(as.numeric(unlist(eipred)))
max(as.numeric(unlist(eipred)))

# eipred <- (as.numeric(unlist(eipred))-1.5)/(5.5-1.5)
eipred_norm <- eipred %>%
  mutate(ei_norm = (`E[ei_qnint_map]` - 1.5) / (5.5 - 1.5)) %>%
  select(ei_norm) %>%
  unlist(use.names = FALSE)

# Suponiendo dat y eipred_norm son idénticos en longitudy orden
dat$ei_norm <- eipred_norm

hist(eipred_norm)
# Load corales shapefile.
corales_shp <- paste0(dropbox_dir, "/data/data_crude/08_Corales/coral-global.shp")
corales <- vect(corales_shp)

# List coastal refetence grids.
ref_grids_dir <- paste0(dropbox_dir, "/data/data_crude/DunasCost250116_malla_ref_50m/")
c_list <- list.files(ref_grids_dir,
                     pattern = "\\.tif$",
                     full.names = TRUE,
                     recursive = TRUE)

for (i in 1:length(regiones)){

    region_ <- rast(c_list[i])  # No se vuelve a usar
    
    region <- regiones[i]
    print(region)
    
    region_dat <- dat[dat$regionId==region,]

    ei_df <- data.frame(x=region_dat$x, y=region_dat$y, z=eipred_norm[region_dat$idx])
    #ei_df <- vect(ei_df, geom=c("x", "y"), crs=crs(corales), keepgeom=FALSE)
    #ei_df <- terra::project(x = ei_df, y = region_)
    
    hist(ei_df$z)

    #ei_rast <- rasterize(ei_df, region_, field = "z")

    ei_rast <- rast(ei_df, type="xyz", digits = 5)
    crs(ei_rast) <- crs(corales)
    ei_map_dir <- paste0(dropbox_dir, "/BN-results/BN_maps/cei_final_ie_expected_port_5_equal_2026/R/")
    output <- paste0(ei_map_dir, "eicoastal_", region,".tif")
    writeRaster(filename = output, x = ei_rast, overwrite=TRUE)
}

