#library("terra")
#library("sf")
#library("kknn")

# Load estructuras shapefile.
#struct <- vect("./data_crude/05_InventarioEstructuras/estructuras_final_unido_.shp")

pacman::p_load(terra, dplyr, readr, kknn)

# Insumos
ref_grids <- "c:/wf-ie-data/results/reference/"
structures_shp <- "c:/wf-ie-data/varsIni/estructuras/estructuras_final_unido_.shp"
struct <- vect(structures_shp)


struct$Tipo[struct$Tipo=="Escollera2"] <- "Escollera"
struct$Tipo[struct$Tipo=="Espigób"] <- "Espigón"
struct$Tipo[struct$Tipo=="espigón"] <- "Espigón"
struct$Tipo[struct$Tipo=="Espigón de M"] <- "Espigón"
struct$Tipo[struct$Tipo=="Muelle"] <- "Puerto"
struct$Tipo[struct$Tipo=="Rompeolas2"] <- "Rompeolas"

unique_strus <- unique(struct$Tipo)

# List coastal refetence grids.
#c_list <- list.files("./data/06_DunasCost250116_malla_ref_50m/",
c_list <- list.files(ref_grids,
                     pattern = "\\.tif$",
                     full.names = TRUE,
                     recursive = TRUE)

df_list <- list()
counter = 0
for (region in c_list){
    print(region)

    region_ <- rast(region)
    
    region_ <- project(region_, y  = crs(struct), method = "near")

    region_id <- strsplit(region, split = "/")[[1]][4]
    
    region_points <- as.data.frame(region_, xy = TRUE)

    counter = counter+1
    region_points$pixid <- 1:nrow(region_points)
    region_points$regionid <- region_id
    
    for (estructura in unique_strus){
      estructura <- unique_strus[1]
      struct_tipo <- struct[struct$Tipo==estructura]
      struct_coords <- as.data.frame(geom(struct_tipo))

      modelkknn <- kknn(part~x+y, struct_coords, region_points, distance = 2, k=1,
                   kernel = "rectangular")
      
      distances <- modelkknn$D
      
      region_points[, estructura] <- distances

    }

    df_list[[counter]] <- region_points
}

full_df <- dplyr::bind_rows(df_list)
names(full_df) <- c("x","y","ref_grid","pixid","regionid","Escollera")

reg_1 <- full_df %>% 
  filter((ref_grid == "1") & !is.na(Escollera))

full_df$Escollera[is.null(full_df$Escollera)]

saveRDS(full_df, "./data_features/2_infraestructura.rds")
