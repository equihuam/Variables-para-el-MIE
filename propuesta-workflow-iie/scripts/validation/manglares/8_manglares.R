library("terra")
library("sf")
library("kknn")
library("readxl")
library("stringr")
library("data.table")

setwd("C:/wf-ie-data/")

# Load manglares shapefile.
manglares <- vect("../../../../../../../../wf-ie-data/varsIni/manglares/cm-conabio.shp")
crs(manglares)
# List coastal refetence grids.
c_list <- list.files("../../../../../../../../wf-ie-data/results/reference",
                     pattern = "\\.tif$",
                     full.names = TRUE,
                     recursive = TRUE)

df_list <- list()
counter = 0

for (region in c_list[1]){
  #region <- c_list[1]
    print(region)

    region_ <- rast(region)
    region_ <- project(region_, y  = crs(manglares), method = "near")
    
    region_id <- strsplit(region, split = "/")[[1]][4]
    
    region_points <- as.data.frame(region_, xy = TRUE)
    region_points_na <- as.data.frame(region_, xy = TRUE, na.rm = FALSE)
    
    counter = counter+1
    region_points$pixid <- 1:nrow(region_points)
    region_points$regionid <- region_id
    region_points$manglares <- 0

    manglares_rast <- rasterize(manglares, region_)

    if (sum(!is.nan(values(manglares_rast)))>0){
      manglares_points <- as.data.frame(manglares_rast, xy = TRUE, na.rm = FALSE)
      manglares_points$layer[is.na(manglares_points$layer) & !is.na(region_points_na$OID_1)] <- 0
      table(manglares_points$layer)

      if (nrow(manglares_points)==1){
        manglares_points <- manglares_points[c(1,1),]
      }
      
      manglares_points$layer <- as.factor(manglares_points$layer)
      
      modelkknn <- kknn(layer~x+y, manglares_points, region_points, distance = 2, k=30,
                        kernel = "optimal")

      region_points$manglares <- modelkknn$prob[,2]

    }


    df_list[[counter]] <- region_points
}

full_df <- dplyr::bind_rows(df_list)

hist(full_df$manglares)

saveRDS(full_df, "./data_features/8_manglares.rds")
