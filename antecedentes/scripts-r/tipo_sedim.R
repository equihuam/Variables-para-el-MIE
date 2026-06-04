#Cargar librerias
library(sf)
library(terra)

#rm(list=ls())

pts <- st_read("../data/tipo_sed/tipo_sed_reproy.shp")

base <- rast("../data/regiones_unidas/reg_unidas.tif")

#Mismo src
pts <- st_transform(pts, crs = st_crs(base))

#aseguro que es numerico
pts$id_estruct <- as.numeric(as.factor(pts$raster))

#Rasterizo los puntos usando la malla como mascara
r_valor <- rasterize(vect(pts), base, field = "raster", mask = TRUE)

#Guardo 
writeRaster(r_valor, "C:.../tipo_sed.tif", overwrite = TRUE)


