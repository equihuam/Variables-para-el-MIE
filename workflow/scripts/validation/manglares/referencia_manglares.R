library(terra)
library(dplyr)
library(readr)
library(kknn)

ref_grid <- "../../../../../../../../wf-ie-data/results/reference/region_1/ref_grid.tif"
manglares_shp <- "../../../../../../../../wf-ie-data/varsIni/manglares/cm-conabio.shp"

manglares <- vect(manglares_shp)

region_1 <- rast(ref_grid)
region_1_pr <- project(region_1, y = crs(manglares), method = "near")

# Grilla válida regional
region_points <- as.data.frame(region_1_pr, xy = TRUE, na.rm = TRUE)
region_points$pixid <- seq_len(nrow(region_points))
region_points$regionid <- "region_1"

# Rasterizar manglares sobre la grilla regional
manglares_rast <- rasterize(manglares, region_1_pr)

manglares_points <- as.data.frame(manglares_rast, xy = TRUE, na.rm = FALSE)
names(manglares_points)[3] <- "manglar_raw"

# Grilla completa para saber qué celdas pertenecen a la región
region_df_all <- as.data.frame(region_1_pr, xy = TRUE, na.rm = FALSE)
names(region_df_all)[3] <- "ref_value"

manglares_points$ref_value <- region_df_all$ref_value

manglares_train <- manglares_points %>%
  filter(!is.na(ref_value)) %>%
  mutate(
    manglar = if_else(is.na(manglar_raw), 0, 1)
  ) %>%
  select(x, y, manglar)

cat("validos region reproyectada:\n")
print(global(region_1_pr, fun = function(x) sum(!is.na(x))))

cat("celdas manglar:\n")
print(sum(manglares_train$manglar == 1))

cat("celdas no manglar:\n")
print(sum(manglares_train$manglar == 0))

# kknn binomial equivalente
manglares_train$manglar <- as.factor(manglares_train$manglar)

model <- kknn(
  manglar ~ x + y,
  train = manglares_train,
  test = region_points,
  distance = 2,
  k = 30,
  kernel = "optimal"
)

# Probabilidad de clase "1"
prob <- as.data.frame(model$prob)

print(names(prob))
print(head(prob))

if ("1" %in% names(prob)) {
  p_manglares <- prob[["1"]]
} else if ("X1" %in% names(prob)) {
  p_manglares <- prob[["X1"]]
} else {
  stop("No encuentro la columna de probabilidad para la clase 1 en model$prob")
}

out_R <- region_points %>%
  mutate(
    p_manglares = as.numeric(p_manglares)
  ) %>%
  select(regionid, pixid, x, y, p_manglares)

dir.create(
  "C:/wf-ie-data/validar/manglares",
  recursive = TRUE,
  showWarnings = FALSE
)

write_csv(
  out_R,
  "C:/wf-ie-data/validar/manglares/region_1_R.csv"
)

write_csv(
  manglares_train,
  "C:/wf-ie-data/validar/manglares/manglares_region_1_train_R.csv"
)

cat("OK -> C:/wf-ie-data/validar/manglares/region_1_R.csv\n")
cat("filas:", nrow(out_R), "\n")
cat("p_min:", min(out_R$p_manglares, na.rm = TRUE), "\n")
cat("p_max:", max(out_R$p_manglares, na.rm = TRUE), "\n")
cat("p_mean:", mean(out_R$p_manglares, na.rm = TRUE), "\n")