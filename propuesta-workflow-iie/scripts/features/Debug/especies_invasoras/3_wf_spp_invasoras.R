library(terra)
library(dplyr)
library(readr)
library(kknn)

ref_grid <- "C:/wf-ie-data/results/reference/region_1/ref_grid.tif"
species_csv <- "C:/wf-ie-data/varsIni/plantas_snib/plantas_invasoras.csv"

normalize <- function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}

sp_inv <- read.csv(species_csv)

# Replicar el script R: columnas 12 y 13 como x/y
names(sp_inv)[12] <- "x"
names(sp_inv)[13] <- "y"

region_1 <- rast(ref_grid)
region_1_pr <- project(region_1, "EPSG:4326", method = "near")

region_points <- as.data.frame(region_1_pr, xy = TRUE, na.rm = TRUE)
region_points$pixid <- seq_len(nrow(region_points))
region_points$regionid <- "region_1"

unique_inv <- unique(sp_inv$especievalida)

wide_R <- region_points %>%
  select(regionid, pixid, x, y)

for (sp in unique_inv) {
  sp_inv_f <- sp_inv[sp_inv$especievalida == sp, ]

  sp_inv_f$part <- seq_len(nrow(sp_inv_f))

  modelkknn <- kknn(
    part ~ x + y,
    train = sp_inv_f,
    test = region_points,
    distance = 2,
    k = 1,
    kernel = "rectangular"
  )

  sp_col <- make.names(sp)
  dist_col <- paste0("dist__", sp_col)
  score_col <- paste0("score__", sp_col)

  wide_R[[dist_col]] <- as.numeric(modelkknn$D)
  wide_R[[score_col]] <- 1 - normalize(wide_R[[dist_col]])
}

score_cols <- grep("^score__", names(wide_R), value = TRUE)

out_R <- wide_R %>%
  mutate(
    sp_inv_pot = rowSums(across(all_of(score_cols)), na.rm = TRUE)
  ) %>%
  select(regionid, pixid, x, y, sp_inv_pot)

dir.create(
  "C:/wf-ie-data/validar/spp_invasoras",
  recursive = TRUE,
  showWarnings = FALSE
)

write_csv(
  out_R,
  "C:/wf-ie-data/validar/spp_invasoras/region_1_R_regionnorm.csv"
)

write_csv(
  wide_R,
  "C:/wf-ie-data/validar/spp_invasoras/region_1_R_regionnorm_wide.csv"
)

cat("OK -> C:/wf-ie-data/validar/spp_invasoras/region_1_R_regionnorm.csv\n")
cat("filas:", nrow(out_R), "\n")
cat("especies:", length(unique_inv), "\n")