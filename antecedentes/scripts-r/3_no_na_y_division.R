library(readr)

ruta <- "C:/Users/Octavio/Dropbox/variables_ie_c/tablas_finales/final_data_v.csv"

dat <- read_csv(ruta)

write_csv(
  dat,
  "C:/Users/Octavio/Dropbox/variables_ie_c/tablas_finales/final_data_v3_s_NA.csv"
)

###############DIVIDIR 70/30#############
library(readr)
library(here)

# Ruta al archivo (texto)
ruta <- here("data", "tablas_finales", "final_data_v5.csv")

# Leer datos
dat <- read_csv(ruta, show_col_types = FALSE)

# Semilla (opcional)
set.seed(123)

# Índices 70%
idx_70 <- sample(seq_len(nrow(dat)), size = floor(0.7 * nrow(dat)))

# Subconjuntos
dat_70 <- dat[idx_70, ]
dat_30 <- dat[-idx_70, ]

# Guardar CSVs
write_csv(dat_70,
          here("data", "tablas_finales", "final_data_v5_70.csv")
)

write_csv(dat_30,
          here("data", "tablas_finales", "final_data_v5_30.csv")
)
