
# =============================================================================
# 1_leer_datos.R
# -----------------------------------------------------------------------------
# Propósito:
#     Limpieza y validación inicial de los datos brutos. Garantiza que 
#     los registros tengan coordenadas válidas antes de la proyección.
# 
# Rol en el workflow:
#     Capa de validación. Filtrado de datos atípicos o incompletos.
# =============================================================================


library(readr)
library(dplyr)

# =============================================================================
# 1. Lectura y Validación
# =============================================================================
datos <- read_csv(snakemake@input[["csv"]], show_col_types = FALSE)

# =============================================================================
# 2. Limpieza de datos (Quality Control)
# =============================================================================
datos_filtrados <- datos %>%
  filter(!is.na(longitud) & !is.na(latitud))

# =============================================================================
# 3. Exportación
# =============================================================================
write_csv(datos_filtrados, snakemake@output[["csv_filtrado"]])
cat("Datos validados correctamente en:", snakemake@output[["csv_filtrado"]], "\n")