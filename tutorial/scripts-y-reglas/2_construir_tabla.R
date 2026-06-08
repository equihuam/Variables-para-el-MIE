
# =============================================================================
# 2_construir_tabla.R
# -----------------------------------------------------------------------------
# Propósito:
#     Agregación estadística para el resumen de atributos. Genera las métricas
#     necesarias para el modelado regional y reportes de calidad.
# 
# Rol en el workflow:
#     Procesamiento estadístico. Construcción de tablas de resumen.
# =============================================================================


library(readr)
library(dplyr)

# =============================================================================
# 1. Carga de datos validados
# =============================================================================
datos_f <- read_csv(snakemake@input[["csv_filtrado"]], show_col_types = FALSE)

# =============================================================================
# 2. Generación de métricas
# =============================================================================
tabla_atributos <- datos_f %>%
  group_by(id_region, tipo_ecosistema) %>%
  summarise(
    media_integridad = mean(valor_indice, na.rm = TRUE),
    conteo_registros = n(),
    .groups = 'drop'
  )

# =============================================================================
# 3. Exportación
# =============================================================================
write_csv(tabla_atributos, snakemake@output[["tabla_resumen"]])
cat("Resumen de atributos generado en:", snakemake@output[["tabla_resumen"]], "\n")