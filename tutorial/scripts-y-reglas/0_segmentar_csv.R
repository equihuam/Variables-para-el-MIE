
#=============================================================================
#0_segmentar_csv.R
#-----------------------------------------------------------------------------
#Propósito:
#    Dividir un dataset maestro masivo en subconjuntos regionales. 
#    Facilita el paralelismo del workflow y la distribución de carga.
#
#Rol en el workflow:
#    Pre-procesamiento. Segmentación (Scatter) para procesos independientes.
#=============================================================================


library(readr)
library(dplyr)
library(purrr)

# =============================================================================
# 1. Captura de argumentos Snakemake
# =============================================================================
archivo_entrada <- snakemake@input[["csv"]]
cat("Iniciando segmentación masiva. Leyendo:", archivo_entrada, "\n")

datos_maestros <- read_csv(archivo_entrada, show_col_types = FALSE)

# =============================================================================
# 2. Lógica de segmentación
# =============================================================================
datos_maestros %>%
  group_by(id_region) %>%
  group_walk(~ {
    # Definición de ruta dinámica basada en el grupo
    ruta_salida <- paste0("procesados/datos_", .y$id_region, ".csv")
    
    # Escritura del fragmento
    write_csv(.x, ruta_salida)
    cat(" -> Segmento escrito:", ruta_salida, " | Registros:", nrow(.x), "\n")
  })

cat("¡División completada con éxito!\n")