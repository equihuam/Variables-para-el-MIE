library(readr)
library(dplyr)
library(here)

# Load data.
variables_ie <- read_csv(here("data", "tablas_finales", "final_data_v5.csv"))

# Verificar tipo de dato
class(variables_ie$zonas_vida_h_aligned)

# Obtener todos los valores únicos
valores_unicos <- variables_ie %>%
  distinct(zonas_vida_h_aligned) %>%
  arrange(zonas_vida_h_aligned)

print(valores_unicos)

# Crear diccionario de correspondencia pa zvh
zonas_dict <- c(
  "4"  = "desierto templado calido",
  "5"  = "desierto subtropical",
  "10" = "matorral desertico",
  "11" = "matorral desertico premontano",
  "12" = "matorral desertico montano bajo",
  "13" = "bosque espinoso",
  "14" = "bosque muy seco",
  "15" = "bosque seco premontano",
  "17" = "bosque subhumedo",
  "18" = "bosque subhumedo premontano",
  "22" = "bosque humedo premontano",
  "26" = "bosque lluvioso",
  "27" = "bosque lluvioso premontano"
)

# Asegurar que la columna sea character para que coincida con el diccionario
variables_ie <- variables_ie %>%
  mutate(
    zonas_vida_h_aligned = as.character(zonas_vida_h_aligned),
    zvh_legend = zonas_dict[zonas_vida_h_aligned]
  )

# Revisar posibles valores sin correspondencia
variables_ie %>%
  filter(is.na(zvh_legend)) %>%
  distinct(zonas_vida_h_aligned)

# Crear diccionario de correspondencia pa tipo de costa
conserv_dict <- c(
  "1"  = "muy bueno",
  "2"  = "bueno",
  "3" = "regular",
  "4" = "malo",
  "5" = "muy malo"
)

# Asegurar que la columna sea character para que coincida con el diccionario
variables_ie <- variables_ie %>%
  mutate(
    dunas_2024_conserv_ed = as.character(dunas_2024_conserv_ed),
    conserv_dun_legend = conserv_dict[dunas_2024_conserv_ed]
  )

# Revisar posibles valores sin correspondencia
variables_ie %>%
  filter(is.na(conserv_dun_legend)) %>%
  distinct(dunas_2024_conserv_ed)


# Guardar nuevo CSV
write_csv(
  variables_ie,
  here("data", "tablas_finales", "final_data_v6.csv"),
  na = ""
)
