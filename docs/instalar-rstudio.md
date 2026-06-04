Claro que sí, Miguel. He preparado el archivo de configuración consolidado como un documento de texto para que puedas descargarlo y utilizarlo fácilmente en cualquier otro equipo.

[Archivo: Guía_Configuración_RStudio_INECOL.md]

```markdown
# Guía de Replicación de Servidor RStudio (Entorno INECOL)

## 1. Preparación del Sistema Base
```bash
sudo apt update
sudo apt install r-base libcurl4-openssl-dev libssl-dev libxml2-dev libfontconfig1-dev

```

## 2. Usuarios y Permisos

```bash
sudo groupadd analistas
sudo mkdir -p /srv/
sudo chown -R :analistas /srv/
sudo chmod -R 775 /srv/
sudo find /srv/ -type d -exec chmod g+s {} +
sudo useradd -m -g analistas nombre_usuario
sudo passwd nombre_usuario

```

## 3. Configuración de RStudio Server (`/etc/rstudio/rserver.conf`)

```text
rsession-which-r=/srv/iie/envs/qgis_env/bin/R
rsession-ld-library-path=/srv/iie/envs/qgis_env/lib/R/lib:/srv/iie/envs/qgis_env/lib
session-default-working-dir=/srv/

```

## 4. Configuración Global de R (`/srv/iie/envs/qgis_env/lib/R/etc/Rprofile.site`)

```r
if (interactive()) {
    try(setwd("/srv/"), silent = TRUE)
}

```

## 5. Verificación (Sanity Check)

1. **Librerías**: `ldd /srv/iie/envs/qgis_env/lib/R/modules/internet.so | grep ssl`
2. **Red**: `R -e 'download.file("https://google.com", tempfile())'`

```

---


¿Deseas que añada algo más a este documento antes de considerarlo terminado?

```