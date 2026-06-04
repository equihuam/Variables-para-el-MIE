## Configuración d PowerShell


A continuación se documenta y consolida de forma estricta los pasos de ingeniería de software, las rutas de los archivos, los scripts de automatización y las configuraciones de red e interfaz gráfica que implementamos para configurar **PowerShell 7** con **Oh My Posh**.

## 1. Comandos de PowerShell (Configuración del Anfitrión Windows)

Para lograr que **PowerShell 7 (`pwsh`)** actúe como el motor predeterminado y conéctarlo dinámicamente con los servicios de red de Windows, se deben ejecutar los siguientes comandos en la máquina anfitriona.

### Establecer PowerShell 7 por Defecto en Windows Terminal

Para evitar que se abra la versión clásica (5.1), la configuración global se edita presionando `Ctrl + ,` en la interfaz gráfica. Si prefieres validar el identificador nativo (`guid`) del perfil moderno vía PowerShell, ejecuta:

```powershell
# Listar los perfiles de terminal disponibles para verificar el identificador de pwsh
Get-CimInstance Win32_Process | Where-Object Name -eq "pwsh.exe"

```

### Script del Puente de Red (Port Forwarding) e Inyección de Firewall

Dado que el entorno de red de subprocesos se ejecuta de forma interna, se configuró un Port Forwarding dinámico en el puerto alternativo **2222** para redirigir el tráfico hacia la IP variable.

> **Requisito:** Este bloque debe ejecutarse en una consola de **PowerShell 7 abierta como Administrador**.

```powershell
# 1. Definición de variables operativas
$wslIp = (wsl exec hostname -I).Split(" ")[0]
$port = 2222

# 2. Limpieza preventiva de reglas previas (Es normal si arroja que no encuentra el archivo)
netsh interface portproxy delete v4tov4 listenport=$port listenaddress=0.0.0.0 2>$null

# 3. Creación del redireccionamiento IPv4 activo
netsh interface portproxy add v4tov4 listenport=$port listenaddress=0.0.0.0 connectport=$port connectaddress=$wslIp

# 4. Inyección de regla de tráfico entrante en el Firewall de Windows
New-NetFirewallRule -DisplayName "WSL SSH" -Direction Inbound -Action Allow -Protocol TCP -LocalPort $port -ErrorAction SilentlyContinue

# 5. Confirmación visual en consola
Write-Host "Puente configurado con éxito hacia la IP interna: $wslIp" -ForegroundColor Green

```

---

## 2. Archivos de Configuración Funcionales y Ubicaciones

### Archivo 1: Perfil de Inicio de PowerShell 7 (`$PROFILE`)

* **Ubicación de red física:** `C:\Users\equih\Documents\PowerShell\Microsoft.PowerShell_profile.ps1`
* **Comando de acceso directo:** `notepad $PROFILE` (desde la consola de PowerShell 7)
* **Permisos en Windows:** Control total para el usuario autenticado (`equih`).

Este archivo unifica de forma secuencial la carga del entorno interactivo. Fuerza la importación del módulo interno de Miniconda y levanta la inicialización estética de Oh My Posh sin generar colisiones ni errores de teclado (`PSReadLine`):

```powershell
# =====================================================================
# PERFIL DE ARRANQUE UNIFICADO - POWERSHELL 7
# =====================================================================

# 1. Silenciar temporalmente errores de lectura de consola (PSReadLine antiguo)
$OldAction = $ErrorActionPreference
$ErrorActionPreference = 'SilentlyContinue'

# 2. Inicialización y puente manual del motor de Miniconda
if (Test-Path "C:\Users\equih\miniconda3\shell\condabin\Conda.psm1") {
    Import-Module "C:\Users\equih\miniconda3\shell\condabin\Conda.psm1"
    & "C:\Users\equih\miniconda3\Scripts\conda.exe" "shell.powershell" "activate" "base" | Invoke-Expression
}

# 3. Inicializar el Prompt estético apuntando al archivo de tema local parcheado
oh-my-posh init pwsh --config "$HOME\.mi_tema.json" | Invoke-Expression

# 4. Restaurar el control de errores por defecto del sistema
$ErrorActionPreference = $OldAction

```

---

### Archivo 2: Esquema JSON del Tema de Oh My Posh (`.mi_tema.json`)

* **Ubicación de red física:** `C:\Users\equih\.mi_tema.json` (Equivalente a `$HOME\.mi_tema.json`)
* **Permisos:** Lectura y escritura para el usuario de desarrollo.

Este archivo contiene el diseño completo y depurado del tema original de Luise Freese (`M365Princess`). Se eliminó el *Prompt Transitorio* que encogía los comandos y generaba distorsión en el historial, y se implementó un **parche quirúrgico de lectura de variables de entorno (`.Env.CONDA_DEFAULT_ENV`)** para forzar el despliegue del bloque morado con el logotipo de Python (``) cada vez que un entorno de Conda es activado:

```json
{
  "$schema": "https://raw.githubusercontent.com/JanDeDobbeleer/oh-my-posh/main/themes/schema.json",
  "blocks": [
    {
      "type": "prompt",
      "alignment": "left",
      "segments": [
        {
          "type": "text",
          "style": "plain",
          "foreground": "#8A2BE2",
          "template": "\uE216 "
        },
        {
          "type": "text",
          "style": "powerline",
          "powerline_symbol": "\uE0B0",
          "foreground": "#ffffff",
          "background": "#8A2BE2",
          "template": " \uE73C {{ .Env.CONDA_DEFAULT_ENV }} ",
          "properties": {
            "display_default": false
          }
        },
        {
          "type": "path",
          "style": "powerline",
          "powerline_symbol": "\uE0B0",
          "foreground": "#ffffff",
          "background": "#D0A9F5",
          "template": " \uE5FF {{ .Path }} "
        },
        {
          "type": "git",
          "style": "powerline",
          "powerline_symbol": "\uE0B0",
          "foreground": "#100e23",
          "background": "#fff700",
          "template": " {{ .HEAD }} "
        }
      ]
    },
    {
      "type": "rprompt",
      "segments": [
        {
          "type": "time",
          "style": "plain",
          "foreground": "#D0A9F5",
          "template": " \uE38F {{ .CurrentDate | date \"15:04\" }} "
        },
        {
          "type": "text",
          "style": "plain",
          "foreground": "#FF69B4",
          "template": " \u2665 "
        }
      ]
    }
  ],
  "version": 2
}

```

---

## 3. Estado Operativo del Ecosistema

Tras la recarga de los perfiles y la sincronización arquitectónica, el flujo de comandos responde exactamente con este comportamiento matemático en tu pantalla:

1. **Arranque Directo:** Al abrir Windows Terminal, se inicia de inmediato **PowerShell 7** cargando las flechas pastel del extremo izquierdo y el reloj alineado con el corazón en el extremo derecho.
2. **Activación de Entorno Exitoso:** Al ejecutar el comando:
```powershell
conda activate r_py

```



```
    El sistema inyecta la variable global en la memoria, provocando que el segmento morado tome su lugar de inmediato al frente de tu terminal:
    ```text
       r_py   C:\                                                   14:20  ♥

```

3. **Historial Conservado:** Al presionar *Enter*, los bloques de color lila y morado se mantienen estáticos en las líneas de comandos previas, ofreciéndote un registro visual claro de en qué entorno ejecutaste cada tarea de análisis.