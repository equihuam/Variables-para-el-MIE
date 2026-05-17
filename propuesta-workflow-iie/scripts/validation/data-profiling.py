import pandas as pd
from pathlib import Path
from ydata_profiling import ProfileReport

DATOS_DIR = Path("C:/wf-ie-data/")

datos = pd.read_csv(DATOS_DIR / "results" / "training" / "bn_input-r.csv", na_values=["*"])

print(datos['p_manglares'].describe())

if 'p_manglares' in datos.columns:
    datos['p_manglares'] = datos['p_manglares'].astype(float)

profile = ProfileReport(
    datos, 
    minimal=True,
    title="IIE Training Profile  —R—",
    plot={"histogram": {"bins": 50, "max_bins": 100}}, # Detiene la locura de los miles de millones de bins
    vars={
        "num": {
            "chi_squared_threshold": 0 # Desactiva el test de Chi-cuadrado que causa el crash
        }
    }
)    

profile.to_file(DATOS_DIR / "results" / "iie-training-profile-R.html")