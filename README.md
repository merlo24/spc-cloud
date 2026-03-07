# spc-monitoring

Librería de **Statistical Process Control (SPC)** en Python.  
Incluye un flujo claro de **Phase I (construcción/calibración)** y **Phase II (monitoreo)**.

---

## Instalación (desde GitHub)

```bash
pip install git+https://github.com/<TU_USUARIO>/<TU_REPO>.git@v0.1.0


Quickstart
Phase I — Construir carta y calibrar límites
import pandas as pd
from spc_monitor import qcc

df_ref = pd.read_csv("reference.csv")
feature_cols = ["x1", "x2", "x3"]

chart = qcc(
    df_ref,
    chart="mw_location",
    feature_cols=feature_cols,
    limits="calibrate",
    calibrate="arl0",
    target_arl0=200,
    iters=10000,
    seed=123,
)

print(chart.summary())
chart.plot(reference=True)
Phase II — Monitorear nuevos datos (batch)
import pandas as pd

df_new = pd.read_csv("stream.csv")
res = chart.monitor(df_new, feature_cols=["x1", "x2", "x3"])

print(res.summary())
print("Primeras alarmas en:", res.violations_idx[:10])

res.plot()
Guardar y cargar el chart (recomendado para producción)

Phase I (una vez):

chart.save("artifacts/mw_location_chart.joblib")

Operación diaria (Phase II):

from spc_monitor import MWLocationChart

chart = MWLocationChart.load("artifacts/mw_location_chart.joblib")
res = chart.monitor(df_new, feature_cols=["x1","x2","x3"])
print("Alarmas:", res.violations.sum())