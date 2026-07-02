# AQR-HNSW — Implementación del Search Phase

Implementación del algoritmo AQR-HNSW basada en el paper:
> *AQR-HNSW: Accelerating Approximate Nearest Neighbor Search via Density-aware Quantization and Multi-stage Re-ranking* — Tewary, Gantayat, Zhang (2026)

---

## Estructura del proyecto

```
InicializadordeChroma/
├── ProyectoEDAModificado/       ← hnswlib modificado con AQR
│   └── hnswlib-master/
│       ├── hnswlib/
│       │   └── hnswalg.h        ← implementacion del AQR
│       ├── python_bindings/
│       │   └── bindings.cpp     ← exposicion a Python
│       └── setup.py
├── hnsworiganltest/             ← hnswlib original sin modificar
│   └── hnswlib-master/
├── test.py                      ← benchmark del AQR-HNSW
├── testoriginal.py              ← benchmark del baseline HNSW
└── .venv/                       ← venv (NO usar este, ver instrucciones)
```

---

## Requisitos

- Python 3.12
- Visual Studio Build Tools con C++ (para compilar el C++)
- numpy
- pandas

---

## Instalación desde cero

El venv incluido en el repositorio **no funcionará** en otra máquina porque tiene rutas absolutas. Debes crear uno nuevo.

**Paso 1 — Crear venv nuevo:**
```powershell
cd "C:\ruta\donde\clonaste\InicializadordeChroma"
python -m venv .venv
.venv\Scripts\activate
```

**Paso 2 — Instalar el hnswlib modificado (AQR):**
```powershell
pip install -e ProyectoEDAModificado\hnswlib-master
```

**Paso 3 — Instalar dependencias:**
```powershell
pip install numpy pandas
```

**Paso 4 — Verificar que cargó el modificado:**
```powershell
python -c "import hnswlib; print(hnswlib.__file__)"
```
Debe mostrar una ruta dentro de tu carpeta del proyecto.

---

## Uso

**Correr el AQR-HNSW** (usa el hnswlib modificado):
```powershell
.venv\Scripts\activate
python test.py
```

**Correr el baseline HNSW original:**

El baseline corre con el mismo hnswlib modificado — cuando no se llama `initAQR` el índice usa el camino original sin AQR. Solo corre:
```powershell
python testoriginal.py
```

---

## Qué hace el AQR modificado

El hnswlib en `ProyectoEDAModificado` agrega estas funciones al índice:

- `index.initAQR(data)` — ejecuta el Build Phase: calcula densidades, cuantiza el dataset a uint8, y construye el grafo
- `index.resetTiempos()` — resetea los contadores de tiempo internos
- `index.getTiempoBase()` — retorna el tiempo acumulado del Stage 1 (búsqueda cuantizada) en microsegundos
- `index.getTiempoAQR()` — retorna el tiempo total del searchKnn con AQR en microsegundos
- `index.getTiempoSoloAQR()` — retorna el tiempo solo del refinamiento (Stages 2 y 3) en microsegundos

Cuando se llama `initAQR`, cada `knn_query` ejecuta automáticamente el Search Phase completo del Algorithm 2: búsqueda coarse con uint8, refinamiento asimétrico, early termination, y exact reranking.

---

## Dataset

Los scripts esperan el archivo `fashion-mnist_train.csv` en:
```
C:\Users\Usuario\Desktop\dataset\fashion-mnist_train.csv
```

Si tu dataset está en otra ruta, cambia esta línea en `test.py` y `testoriginal.py`:
```python
df = pd.read_csv(r"C:\Users\Usuario\Desktop\dataset\fashion-mnist_train.csv")
```

---

## Parámetros AQR configurados

Los parámetros están hardcodeados en `hnswalg.h` según la Tabla 3 del paper para recall 0.85-0.90:

| Parámetro | Valor | Descripción |
|---|---|---|
| efMultiplier | 2 | multiplicador del ef para búsqueda coarse |
| parametroUmbralGap | 0.020 | τ_gap para early termination |
| parametroUmbralRatio | 1.015 | τ_ratio para early termination |
| parametroReranK | 12 | candidatos para exact reranking |

Para cambiarlos edita los campos al inicio de la clase `HierarchicalNSW` en `hnswalg.h` y recompila con `pip install -e ProyectoEDAModificado\hnswlib-master`.
