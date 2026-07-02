# AQR-HNSW — Implementación del Search Phase

Implementación del algoritmo AQR-HNSW basada en el paper:

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
```

---

## Requisitos

- Python 3.12
- Visual Studio Build Tools con C++ (para compilar el C++)
- numpy
- pandas
- descargar el fashion mnist de https://www.kaggle.com/datasets/zalando-research/fashionmnist?resource=download
---

## Instalación desde cero

El venv incluido en el repositorio **no funcionará** en otra máquina porque tiene rutas absolutas. Debes crear uno nuevo.

**Paso 1 — Crear venv nuevo:**
```powershell
cd "C:\ruta\InicializadordeChroma"
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


Cuando se llama `initAQR`, cada `knn_query` ejecuta automáticamente el Search Phase completo del Algorithm 2: búsqueda coarse con uint8, refinamiento asimétrico, early termination, y exact reranking.

---

## Dataset

Los scripts esperan el archivo `fashion-mnist_train.csv` en:
```
C:\Users\Usuario\Desktop\dataset\fashion-mnist_train.csv
```

Si el dataset está en otra ruta, cambia esta línea en `test.py` y `testoriginal.py`:
```python
df = pd.read_csv(r"fashion-mnist_train.csv")
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
