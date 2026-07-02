# AQR-HNSW — Implementación del Search Phase

Implementación del algoritmo AQR-HNSW basada en el paper:

---

## Estructura del proyecto

```
ProyectoEDAFinalAQR10/
|-- ProyectoEDAModificado/       -- hnswlib modificado con AQR
|   └── hnswlib-master/
|       |-- hnswlib/
|       │   |-- hnswalg.h        |-- implementacion del AQR
│       |-- python_bindings/
│       │   |-- bindings.cpp     |-- exposicion a Python
│       |-- setup.py
|-- hnsworiganltest/             |-- hnswlib original sin modificar
│   |-- hnswlib-master/
|-- test.py                      |-- benchmark del AQR-HNSW
|-- testoriginal.py              |-- benchmark del baseline HNSW
|--fashionmnist train.csv        |-- archivo descargado desde la web 
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

El data set es el de fashion mnist y el el archivo train, poner el archivo dentro del .


**Paso 1 — Crear venv nuevo:**
```powershell
cd "C:\ruta\InicializadordeChroma"
python -m venv .venv
.venv\Scripts\activate
```

**Paso 2 — Instalar dependencias, instalar el hnswlib modificado (AQR):**
```powershell
pip install chromadb numpy pandas
pip install -e .\ProyectoEDAModificado\hnswlib-master\ 
```

**Paso 3 — Ejecutar el archivo del AQR**
```powershell
python .\test.py
```
**Paso 3 — Ejecutar el archivo del normal-Baseline**
```powershell
pip install -e .\hnsworiganltest\hnswlib-master\
python .\testoriginal.py
```

## Qué hace el AQR modificado


Cuando se llama `initAQR`, cada `knn_query` ejecuta automáticamente el Search Phase completo del Algorithm 2: búsqueda coarse con uint8, refinamiento asimétrico, early termination, y exact reranking.

---


Si el dataset está en otra ruta, cambia el archivo a del proyectofinalEDAAQR10:
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
