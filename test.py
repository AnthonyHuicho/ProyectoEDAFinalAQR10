import time
import hnswlib
import numpy as np
import pandas as pd

K = 10
QUERIES = 1000


def groundtruth(data, queries):
    gt = []
    for q in queries:
        dist = np.sum((data - q) ** 2, axis=1)
        vecinos = np.argsort(dist)[:K]
        gt.append(vecinos)
    return np.array(gt)


print("Cargando Fashion-MNIST...")
df = pd.read_csv(r"fashion-mnist_train.csv")
embeddings = df.iloc[:, 1:].values.astype(np.float32) / 255.0
queries = embeddings[:QUERIES]

print("HNSW:", hnswlib.__file__)
print(f"Dataset: {len(embeddings)} vectores dim {embeddings.shape[1]}")

print("Calculando ground truth...")
gt = groundtruth(embeddings, queries)

dim = embeddings.shape[1]
index = hnswlib.Index(space="l2", dim=dim)
index.init_index(max_elements=len(embeddings), ef_construction=200, M=32)
index.set_ef(200)

t0 = time.perf_counter()
index.initAQR(embeddings)
build = time.perf_counter() - t0

index.resetTiempos()
t0 = time.perf_counter()
todas_las_labels, _ = index.knn_query(queries, k=K)
search = time.perf_counter() - t0

aciertos = 0
for i in range(len(queries)):
    aciertos += len(set(todas_las_labels[i]) & set(gt[i]))

qpstotal = len(queries) / search
recall = aciertos / (len(queries) * K)

tsoloaqrus = index.getTiempoAQR()
print(tsoloaqrus/1000000)
if tsoloaqrus > 0:
    qpsaqrpuro = len(queries) / (tsoloaqrus / 1000000)
else:
    qpsaqrpuro = 0.0
    print("tiempoAQr es 0, algo hiciste mal xd")

print("\n=========================================")
print(f"Build         : {build:.4f}s")
print(f"QPS total     : {qpstotal:.2f}")
print(f"Recall@10     : {recall:.4f}")
print("=========================================")