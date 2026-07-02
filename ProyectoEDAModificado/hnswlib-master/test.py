import hnswlib
import numpy as np

dim = 64
num_elements = 1000

data = np.random.random((num_elements, dim)).astype(np.float32)

index = hnswlib.Index(space='l2', dim=dim)

index.init_index(max_elements=num_elements, ef_construction=200, M=16)
index.add_items(data)

index.set_ef(100)

labels, distances = index.knn_query(data[0], k=10)

print(labels)
print(distances)