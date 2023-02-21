from openTSNE import TSNE
import utils

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# import gzip
# import pickle
#
# with gzip.open("data/macosko_2015.pkl.gz", "rb") as f:
#     data = pickle.load(f)
#
# x = data["pca_50"]
# y = data["CellType1"].astype(str)


import numpy as np

x = np.loadtxt("./data/planaria/R_pca_seurat.txt", delimiter="\t")
labels_str = np.loadtxt("./data/planaria/R_annotation.txt", delimiter=",", dtype=str)
_, y = np.unique(labels_str, return_inverse=True)

print("Data set contains %d samples with %d features" % x.shape)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.90, random_state=42)

print("%d training samples" % x_train.shape[0])
print("%d test samples" % x_test.shape[0])

tsne = TSNE(
    perplexity=30,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
)

embedding_train = tsne.fit(x_train)

# utils.plot(embedding_train, y_train, colors=utils.MACOSKO_COLORS)
plt.scatter(embedding_train[:, 0], embedding_train[:, 1], c=y_train)

plt.show()
