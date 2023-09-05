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

# x = np.loadtxt("./data/planaria/R_pca_seurat.txt", delimiter="\t")
# labels_str = np.loadtxt("./data/planaria/R_annotation.txt", delimiter=",", dtype=str)
# _, y = np.unique(labels_str, return_inverse=True)

x = np.loadtxt("./data/myeloid-progenitors/MyeloidProgenitors.csv", delimiter=",", skiprows=1,  usecols=np.arange(11))
labels_str = np.loadtxt("./data/myeloid-progenitors/MyeloidProgenitors.csv", delimiter=",", skiprows=1, usecols=11, dtype=str)
_, y = np.unique(labels_str, return_inverse=True)

print("Data set contains %d samples with %d features" % x.shape)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.90, random_state=42)
#
# print("%d training samples" % x_train.shape[0])
# print("%d test samples" % x_test.shape[0])

tsne = TSNE(
    perplexity=30,
    metric="euclidean",
    learning_rate=1,
    n_jobs=12,
    random_state=42,
    verbose=True,
    max_grad_norm=1,
    # n_iter=100,
    # early_exaggeration_iter=5
)

embedding_train = tsne.fit(x_train)

# utils.plot(embedding_train, y_train, colors=utils.MACOSKO_COLORS)

fig, ax = plt.subplots()
ax.scatter(embedding_train[:, 0], embedding_train[:, 1],
           c=y_train,
           marker=".")
ax.axis("square")
ax.add_patch(plt.Circle((0, 0), radius=1, edgecolor="b", facecolor="None"))

plt.savefig("embedding.pdf")
plt.show()
