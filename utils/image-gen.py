import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# 生成数据
x = [
    137.6, 534.4, 300.8, 332.8, 742.4, 928.0, 508.8, 649.6, 288, 460.8, -12.8,
    128, 544, 806.4
]
y = [
    272, 272, 147.2, 377.6, 339.2, 300.8, 217.6, 553.6, 822.4, 854.4, 694.4,
    899.2, 499.6, 950.8
]
fig, ax = plt.subplots()
ax.scatter(x, y)
plt.show()
