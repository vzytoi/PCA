# Principal Component Analysis (PCA) implementation in C++

**Note:** This code is **not meant for production use**. It is neither as fast nor as numerically reliable as  
[`numpy.linalg.eigh`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html).

## Usage

For a minimal example, see the following:

```python
import pca

# The matrix to reduce.
# It has shape N × p, where:
# - N is the number of samples in the dataset
# - p is the number of features per sample
M = [
    [2, 1, -1],
    [-4, 2, 0],
    [9, 2, 1],
    [0, -1, 2]
]

# Target dimension
k = 2

M_reduced = pca.run(M, k)

```

If you want to quickly understand why dimensionality reduction is useful, you can follow the example below.

```python
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import pca

d = 0.05
data3d = np.array([
    [r + uniform(-d, d), r + uniform(-d, d), i / 9]
    for i in range(10)
    for _ in range(5)
    for r in [uniform(0, 1)]
])

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(data3d[:, 0], data3d[:, 1], data3d[:, 2], s=35)
plt.show()

data2d = np.array(pca.run(data3d.tolist(), 2), dtype=float)

plt.figure()
plt.scatter(data2d[:, 0], data2d[:, 1], s=35)
plt.axis("equal")
plt.show()
```

The left image represents the generated data in 3D. You may notice that one axis does not carry much information. In this case, reducing the dimension to 2 is especially interesting because very little information is lost. 
The right image shows the same dataset after applying PCA.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="docs/images/pca_3d.png" height="350px" />
    </td>
    <td align="center" width="50%">
      <img src="docs/images/pca_2d.png" height="350px" />
    </td>
  </tr>
</table>

## How it works

If you want more detail on how TCA works and the implementation choices I made, checkout  [How PCA works (PDF)](docs/how_it_works.pdf)