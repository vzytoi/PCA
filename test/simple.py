import pca

dataset = [
    [12, 82, 0, 62, 7, -23],
    [9, -1, 823, -182, 0, 91],
    [10, 30, 200, -10, 3, 12],
]

res = pca.fit(dataset, 2)
X_proj = res["projected"]
W = res["W"]
means = res["means"]

print(res)