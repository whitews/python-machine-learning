from ch02.adaline import AdalineGD

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

# extract first 100 data values from iris data set,
# and only the 1st and 3rd features
x = iris['data'][:100, [0, 2]]

# re-map targets to -1 and 1
y = np.where(iris['target'][:100] == 0, -1, 1)

# plt.scatter(
#     x[:50, 0],
#     x[:50, 1],
#     color='red',
#     marker='o',
#     label='setosa'
# )
# plt.scatter(
#     x[50:100, 0],
#     x[50:100, 1],
#     color='blue',
#     marker='x',
#     label='versicolor'
# )
#
# plt.xlabel('sepal length')
# plt.ylabel('petal length')
#
# plt.legend(loc='upper left')

ada = AdalineGD(eta=0.00048, n_iter=100)

ada.fit(x, y)

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('# of mis-classifications')

plt.show()
pass