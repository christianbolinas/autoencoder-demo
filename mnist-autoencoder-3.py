# %%[markdown]
'''# SVM, RBF kernel, MNIST'''

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score

import time
from tqdm.notebook import tqdm

SEED = 69
np.random.seed(SEED)
check_random_state(SEED)

# %%
X_tr = np.load('x_train.npy')
Y_tr = np.load('y_train.npy')
X_te = np.load('x_test.npy')
Y_te = np.load('y_test.npy')

# %%
szs = []
accs = [] 

for split in range(500, 5_000, 500):
    X_tr_small = X_tr[:split]
    Y_tr_small = Y_tr[:split]
    svm = SVC()
    svm.fit(X_tr_small, Y_tr_small)
    acc = accuracy_score(svm.predict(X_te), Y_te)
    print(f'RBF SVM accuracy for tr sz {split} was {acc * 100:.2f}%.')
    szs.append(split)
    accs.append(acc)

# %%
plt.plot(szs, accs)
plt.show()