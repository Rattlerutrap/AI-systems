import numpy as np
import matplotlib.pyplot as plt



n_neg = 20
n_pos = 80


mean_neg = [10, 14]
std_neg = 3


mean_pos = [19, 16]
std_pos = 4


np.random.seed(42)
X_neg = np.random.normal(mean_neg, std_neg, size=(n_neg, 2))
X_pos = np.random.normal(mean_pos, std_pos, size=(n_pos, 2))

