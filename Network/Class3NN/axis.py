import numpy as np

a = np.array([[1,2,3],[4,5,6]])
print(a)
print(np.sum(a, axis=0, keepdims=True))
print(np.sum(a, axis=1, keepdims=True))