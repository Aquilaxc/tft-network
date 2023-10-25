import numpy as np

a = np.array([2, 4, 6, 8, 10])

x = np.arange(len(a))
xx = np.arange(0, len(a), 0.3)

aa = np.interp(xx, x, a)
print(aa)