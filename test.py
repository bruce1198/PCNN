import numpy as np
x = np.ones((4,3))
x[0][0] = 5
print(x)
y = x[0:2]
print(y.shape)