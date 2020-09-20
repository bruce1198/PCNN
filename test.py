import numpy as np
import pickle
# x = np.ones((4,3))
# x[0][0] = 5
# print(x)
# y = x[0:2]
# print(y.shape)

msg = pickle.dumps(np.arange(12).reshape(3, 4))
ary = pickle.loads(msg)
print(ary, ary.shape)