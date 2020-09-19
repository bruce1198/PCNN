import math
import numpy as np

def relu(x):
    return np.maximum(x, 0)
    # return max(x, 0)

device_num = 5

X = np.array(np.random.uniform(0, 1, [9216]))
w1 = np.array(np.random.uniform(-0.1, 0.1, [9216, 4096]))
w2 = np.array(np.random.uniform(-0.1, 0.1, [4096, 4096]))
w3 = np.array(np.random.uniform(-0.1, 0.1, [4096, 1000]))
a1 = relu(np.matmul(X, w1))
a2 = relu(np.matmul(a1, w2))
h  = relu(np.matmul(a2, w3))
# print(h.shape)
# print(h[:10])
a1_tmp = [0 for i in range(w1.shape[1])]
# data parallelism
x = []
# split the data
for device in range(device_num):
    size = len(X)
    b = int(device*math.ceil(size/device_num))
    e = int(min((device+1)*math.ceil(size/device_num), size))
    x.append(X[b:e])
# calculate the partial sum
for device in range(device_num):
    # a1_tmp[i]
    size = len(X)
    b = int(device*math.ceil(size/device_num))
    e = int(min((device+1)*math.ceil(size/device_num), size))
    a1_tmp += np.matmul(x[device], w1[b:e, :])
a1_tmp = relu(a1_tmp)


# last block, hybrid mode

# first, split the weights
fc2 = []
for device in range(device_num):
    size = len(a1)
    b = int(device*math.ceil(size/device_num))
    e = int(min((device+1)*math.ceil(size/device_num), size))
    fc2.append(w2[:, b:e])

a2_tmp = [None for i in range(device_num)]

for device in range(device_num):
    a2_tmp[device] = relu(np.matmul(a1, fc2[device]))

# print(a2_tmp[device_num-1][-10:])
# print(a2[-10:])

# a2_tmp has already been splitted into device_num pieces
# calculate the partial sum
h_tmp = [0 for i in range(w3.shape[1])]
for device in range(device_num):
    size = 4096
    b = int(device*math.ceil(size/device_num))
    e = int(min((device+1)*math.ceil(size/device_num), size))
    h_tmp += np.matmul(a2_tmp[device], w3[b:e, :])
h_tmp = relu(h_tmp)

print(h_tmp[:10])
print(h[:10])
