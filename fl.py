import math
import numpy as np

def relu(x):
    return np.maximum(x, 0)
        
class FCBlock:
    def __init__(self, mode, idx, total_device_num):
        self.mode = mode
        self.layers = []
        self.idx = idx
        self.device_num = total_device_num

    # total device number
    def set_device_num(self, num):
        self.device_num = num
    
    def append_layer(self, layer):
        self.layers.append(layer)

    def process(self, X):
        if self.mode == 'normal':
            if len(self.layers) != 1:
                print('Total layer should be one on normal mode!')
                return
            else:
                size = self.layers[0].shape[0]
                # print(size)
                b = int(self.idx*math.ceil(size/self.device_num))
                e = int(min((self.idx+1)*math.ceil(size/self.device_num), size))
                ans = np.matmul(X, self.layers[0][b:e, :])

                return ans

        elif self.mode == 'hybrid':
            if len(self.layers) != 2:
                print('Total layer should be two on hybrid mode!')
                return
            else:
                # print('process device:', self.idx)
                size = self.layers[0].shape[1]
                # print(size)
                b = int(self.idx*math.ceil(size/self.device_num))
                e = int(min((self.idx+1)*math.ceil(size/self.device_num), size))
                # print(b, e)
                weights = self.layers[0][:, b:e]
                # print(weights.shape)
                a1 = relu(np.matmul(X, weights))

                # print(a1[-10:])
                # print(a1.shape)
                h = [0 for i in range(self.layers[1].shape[1])]
                size = self.layers[1].shape[0]
                # print(size)
                b = int(self.idx*math.ceil(size/self.device_num))
                e = int(min((self.idx+1)*math.ceil(size/self.device_num), size))
                # print(b, e)
                # print(self.layers[1][b:e, :].shape)
                h = np.matmul(a1, self.layers[1][b:e, :])
                # print(h[:10])
                # print(h.shape)
                return h

if __name__ == '__main__':
    device_num = 5

    input_channel = [9216, 4096, 4096]
    output_channel = [4096, 4096, 1000]

    X = np.array(np.random.uniform(0, 1, [9216]))
    w1 = np.array(np.random.uniform(-0.1, 0.1, [9216, 4096]))
    w2 = np.array(np.random.uniform(-0.1, 0.1, [4096, 4096]))
    w3 = np.array(np.random.uniform(-0.1, 0.1, [4096, 1000]))
    w = [
        w2,
        w3
    ]

    # correct answer
    a1 = relu(np.matmul(X, w1))
    a2 = relu(np.matmul(a1, w2))
    h  = relu(np.matmul(a2, w3))

    # data parallelism
    x = []
    # split the data
    for device in range(device_num):
        size = len(X)
        b = int(device*math.ceil(size/device_num))
        e = int(min((device+1)*math.ceil(size/device_num), size))
        x.append(X[b:e])

    ###
    a1_tmp = [0 for i in range(output_channel[0])]

    for device in range(device_num):
        # size of input_channel
        fblk = FCBlock('normal', device, device_num)
        fblk.append_layer(w1)
        a1_tmp += fblk.process(x[device])

    a1_tmp = relu(a1_tmp)
    # print(a1_tmp[:10])

    # last block, hybrid mode

    h_tmp = [0 for i in range(output_channel[2])]

    for device in range(device_num):
        fblk = FCBlock('hybrid', device, device_num)
        for i in range(2):
            fblk.append_layer(w[i])
        h_tmp += fblk.process(a1_tmp)

    h_tmp = relu(h_tmp)

    # print(h_tmp[:10])
    # print(h[:10])

    #check
    for i in range(len(h)):
        if round(h[i], 6) != round(h_tmp[i], 6):
            print(i, 'is not equal!')
