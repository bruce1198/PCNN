import math
import numpy as np

def relu(x):
    return np.maximum(x, 0)

class Layer:
    def __init__(self, input_channel, output_channel, weights):
        self.weights = weights
        self.input_channel = input_channel
        self.output_channel = output_channel
    
    def set_input_channel(self, input_channel):
        self.input_channel = input_channel

    def set_output_channel(self, output_channel):
        self.output_channel = output_channel

    def set_weights(self, weights):
        self.weights = weights
        

class FCBlock:
    def __init__(self, mode, idx):
        self.mode = mode
        self.layers = []
        self.idx = idx

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
                size = self.layers[0].input_channel

                # print(size)

                b = int(self.idx*math.ceil(size/self.device_num))
                e = int(min((self.idx+1)*math.ceil(size/self.device_num), size))

                ans = np.matmul(X, self.layers[0].weights[b:e, :])

                return ans

        elif self.mode == 'hybrid':
            if len(self.layers) != 2:
                print('Total layer should be two on hybrid mode!')
                return
            else:
                # print('process device:', self.idx)

                size = self.layers[0].output_channel

                # print(size)

                b = int(self.idx*math.ceil(size/self.device_num))
                e = int(min((self.idx+1)*math.ceil(size/self.device_num), size))

                # print(b, e)

                weights = self.layers[0].weights[:, b:e]

                # print(weights.shape)

                a1 = relu(np.matmul(X, weights))

                # print(a1[-10:])

                # print(a1.shape)

                h = [0 for i in range(self.layers[1].output_channel)]

                size = self.layers[1].input_channel

                # print(size)

                b = int(self.idx*math.ceil(size/self.device_num))
                e = int(min((self.idx+1)*math.ceil(size/self.device_num), size))

                # print(b, e)

                # print(self.layers[1].weights[b:e, :].shape)

                h = np.matmul(a1, self.layers[1].weights[b:e, :])

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

    a1_tmp = [0 for i in range(output_channel[0])]
    # data parallelism
    x = []
    # split the data
    for device in range(device_num):
        size = len(X)
        b = int(device*math.ceil(size/device_num))
        e = int(min((device+1)*math.ceil(size/device_num), size))
        x.append(X[b:e])

    # calculate the partial sum
    # for device in range(device_num):
    #     # size of input_channel
    #     size = input_channel[0]
    #     b = int(device*math.ceil(size/device_num))
    #     e = int(min((device+1)*math.ceil(size/device_num), size))
    #     a1_tmp += np.matmul(x[device], w1[b:e, :])
    # a1_tmp = relu(a1_tmp)
    # print(a1_tmp[:10])

    ###
    a1_tmp = [0 for i in range(output_channel[0])]

    for device in range(device_num):
        # size of input_channel
        fblk = FCBlock('normal', device)
        fblk.set_device_num(device_num)
        layer = Layer(
            input_channel = input_channel[0],
            output_channel = output_channel[0],
            weights = w1
        )
        fblk.append_layer(layer)
        a1_tmp += fblk.process(x[device])
    a1_tmp = relu(a1_tmp)
    # print(a1_tmp[:10])

    ###

    # last block, hybrid mode

    ####

    h_tmp = [0 for i in range(output_channel[2])]

    for device in range(device_num):
        fblk = FCBlock('hybrid', device)
        fblk.set_device_num(device_num)
        for i in range(2):
            layer = Layer(
                input_channel = input_channel[i+1],
                output_channel = output_channel[i+1],
                weights = w[i]
            )
            fblk.append_layer(layer)
        h_tmp += fblk.process(a1_tmp)
    ####
    h_tmp = relu(h_tmp)
    # print(h_tmp[:10])

    # first, split the weights
    # fc2 = []
    # for device in range(device_num):
    #     # size of output_channel
    #     size = output_channel[1]
    #     b = int(device*math.ceil(size/device_num))
    #     e = int(min((device+1)*math.ceil(size/device_num), size))
    #     fc2.append(w2[:, b:e])

    # a2_tmp = [None for i in range(device_num)]

    # for device in range(device_num):
    #     a2_tmp[device] = relu(np.matmul(a1, fc2[device]))

    # # print(a2_tmp[0][-10:])
    # # print(a2[-10:])

    # # a2_tmp has already been splitted into device_num pieces
    # # calculate the partial sum
    # h_tmp = [0 for i in range(output_channel[2])]
    # for device in range(device_num):
    #     # size of input_channel
    #     size = input_channel[2]
    #     b = int(device*math.ceil(size/device_num))
    #     e = int(min((device+1)*math.ceil(size/device_num), size))
    #     h_tmp += np.matmul(a2_tmp[device], w3[b:e, :])
    #     # print(np.matmul(a2_tmp[device], w3[b:e, :])[-10:])
    # h_tmp = relu(h_tmp)

    # print(h_tmp[:10])
    # print(h[:10])

    for i in range(len(h)):
        if round(h[i], 6) != round(h_tmp[i], 6):
            print(i, 'is not equal!')
