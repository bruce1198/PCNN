import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
# works
from PIL import Image
import numpy as np
import pickle
from os.path import abspath, dirname
# estimate
import time
load_time = 0
cal_time = 0

pcnn_path = dirname(dirname(abspath(__file__)))
sys.path.insert(0, pcnn_path)
from fl import FCBlock

def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def set_pre_cal_w(self, w):
        self.w = w

    def forward_origin(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        # x = x.view(-1, 256 * 6 * 6)
        x = x.view(-1).detach().numpy()
        # w = self.fc1.weight.data.numpy().transpose()
        fblk = FCBlock('normal', 0, 1)
        # fblk.set_input_size(6.0)
        fblk.set_pre_cal_w(self.w)
        # start_time = time.time()
        x = relu(fblk.process(x) + self.fc1.bias.detach().numpy())
        # print('FC1:', time.time() - start_time)
        w1 = self.fc2.weight.data.numpy().transpose()
        w2 = self.fc3.weight.data.numpy().transpose()
        fblk = FCBlock('hybrid', 0, 1)
        fblk.set_bias(self.fc2.bias.detach().numpy())
        fblk.append_layer(w1)
        fblk.append_layer(w2)
        # # start_time = time.time()
        x = fblk.process(x)
        x += self.fc3.bias.detach().numpy()
        # print('FC2:', time.time() - start_time)
        return x

if __name__ == "__main__":
    import math
    start_time = time.time()
    net = Net()
    net.load_state_dict(torch.load(os.path.join(pcnn_path, 'models', 'alexnet.h5')))
    pre_cal_w = net.fc1.weight.data.numpy().transpose()
    size = pre_cal_w.shape[0]
    size2 = pre_cal_w.shape[1]
    input_size = 6
    avg = 6
    total = avg
    start = 0
    height = total
    stride = input_size * input_size
    # print(size)
    # print(stride)
    height1 = int(size * height / input_size)
    w = np.float32(np.zeros(shape=(height1, size2)))
    cnt = 0
    for i in range(start*input_size, size, stride):
        pos = cnt * height*input_size
        w[pos:pos+height*input_size, :] = pre_cal_w[i:i+height*input_size, :]
        # print('w['+str(pos)+':'+str(pos+height*input_size)+'] = layer['+str(i)+':'+str(i+height*input_size)+']')
        cnt += 1
    net.set_pre_cal_w(w)
    load_time = time.time() - start_time
    print(load_time)

    if len(sys.argv) == 2:
        if sys.argv[1] == '-g':
            torch.save(net.state_dict(), os.path.join(pcnn_path, 'models', 'alexnet.h5'))
            exit(0)
        else:
            image_path = sys.argv[1]
            image = Image.open(image_path)
            # image = image.resize((224, 224), Image.ANTIALIAS )
            # # convert image to numpy array
            # x = np.array([np.asarray(image)[:, :, :3]])
            # x = torch.Tensor(list(x)).permute(0, 3, 2, 1)
            from torchvision import transforms
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(image)
            x = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    start_time = time.time()
    # y = net.forward_origin(x)
    # print(y.view(-1).detach().numpy()[:50])
    # y = net.forward_origin(x).view(-1).detach().numpy()
    # print(y[:50])
    y = net(x)
    # print(y.shape)
    # print(y[:50])
    y = softmax(y)
    index = np.argmax(y)
    # index = 0

    cal_time = time.time() - start_time
    import json
    print(json.dumps({
        'index': int(index),
        'load_time': int(1000*load_time),
        'cal_time': int(1000*cal_time)
    }))


