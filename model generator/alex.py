import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
# works
from PIL import Image
import numpy as np
import pickle
from pathlib import Path
# estimate
import time
load_time = 0
cal_time = 0

pcnn_path = str(Path(__file__).parent.parent.absolute())

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
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



start_time = time.time()
net = Net()
net.load_state_dict(torch.load(os.path.join(pcnn_path, 'models', 'alexnet')))
load_time = time.time() - start_time

if len(sys.argv) == 2:
    if sys.argv[1] == '-g':
        torch.save(net.state_dict(), os.path.join(pcnn_path, 'models', 'alexnet'))
        exit(0)
    else:
        image_path = sys.argv[1]
        image = Image.open(image_path)
        image = image.resize((224, 224), Image.ANTIALIAS )
        # convert image to numpy array
        x = np.array([np.asarray(image)[:, :, :3]])
        x = torch.Tensor(list(x)).permute(0, 3, 2, 1)

start_time = time.time()
y = net(x)
y = softmax(y.view(-1).detach().numpy())
index = np.argmax(y)
# print(y.view(-1).detach().numpy()[:50])

cal_time = time.time() - start_time
import json
print(json.dumps({
    'index': int(index),
    'load_time': load_time,
    'cal_time': cal_time
}))


