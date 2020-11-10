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
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
		self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
		self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
		self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
		self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
		self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
		self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
		self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
		self.conv14 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
		self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
		self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
		self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.fc1 = nn.Linear(25088, 4096)
		self.fc2 = nn.Linear(4096, 4096)
		self.fc3 = nn.Linear(4096, 1000)

	def forward_origin(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.pool1(x)
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = self.pool2(x)
		x = F.relu(self.conv5(x))
		x = F.relu(self.conv6(x))
		x = F.relu(self.conv7(x))
		x = F.relu(self.conv8(x))
		x = self.pool3(x)
		x = F.relu(self.conv9(x))
		x = F.relu(self.conv10(x))
		x = F.relu(self.conv11(x))
		x = F.relu(self.conv12(x))
		x = self.pool4(x)
		x = F.relu(self.conv13(x))
		x = F.relu(self.conv14(x))
		x = F.relu(self.conv15(x))
		x = F.relu(self.conv16(x))
		x = self.pool5(x)
		x = x.view(-1, 25088)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.pool1(x)
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = self.pool2(x)
		x = F.relu(self.conv5(x))
		x = F.relu(self.conv6(x))
		x = F.relu(self.conv7(x))
		x = F.relu(self.conv8(x))
		x = self.pool3(x)
		x = F.relu(self.conv9(x))
		x = F.relu(self.conv10(x))
		x = F.relu(self.conv11(x))
		x = F.relu(self.conv12(x))
		x = self.pool4(x)
		x = F.relu(self.conv13(x))
		x = F.relu(self.conv14(x))
		x = F.relu(self.conv15(x))
		x = F.relu(self.conv16(x))
		x = self.pool5(x)
		x = x.view(-1).detach().numpy()
		w = self.fc1.weight.data.numpy().transpose()
		fblk = FCBlock('normal', 0, 1)
		fblk.set_input_size(7.0)
		fblk.append_layer(w)
		x = relu(fblk.process(x) + self.fc1.bias.detach().numpy())
		w1 = self.fc2.weight.data.numpy().transpose()
		w2 = self.fc3.weight.data.numpy().transpose()
		fblk = FCBlock('hybrid', 0, 1)
		fblk.set_bias(self.fc2.bias.detach().numpy())
		fblk.append_layer(w1)
		fblk.append_layer(w2)
		x = fblk.process(x)
		x += self.fc3.bias.detach().numpy()
		return x

start_time = time.time()
net = Net()
net.load_state_dict(torch.load(os.path.join(pcnn_path, 'models', 'vgg19.h5')))
load_time = time.time() - start_time

if len(sys.argv) == 2:
	if sys.argv[1] == '-g':
		torch.save(net.state_dict(), os.path.join(pcnn_path, 'models', 'vgg19.h5'))
		exit(0)
	else:
		image_path = sys.argv[1]
		image = Image.open(image_path)
		image = image.resize((224, 224), Image.ANTIALIAS )
		# convert image to numpy array
		x = np.array([np.asarray(image)[:, :, :3]])
		x = torch.Tensor(list(x)).permute(0, 3, 2, 1)

start_time = time.time()
# y = net.forward_origin(x)
# print(y.view(-1).detach().numpy()[:50])
# print(y.view(-1).detach().numpy()[-50:])
y = net(x)
# print(y.shape)
# print(y[:50])
# print(y[-50:])
y = softmax(y)
index = np.argmax(y)

cal_time = time.time() - start_time
import json
print(json.dumps({
    'index': int(index),
    'load_time': int(1000*load_time),
    'cal_time': int(1000*cal_time)
}))