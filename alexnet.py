import torch
import torch.nn as nn
import torch.nn.functional as F
from fl import FCBlock
import numpy as np

import json

def relu(x):
	return np.maximum(x, 0)

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
		self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
		self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=0)
		self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
		self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=0)
		self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=0)
		self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=0)
		self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
		self.fc1 = nn.Linear(9216, 4096)
		self.fc2 = nn.Linear(4096, 4096)
		self.fc3 = nn.Linear(4096, 1000)

	def b0_forward(self, x, device_num):
		self.device_num = device_num
		x = self.pad(x, padding_value=2)
		x = F.relu(self.conv1(x))
		x = self.pool1(x)
		return x

	def b1_forward(self, x, device_num):
		self.device_num = device_num
		x = self.pad(x, padding_value=2)
		x = F.relu(self.conv2(x))
		x = self.pool2(x)
		return x

	def b2_forward(self, x, device_num):
		self.device_num = device_num
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv3(x))
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv4(x))
		return x

	def b3_forward(self, x, device_num):
		self.device_num = device_num
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv5(x))
		x = self.pool3(x)
		return x

	def b4_forward(self, x, device_num):
		self.device_num = device_num
		return x

	def pad(self, x, padding_value):
		if self.device_num == 0:
			m = nn.ConstantPad2d((padding_value, padding_value, padding_value, 0), 0)
		elif self.device_num == 1:
			m = nn.ConstantPad2d((padding_value, padding_value, 0, padding_value), 0)
		else:
			m = nn.ConstantPad2d((padding_value, padding_value, 0, 0), 0)
		x = m(x)
		return x

net = Net()
net.load_state_dict(torch.load('models/alexnet'))
################# setting ####################
num_of_devices = 2
num_of_blocks = 5
################# block 0 ####################

y = torch.ones(1, 3, 224, 224)
x1 = y[:, :, 0:121, :]
y1 = net.b0_forward(x1, 0)
x2 = y[:, :, 110:5, :]
y2 = net.b0_forward(x2, 1)

y = torch.ones(1, 96, 27, 27)
offset = 0
y[:, :, offset: offset+y1.shape[2], :] = y1
offset += y1.shape[2]
y[:, :, offset: offset+y2.shape[2], :] = y2
offset += y2.shape[2]
################# block 1 ####################

x1 = y[:, :, 0:17, :]
y1 = net.b1_forward(x1, 0)
x2 = y[:, :, 12:2, :]
y2 = net.b1_forward(x2, 1)

y = torch.ones(1, 256, 13, 13)
offset = 0
y[:, :, offset: offset+y1.shape[2], :] = y1
offset += y1.shape[2]
y[:, :, offset: offset+y2.shape[2], :] = y2
offset += y2.shape[2]
################# block 2 ####################

x1 = y[:, :, 0:9, :]
y1 = net.b2_forward(x1, 0)
x2 = y[:, :, 5:1, :]
y2 = net.b2_forward(x2, 1)

y = torch.ones(1, 384, 13, 13)
offset = 0
y[:, :, offset: offset+y1.shape[2], :] = y1
offset += y1.shape[2]
y[:, :, offset: offset+y2.shape[2], :] = y2
offset += y2.shape[2]
################# block 3 ####################

x1 = y[:, :, 0:8, :]
y1 = net.b3_forward(x1, 0)
x2 = y[:, :, 5:13, :]
y2 = net.b3_forward(x2, 1)

################# block 4 ####################

x1 = y[:, :, 0:4096, :]
y1 = net.b4_forward(x1, 0)
x2 = y[:, :, 0:4096, :]
y2 = net.b4_forward(x2, 1)

