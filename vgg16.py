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
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0)
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
		self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
		self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
		self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0)
		self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
		self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
		self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
		self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
		self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
		self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.fc1 = nn.Linear(25088, 4096)
		self.fc2 = nn.Linear(4096, 4096)
		self.fc3 = nn.Linear(4096, 1000)

	def b0_forward(self, x, device_num):
		self.device_num = device_num
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv1(x))
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv2(x))
		x = self.pool1(x)
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv3(x))
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv4(x))
		x = self.pool2(x)
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv5(x))
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv6(x))
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv7(x))
		x = self.pool3(x)
		return x

	def b1_forward(self, x, device_num):
		self.device_num = device_num
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv8(x))
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv9(x))
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv10(x))
		x = self.pool4(x)
		return x

	def b2_forward(self, x, device_num):
		self.device_num = device_num
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv11(x))
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv12(x))
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv13(x))
		x = self.pool5(x)
		return x

	def b3_forward(self, x, device_num):
		self.device_num = device_num
		return x

	def pad(self, x, padding_value):
		if self.device_num == 0:
			m = nn.ConstantPad2d((padding_value, padding_value, padding_value, 0), 0)
		elif self.device_num == 6:
			m = nn.ConstantPad2d((padding_value, padding_value, 0, padding_value), 0)
		else:
			m = nn.ConstantPad2d((padding_value, padding_value, 0, 0), 0)
		x = m(x)
		return x

net = Net()
net.load_state_dict(torch.load('models/vgg16'))
################# setting ####################
num_of_devices = 7
num_of_blocks = 4
################# block 0 ####################

y = torch.ones(1, 3, 224, 224)
x1 = y[:, :, 0:50, :]
y1 = net.b0_forward(x1, 0)
x2 = y[:, :, 14:18, :]
y2 = net.b0_forward(x2, 1)
x3 = y[:, :, 46:22, :]
y3 = net.b0_forward(x3, 2)
x4 = y[:, :, 78:26, :]
y4 = net.b0_forward(x4, 3)
x5 = y[:, :, 110:30, :]
y5 = net.b0_forward(x5, 4)
x6 = y[:, :, 142:34, :]
y6 = net.b0_forward(x6, 5)
x7 = y[:, :, 174:38, :]
y7 = net.b0_forward(x7, 6)

y = torch.ones(1, 256, 28, 28)
offset = 0
y[:, :, offset: offset+y1.shape[2], :] = y1
offset += y1.shape[2]
y[:, :, offset: offset+y2.shape[2], :] = y2
offset += y2.shape[2]
y[:, :, offset: offset+y3.shape[2], :] = y3
offset += y3.shape[2]
y[:, :, offset: offset+y4.shape[2], :] = y4
offset += y4.shape[2]
y[:, :, offset: offset+y5.shape[2], :] = y5
offset += y5.shape[2]
y[:, :, offset: offset+y6.shape[2], :] = y6
offset += y6.shape[2]
y[:, :, offset: offset+y7.shape[2], :] = y7
offset += y7.shape[2]
################# block 1 ####################

x1 = y[:, :, 0:7, :]
y1 = net.b1_forward(x1, 0)
x2 = y[:, :, 1:3, :]
y2 = net.b1_forward(x2, 1)
x3 = y[:, :, 5:4, :]
y3 = net.b1_forward(x3, 2)
x4 = y[:, :, 9:5, :]
y4 = net.b1_forward(x4, 3)
x5 = y[:, :, 13:6, :]
y5 = net.b1_forward(x5, 4)
x6 = y[:, :, 17:7, :]
y6 = net.b1_forward(x6, 5)
x7 = y[:, :, 21:8, :]
y7 = net.b1_forward(x7, 6)

y = torch.ones(1, 512, 14, 14)
offset = 0
y[:, :, offset: offset+y1.shape[2], :] = y1
offset += y1.shape[2]
y[:, :, offset: offset+y2.shape[2], :] = y2
offset += y2.shape[2]
y[:, :, offset: offset+y3.shape[2], :] = y3
offset += y3.shape[2]
y[:, :, offset: offset+y4.shape[2], :] = y4
offset += y4.shape[2]
y[:, :, offset: offset+y5.shape[2], :] = y5
offset += y5.shape[2]
y[:, :, offset: offset+y6.shape[2], :] = y6
offset += y6.shape[2]
y[:, :, offset: offset+y7.shape[2], :] = y7
offset += y7.shape[2]
################# block 2 ####################

x1 = y[:, :, 0:5, :]
y1 = net.b2_forward(x1, 0)
x2 = y[:, :, 0:7, :]
y2 = net.b2_forward(x2, 1)
x3 = y[:, :, 1:9, :]
y3 = net.b2_forward(x3, 2)
x4 = y[:, :, 3:11, :]
y4 = net.b2_forward(x4, 3)
x5 = y[:, :, 5:13, :]
y5 = net.b2_forward(x5, 4)
x6 = y[:, :, 7:14, :]
y6 = net.b2_forward(x6, 5)
x7 = y[:, :, 9:14, :]
y7 = net.b2_forward(x7, 6)

y = torch.ones(1, 4096, 1, 1)
offset = 0
y[:, :, offset: offset+y1.shape[2], :] = y1
offset += y1.shape[2]
y[:, :, offset: offset+y2.shape[2], :] = y2
offset += y2.shape[2]
y[:, :, offset: offset+y3.shape[2], :] = y3
offset += y3.shape[2]
y[:, :, offset: offset+y4.shape[2], :] = y4
offset += y4.shape[2]
y[:, :, offset: offset+y5.shape[2], :] = y5
offset += y5.shape[2]
y[:, :, offset: offset+y6.shape[2], :] = y6
offset += y6.shape[2]
y[:, :, offset: offset+y7.shape[2], :] = y7
offset += y7.shape[2]
################# block 3 ####################

x1 = y[:, :, 0:4096, :]
y1 = net.b3_forward(x1, 0)
x2 = y[:, :, 0:4096, :]
y2 = net.b3_forward(x2, 1)
x3 = y[:, :, 0:4096, :]
y3 = net.b3_forward(x3, 2)
x4 = y[:, :, 0:4096, :]
y4 = net.b3_forward(x4, 3)
x5 = y[:, :, 0:4096, :]
y5 = net.b3_forward(x5, 4)
x6 = y[:, :, 0:4096, :]
y6 = net.b3_forward(x6, 5)
x7 = y[:, :, 0:4096, :]
y7 = net.b3_forward(x7, 6)

y = torch.ones(1, 1000, 1, 1)
offset = 0
y[:, :, offset: offset+y1.shape[2], :] = y1
offset += y1.shape[2]
y[:, :, offset: offset+y2.shape[2], :] = y2
offset += y2.shape[2]
y[:, :, offset: offset+y3.shape[2], :] = y3
offset += y3.shape[2]
y[:, :, offset: offset+y4.shape[2], :] = y4
offset += y4.shape[2]
y[:, :, offset: offset+y5.shape[2], :] = y5
offset += y5.shape[2]
y[:, :, offset: offset+y6.shape[2], :] = y6
offset += y6.shape[2]
y[:, :, offset: offset+y7.shape[2], :] = y7
offset += y7.shape[2]
