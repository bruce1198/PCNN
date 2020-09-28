import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from fl import FCBlock

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
		self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0)
		self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
		self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
		self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
		self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
		self.conv14 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
		self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
		self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
		self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.fc1 = nn.Linear(25088, 4096)
		self.fc2 = nn.Linear(4096, 4096)
		self.fc3 = nn.Linear(4096, 1000)

	def b0_forward(self, x):
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
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv8(x))
		x = self.pool3(x)
		return x

	def b1_forward(self, x):
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv9(x))
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv10(x))
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv11(x))
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv12(x))
		x = self.pool4(x)
		return x

	def b2_forward(self, x):
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv13(x))
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv14(x))
		return x

	def b3_forward(self, x):
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv15(x))
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv16(x))
		x = self.pool5(x)
		return x

	def b4_forward(self, x):
		return x

	def pad(self, x, padding_value):
		m = nn.ConstantPad2d((padding_value, padding_value, 0, 0), 0)
		x = m(x)
		return x

net = Net()
net.load_state_dict(torch.load('models/model'))
################# setting ####################
num_of_devices = 7
num_of_blocks = 5
################# read json ##################

################# block 0 ####################

x = torch.ones(1, 3, 76, 224)
y = net.b0_forward(x)

#TODO
#Send y to the server and get the new input.

################# block 1 ####################

x = y[:, :, 0:12, :]
y = net.b1_forward(x)

#TODO
#Send y to the server and get the new input.

################# block 2 ####################

x = y[:, :, 0:6, :]
y = net.b2_forward(x)

#TODO
#Send y to the server and get the new input.

################# block 3 ####################

x = y[:, :, 0:6, :]
y = net.b3_forward(x)

#TODO
#Send y to the server and get the new input.

################# block 4 ####################

x = y[:, :, 0:4096, :]
y = net.b4_forward(x)

#TODO
#Send y to the server and get the new input.

