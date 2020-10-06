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
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
		self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
		self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
		self.conv7 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
		self.conv8 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
		self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0)
		self.conv10 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
		self.conv11 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0)
		self.conv12 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
		self.conv13 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0)
		self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv14 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0)
		self.conv15 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
		self.conv16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0)
		self.conv17 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
		self.conv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0)
		self.conv19 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0)
		self.conv20 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0)
		self.conv21 = nn.Conv2d(in_channels=1280, out_channels=1024, kernel_size=3, stride=1, padding=0)
		self.conv22 = nn.Conv2d(in_channels=1024, out_channels=425, kernel_size=1, stride=1, padding=0)

	def b0_forward(self, x, device_num):
		self.device_num = device_num
		if device_num == 0:
			m = nn.ConstantPad2d((1, 1, 1, 0), 0)
		elif device_num == 5:
			m = nn.ConstantPad2d((1, 1, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv1(x))
		x = self.pool1(x)
		if device_num == 0:
			m = nn.ConstantPad2d((1, 1, 1, 0), 0)
		elif device_num == 5:
			m = nn.ConstantPad2d((1, 1, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv2(x))
		x = self.pool2(x)
		if device_num == 0:
			m = nn.ConstantPad2d((1, 1, 1, 0), 0)
		elif device_num == 5:
			m = nn.ConstantPad2d((1, 1, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv3(x))
		if device_num == 			m = nn.ConstantPad2d((0, 0, 1, 0), 0)
		elif device_num == 			m = nn.ConstantPad2d((0, 0, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((0, 0, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv4(x))
		if device_num == 0:
			m = nn.ConstantPad2d((1, 1, 1, 0), 0)
		elif device_num == 5:
			m = nn.ConstantPad2d((1, 1, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv5(x))
		x = self.pool3(x)
		return x

	def b1_forward(self, x, device_num):
		self.device_num = device_num
		if device_num == 0:
			m = nn.ConstantPad2d((1, 1, 1, 0), 0)
		elif device_num == 5:
			m = nn.ConstantPad2d((1, 1, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv6(x))
		if device_num == 			m = nn.ConstantPad2d((0, 0, 1, 0), 0)
		elif device_num == 			m = nn.ConstantPad2d((0, 0, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((0, 0, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv7(x))
		if device_num == 0:
			m = nn.ConstantPad2d((1, 1, 1, 0), 0)
		elif device_num == 5:
			m = nn.ConstantPad2d((1, 1, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv8(x))
		x = self.pool4(x)
		return x

	def b2_forward(self, x, device_num):
		self.device_num = device_num
		if device_num == 0:
			m = nn.ConstantPad2d((1, 1, 1, 0), 0)
		elif device_num == 5:
			m = nn.ConstantPad2d((1, 1, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv9(x))
		if device_num == 			m = nn.ConstantPad2d((0, 0, 1, 0), 0)
		elif device_num == 			m = nn.ConstantPad2d((0, 0, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((0, 0, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv10(x))
		return x

	def b3_forward(self, x, device_num):
		self.device_num = device_num
		if device_num == 0:
			m = nn.ConstantPad2d((1, 1, 1, 0), 0)
		elif device_num == 5:
			m = nn.ConstantPad2d((1, 1, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv11(x))
		if device_num == 			m = nn.ConstantPad2d((0, 0, 1, 0), 0)
		elif device_num == 			m = nn.ConstantPad2d((0, 0, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((0, 0, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv12(x))
		if device_num == 0:
			m = nn.ConstantPad2d((1, 1, 1, 0), 0)
		elif device_num == 5:
			m = nn.ConstantPad2d((1, 1, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv13(x))
		x = self.pool5(x)
		return x

	def b4_forward(self, x, device_num):
		self.device_num = device_num
		if device_num == 0:
			m = nn.ConstantPad2d((1, 1, 1, 0), 0)
		elif device_num == 5:
			m = nn.ConstantPad2d((1, 1, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv14(x))
		if device_num == 			m = nn.ConstantPad2d((0, 0, 1, 0), 0)
		elif device_num == 			m = nn.ConstantPad2d((0, 0, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((0, 0, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv15(x))
		return x

	def b5_forward(self, x, device_num):
		self.device_num = device_num
		if device_num == 0:
			m = nn.ConstantPad2d((1, 1, 1, 0), 0)
		elif device_num == 5:
			m = nn.ConstantPad2d((1, 1, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv16(x))
		if device_num == 			m = nn.ConstantPad2d((0, 0, 1, 0), 0)
		elif device_num == 			m = nn.ConstantPad2d((0, 0, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((0, 0, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv17(x))
		return x

	def b6_forward(self, x, device_num):
		self.device_num = device_num
		if device_num == 0:
			m = nn.ConstantPad2d((1, 1, 1, 0), 0)
		elif device_num == 4 or device_num == 5:
			m = nn.ConstantPad2d((1, 1, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv18(x))
		if device_num == 0:
			m = nn.ConstantPad2d((1, 1, 1, 0), 0)
		elif device_num == 5:
			m = nn.ConstantPad2d((1, 1, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv19(x))
		if device_num == 0:
			m = nn.ConstantPad2d((1, 1, 1, 0), 0)
		elif device_num == 5:
			m = nn.ConstantPad2d((1, 1, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv20(x))
		if device_num == 0:
			m = nn.ConstantPad2d((1, 1, 1, 0), 0)
		elif device_num == 5:
			m = nn.ConstantPad2d((1, 1, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv21(x))
		if device_num == 			m = nn.ConstantPad2d((0, 0, 1, 0), 0)
		elif device_num == 			m = nn.ConstantPad2d((0, 0, 0, 1), 0)
		else:
			m = nn.ConstantPad2d((0, 0, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv22(x))
		return x

net = Net()
net.load_state_dict(torch.load('models/yolov2'))
################# setting ####################
num_of_devices = 6
num_of_blocks = 7
################# block 0 ####################

y = torch.ones(1, 3, 608, 608)
x1 = y[:, :, 0:115, :]
y1 = net.b0_forward(x1, 0)
x2 = y[:, :, 93:219, :]
y2 = net.b0_forward(x2, 1)
x3 = y[:, :, 197:323, :]
y3 = net.b0_forward(x3, 2)
x4 = y[:, :, 301:427, :]
y4 = net.b0_forward(x4, 3)
x5 = y[:, :, 405:523, :]
y5 = net.b0_forward(x5, 4)
x6 = y[:, :, 501:608, :]
y6 = net.b0_forward(x6, 5)

y = torch.ones(1, 128, 76, 76)
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
################# block 1 ####################

x1 = y[:, :, 0:16, :]
y1 = net.b1_forward(x1, 0)
x2 = y[:, :, 12:30, :]
y2 = net.b1_forward(x2, 1)
x3 = y[:, :, 26:42, :]
y3 = net.b1_forward(x3, 2)
x4 = y[:, :, 38:54, :]
y4 = net.b1_forward(x4, 3)
x5 = y[:, :, 50:66, :]
y5 = net.b1_forward(x5, 4)
x6 = y[:, :, 62:76, :]
y6 = net.b1_forward(x6, 5)

y = torch.ones(1, 256, 38, 38)
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
################# block 2 ####################

x1 = y[:, :, 0:8, :]
y1 = net.b2_forward(x1, 0)
x2 = y[:, :, 6:15, :]
y2 = net.b2_forward(x2, 1)
x3 = y[:, :, 13:21, :]
y3 = net.b2_forward(x3, 2)
x4 = y[:, :, 19:27, :]
y4 = net.b2_forward(x4, 3)
x5 = y[:, :, 25:33, :]
y5 = net.b2_forward(x5, 4)
x6 = y[:, :, 31:38, :]
y6 = net.b2_forward(x6, 5)

y = torch.ones(1, 256, 38, 38)
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
################# block 3 ####################

x1 = y[:, :, 0:10, :]
y1 = net.b3_forward(x1, 0)
x2 = y[:, :, 6:16, :]
y2 = net.b3_forward(x2, 1)
x3 = y[:, :, 12:22, :]
y3 = net.b3_forward(x3, 2)
x4 = y[:, :, 18:28, :]
y4 = net.b3_forward(x4, 3)
x5 = y[:, :, 24:34, :]
y5 = net.b3_forward(x5, 4)
x6 = y[:, :, 30:38, :]
y6 = net.b3_forward(x6, 5)

y = torch.ones(1, 512, 19, 19)
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
################# block 4 ####################

x1 = y[:, :, 0:5, :]
y1 = net.b4_forward(x1, 0)
x2 = y[:, :, 3:8, :]
y2 = net.b4_forward(x2, 1)
x3 = y[:, :, 6:11, :]
y3 = net.b4_forward(x3, 2)
x4 = y[:, :, 9:14, :]
y4 = net.b4_forward(x4, 3)
x5 = y[:, :, 12:17, :]
y5 = net.b4_forward(x5, 4)
x6 = y[:, :, 15:19, :]
y6 = net.b4_forward(x6, 5)

y = torch.ones(1, 512, 19, 19)
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
################# block 5 ####################

x1 = y[:, :, 0:5, :]
y1 = net.b5_forward(x1, 0)
x2 = y[:, :, 3:8, :]
y2 = net.b5_forward(x2, 1)
x3 = y[:, :, 6:11, :]
y3 = net.b5_forward(x3, 2)
x4 = y[:, :, 9:14, :]
y4 = net.b5_forward(x4, 3)
x5 = y[:, :, 12:17, :]
y5 = net.b5_forward(x5, 4)
x6 = y[:, :, 15:19, :]
y6 = net.b5_forward(x6, 5)

y = torch.ones(1, 512, 19, 19)
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
################# block 6 ####################

x1 = y[:, :, 0:8, :]
y1 = net.b6_forward(x1, 0)
x2 = y[:, :, 0:11, :]
y2 = net.b6_forward(x2, 1)
x3 = y[:, :, 3:14, :]
y3 = net.b6_forward(x3, 2)
x4 = y[:, :, 6:17, :]
y4 = net.b6_forward(x4, 3)
x5 = y[:, :, 9:19, :]
y5 = net.b6_forward(x5, 4)
x6 = y[:, :, 12:19, :]
y6 = net.b6_forward(x6, 5)

y = torch.ones(1, 425, 19, 19)
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

print(y[:50])
