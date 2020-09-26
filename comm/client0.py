import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pickle
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from pathlib import Path

path = str(Path(__file__).parent.parent.absolute())
from fl import FCBlock

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

	def b0_forward(self, x):
		x = self.pad(x, padding_value=2)
		x = F.relu(self.conv1(x))
		x = self.pool1(x)
		return x

	def b1_forward(self, x):
		x = self.pad(x, padding_value=2)
		x = F.relu(self.conv2(x))
		x = self.pool2(x)
		return x

	def b2_forward(self, x):
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv3(x))
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv4(x))
		return x

	def b3_forward(self, x):
		x = self.pad(x, padding_value=1)
		x = F.relu(self.conv5(x))
		x = self.pool3(x)
		x = x.view(-1).detach().numpy()
		w = self.fc1.weight.data.numpy().transpose()
		fblk = FCBlock('normal', 0, 2)
		fblk.set_input_size(6.0)
		fblk.append_layer(w)
		x = fblk.process(x)
		return x

	def b4_forward(self, x):
		w1 = self.fc2.weight.data.numpy().transpose()
		w2 = self.fc3.weight.data.numpy().transpose()
		fblk = FCBlock('hybrid', 0, 2)
		fblk.set_bias(self.fc2.bias.detach().numpy())
		fblk.append_layer(w1)
		fblk.append_layer(w2)
		x = fblk.process(x)
		return x

	def pad(self, x, padding_value):
		m = nn.ConstantPad2d((padding_value, padding_value, padding_value, 0), 0)
		x = m(x)
		return x

net = Net()
net.load_state_dict(torch.load('../models/alexnet'))


import socket
 
s = socket.socket()
host = 'localhost'
port = 65432

def recvall(sock):
    BUFF_SIZE = 4096 # 4 KiB
    data = b''
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            # either 0 or end of data
            break
    return data

s.connect((host, port))
x = None
for i in range(6):
	s.send(pickle.dumps({
		'key': 'get',
		'blkId': i,
		'id': 0,
		'data': x
	}))
	if i != 5:
		bytes = recvall(s)
		data = pickle.loads(bytes)
		key = data['key']
		if key == 'data':
			x = data[key]
			print(x.shape)
			if i == 0:
				x = net.b0_forward(x)
			elif i == 1:
				x = net.b1_forward(x)
			elif i == 2:
				x = net.b2_forward(x)
			elif i == 3:
				x = net.b3_forward(x)
			elif i == 4:
				x = net.b4_forward(x)
			print(x.shape)
			# do calulate
s.close()
################# block 0 ####################

# x = torch.ones(1, 3, 121, 224)
# y = net.b0_forward(x)

# #TODO
# #Send y to the server and get the new input.

# ################# block 1 ####################

# x = y[:, :, 0:17, :]
# y = net.b1_forward(x)

# #TODO
# #Send y to the server and get the new input.

# ################# block 2 ####################

# x = y[:, :, 0:9, :]
# y = net.b2_forward(x)

# #TODO
# #Send y to the server and get the new input.

# ################# block 3 ####################

# x = y[:, :, 0:8, :]
# y = net.b3_forward(x)

# #TODO
# #Send y to the server and get the new input.

# ################# block 4 ####################

# x = y[:, :, 0:4096, :]
# y = net.b4_forward(x)

#TODO
#Send y to the server and get the new input.

