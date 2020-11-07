import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pickle
import os, sys, struct
from pathlib import Path

path = dirname(dirname(abspath(__file__)))
sys.path.insert(0, path)
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
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv1(x))
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv2(x))
		x = self.pool1(x)
		m = nn.ConstantPad2d((1, 1, 0, 70), 0)
		x = m(x)
		x = F.relu(self.conv3(x))
		m = nn.ConstantPad2d((1, 1, 0, 70), 0)
		x = m(x)
		x = F.relu(self.conv4(x))
		x = self.pool2(x)
		m = nn.ConstantPad2d((1, 1, 0, 126), 0)
		x = m(x)
		x = F.relu(self.conv5(x))
		m = nn.ConstantPad2d((1, 1, 0, 126), 0)
		x = m(x)
		x = F.relu(self.conv6(x))
		m = nn.ConstantPad2d((1, 1, 0, 126), 0)
		x = m(x)
		x = F.relu(self.conv7(x))
		m = nn.ConstantPad2d((1, 1, 0, 126), 0)
		x = m(x)
		x = F.relu(self.conv8(x))
		x = self.pool3(x)
		return x

	def b1_forward(self, x):
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv9(x))
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv10(x))
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv11(x))
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv12(x))
		x = self.pool4(x)
		return x

	def b2_forward(self, x):
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv13(x))
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv14(x))
		return x

	def b3_forward(self, x):
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv15(x))
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv16(x))
		x = self.pool5(x)
		x = x.view(-1).detach().numpy()
		w1 = self.fc1.weight.data.numpy().transpose()
		fblk = FCBlock('normal', 4, 7)
		fblk.set_input_size(7.0)
		fblk.append_layer(w1)
		x = fblk.process(x)
		return x

	def b4_forward(self, x):
		x = x.view(-1).detach().numpy()
		fblk = FCBlock('hybrid', 4, 7)
		fblk.set_bias(self.fc2.bias.detach().numpy())
		w2 = self.fc2.weight.data.numpy().transpose()
		w3 = self.fc3.weight.data.numpy().transpose()
		fblk.append_layer(w2)
		fblk.append_layer(w3)
		x = fblk.process(x)
		return x

def sendall(sock, msg):
	# Prefix each message with a 4-byte length (network byte order)
	msg = struct.pack('>I', len(msg)) + msg
	sock.sendall(msg)

def recvall(sock):
	# Read message length and unpack it into an integer
	raw_msglen = recv(sock, 4)
	if not raw_msglen:
		return None
	msglen = struct.unpack('>I', raw_msglen)[0]
	# Read the message data
	return recv(sock, msglen)

def recv(sock, n):
	# Helper function to recv n bytes or return None if EOF is hit
	data = bytearray()
	while len(data) < n:
		packet = sock.recv(n - len(data))
		if not packet:
			return None
		data.extend(packet)
	return data

net = Net()
net.load_state_dict(torch.load(os.path.join(path, 'models', 'vgg19')))


import socket

s = socket.socket()
host = sys.argv[1]
port = int(sys.argv[2])
print(host, port)

s.connect((host, port))
x = None
send_data = None
for i in range(6):
	sendall(s, pickle.dumps({
		'key': 'get',
		'blkId': i,
		'id': 4,
		'data': send_data
	}))
	if i != 5:
		try:
			bytes = recvall(s)
			if bytes is None:
				break
		except ConnectionResetError:
			break
		data = pickle.loads(bytes)
		key = data['key']
		if key == 'data':
			if i == 0:
				x = net.b0_forward(data[key])
				send_data = x[:, :, 0:4, :]
			elif i == 1:
				x = torch.cat((x, data[key]), dim=2)
				x = net.b1_forward(data[key])
				send_data = x[:, :, 0:2, :]
			elif i == 2:
				x = torch.cat((x, data[key]), dim=2)
				x = net.b2_forward(data[key])
				send_data = x[:, :, 0:2, :]
			elif i == 3:
				x = torch.cat((x, data[key]), dim=2)
				x = net.b3_forward(data[key])
				send_data = x
			elif i == 4:
				x = net.b4_forward(data[key])
				send_data = x
			# print(x.shape)
			# do calculate
s.close()
