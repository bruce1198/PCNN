import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pickle
import os, sys, struct
from os.path import dirname, abspath

path = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, path)
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

	def set_pre_cal_w(self, w):
		self.w = w

	def b0_forward(self, x):
		m = nn.ConstantPad2d((2, 2, 2, 0), 0)
		x = m(x)
		x = F.relu(self.conv1(x))
		x = self.pool1(x)
		return x

	def b1_forward(self, x):
		m = nn.ConstantPad2d((2, 2, 2, 0), 0)
		x = m(x)
		x = F.relu(self.conv2(x))
		x = self.pool2(x)
		return x

	def b2_forward(self, x):
		m = nn.ConstantPad2d((1, 1, 1, 0), 0)
		x = m(x)
		x = F.relu(self.conv3(x))
		return x

	def b3_forward(self, x):
		m = nn.ConstantPad2d((1, 1, 1, 0), 0)
		x = m(x)
		x = F.relu(self.conv4(x))
		return x

	def b4_forward(self, x):
		m = nn.ConstantPad2d((1, 1, 1, 0), 0)
		x = m(x)
		x = F.relu(self.conv5(x))
		x = self.pool3(x)
		x = x.view(-1).detach().numpy()
		fblk = FCBlock('normal', 0, 3)
		x = fblk.process(x)
		return x

	def b5_forward(self, x):
		fblk = FCBlock('hybrid', 0, 3)
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

import math
def pre_cal_weight(idx, device_num, input_size, originw):
	size = originw.shape[0]
	size2 = originw.shape[1]
	input_size = int(input_size)
	avg = int(math.floor(input_size/device_num))
	total = avg
	mod = input_size % device_num
	start = 0
	for ii in range(idx):
		if ii < mod:
			start += avg+1
		else:
			start += avg
	if idx < mod:
		total += 1
	height = total
	stride = input_size * input_size
	height1 = int(size * height / input_size)
	w = np.float32(np.zeros(shape=(height1, size2)))
	cnt = 0
	for i in range(start*input_size, size, stride):
		pos = cnt * height*input_size
		w[pos:pos+height*input_size, :] = originw[i:i+height*input_size, :]
		cnt += 1
	return w

import time
load = 0
comm = 0
cal  = 0
start = time.time()
net = Net()
net.load_state_dict(torch.load(os.path.join(path, 'models', 'alexnet.h5')))
pre_cal_w = pre_cal_weight(0, 3, 6, net.fc1.weight.data.numpy().transpose())
net.set_pre_cal_w(pre_cal_w)
load = time.time() - start


import socket

s = socket.socket()
host = sys.argv[1]
port = int(sys.argv[2])
# print(host, port)

s.connect((host, port))
x = None
send_data = None
for i in range(7):
	sendall(s, pickle.dumps({
		'key': 'get',
		'blkId': i,
		'id': 0,
		'data': send_data
	}))
	comm += time.time() - start
	if i != 6:
		try:
			bytes = recvall(s)
			if bytes is None:
				break
		except ConnectionResetError:
			break
		data = pickle.loads(bytes)
		comm += time.time() - start
		key = data['key']
		start = time.time()
		if key == 'data':
			if i == 0:
				x = net.b0_forward(data[key])
				send_data = x[:, :, 8:9, :]
			elif i == 1:
				x = torch.cat((x, data[key]), dim=2)
				x = net.b1_forward(x)
				send_data = x[:, :, 4:5, :]
			elif i == 2:
				x = torch.cat((x, data[key]), dim=2)
				x = net.b2_forward(x)
				send_data = x[:, :, 4:5, :]
			elif i == 3:
				x = torch.cat((x, data[key]), dim=2)
				x = net.b3_forward(x)
				send_data = x[:, :, 3:5, :]
			elif i == 4:
				x = torch.cat((x, data[key]), dim=2)
				x = net.b4_forward(x)
				send_data = x
			elif i == 5:
				x = net.b5_forward(data[key])
				send_data = x
			# print(x.shape)
			# do calculate
		cal += time.time() - start
s.close()
print(json.dumps({
	'load': int(1000*load),
	'comm': int(1000*comm),
	'cal': int(1000*cal),
}))
