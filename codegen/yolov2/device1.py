import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pickle
import os, sys, struct
from pathlib import Path

path = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, path)
from fl import FCBlock

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

	def b0_forward(self, x):
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv1(x))
		x = self.pool1(x)
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv2(x))
		x = self.pool2(x)
		m = nn.ConstantPad2d((1, 1, 0, 67), 0)
		x = m(x)
		x = F.relu(self.conv3(x))
		m = nn.ConstantPad2d((0, 0, 0, 67), 0)
		x = m(x)
		x = F.relu(self.conv4(x))
		m = nn.ConstantPad2d((1, 1, 0, 67), 0)
		x = m(x)
		x = F.relu(self.conv5(x))
		x = self.pool3(x)
		return x

	def b1_forward(self, x):
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv6(x))
		m = nn.ConstantPad2d((0, 0, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv7(x))
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv8(x))
		x = self.pool4(x)
		return x

	def b2_forward(self, x):
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv9(x))
		m = nn.ConstantPad2d((0, 0, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv10(x))
		return x

	def b3_forward(self, x):
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv11(x))
		m = nn.ConstantPad2d((0, 0, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv12(x))
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv13(x))
		x = self.pool5(x)
		return x

	def b4_forward(self, x):
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv14(x))
		m = nn.ConstantPad2d((0, 0, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv15(x))
		return x

	def b5_forward(self, x):
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv16(x))
		m = nn.ConstantPad2d((0, 0, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv17(x))
		return x

	def b6_forward(self, x):
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv18(x))
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv19(x))
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv20(x))
		m = nn.ConstantPad2d((1, 1, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv21(x))
		m = nn.ConstantPad2d((0, 0, 0, 0), 0)
		x = m(x)
		x = F.relu(self.conv22(x))
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
net.load_state_dict(torch.load(os.path.join(path, 'models', 'yolov2')))


import socket

s = socket.socket()
host = sys.argv[1]
port = int(sys.argv[2])
print(host, port)

s.connect((host, port))
x = None
for i in range(8):
	sendall(s, pickle.dumps({
		'key': 'get',
		'blkId': i,
		'id': 1,
		'data': x
	}))
	if i != 7:
		try:
			bytes = recvall(s)
			if bytes is None:
				break
		except ConnectionResetError:
			break
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
			elif i == 5:
				x = net.b5_forward(x)
			elif i == 6:
				x = net.b6_forward(x)
			# print(x.shape)
			# do calculate
s.close()
