# argv
import sys, os, inspect
device_num = int(sys.argv[1])
port = int(sys.argv[3])
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# communication
from socket import *
import struct
HOST = sys.argv[2]
PORT = port
# works
from PIL import Image
import numpy as np
import threading
import pickle
from pathlib import Path

pcnn_path = str(Path(__file__).parent.parent.parent.absolute())

image_path = sys.argv[4]
image = Image.open(image_path)
image = image.resize((224, 224), Image.ANTIALIAS)
# convert image to numpy array
x = np.array([np.asarray(image)[:, :, :3]])
x = torch.Tensor(list(x)).permute(0, 3, 2, 1)


y = None
cnt = 0
offset = 0

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

net = Net()
net.load_state_dict(torch.load(os.path.join(pcnn_path, 'models', 'alexnet')))

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

def sendall(sock, msg):
	msg = struct.pack('>I', len(msg)) + msg
	sock.sendall(msg)

def job(conn, condition):
	# print(conn)
	global cnt
	global offset
	global x
	global y
	global device_num
	while True:
		try:
			bytes = recvall(conn)
			if bytes is None:
				break
		except ConnectionResetError:
			break
		data = pickle.loads(bytes)
		key = data['key']
		block_id = data['blkId']
		idx = data['id']
		data_from_device = data['data']
		if key == 'get':
			# merge data
			condition.acquire()
			cnt += 1
			if data_from_device is not None:
				# print(data_from_device.shape)
				if block_id == 1:
					if cnt == 1:
						x = torch.ones(1, 256, 28, 28)
					if idx == 0:
						x[:, :, 0:4, :] = data_from_device
					elif idx == 1:
						x[:, :, 4:8, :] = data_from_device
					elif idx == 2:
						x[:, :, 8:12, :] = data_from_device
					elif idx == 3:
						x[:, :, 12:16, :] = data_from_device
					elif idx == 4:
						x[:, :, 16:20, :] = data_from_device
					elif idx == 5:
						x[:, :, 20:24, :] = data_from_device
					elif idx == 6:
						x[:, :, 24:28, :] = data_from_device
				elif block_id == 2:
					if cnt == 1:
						x = torch.ones(1, 512, 14, 14)
					if idx == 0:
						x[:, :, 0:2, :] = data_from_device
					elif idx == 1:
						x[:, :, 2:4, :] = data_from_device
					elif idx == 2:
						x[:, :, 4:6, :] = data_from_device
					elif idx == 3:
						x[:, :, 6:8, :] = data_from_device
					elif idx == 4:
						x[:, :, 8:10, :] = data_from_device
					elif idx == 5:
						x[:, :, 10:12, :] = data_from_device
					elif idx == 6:
						x[:, :, 12:14, :] = data_from_device
				elif block_id == 3:
					if cnt == 1:
						x = torch.ones(1, 512, 14, 14)
					if idx == 0:
						x[:, :, 0:2, :] = data_from_device
					elif idx == 1:
						x[:, :, 2:4, :] = data_from_device
					elif idx == 2:
						x[:, :, 4:6, :] = data_from_device
					elif idx == 3:
						x[:, :, 6:8, :] = data_from_device
					elif idx == 4:
						x[:, :, 8:10, :] = data_from_device
					elif idx == 5:
						x[:, :, 10:12, :] = data_from_device
					elif idx == 6:
						x[:, :, 12:14, :] = data_from_device
				elif block_id == 4:
					if cnt == 1:
						x = np.zeros(4096)
					x += data_from_device
				elif block_id == 5:
					if cnt == 1:
						x = np.zeros(1000)
					x += data_from_device
			if cnt < device_num:
				condition.wait()
			if cnt == device_num:
				condition.notifyAll()
				cnt = 0
			condition.release()
			# print(idx, cnt)
			# group[data['id']] = conn
			# assign data
			if block_id == 0:
				if idx == 0:
					y = x[:, :, 0:54, :]
				elif idx == 1:
					y = x[:, :, 10:86, :]
				elif idx == 2:
					y = x[:, :, 42:118, :]
				elif idx == 3:
					y = x[:, :, 74:150, :]
				elif idx == 4:
					y = x[:, :, 106:182, :]
				elif idx == 5:
					y = x[:, :, 138:214, :]
				elif idx == 6:
					y = x[:, :, 170:224, :]
			elif block_id == 1:
				if idx == 0:
					y = x[:, :, 0:8, :]
				elif idx == 1:
					y = x[:, :, 0:12, :]
				elif idx == 2:
					y = x[:, :, 4:16, :]
				elif idx == 3:
					y = x[:, :, 8:20, :]
				elif idx == 4:
					y = x[:, :, 12:24, :]
				elif idx == 5:
					y = x[:, :, 16:28, :]
				elif idx == 6:
					y = x[:, :, 20:28, :]
			elif block_id == 2:
				if idx == 0:
					y = x[:, :, 0:4, :]
				elif idx == 1:
					y = x[:, :, 0:6, :]
				elif idx == 2:
					y = x[:, :, 2:8, :]
				elif idx == 3:
					y = x[:, :, 4:10, :]
				elif idx == 4:
					y = x[:, :, 6:12, :]
				elif idx == 5:
					y = x[:, :, 8:14, :]
				elif idx == 6:
					y = x[:, :, 10:14, :]
			elif block_id == 3:
				if idx == 0:
					y = x[:, :, 0:4, :]
				elif idx == 1:
					y = x[:, :, 0:6, :]
				elif idx == 2:
					y = x[:, :, 2:8, :]
				elif idx == 3:
					y = x[:, :, 4:10, :]
				elif idx == 4:
					y = x[:, :, 6:12, :]
				elif idx == 5:
					y = x[:, :, 8:14, :]
				elif idx == 6:
					y = x[:, :, 10:14, :]
			elif block_id == 4:
				if idx == 0:
					y = relu(x + net.fc2.bias.detach().numpy())
				elif idx == 1:
					y = relu(x + net.fc2.bias.detach().numpy())
				elif idx == 2:
					y = relu(x + net.fc2.bias.detach().numpy())
				elif idx == 3:
					y = relu(x + net.fc2.bias.detach().numpy())
				elif idx == 4:
					y = relu(x + net.fc2.bias.detach().numpy())
				elif idx == 5:
					y = relu(x + net.fc2.bias.detach().numpy())
				elif idx == 6:
					y = relu(x + net.fc2.bias.detach().numpy())
			elif block_id == 5:
				y = x + net.fc3.bias.detach().numpy()
				break
			# print('to', idx, y.shape)
			sendall(conn, pickle.dumps({
				'key': 'data',
				'data': y
			}))
	conn.close()

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)

with socket(AF_INET, SOCK_STREAM) as s:
	try:
		s.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
		s.bind((HOST, PORT))
		s.listen()
		condition = threading.Condition()
		for i in range(device_num):
			conn, addr = s.accept()
			# print('a device connect')
			t = threading.Thread(
				target = job,
				args = (conn, condition)
			)
			t.start()
		for i in range(device_num):
			t.join()
		# print(y[:50])
		# print(y.view(-1).detach().numpy()[:50])
		y = softmax(y)
		index = np.argmax(y)
		print(index)
	except error:
		s.close()
