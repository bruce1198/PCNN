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
count = 0
offset = 0

def relu(x)
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
					if cnt == 1
						x = torch.ones(1, 128, 76, 76)
					if idx == 0
						x[:, :, 0:13, :] = data_from_device
					elif idx == 1
						x[:, :, 13:26, :] = data_from_device
					elif idx == 2
						x[:, :, 26:39, :] = data_from_device
					elif idx == 3
						x[:, :, 39:52, :] = data_from_device
					elif idx == 4
						x[:, :, 52:64, :] = data_from_device
					elif idx == 5
						x[:, :, 64:76, :] = data_from_device
				elif block_id == 2:
					if cnt == 1
						x = torch.ones(1, 256, 38, 38)
					if idx == 0
						x[:, :, 0:7, :] = data_from_device
					elif idx == 1
						x[:, :, 7:14, :] = data_from_device
					elif idx == 2
						x[:, :, 14:20, :] = data_from_device
					elif idx == 3
						x[:, :, 20:26, :] = data_from_device
					elif idx == 4
						x[:, :, 26:32, :] = data_from_device
					elif idx == 5
						x[:, :, 32:38, :] = data_from_device
				elif block_id == 3:
					if cnt == 1
						x = torch.ones(1, 256, 38, 38)
					if idx == 0
						x[:, :, 0:7, :] = data_from_device
					elif idx == 1
						x[:, :, 7:14, :] = data_from_device
					elif idx == 2
						x[:, :, 14:20, :] = data_from_device
					elif idx == 3
						x[:, :, 20:26, :] = data_from_device
					elif idx == 4
						x[:, :, 26:32, :] = data_from_device
					elif idx == 5
						x[:, :, 32:38, :] = data_from_device
				elif block_id == 4:
					if cnt == 1
						x = torch.ones(1, 512, 19, 19)
					if idx == 0
						x[:, :, 0:4, :] = data_from_device
					elif idx == 1
						x[:, :, 4:7, :] = data_from_device
					elif idx == 2
						x[:, :, 7:10, :] = data_from_device
					elif idx == 3
						x[:, :, 10:13, :] = data_from_device
					elif idx == 4
						x[:, :, 13:16, :] = data_from_device
					elif idx == 5
						x[:, :, 16:19, :] = data_from_device
				elif block_id == 5:
					if cnt == 1
						x = torch.ones(1, 512, 19, 19)
					if idx == 0
						x[:, :, 0:4, :] = data_from_device
					elif idx == 1
						x[:, :, 4:7, :] = data_from_device
					elif idx == 2
						x[:, :, 7:10, :] = data_from_device
					elif idx == 3
						x[:, :, 10:13, :] = data_from_device
					elif idx == 4
						x[:, :, 13:16, :] = data_from_device
					elif idx == 5
						x[:, :, 16:19, :] = data_from_device
				elif block_id == 6:
					if cnt == 1
						x = torch.ones(1, 512, 19, 19)
					if idx == 0
						x[:, :, 0:4, :] = data_from_device
					elif idx == 1
						x[:, :, 4:7, :] = data_from_device
					elif idx == 2
						x[:, :, 7:10, :] = data_from_device
					elif idx == 3
						x[:, :, 10:13, :] = data_from_device
					elif idx == 4
						x[:, :, 13:16, :] = data_from_device
					elif idx == 5
						x[:, :, 16:19, :] = data_from_device
				elif block_id == 7:
					if cnt == 1
						x = torch.ones(1, 425, 19, 19)
					if idx == 0
						x[:, :, 0:4, :] = data_from_device
					elif idx == 1
						x[:, :, 4:7, :] = data_from_device
					elif idx == 2
						x[:, :, 7:10, :] = data_from_device
					elif idx == 3
						x[:, :, 10:13, :] = data_from_device
					elif idx == 4
						x[:, :, 13:16, :] = data_from_device
					elif idx == 5
						x[:, :, 16:19, :] = data_from_device
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
					y = x[:, :, 0:115, :]
				elif idx == 1:
					y = x[:, :, 93:219, :]
				elif idx == 2:
					y = x[:, :, 197:323, :]
				elif idx == 3:
					y = x[:, :, 301:427, :]
				elif idx == 4:
					y = x[:, :, 405:523, :]
				elif idx == 5:
					y = x[:, :, 501:608, :]
			elif block_id == 1:
				if idx == 0:
					y = x[:, :, 0:16, :]
				elif idx == 1:
					y = x[:, :, 12:30, :]
				elif idx == 2:
					y = x[:, :, 26:42, :]
				elif idx == 3:
					y = x[:, :, 38:54, :]
				elif idx == 4:
					y = x[:, :, 50:66, :]
				elif idx == 5:
					y = x[:, :, 62:76, :]
			elif block_id == 2:
				if idx == 0:
					y = x[:, :, 0:8, :]
				elif idx == 1:
					y = x[:, :, 6:15, :]
				elif idx == 2:
					y = x[:, :, 13:21, :]
				elif idx == 3:
					y = x[:, :, 19:27, :]
				elif idx == 4:
					y = x[:, :, 25:33, :]
				elif idx == 5:
					y = x[:, :, 31:38, :]
			elif block_id == 3:
				if idx == 0:
					y = x[:, :, 0:10, :]
				elif idx == 1:
					y = x[:, :, 6:16, :]
				elif idx == 2:
					y = x[:, :, 12:22, :]
				elif idx == 3:
					y = x[:, :, 18:28, :]
				elif idx == 4:
					y = x[:, :, 24:34, :]
				elif idx == 5:
					y = x[:, :, 30:38, :]
			elif block_id == 4:
				if idx == 0:
					y = x[:, :, 0:5, :]
				elif idx == 1:
					y = x[:, :, 3:8, :]
				elif idx == 2:
					y = x[:, :, 6:11, :]
				elif idx == 3:
					y = x[:, :, 9:14, :]
				elif idx == 4:
					y = x[:, :, 12:17, :]
				elif idx == 5:
					y = x[:, :, 15:19, :]
			elif block_id == 5:
				if idx == 0:
					y = x[:, :, 0:5, :]
				elif idx == 1:
					y = x[:, :, 3:8, :]
				elif idx == 2:
					y = x[:, :, 6:11, :]
				elif idx == 3:
					y = x[:, :, 9:14, :]
				elif idx == 4:
					y = x[:, :, 12:17, :]
				elif idx == 5:
					y = x[:, :, 15:19, :]
			elif block_id == 6:
				if idx == 0:
					y = x[:, :, 0:8, :]
				elif idx == 1:
					y = x[:, :, 0:11, :]
				elif idx == 2:
					y = x[:, :, 3:14, :]
				elif idx == 3:
					y = x[:, :, 6:17, :]
				elif idx == 4:
					y = x[:, :, 9:19, :]
				elif idx == 5:
					y = x[:, :, 12:19, :]
			elif block_id == 7:
				y = x
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
