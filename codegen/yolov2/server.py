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
from os.path import abspath, dirname
# estimate
import time
load_time = 0
cal_time = 0
pcnn_path = dirname(dirname(dirname(abspath(__file__))))

image_path = sys.argv[4]
image = Image.open(image_path)
image = image.resize((224, 224), Image.ANTIALIAS)
# convert image to numpy array
x = np.array([np.asarray(image)[:, :, :3]])
x = torch.Tensor(list(x)).permute(0, 3, 2, 1)


y = None
cnt = 0

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

comm_time = 0

def job(conn, condition):
	# print(conn)
	global cnt
	global x
	global y
	global device_num
	global comm_time
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
						x = torch.ones(1, 128, 76, 76)
					if idx == 0:
						x[:, :, 8: 9, :] = data_from_device
					if idx == 1:
						x[:, :, 9: 12, :] = data_from_device[:, :, 0:3, :]
						x[:, :, 17: 18, :] = data_from_device[:, :, 3:4, :]
					if idx == 2:
						x[:, :, 18: 22, :] = data_from_device[:, :, 0:4, :]
						x[:, :, 26: 27, :] = data_from_device[:, :, 4:5, :]
					if idx == 3:
						x[:, :, 27: 30, :] = data_from_device[:, :, 0:3, :]
						x[:, :, 34: 36, :] = data_from_device[:, :, 3:5, :]
					if idx == 4:
						x[:, :, 36: 38, :] = data_from_device[:, :, 0:2, :]
						x[:, :, 42: 44, :] = data_from_device[:, :, 2:4, :]
					if idx == 5:
						x[:, :, 44: 46, :] = data_from_device[:, :, 0:2, :]
						x[:, :, 50: 52, :] = data_from_device[:, :, 2:4, :]
					if idx == 6:
						x[:, :, 52: 54, :] = data_from_device[:, :, 0:2, :]
						x[:, :, 58: 60, :] = data_from_device[:, :, 2:4, :]
					if idx == 7:
						x[:, :, 60: 62, :] = data_from_device[:, :, 0:2, :]
						x[:, :, 66: 68, :] = data_from_device[:, :, 2:4, :]
					if idx == 8:
						x[:, :, 68: 70, :] = data_from_device
				elif block_id == 2:
					if cnt == 1:
						x = torch.ones(1, 256, 38, 38)
					if idx == 0:
						x[:, :, 3: 5, :] = data_from_device
					if idx == 1:
						x[:, :, 5: 10, :] = data_from_device
					if idx == 2:
						x[:, :, 10: 14, :] = data_from_device
					if idx == 3:
						x[:, :, 14: 18, :] = data_from_device
					if idx == 4:
						x[:, :, 18: 22, :] = data_from_device
					if idx == 5:
						x[:, :, 22: 26, :] = data_from_device
					if idx == 6:
						x[:, :, 26: 30, :] = data_from_device
					if idx == 7:
						x[:, :, 30: 34, :] = data_from_device
					if idx == 8:
						x[:, :, 34: 37, :] = data_from_device
				elif block_id == 3:
					if cnt == 1:
						x = torch.ones(1, 512, 19, 19)
					if idx == 0:
						x[:, :, 2: 3, :] = data_from_device
					if idx == 1:
						x[:, :, 3: 5, :] = data_from_device
					if idx == 2:
						x[:, :, 5: 7, :] = data_from_device
					if idx == 3:
						x[:, :, 7: 9, :] = data_from_device
					if idx == 4:
						x[:, :, 9: 11, :] = data_from_device
					if idx == 5:
						x[:, :, 11: 13, :] = data_from_device
					if idx == 6:
						x[:, :, 13: 15, :] = data_from_device
					if idx == 7:
						x[:, :, 15: 17, :] = data_from_device
					if idx == 8:
						x[:, :, 17: 18, :] = data_from_device
				elif block_id == 4:
					if cnt == 1:
						x = torch.ones(1, 512, 19, 19)
					if idx == 0:
						x[:, :, 2: 3, :] = data_from_device
					if idx == 1:
						x[:, :, 3: 5, :] = data_from_device
					if idx == 2:
						x[:, :, 5: 7, :] = data_from_device
					if idx == 3:
						x[:, :, 7: 9, :] = data_from_device
					if idx == 4:
						x[:, :, 9: 11, :] = data_from_device
					if idx == 5:
						x[:, :, 11: 13, :] = data_from_device
					if idx == 6:
						x[:, :, 13: 15, :] = data_from_device
					if idx == 7:
						x[:, :, 15: 17, :] = data_from_device
					if idx == 8:
						x[:, :, 17: 18, :] = data_from_device
				elif block_id == 5:
					if cnt == 1:
						x = torch.ones(1, 512, 19, 19)
					if idx == 0:
						x[:, :, 1: 3, :] = data_from_device
					if idx == 1:
						x[:, :, 3: 5, :] = data_from_device
					if idx == 2:
						x[:, :, 5: 7, :] = data_from_device
					if idx == 3:
						x[:, :, 7: 9, :] = data_from_device
					if idx == 4:
						x[:, :, 9: 11, :] = data_from_device
					if idx == 5:
						x[:, :, 11: 13, :] = data_from_device
					if idx == 6:
						x[:, :, 13: 15, :] = data_from_device
					if idx == 7:
						x[:, :, 15: 17, :] = data_from_device
					if idx == 8:
						x[:, :, 17: 19, :] = data_from_device
				elif block_id == 6:
					if cnt == 1:
						x = torch.ones(1, 1024, 19, 19)
					if idx == 0:
						x[:, :, 1: 3, :] = data_from_device
					if idx == 1:
						x[:, :, 3: 5, :] = data_from_device
					if idx == 2:
						x[:, :, 5: 7, :] = data_from_device
					if idx == 3:
						x[:, :, 7: 9, :] = data_from_device
					if idx == 4:
						x[:, :, 9: 11, :] = data_from_device
					if idx == 5:
						x[:, :, 11: 13, :] = data_from_device
					if idx == 6:
						x[:, :, 13: 15, :] = data_from_device
					if idx == 7:
						x[:, :, 15: 17, :] = data_from_device
					if idx == 8:
						x[:, :, 17: 19, :] = data_from_device
				elif block_id == 7:
					if cnt == 1:
						x = torch.ones(1, 425, 19, 19)
					if idx == 0:
						x[:, :, 0:3, :] = data_from_device
					elif idx == 1:
						x[:, :, 3:5, :] = data_from_device
					elif idx == 2:
						x[:, :, 5:7, :] = data_from_device
					elif idx == 3:
						x[:, :, 7:9, :] = data_from_device
					elif idx == 4:
						x[:, :, 9:11, :] = data_from_device
					elif idx == 5:
						x[:, :, 11:13, :] = data_from_device
					elif idx == 6:
						x[:, :, 13:15, :] = data_from_device
					elif idx == 7:
						x[:, :, 15:17, :] = data_from_device
					elif idx == 8:
						x[:, :, 17:19, :] = data_from_device
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
					y = x[:, :, 0:83, :]
				elif idx == 1:
					y = x[:, :, 61:155, :]
				elif idx == 2:
					y = x[:, :, 133:227, :]
				elif idx == 3:
					y = x[:, :, 205:299, :]
				elif idx == 4:
					y = x[:, :, 277:363, :]
				elif idx == 5:
					y = x[:, :, 341:427, :]
				elif idx == 6:
					y = x[:, :, 405:491, :]
				elif idx == 7:
					y = x[:, :, 469:555, :]
				elif idx == 8:
					y = x[:, :, 533:608, :]
			elif block_id == 1:
				if idx == 0:
					y = x[:, :, 9:12, :]
				elif idx == 1:
					y = torch.cat((x[:, :, 8:9, :], x[:, :, 18:22, :]), dim=2)
				elif idx == 2:
					y = x[:, :, 27:30, :]
				elif idx == 3:
					y = torch.cat((x[:, :, 26:27, :], x[:, :, 36:38, :]), dim=2)
				elif idx == 4:
					y = torch.cat((x[:, :, 34:36, :], x[:, :, 44:46, :]), dim=2)
				elif idx == 5:
					y = torch.cat((x[:, :, 42:44, :], x[:, :, 52:54, :]), dim=2)
				elif idx == 6:
					y = torch.cat((x[:, :, 50:52, :], x[:, :, 60:62, :]), dim=2)
				elif idx == 7:
					y = torch.cat((x[:, :, 58:60, :], x[:, :, 68:70, :]), dim=2)
				elif idx == 8:
					y = x[:, :, 66:68, :]
			elif block_id == 2:
				if idx == 0:
					y = x[:, :, 5:9, :]
				elif idx == 1:
					y = torch.cat((x[:, :, 3:5, :], x[:, :, 10:13, :]), dim=2)
				elif idx == 2:
					y = torch.cat((x[:, :, 7:10, :], x[:, :, 14:17, :]), dim=2)
				elif idx == 3:
					y = torch.cat((x[:, :, 11:14, :], x[:, :, 18:21, :]), dim=2)
				elif idx == 4:
					y = torch.cat((x[:, :, 15:18, :], x[:, :, 22:25, :]), dim=2)
				elif idx == 5:
					y = torch.cat((x[:, :, 19:22, :], x[:, :, 26:29, :]), dim=2)
				elif idx == 6:
					y = torch.cat((x[:, :, 23:26, :], x[:, :, 30:33, :]), dim=2)
				elif idx == 7:
					y = torch.cat((x[:, :, 27:30, :], x[:, :, 34:37, :]), dim=2)
				elif idx == 8:
					y = x[:, :, 31:34, :]
			elif block_id == 3:
				if idx == 0:
					y = x[:, :, 3:4, :]
				elif idx == 1:
					y = torch.cat((x[:, :, 2:3, :], x[:, :, 5:6, :]), dim=2)
				elif idx == 2:
					y = torch.cat((x[:, :, 4:5, :], x[:, :, 7:8, :]), dim=2)
				elif idx == 3:
					y = torch.cat((x[:, :, 6:7, :], x[:, :, 9:10, :]), dim=2)
				elif idx == 4:
					y = torch.cat((x[:, :, 8:9, :], x[:, :, 11:12, :]), dim=2)
				elif idx == 5:
					y = torch.cat((x[:, :, 10:11, :], x[:, :, 13:14, :]), dim=2)
				elif idx == 6:
					y = torch.cat((x[:, :, 12:13, :], x[:, :, 15:16, :]), dim=2)
				elif idx == 7:
					y = torch.cat((x[:, :, 14:15, :], x[:, :, 17:18, :]), dim=2)
				elif idx == 8:
					y = x[:, :, 16:17, :]
			elif block_id == 4:
				if idx == 0:
					y = x[:, :, 3:4, :]
				elif idx == 1:
					y = torch.cat((x[:, :, 2:3, :], x[:, :, 5:6, :]), dim=2)
				elif idx == 2:
					y = torch.cat((x[:, :, 4:5, :], x[:, :, 7:8, :]), dim=2)
				elif idx == 3:
					y = torch.cat((x[:, :, 6:7, :], x[:, :, 9:10, :]), dim=2)
				elif idx == 4:
					y = torch.cat((x[:, :, 8:9, :], x[:, :, 11:12, :]), dim=2)
				elif idx == 5:
					y = torch.cat((x[:, :, 10:11, :], x[:, :, 13:14, :]), dim=2)
				elif idx == 6:
					y = torch.cat((x[:, :, 12:13, :], x[:, :, 15:16, :]), dim=2)
				elif idx == 7:
					y = torch.cat((x[:, :, 14:15, :], x[:, :, 17:18, :]), dim=2)
				elif idx == 8:
					y = x[:, :, 16:17, :]
			elif block_id == 5:
				if idx == 0:
					y = x[:, :, 3:5, :]
				elif idx == 1:
					y = torch.cat((x[:, :, 1:3, :], x[:, :, 5:7, :]), dim=2)
				elif idx == 2:
					y = torch.cat((x[:, :, 3:5, :], x[:, :, 7:9, :]), dim=2)
				elif idx == 3:
					y = torch.cat((x[:, :, 5:7, :], x[:, :, 9:11, :]), dim=2)
				elif idx == 4:
					y = torch.cat((x[:, :, 7:9, :], x[:, :, 11:13, :]), dim=2)
				elif idx == 5:
					y = torch.cat((x[:, :, 9:11, :], x[:, :, 13:15, :]), dim=2)
				elif idx == 6:
					y = torch.cat((x[:, :, 11:13, :], x[:, :, 15:17, :]), dim=2)
				elif idx == 7:
					y = torch.cat((x[:, :, 13:15, :], x[:, :, 17:19, :]), dim=2)
				elif idx == 8:
					y = x[:, :, 15:17, :]
			elif block_id == 6:
				if idx == 0:
					y = x[:, :, 3:5, :]
				elif idx == 1:
					y = torch.cat((x[:, :, 1:3, :], x[:, :, 5:7, :]), dim=2)
				elif idx == 2:
					y = torch.cat((x[:, :, 3:5, :], x[:, :, 7:9, :]), dim=2)
				elif idx == 3:
					y = torch.cat((x[:, :, 5:7, :], x[:, :, 9:11, :]), dim=2)
				elif idx == 4:
					y = torch.cat((x[:, :, 7:9, :], x[:, :, 11:13, :]), dim=2)
				elif idx == 5:
					y = torch.cat((x[:, :, 9:11, :], x[:, :, 13:15, :]), dim=2)
				elif idx == 6:
					y = torch.cat((x[:, :, 11:13, :], x[:, :, 15:17, :]), dim=2)
				elif idx == 7:
					y = torch.cat((x[:, :, 13:15, :], x[:, :, 17:19, :]), dim=2)
				elif idx == 8:
					y = x[:, :, 15:17, :]
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

start_time = time.time()
index = -1
with socket(AF_INET, SOCK_STREAM) as s:
	try:
		s.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
		s.bind((HOST, PORT))
		s.listen()
		start_time = time.time()
		net = Net()
		net.load_state_dict(torch.load(os.path.join(pcnn_path, 'models', 'yolov2.h5')))
		load_time = time.time() - start_time
		condition = threading.Condition()
		threads = []
		for i in range(device_num):
			conn, addr = s.accept()
			# print('a device connect')
			t = threading.Thread(
				target = job,
				args = (conn, condition)
			)
			threads.append(t)
			t.start()
		start_time = time.time()
		for t in threads:
			t.join()
		# print(y[:50])
		# print(y.view(-1).detach().numpy()[:50])
		y = softmax(y)
		index = np.argmax(y)
		# print(index)
	except error:
		s.close()
cal_time = time.time() - start_time
import json
print(json.dumps({
	'index': int(index),
	'load_time': int(1000*load_time),
	'cal_time': int(1000*cal_time)
}))
