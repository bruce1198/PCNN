# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# communication
from socket import *
import sys
HOST = 'localhost'
PORT = 65432
# works
from PIL import Image
import numpy as np
import threading
import pickle
# image = Image.open('images/input.jpg')
# convert image to numpy array
# x = np.asarray(image)[:, :, :3]
# x = np.array([np.arange(224*224*3).reshape(224, 224, 3)])
x = torch.ones(1, 3, 224, 224)
# print(x.shape)

cnt = 0
offset = 0
group = {

}

def relu(x):
	return np.maximum(x, 0)

def recvall(sock):
    BUFF_SIZE = 4096 # 4 KiB
    data = b''
    while True:
        part = sock.recv(BUFF_SIZE)
        # print(len(part))
        data += part
        if len(part) < BUFF_SIZE:
            # either 0 or end of data
            break
    return data

def job(conn, condition):
    # print(conn)
    global cnt
    global offset
    global group
    global x
    global info
    while True:
        try:
            bytes = recvall(conn)
        except ConnectionResetError:
            break
        if not bytes:
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
                    x = torch.ones(1, 96, 27, 27)
                    if idx == 0:
                        x[:, :, 0: 14, :] = data_from_device
                    elif idx == 1:
                        x[:, :, 14: 27, :] = data_from_device
                elif block_id == 2:
                    x = torch.ones(1, 256, 13, 13)
                    if idx == 0:
                        x[:, :, 0: 7, :] = data_from_device
                    elif idx == 1:
                        x[:, :, 7: 13, :] = data_from_device
                elif block_id == 3:
                    x = torch.ones(1, 384, 13, 13)
                    if idx == 0:
                        x[:, :, 0: 7, :] = data_from_device
                    elif idx == 1:
                        x[:, :, 7: 13, :] = data_from_device
                elif block_id == 4:
                    x = torch.zeros(1, 4096)
                    x += data_from_device
                elif block_id == 5:
                    x = torch.zeros(1, 1000)
                    x += data_from_device
                # print(data_from_device.shape)
                # y[:, :, offset: offset+y1.shape[2], :] = y1
                # offset += y1.shape[2]
                # y[:, :, offset: offset+y2.shape[2], :] = y2
                # offset += y2.shape[2]
            if cnt < 2:
                condition.wait()
            if cnt == 2:
                condition.notifyAll()
                cnt = 0
            condition.release()
            # print(idx, cnt)
            # group[data['id']] = conn
            # assign data
            if block_id == 0:
                if idx == 0:
                    y = x[:, :, 0:121, :]
                elif idx == 1:
                    y = x[:, :, 110:224, :]
            elif block_id ==1:
                if idx == 0:
                    y = x[:, :, 0:17, :]
                elif idx == 1:
                    y = x[:, :, 12:27, :]
            elif block_id == 2:
                if idx == 0:
                    y = x[:, :, 0:9, :]
                elif idx == 1:
                    y = x[:, :, 5:13, :]
            elif block_id ==3:
                if idx == 0:
                    y = x[:, :, 0:8, :]
                elif idx == 1:
                    y = x[:, :, 5:13, :]
            elif block_id ==4:
                if idx == 0:
                    y = x
                elif idx == 1:
                    y = x
            # print('to', idx, y.shape)
            conn.sendall(pickle.dumps({
                'key': 'data',
                'data': y
            }))
        # conn.
    conn.close()

device_num = int(sys.argv[1])
# print(device_num)


with socket(AF_INET, SOCK_STREAM) as s:
    try:
        s.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        condition = threading.Condition()
        for i in range(device_num):
            conn, addr = s.accept()
            t = threading.Thread(
                target = job,
                args = (conn, condition)
            )
            t.start()
        for i in range(device_num):
            t.join()
        print(x.view(-1).detach().numpy())
    except error:
        s.close()