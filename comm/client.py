import socket
import numpy as np
import pickle
HOST = 'localhost'
PORT = 8000

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))

while True:
    sendmsg = input()
    if sendmsg == 'bye':
        break
    else:
        msg = pickle.dumps(np.arange(12).reshape(3, 4))
    client.send(msg)
    # bytes = client.recv(4096)
    # msg = str(bytes, encoding='utf-8')
    # print(msg)
client.close()

class Client():
    def __init__(self, ip, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip = ip
        self.port = port
        
    def connect(self):
        self.socket.connect(self.ip, self.port)
    
    def send(self,data):
        client.send(pickle.dumps(data))
        bytes = client.recv(4096)
        return pickle.loads(bytes)
        # return np.fromstring()