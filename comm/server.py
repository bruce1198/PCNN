import socket
from threading import Thread
import numpy as np
import pickle

HOST = 'localhost'
PORT = 8000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen()

group = []

def job(conn):
    while True:
        bytes = conn.recv(4096)
        if not bytes:
            break
        ary = pickle.loads(bytes)
        print(ary, ary.shape)
    conn.close()

print('server start')
while True:
    conn, addr = s.accept()
    print(addr, 'connected!')
    Thread(target=job, args=(conn, )).start()
    
