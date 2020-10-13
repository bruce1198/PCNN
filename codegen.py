import json
import os
import sys
import numpy as np
def write_header():
    f.write('import torch\n')
    f.write('import torch.nn as nn\n')
    f.write('import torch.nn.functional as F\n')
    f.write('import numpy as np\n')
    f.write('import json\n')
    f.write('import pickle\n')
    f.write('import os, sys, struct\n')
    f.write('from pathlib import Path\n\n')
    f.write('path = str(Path(__file__).parent.parent.parent.absolute())\n')
    f.write('sys.path.insert(0, path)\n')
    f.write('from fl import FCBlock\n\n')

def write_relu():
    f.write('def relu(x):\n')
    f.write('\treturn np.maximum(x, 0)\n\n')

def write_init():
    conv_idx = 1
    pool_idx = 1
    fc_idx   = 1
    for idx, layer in enumerate(data['layers']):
        if layer == 'conv':
            f.write('\t\tself.conv'+str(conv_idx)+' = nn.Conv2d(in_channels='+str(int(data['in_channel'][idx]))+', out_channels='+str(int(data['out_channel'][idx]))+', kernel_size='+str(int(data['filter'][idx]))+', stride='+str(int(data['stride'][idx]))+', padding=0)\n')
            conv_idx += 1
        elif layer == 'pool':
            f.write('\t\tself.pool'+str(pool_idx)+' = nn.MaxPool2d(kernel_size='+str(int(data['filter'][idx]))+', stride='+str(int(data['stride'][idx]))+')\n')
            pool_idx += 1
        elif layer == 'FL':
            f.write('\t\tself.fc'+str(fc_idx)+' = nn.Linear('+str(int(data['in_channel'][idx]))+', '+str(int(data['out_channel'][idx]))+')\n')
            fc_idx += 1
    f.write('\n')
def write_forward():
    conv_idx = 1
    pool_idx = 1
    fc_idx   = 1
    # count fully connected layer in each block
    for idx, key in enumerate(device.keys()):
        begin = int(key.split(',')[0])
        end = int(key.split(',')[1])
        for i in range(begin, end+1):
            if data['layers'][i] == 'FL':
                num_of_fc_in_block[idx] += 1

    for idx, key in enumerate(device.keys()):
        f.write('\tdef b'+str(idx)+'_forward(self, x):\n')
        layer_idx_in_block = 0
        
        begin_idx_in_layer = data['padding_info'][device_idx][key][layer_idx_in_block][0]
        end_idx_in_layer = data['padding_info'][device_idx][key][layer_idx_in_block][1]
        begin = int(key.split(',')[0])
        end = int(key.split(',')[1])
        for i in range(begin, end+1):
            if data['layers'][i] == 'conv':
                if begin_idx_in_layer < 0:
                    f.write('\t\tm = nn.ConstantPad2d(('+str(int(data['padding'][i]))+', '+str(int(data['padding'][i]))+', '+str(int(abs(begin_idx_in_layer)))+', 0), 0)\n')
                elif end_idx_in_layer >= data['input'][i]:
                    f.write('\t\tm = nn.ConstantPad2d(('+str(int(data['padding'][i]))+', '+str(int(data['padding'][i]))+', 0, '+str(int(end_idx_in_layer - data['input'][i] + 1))+'), 0)\n')
                else:
                    f.write('\t\tm = nn.ConstantPad2d(('+str(int(data['padding'][i]))+', '+str(int(data['padding'][i]))+', 0, 0), 0)\n')
                # f.write('\t\tx = self.pad(x, padding_value='+str(int(data['padding'][i]))+')\n')
                f.write('\t\tx = m(x)\n')
                f.write('\t\tx = F.relu(self.conv'+str(conv_idx)+'(x))\n')
                conv_idx += 1
            elif data['layers'][i] == 'pool':
                f.write('\t\tx = self.pool'+str(pool_idx)+'(x)\n')
                pool_idx += 1
            elif data['layers'][i] == 'FL':
                # if block have two fc layer, only ente this write function one time
                if i != begin and (data['layers'][i-1] == 'FL'):
                    continue

                
                if num_of_fc_in_block[idx] == 1:
                    f.write('\t\tx = x.view(-1).detach().numpy()\n')
                    f.write('\t\tw%d = self.fc%d.weight.data.numpy().transpose()\n' % (fc_idx, fc_idx))
                    f.write('\t\tfblk = FCBlock(\'normal\', %d, %d)\n' % (device_idx, total_device_num))
                    f.write('\t\tfblk.set_input_size(%.1f)\n' % (data['output'][i-1]))
                    f.write('\t\tfblk.append_layer(w%d)\n' % (fc_idx))
                    fc_idx += 1
                elif num_of_fc_in_block[idx] == 2:
                    f.write('\t\tx = x.view(-1).detach().numpy()\n')
                    f.write('\t\tfblk = FCBlock(\'hybrid\', %d, %d)\n' % (device_idx, total_device_num))
                    for weight_idx in range(2):
                        if weight_idx == 0:
                            f.write('\t\tfblk.set_bias(self.fc2.bias.detach().numpy())\n')
                        f.write('\t\tw%d = self.fc%d.weight.data.numpy().transpose()\n' % (fc_idx + weight_idx, fc_idx + weight_idx))
                        
                    for weight_idx in range(2):
                        f.write('\t\tfblk.append_layer(w%d)\n' % (fc_idx + weight_idx))
                    fc_idx += 2
                f.write('\t\tx = fblk.process(x)\n')
                
            layer_idx_in_block
        f.write('\t\treturn x\n')
        f.write('\n')

def write_main():
    f.write('net = Net()\n')
    f.write('net.load_state_dict(torch.load(os.path.join(path, \'models\', \'%s\')))\n\n\n' % (path))
    f.write('import socket\n\n')
    f.write('s = socket.socket()\n')
    f.write('host = sys.argv[1]\n')
    f.write('port = int(sys.argv[2])\n')
    f.write('print(host, port)\n\n')
    f.write('s.connect((host, port))\n')
    f.write('x = None\n')
    f.write('for i in range(%d):\n' % (total_block_num+1))
    f.write('\tsendall(s, pickle.dumps({\n')
    f.write('\t\t\'key\': \'get\',\n')
    f.write('\t\t\'blkId\': i,\n')
    f.write('\t\t\'id\': %d,\n' % (device_idx))
    f.write('\t\t\'data\': x\n')
    f.write('\t}))\n')
    f.write('\tif i != %d:\n' % (total_block_num))
    f.write('\t\ttry:\n')
    f.write('\t\t\tbytes = recvall(s)\n')
    f.write('\t\t\tif bytes is None:\n')
    f.write('\t\t\t\tbreak\n')
    f.write('\t\texcept ConnectionResetError:\n')
    f.write('\t\t\tbreak\n')
    f.write('\t\tdata = pickle.loads(bytes)\n')
    f.write('\t\tkey = data[\'key\']\n')
    f.write('\t\tif key == \'data\':\n')
    f.write('\t\t\tx = data[key]\n')
    f.write('\t\t\tprint(x.shape)\n')
    for block_idx in range(total_block_num):
        if block_idx == 0:
            f.write('\t\t\tif i == %d:\n' % (block_idx))
            f.write('\t\t\t\tx = net.b%d_forward(x)\n' % (block_idx))
        else :
            f.write('\t\t\telif i == %d:\n' % (block_idx))
            f.write('\t\t\t\tx = net.b%d_forward(x)\n' % (block_idx))
    f.write('\t\t\t# print(x.shape)\n')
    f.write('\t\t\t# do calculate\n')
    f.write('s.close()\n')


def write_sendall():
    f.write('def sendall(sock, msg):\n')
    f.write('\t# Prefix each message with a 4-byte length (network byte order)\n')
    f.write('\tmsg = struct.pack(\'>I\', len(msg)) + msg\n')
    f.write('\tsock.sendall(msg)\n\n')

def write_recvall():
    f.write('def recvall(sock):\n')
    f.write('\t# Read message length and unpack it into an integer\n')
    f.write('\traw_msglen = recv(sock, 4)\n')
    f.write('\tif not raw_msglen:\n')
    f.write('\t\treturn None\n')
    f.write('\tmsglen = struct.unpack(\'>I\', raw_msglen)[0]\n')
    f.write('\t# Read the message data\n')
    f.write('\treturn recv(sock, msglen)\n\n')

def write_recv():
    f.write('def recv(sock, n):\n')
    f.write('\t# Helper function to recv n bytes or return None if EOF is hit\n')
    f.write('\tdata = bytearray()\n')
    f.write('\twhile len(data) < n:\n')
    f.write('\t\tpacket = sock.recv(n - len(data))\n')
    f.write('\t\tif not packet:\n')
    f.write('\t\t\treturn None\n')
    f.write('\t\tdata.extend(packet)\n')
    f.write('\treturn data\n\n')

if not os.path.exists('codegen'):
    os.mkdir('codegen')

for model in range(4):
    with open('data/prefetch'+str(model)+'.json') as f:
        data = json.loads(f.read())

    # print(data)
    total_device_num = len(data['devices'])
    total_block_num  = len(data['devices'][0].keys())
    
    for device_idx, device in enumerate(data['devices']):
        num_of_fc_in_block = np.zeros(total_block_num)
        path = {
            0: 'yolov2',
            1: 'alexnet_tmp',
            2: 'vgg16',
            3: 'vgg19',
        }.get(model)
        if not os.path.exists('codegen/'+path):
            os.mkdir('codegen/'+path)
        with open('codegen/'+path+'/device'+str(device_idx)+'.py', 'w') as f:

            write_header()
            write_relu()
            
            ######################## write class Net ########################

            f.write('class Net(nn.Module):\n')
            f.write('\tdef __init__(self):\n')
            f.write('\t\tsuper(Net, self).__init__()\n')
            write_init()
            write_forward()
            write_sendall()
            write_recvall()
            write_recv()
            ##################### main ######################
            write_main()
            
        