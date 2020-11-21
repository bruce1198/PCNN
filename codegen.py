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
    f.write('from os.path import dirname, abspath\n\n')
    f.write('path = dirname(dirname(dirname(abspath(__file__))))\n')
    f.write('sys.path.insert(0, path)\n')
    f.write('from fl import FCBlock\n\n')

def write_relu():
    f.write('def relu(x):\n')
    f.write('\treturn np.maximum(x, 0)\n\n')

def write_init():
    f.write('\tdef __init__(self):\n')
    f.write('\t\tsuper(Net, self).__init__()\n')
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

def write_set_pre_cal_w():
    f.write('\tdef set_pre_cal_w(self, w):\n')
    f.write('\t\tself.w = w\n')
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
        
        
        begin = int(key.split(',')[0])
        end = int(key.split(',')[1])

        for i in range(begin, end+1):
            begin_idx_in_layer = data['padding_info'][device_idx][key][layer_idx_in_block][0]
            end_idx_in_layer = data['padding_info'][device_idx][key][layer_idx_in_block][1]
            
            if data['layers'][i] == 'conv':
                if i == end:
                    if device_idx == 0:
                        f.write('\t\tm = nn.ConstantPad2d((%d, %d, %d, 0), 0)\n' % (int(data['padding'][i]), int(data['padding'][i]), int(data['padding'][i])))
                    elif device_idx == total_device_num-1:
                        f.write('\t\tm = nn.ConstantPad2d((%d, %d, 0, %d), 0)\n' % (int(data['padding'][i]), int(data['padding'][i]), int(data['padding'][i])))
                    else:
                        f.write('\t\tm = nn.ConstantPad2d((%d, %d, 0, 0), 0)\n' % (int(data['padding'][i]), int(data['padding'][i])))
                else:
                    if begin_idx_in_layer < 0:
                        f.write('\t\tm = nn.ConstantPad2d((%d, %d, %d, 0), 0)\n' % (int(data['padding'][i]), int(data['padding'][i]), int(abs(begin_idx_in_layer))))
                    elif end_idx_in_layer >= data['input'][i]:
                        f.write('\t\tm = nn.ConstantPad2d((%d, %d, 0, %d), 0)\n' % (int(data['padding'][i]), int(data['padding'][i]), int(end_idx_in_layer - data['input'][i] + 1)))
                    else:
                        f.write('\t\tm = nn.ConstantPad2d((%d, %d, 0, 0), 0)\n' % (int(data['padding'][i]), int(data['padding'][i])))
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
                    # f.write('\t\tw%d = self.fc%d.weight.data.numpy().transpose()\n' % (fc_idx, fc_idx))
                    f.write('\t\tfblk = FCBlock(\'normal\', %d, %d)\n' % (device_idx, total_device_num))
                    # f.write('\t\tfblk.set_input_size(%.1f)\n' % (data['output'][i-1]))
                    # f.write('\t\tfblk.append_layer(w%d)\n' % (fc_idx))
                    fc_idx += 1
                elif num_of_fc_in_block[idx] == 2:
                    # f.write('\t\tx = x.view(-1).detach().numpy()\n')
                    f.write('\t\tfblk = FCBlock(\'hybrid\', %d, %d)\n' % (device_idx, total_device_num))
                    for weight_idx in range(2):
                        if weight_idx == 0:
                            f.write('\t\tfblk.set_bias(self.fc2.bias.detach().numpy())\n')
                        f.write('\t\tw%d = self.fc%d.weight.data.numpy().transpose()\n' % (fc_idx + weight_idx, fc_idx + weight_idx))
                        
                    for weight_idx in range(2):
                        f.write('\t\tfblk.append_layer(w%d)\n' % (fc_idx + weight_idx))
                    fc_idx += 2
                f.write('\t\tx = fblk.process(x)\n')
                
            layer_idx_in_block += 1
        f.write('\t\treturn x\n')
        f.write('\n')

def write_main():
    f.write('import time\n')
    f.write('load = 0\n')
    f.write('comm = 0\n')
    f.write('cal  = 0\n')
    f.write('start = time.time()\n')
    f.write('net = Net()\n')
    f.write('net.load_state_dict(torch.load(os.path.join(path, \'models\', \'%s.h5\')))\n' % (path))
    # check whether there is fc layer
    have_fc = False
    for layer in data['layers']:
        if layer == 'FL':
            have_fc = True
            break
    # record input size of fc
    input_size = 0
    for idx, output in enumerate(data['output']):
        if output == '':
            input_size = data['output'][idx-1]
            break
    if have_fc == True:
        f.write('pre_cal_w = pre_cal_weight(%d, %d, %d, net.fc1.weight.data.numpy().transpose())\n' % (device_idx, total_device_num, input_size))
        f.write('net.set_pre_cal_w(pre_cal_w)\n')
    f.write('load = time.time() - start\n\n\n')

    f.write('import socket\n\n')
    f.write('s = socket.socket()\n')
    f.write('host = sys.argv[1]\n')
    f.write('port = int(sys.argv[2])\n')
    f.write('# print(host, port)\n\n')
    f.write('s.connect((host, port))\n')
    f.write('x = None\n')
    f.write('send_data = None\n')
    f.write('for i in range(%d):\n' % (total_block_num+1))
    f.write('\tstart = time.time()\n')
    f.write('\tsendall(s, pickle.dumps({\n')
    f.write('\t\t\'key\': \'get\',\n')
    f.write('\t\t\'blkId\': i,\n')
    f.write('\t\t\'id\': %d,\n' % (device_idx))
    f.write('\t\t\'data\': send_data\n')
    f.write('\t}))\n')
    f.write('\tcomm += time.time() - start\n')
    f.write('\tif i != %d:\n' % (total_block_num))
    f.write('\t\ttry:\n')
    f.write('\t\t\tbytes = recvall(s)\n')
    f.write('\t\t\tif bytes is None:\n')
    f.write('\t\t\t\tbreak\n')
    f.write('\t\texcept ConnectionResetError:\n')
    f.write('\t\t\tbreak\n')
    f.write('\t\tdata = pickle.loads(bytes)\n')
    f.write('\t\tcomm += time.time() - start\n')
    f.write('\t\tkey = data[\'key\']\n')
    f.write('\t\tstart = time.time()\n')
    f.write('\t\tif key == \'data\':\n')
    # f.write('\t\t\tx = data[key]\n')
    # f.write('\t\t\tprint(x.shape)\n')
    prev_key = None
    for block_idx, key in enumerate(data['padding_info'][device_idx]):
        if block_idx == 0:
            f.write('\t\t\tif i == %d:\n' % (block_idx))
        else :
            f.write('\t\t\telif i == %d:\n' % (block_idx))

        if block_idx != 0 and data['layers'][int(key.split(',')[0])] == 'conv':
            current_begin = data['devices'][device_idx][key][0]
            current_end = data['devices'][device_idx][key][1]
            prev_begin = data['padding_info'][device_idx][prev_key][-1][0]
            prev_end = data['padding_info'][device_idx][prev_key][-1][1]

            if current_begin < prev_begin and current_end > prev_end:
                f.write('\t\t\t\tx = torch.cat((data[key][:, :, %d:%d, :], x, data[key][:, :, %d:%d, :]), dim=2) \n'% (\
                    0, prev_begin-current_begin, prev_begin-current_begin, (prev_begin-current_begin) + (current_end-prev_end)
                    ))
            elif current_begin < prev_begin:
                f.write('\t\t\t\tx = torch.cat((data[key], x), dim=2)\n')
            elif current_end > prev_end:
                f.write('\t\t\t\tx = torch.cat((x, data[key]), dim=2)\n')
            # if model == 1:
            #     print(current_begin, current_end, prev_begin, prev_end)
            f.write('\t\t\t\tx = net.b%d_forward(x)\n' % (block_idx))
        else:
            f.write('\t\t\t\tx = net.b%d_forward(data[key])\n' % (block_idx))
        if block_idx < len(mask_list):
            output_begin_idx_in_block = data['padding_info'][device_idx][key][-1][0]
            output_end_idx_in_block = data['padding_info'][device_idx][key][-1][1]
            # print(output_begin_idx_in_block, output_end_idx_in_block)
            begin_index_list = [] 
            end_index_list = [] 
            for iter_ in range(output_begin_idx_in_block, output_end_idx_in_block+1):
                # data index needed to be transmitted
                if(mask_list[block_idx][iter_] != -1 and iter_ == output_begin_idx_in_block) \
                    or (mask_list[block_idx][iter_-1] == -1 and mask_list[block_idx][iter_] != -1):
                    begin_index_list.append(iter_)
                if(mask_list[block_idx][iter_] != -1 and iter_ == output_end_idx_in_block) \
                    or (iter_ <= output_end_idx_in_block and mask_list[block_idx][iter_] != -1 and mask_list[block_idx][iter_+1] == -1):
                    end_index_list.append(iter_)
            if len(begin_index_list) > 1:
                f.write('\t\t\t\tsend_data = torch.cat((x[:, :, %d:%d, :], x[:, :, %d:%d, :]), dim=2)\n' % (\
                    begin_index_list[0]-output_begin_idx_in_block, end_index_list[0]-output_begin_idx_in_block+1,\
                    begin_index_list[1]-output_begin_idx_in_block, end_index_list[1]-output_begin_idx_in_block+1\
                        ))
            else:
                f.write('\t\t\t\tsend_data = x[:, :, %d:%d, :]\n' % \
                    (begin_index_list[0]-output_begin_idx_in_block, end_index_list[0]-output_begin_idx_in_block+1))
        else:
            f.write('\t\t\t\tsend_data = x\n')
        prev_key = key
        
        
    f.write('\t\t\t# print(x.shape)\n')
    f.write('\t\t\t# do calculate\n')
    f.write('\t\tcal += time.time() - start\n')
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

def write_pre_cal_weight():
    f.write('import math\n')
    f.write('def pre_cal_weight(idx, device_num, input_size, originw):\n')
    f.write('\tsize = originw.shape[0]\n')
    f.write('\tsize2 = originw.shape[1]\n')
    f.write('\tinput_size = int(input_size)\n')
    f.write('\tavg = int(math.floor(input_size/device_num))\n')
    f.write('\ttotal = avg\n')
    f.write('\tmod = input_size % device_num\n')
    f.write('\tstart = 0\n')
    f.write('\tfor ii in range(idx):\n')
    f.write('\t\tif ii < mod:\n')
    f.write('\t\t\tstart += avg+1\n')
    f.write('\t\telse:\n')
    f.write('\t\t\tstart += avg\n')
    f.write('\tif idx < mod:\n')
    f.write('\t\ttotal += 1\n')
    f.write('\theight = total\n')
    f.write('\tstride = input_size * input_size\n')
    f.write('\theight1 = int(size * height / input_size)\n')
    f.write('\tw = np.float32(np.zeros(shape=(height1, size2)))\n')
    f.write('\tcnt = 0\n')
    f.write('\tfor i in range(start*input_size, size, stride):\n')
    f.write('\t\tpos = cnt * height*input_size\n')
    f.write('\t\tw[pos:pos+height*input_size, :] = originw[i:i+height*input_size, :]\n')
    f.write('\t\tcnt += 1\n')
    f.write('\treturn w\n')
    f.write('\n')

def fastmode_calculation():
    mask_list = [] # record whether data need to be send
    key_list = []
    for index, key in enumerate(data['padding_info'][0]):
        key_list.append(key)
    # print(key_list)
    # print(len(key_list))
    for key_index in range(len(key_list)-1):
        end_idx_in_block = int(key_list[key_index].split(',')[1])
        if data['layers'][end_idx_in_block+1] == 'conv':
            mask = np.zeros((int(data['output'][end_idx_in_block]))) - 1 # initialized with -1
            # mask = np.zeros()
            for device_idx in range(total_device_num):
                current_key = key_list[key_index]
                next_key = key_list[key_index+1]
                # print(device_idx, current_key, data['padding_info'][device_idx][current_key][-1], data['devices'][device_idx][next_key])
                current_begin = data['padding_info'][device_idx][current_key][-1][0]
                next_begin = data['devices'][device_idx][next_key][0]
                if next_begin < current_begin:
                    mask[next_begin:current_begin+1] = 0
                current_end = data['padding_info'][device_idx][current_key][-1][1]
                next_end = data['devices'][device_idx][next_key][1]
                if current_end < next_end:
                    mask[current_end:next_end+1] = 0
            mask_list.append(mask)
    return(mask_list)

def write_dump():
    f.write('print(json.dumps({\n')
    f.write('\t\'load\': int(1000*load),\n')
    f.write('\t\'comm\': int(1000*comm),\n')
    f.write('\t\'cal\': int(1000*cal),\n')
    f.write('}))\n')
if not os.path.exists('codegen'):
    os.mkdir('codegen')

for model in range(4):
    with open('data/prefetch'+str(model)+'.json') as f:
        data = json.loads(f.read())

    # print(data)
    total_device_num = len(data['devices'])
    total_block_num  = len(data['devices'][0].keys())
    
    mask_list = fastmode_calculation()
    
    for device_idx, device in enumerate(data['devices']):
        num_of_fc_in_block = np.zeros(total_block_num)
        path = {
            0: 'yolov2',
            1: 'alexnet',
            2: 'vgg16',
            3: 'vgg19',
        }.get(model)

        import shutil
        if device_idx == 0:
            if os.path.exists('codegen/'+path):
                shutil.rmtree('codegen/'+path)
                
        if not os.path.exists('codegen/'+path):
            os.mkdir('codegen/'+path)
        with open('codegen/'+path+'/device'+str(device_idx)+'.py', 'w') as f:

            write_header()
            write_relu()
            
            ######################## write class Net ########################

            f.write('class Net(nn.Module):\n')
            write_init()
            write_set_pre_cal_w()
            write_forward()
            write_sendall()
            write_recvall()
            write_recv()
            write_pre_cal_weight()
            ##################### main ######################
            write_main()
            write_dump()