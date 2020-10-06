import json
import os
import sys

try:
    if sys.argv[1] == '-g':
        for model in range(4):
            with open('data/prefetch'+str(model)+'.json') as f:
                data = json.loads(f.read())
            # print(data)
            total_device_num = len(data['devices'])
            total_block_num  = len(data['devices'][0].keys())
            path = {
                0: 'yolov2',
                1: 'alexnet',
                2: 'vgg16',
                3: 'vgg19',
            }.get(model)
            with open(path+'.py', 'w') as f:
                ######################### write header ########################
                f.write('import torch\n')
                f.write('import torch.nn as nn\n')
                f.write('import torch.nn.functional as F\n')
                f.write('from fl import FCBlock\n')
                f.write('import numpy as np\n\n')
                f.write('import json\n\n')
                ######################## write relu function ####################

                f.write('def relu(x):\n')
                f.write('\treturn np.maximum(x, 0)\n\n')

                ######################## write class Net ########################

                f.write('class Net(nn.Module):\n')
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
                conv_idx = 1
                pool_idx = 1
                fc_idx   = 1
                for idx, key in enumerate(data['devices'][0].keys()):
                    f.write('\tdef b'+str(idx)+'_forward(self, x, device_num):\n')
                    f.write('\t\tself.device_num = device_num\n')
                    begin = int(key.split(',')[0])
                    end = int(key.split(',')[1])
                    layer_idx_in_block = 0
                    hybrid = False
                    for i in range(begin, end+1):
                        if data['layers'][i] == 'conv':
#################################################################### TODO ##############################################################################
                            padding_top = []
                            padding_bottom = []
                            for device_idx in range(total_device_num):
                                begin_idx_in_layer = data['padding_info'][device_idx][key][layer_idx_in_block][0]
                                end_idx_in_layer = data['padding_info'][device_idx][key][layer_idx_in_block][1]
                                if begin_idx_in_layer < 0:
                                    padding_top.append(device_idx)
                                elif end_idx_in_layer >= data['input'][i]:
                                    padding_bottom.append(device_idx)
                            # print(i, padding_top)
                            # top padding
                            f.write('\t\tif device_num == ')
                            for x in range(len(padding_top)):
                                if x == len(padding_top) - 1:
                                    f.write(str(padding_top[x])+':\n')
                                else:
                                    f.write(str(padding_top[x]) + ' or device_num == ')
                            f.write('\t\t\tm = nn.ConstantPad2d(('+str(int(data['padding'][i]))+', '+str(int(data['padding'][i]))+', '+str(1)+', 0), 0)\n')
                            # bottom padding
                            f.write('\t\telif device_num == ')
                            for x in range(len(padding_bottom)):
                                if x == len(padding_bottom) - 1:
                                    f.write(str(padding_bottom[x])+':\n')
                                else:
                                    f.write(str(padding_bottom[x]) + ' or device_num == ')
                            f.write('\t\t\tm = nn.ConstantPad2d(('+str(int(data['padding'][i]))+', '+str(int(data['padding'][i]))+', 0, '+str(1)+'), 0)\n')
                            # else
                            f.write('\t\telse:\n')
                            f.write('\t\t\tm = nn.ConstantPad2d(('+str(int(data['padding'][i]))+', '+str(int(data['padding'][i]))+', 0, 0), 0)\n')
                            # f.write('\t\tx = self.pad(x, padding_value='+str(int(data['padding'][i]))+')\n')
                            f.write('\t\tx = m(x)\n')
                            f.write('\t\tx = F.relu(self.conv'+str(conv_idx)+'(x))\n')
#################################################################### TODO ##############################################################################
                            conv_idx += 1
                        elif data['layers'][i] == 'pool':
                            f.write('\t\tx = self.pool'+str(pool_idx)+'(x)\n')
                            pool_idx += 1
                        elif data['layers'][i] == 'FL':
                            if hybrid:
                                continue
                            if i+1 <= end:
                                # hybrid
                                if data['layers'][i+1] == 'FL':
                                    f.write('\t\tw1 = self.fc'+str(fc_idx)+'.weight.data.numpy().transpose()\n')
                                    f.write('\t\tw2 = self.fc'+str(fc_idx+1)+'.weight.data.numpy().transpose()\n')
                                    f.write('\t\tfblk = FCBlock(\'hybrid\', device_num, '+str(total_device_num)+')\n')
                                    f.write('\t\tfblk.set_bias(self.fc'+str(fc_idx)+'.bias.detach().numpy())\n')
                                    f.write('\t\tfblk.append_layer(w1)\n')
                                    f.write('\t\tfblk.append_layer(w2)\n')
                                    f.write('\t\tx = fblk.process(x)\n')
                                    hybrid = True
                            # single fl
                            else:
                                if i-1 >= begin:
                                    if data['layers'][i-1] == 'conv' or data['layers'][i-1] == 'pool':
                                        f.write('\t\tx = x.view(-1).detach().numpy()\n')
                                f.write('\t\tw = self.fc'+str(fc_idx)+'.weight.data.numpy().transpose()\n')
                                f.write('\t\tfblk = FCBlock(\'normal\', device_num, '+str(total_device_num)+')\n')
                                f.write('\t\tfblk.set_input_size('+str(data['output'][i-1])+')\n')
                                f.write('\t\tfblk.append_layer(w)\n')
                                f.write('\t\tx = fblk.process(x)\n')
                            # f.write('\t\tself.fc'+str(fc_idx)+' = nn.Linear('+str(int(data['in_channel'][idx]))+', '+str(int(data['out_channel'][idx]))+')\n')
                            fc_idx += 1
                        layer_idx_in_block += 1
                    f.write('\t\treturn x\n')
                    f.write('\n')
                
                # # padding function
                # f.write('\tdef pad(self, x, padding_value):\n')
                # f.write('\t\tif self.device_num == 0:\n')
                # f.write('\t\t\tm = nn.ConstantPad2d((padding_value, padding_value, padding_value, 0), 0)\n')
                # f.write('\t\telif self.device_num == '+str(total_device_num - 1)+':\n')
                # f.write('\t\t\tm = nn.ConstantPad2d((padding_value, padding_value, 0, padding_value), 0)\n')
                # f.write('\t\telse:\n')
                # f.write('\t\t\tm = nn.ConstantPad2d((padding_value, padding_value, 0, 0), 0)\n')
                # f.write('\t\tx = m(x)\n')
                # f.write('\t\treturn x\n\n')

                ##################### main ######################
                f.write('net = Net()\n')
                f.write('net.load_state_dict(torch.load(\'models/'+path+'\'))\n')
                f.write('################# setting ####################\n')
                f.write('num_of_devices = '+str(total_device_num)+'\n')
                f.write('num_of_blocks = '+str(total_block_num)+'\n')

                fc_idx = 0
                for block_idx, key in enumerate(data['devices'][0].keys()):
                    f.write('################# block '+str(block_idx)+' ####################\n\n')
                    layer_start = int(key.split(',')[0])
                    layer_end = int(key.split(',')[1])
                    # block 0, create input by myself for now
                    # print(device[key])
                    if block_idx == 0:
                        f.write('y = torch.ones(1, 3, '+str(int(data['input'][0]))+', '+str(int(data['input'][0]))+')\n')
                    if data['layers'][layer_start] == 'FL':
                        fc_idx += 1
                        for idx, device in enumerate(data['devices']):
                            f.write('y'+str(idx+1)+' = net.b'+str(block_idx)+'_forward(y, '+str(idx)+')\n')
                    else:
                        max = -1
                        for idx, device in enumerate(data['devices']):
                            start = device[key][0]
                            end = device[key][1]
                            if end - start + 1 > max:
                                max = end - start + 1
                        for idx, device in enumerate(data['devices']):
                            start = device[key][0]
                            end = device[key][1]
                            # not the margin devices
                            # size = data['input'][layer_start]
                            # if (idx != 0 and idx != len(data['devices'])-1) and (end - start + 1)<max:
                            #     f.write('x'+str(idx+1)+' = torch.zeros(1, '+str(int(data['in_channel'][layer_start]))+', '+str(max)+', '+str(int(size))+')\n')
                            #     if start == 0:
                            #         f.write('x'+str(idx+1)+'[:, :, '+str(max-(end-start))+':'+str(max)+', :] = y[:, :, '+str(start)+':'+str(end+1)+', :]\n')
                            #     if end == size:
                            #         f.write('x'+str(idx+1)+'[:, :, '+str(0)+':'+str(end-start)+', :] = y[:, :, '+str(start)+':'+str(end+1)+', :]\n')
                            # else:
                            f.write('x'+str(idx+1)+' = y[:, :, '+str(start)+':'+str(end+1)+', :]\n')
                            f.write('y'+str(idx+1)+' = net.b'+str(block_idx)+'_forward(x'+str(idx+1)+', '+str(idx)+')\n')
                            # f.write('\n')
                    f.write('\n')
                    if data['layers'][layer_end] == 'FL':
                        fc_idx += 1
                        f.write('y = ')
                        if layer_end != len(data['layers'])-1:
                            f.write('relu(')
                        for idx, device in enumerate(data['devices']):
                            f.write('y'+str(idx+1)+' + ') 
                        f.write('net.fc'+str(fc_idx)+'.bias.detach().numpy()')
                        if layer_end != len(data['layers'])-1:
                            f.write(')\n')
                    else:
                        f.write('y = torch.ones(1, '+str(int(data['out_channel'][layer_end]))+', '+str(int(data['output'][layer_end]))+', '+str(int(data['output'][layer_end]))+')\n')
                        f.write('offset = 0\n')
                        for idx in range(len(data['devices'])):
                            f.write('y[:, :, offset: offset+y'+str(idx+1)+'.shape[2], :] = y'+str(idx+1)+'\n')
                            f.write('offset += y'+str(idx+1)+'.shape[2]\n')
                f.write('\nprint(y[:50])\n')
            f.close()
        sys.exit(0)

except IndexError:
    pass

if not os.path.exists('codegen'):
    os.mkdir('codegen')

for model in range(4):
    with open('data/prefetch'+str(model)+'.json') as f:
        data = json.loads(f.read())

    # print(data)
    total_device_num = len(data['devices'])
    total_block_num  = len(data['devices'][0].keys())

    for device_idx, device in enumerate(data['devices']):
        path = {
            0: 'yolov2',
            1: 'alexnet',
            2: 'vgg16',
            3: 'vgg19',
        }.get(model)

        if not os.path.exists('codegen/'+path):
            os.mkdir('codegen/'+path)
        with open('codegen/'+path+'/device'+str(device_idx)+'.py', 'w') as f:
            ########################## write header ########################
            f.write('import torch\n')
            f.write('import torch.nn as nn\n')
            f.write('import torch.nn.functional as F\n')
            f.write('import numpy as np\n')
            f.write('import json\n')
            f.write('from fl import FCBlock\n\n')

            ######################## write relu function ####################

            f.write('def relu(x):\n')
            f.write('\treturn np.maximum(x, 0)\n\n')

            ######################## write class Net ########################

            f.write('class Net(nn.Module):\n')
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
            conv_idx = 1
            pool_idx = 1
            fc_idx   = 1
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
                        # f.write('\t\tself.fc'+str(fc_idx)+' = nn.Linear('+str(int(data['in_channel'][idx]))+', '+str(int(data['out_channel'][idx]))+')\n')
                        fc_idx += 1
                    layer_idx_in_block
                f.write('\t\treturn x\n')
                f.write('\n')
            
            # # padding function
            # f.write('\tdef pad(self, x, padding_value):\n')
            # if device_idx == 0:
            #     f.write('\t\tm = nn.ConstantPad2d((padding_value, padding_value, padding_value, 0), 0)\n')
            #     f.write('\t\tx = m(x)\n')
            # elif device_idx == total_device_num - 1:
            #     f.write('\t\tm = nn.ConstantPad2d((padding_value, padding_value, 0, padding_value), 0)\n')
            #     f.write('\t\tx = m(x)\n')
            # else:
            #     f.write('\t\tm = nn.ConstantPad2d((padding_value, padding_value, 0, 0), 0)\n')
            #     f.write('\t\tx = m(x)\n')
            # f.write('\t\treturn x\n\n')

            ##################### main ######################
            f.write('net = Net()\n')
            f.write('net.load_state_dict(torch.load(\'models/model\'))\n')
            f.write('################# setting ####################\n')
            f.write('num_of_devices = '+str(total_device_num)+'\n')
            f.write('num_of_blocks = '+str(total_block_num)+'\n')
            f.write('################# read json ##################\n\n')

            for idx, key in enumerate(device.keys()):
                f.write('################# block '+str(idx)+' ####################\n\n')
                # block 0, create input by myself for now
                # print(device[key])
                start = device[key][0]
                end = device[key][1]
                if idx == 0:
                    f.write('x = torch.ones(1, 3, '+str(end-start+1)+', '+str(int(data['input'][0]))+')\n')
                else:
                    f.write('x = y[:, :, '+str(start)+':'+str(end+1)+', :]\n')

                f.write('y = net.b'+str(idx)+'_forward(x)\n')
                f.write('\n#TODO\n#Send y to the server and get the new input.\n')
                f.write('\n')
        