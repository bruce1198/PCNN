import json

with open('data/prefetch1.json') as f:
    data = json.loads(f.read())

# print(data)
total_device_num = len(data['devices'])
total_block_num  = len(data['devices'][0].keys())

for device_idx, device in enumerate(data['devices']):
    with open('codegen/alexnet/device'+str(device_idx)+'.py', 'w') as f:
        ########################## write header ########################
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
        for idx, key in enumerate(device.keys()):
            f.write('\tdef b'+str(idx)+'_forward(self, x):\n')
            begin = int(key.split(',')[0])
            end = int(key.split(',')[1])
            for i in range(begin, end+1):
                if data['layers'][i] == 'conv':
                    f.write('\t\tx = self.pad(x, padding_value='+str(int(data['padding'][i]))+')\n')
                    f.write('\t\tx = F.relu(self.conv'+str(conv_idx)+'(x))\n')
                    conv_idx += 1
                elif data['layers'][i] == 'pool':
                    f.write('\t\tx = self.pool'+str(pool_idx)+'(x)\n')
                    pool_idx += 1
                elif data['layers'][i] == 'FL':
                    # f.write('\t\tself.fc'+str(fc_idx)+' = nn.Linear('+str(int(data['in_channel'][idx]))+', '+str(int(data['out_channel'][idx]))+')\n')
                    fc_idx += 1
            f.write('\t\treturn x\n')
            f.write('\n')
        
        # padding function
        f.write('\tdef pad(self, x, padding_value):\n')
        if device_idx == 0:
            f.write('\t\tm = nn.ConstantPad2d((padding_value, padding_value, padding_value, 0), 0)\n')
            f.write('\t\tx = m(x)\n')
        elif device_idx == total_device_num - 1:
            f.write('\t\tm = nn.ConstantPad2d((padding_value, padding_value, 0, padding_value), 0)\n')
            f.write('\t\tx = m(x)\n')
        else:
            f.write('\t\tm = nn.ConstantPad2d((padding_value, padding_value, 0, 0), 0)\n')
            f.write('\t\tx = m(x)\n')
        f.write('\t\treturn x\n\n')

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
        