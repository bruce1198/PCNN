import torch
import torch.nn as nn
import torch.nn.functional as F
from fl import FCBlock
import numpy as np

################# setting ####################
num_of_devices = 2
num_of_blocks = 5

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def b0_forward(self, x):
        x = self.pad(x, padding_value=2)
        x = self.pool1(F.relu(self.conv1(x)))
        return x

    def b1_forward(self, x):
        x = self.pad(x, padding_value=2)
        x = self.pool2(F.relu(self.conv2(x)))
        return x

    def b2_forward(self, x):
        x = self.pad(x, padding_value=1)
        x = F.relu(self.conv3(x))
        x = self.pad(x, padding_value=1)
        x = F.relu(self.conv4(x))
        return x

    def b3_forward(self, x):
        x = self.pad(x, padding_value=1)
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        x = x.view(4608).detach().numpy()
        w = self.fc1.weight.data.numpy().transpose()
        fblk = FCBlock('normal', self.device_idx, num_of_devices)
        fblk.append_layer(w)
        x = fblk.process(x)
        return x

    def b4_forward(self, x):
        w1 = self.fc2.weight.data.numpy().transpose()
        w2 = self.fc3.weight.data.numpy().transpose()

        fblk = FCBlock('hybrid', self.device_idx, num_of_devices)
        fblk.set_bias(self.fc2.bias.detach().numpy())
        fblk.append_layer(w1)
        fblk.append_layer(w2)
        x = fblk.process(x)
        return x


    def forward(self, x, block_idx, device_idx):
        self.device_idx = device_idx
        if block_idx == 0:
            x = self.b0_forward(x)
        elif block_idx == 1:
            x = self.b1_forward(x)
        elif block_idx == 2:
            x = self.b2_forward(x)
        elif block_idx == 3:
            x = self.b3_forward(x)
        elif block_idx == 4:
            x = self.b4_forward(x)
        return x


    def pad(self, x, padding_value):
        if self.device_idx == 0:
            m = nn.ConstantPad2d((padding_value, padding_value, padding_value, 0), 0)
            x = m(x)
        elif self.device_idx == 1:
            m = nn.ConstantPad2d((padding_value, padding_value, 0, padding_value), 0)
            x = m(x)
        return x



net = Net()
net.load_state_dict(torch.load('models/model'))
################# read json ##################
import json
with open('./data/prefetch1.json', 'r', encoding='utf-8') as f:
    index = json.load(f)

start_index = np.zeros((num_of_devices, num_of_blocks), dtype='int32')
end_index = np.zeros((num_of_devices, num_of_blocks), dtype='int32')
# print(start_index.shape)
for i in range(num_of_devices):
    # print(len(index[i]))
    count = 0
    for key in index[i]:
        # print(index[i][key][0])
        start_index[i][count] = index[i][key][0]
        end_index[i][count] = index[i][key][1]
        count = count + 1
print(start_index)
print(end_index)

################# block 0 ####################

x1 = torch.ones(1, 3, end_index[0][0] - start_index[0][0] + 1, 224)
y1 = net.forward(x1, 0, 0)

x2 = torch.ones(1, 3, end_index[1][0] - start_index[1][0] + 1, 224)
y2 = net.forward(x2, 0, 1)


# aggregate block1 output
y = torch.ones(y1.shape[0], y1.shape[1], y1.shape[2]+y2.shape[2], y1.shape[3])
y[:, :, 0:y1.shape[2], :] = y1
y[:, :, y1.shape[2]:y1.shape[2]+y1.shape[2], :] = y2


################# block 1 ####################


x1 = y[:, :, start_index[0][1]:end_index[0][1]+1, :]
y1 = net.forward(x1, 1, 0)


x2 = y[:, :, start_index[1][1]:end_index[1][1]+1, :]
y2 = net.forward(x2, 1, 1)

y = torch.ones(y1.shape[0], y1.shape[1], y1.shape[2]+y2.shape[2], y1.shape[3])
y[:, :, 0:y1.shape[2], :] = y1
y[:, :, y1.shape[2]:y1.shape[2]+y1.shape[2], :] = y2

################# block 2 ####################

x1 = y[:, :, start_index[0][2]:end_index[0][2]+1, :]
y1 = net.forward(x1, 2, 0)


x2 = y[:, :, start_index[1][2]:end_index[1][2]+1, :]
y2 = net.forward(x2, 2, 1)

y = torch.ones(y1.shape[0], y1.shape[1], y1.shape[2]+y2.shape[2], y1.shape[3])
y[:, :, 0:y1.shape[2], :] = y1
y[:, :, y1.shape[2]:y1.shape[2]+y1.shape[2], :] = y2

# print(y.view(-1).detach().numpy()[50:150])

################# block 3 ####################

x1 = y[:, :, start_index[0][3]:end_index[0][3]+1, :]
y1 = net.forward(x1, 3, 0)


x2 = y[:, :, start_index[1][3]:end_index[1][3]+1, :]
y2 = net.forward(x2, 3, 1)

def relu(x):
    return np.maximum(0, x)

block3_output = relu(y1 + y2 + net.fc1.bias.detach().numpy())

# print(block4_output[:50])

# ################# block 4 ####################

y1 = net.forward(block3_output, 4, 0)
y2 = net.forward(block3_output, 4, 1)

block4_output = y1 + y2 + net.fc3.bias.detach().numpy()

print(block4_output[-50:])

##############################################

import xlwt
import numpy as np

# separate output
wb = xlwt.Workbook()

sh = wb.add_sheet('output')

for i in range(10):
    sh.write(i, 0, float(block4_output[i]))

wb.save('output.xls')