import torch
import torch.nn as nn
import torch.nn.functional as F
from fl import FCBlock
import numpy as np

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

    def b1_forward(self, x, device_num):
        self.device_num = device_num
        x = self.pad(x, padding_value=2)
        x = self.pool1(F.relu(self.conv1(x)))
        return x

    def b2_forward(self, x, device_num):
        self.device_num = device_num
        x = self.pad(x, padding_value=2)
        x = self.pool2(F.relu(self.conv2(x)))
        return x

    def b3_forward(self, x, device_num):
        self.device_num = device_num
        x = self.pad(x, padding_value=1)
        x = F.relu(self.conv3(x))
        x = self.pad(x, padding_value=1)
        x = F.relu(self.conv4(x))
        return x

    def b4_forward(self, x, device_num):
        x = self.pad(x, padding_value=1)
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        # x = x.view(-1, 256 * 6 * 6)
        # x = F.relu(self.fc1(x))
        return x

    def b5_forward(self, x, device_num):
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

    def pad(self, x, padding_value):
        if self.device_num == 0:
            m = nn.ConstantPad2d((padding_value, padding_value, padding_value, 0), 0)
            x = m(x)
        elif self.device_num == 1:
            m = nn.ConstantPad2d((padding_value, padding_value, 0, padding_value), 0)
            x = m(x)
        return x



net = Net()
net.load_state_dict(torch.load('models/model'))
################# setting ####################
num_of_devices = 2
################# read json ##################
import json
with open('./data/prefetch1.json', 'r', encoding='utf-8') as f:
    index = json.load(f)

start_index = np.zeros((num_of_devices, len(index[0])))
end_index = np.zeros((num_of_devices, len(index[0])))
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
################# block 1 ####################

x1 = torch.ones(1, 3, 121, 224)
y1 = net.b1_forward(x1, 0)

x2 = torch.ones(1, 3, 114, 224)
y2 = net.b1_forward(x2, 1)


# aggregate block1 output
y = torch.ones(1, 96, 27, 27)
y[:, :, 0:y1.shape[2], :] = y1
y[:, :, y1.shape[2]:y1.shape[2]+y1.shape[2], :] = y2


################# block 2 ####################


x1 = y[:, :, 0:17, :]
y1 = net.b2_forward(x1, 0)


x2 = y[:, :, 12:27, :]
y2 = net.b2_forward(x2, 1)

y = torch.ones(1, 256, 13, 13)
y[:, :, 0:y1.shape[2], :] = y1
y[:, :, y1.shape[2]:y1.shape[2]+y1.shape[2], :] = y2

################# block 3 ####################

x1 = y[:, :, 0:9, :]
y1 = net.b3_forward(x1, 0)


x2 = y[:, :, 5:13, :]
y2 = net.b3_forward(x2, 1)

y = torch.ones(1, 384, 13, 13)
y[:, :, 0:y1.shape[2], :] = y1
y[:, :, y1.shape[2]:y1.shape[2]+y1.shape[2], :] = y2

################# block 4 ####################

x1 = y[:, :, 0:8, :]
y1 = net.b4_forward(x1, 0)


x2 = y[:, :, 5:13, :]
y2 = net.b4_forward(x2, 1)

y = torch.ones(1, 256, 6, 6)

y[:, :, 0:y1.shape[2], :] = y1
y[:, :, y1.shape[2]:y1.shape[2]+y1.shape[2], :] = y2

print(y1.shape)
print(y2.shape)
# replace a1, a2 with pooling result

y_tmp = y.view(9216)
# print(y_tmp[36:54])


a1 = y1.view(4608)
a2 = y2.view(4608)

a1 = a1.detach().numpy()
a2 = a2.detach().numpy()

w = net.fc1.weight.data.numpy().transpose()

fblk = FCBlock('normal', 0, 2)
fblk.append_layer(w)
y_partial0 = fblk.process(a1)
print(np.matmul(a1, fblk.get_weights())[:10])
print(y_partial0[:10])

fblk = FCBlock('normal', 1, 2)
fblk.append_layer(w)
y_partial1 = fblk.process(a2)

def relu(x):
    return np.maximum(0, x)

block4_output = relu(y_partial0 + y_partial1)

print(block4_output[:10])

# ################# block 5 ####################
# w1 = net.fc2.weight.data.numpy().transpose()
# w2 = net.fc3.weight.data.numpy().transpose()


# fblk = FCBlock('hybrid', 0, 2)
# fblk.append_layer(w1)
# fblk.append_layer(w2)
# y_partial0 = fblk.process(block4_output)

# fblk = FCBlock('hybrid', 1, 2)
# fblk.append_layer(w1)
# fblk.append_layer(w2)
# y_partial1 = fblk.process(block4_output)

# block5_output = y_partial0 + y_partial1

# # print(block5_output.shape)

##############################################

import xlwt
import numpy as np

# separate output
wb = xlwt.Workbook()

sh = wb.add_sheet('output')

for i in range(10):
    sh.write(i, 0, float(block4_output[i]))

wb.save('output.xls')