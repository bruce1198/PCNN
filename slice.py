import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x = F.relu(self.conv2(x))
        x = self.pad(x, padding_value=1)
        x = F.relu(self.conv3(x))
        return x

    def b4_forward(self, x, device_num):
        # x = F.relu(self.conv5(x))
        # x = self.pool3(x)
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

# block1
# device 0 output
x1 = torch.ones(1, 3, 121, 224)
y1 = net.b1_forward(x1, 0)
# print(y1.shape)
# device 1 output
x2 = torch.ones(1, 3, 114, 224)
y2 = net.b1_forward(x2, 1)
# print(y2.shape)

# aggregate block1 output
y = torch.ones(1, 96, 27, 27)
y[:, :, 0:y1.shape[2], :] = y1
y[:, :, y1.shape[2]:y1.shape[2]+y1.shape[2], :] = y2

# block2


import xlwt
import numpy as np

# separate output
wb = xlwt.Workbook()

sh = wb.add_sheet('output')
y1 = y1.detach().numpy()
y2 = y2.detach().numpy()

for i in range(27):
    for j in range(27):
        sh.write(i, j, float(y[0][95][i][j]))

wb.save('output.xls')