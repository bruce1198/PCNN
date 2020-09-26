import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
		self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
		self.conv7 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
		self.conv8 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
		self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
		self.conv10 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
		self.conv11 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
		self.conv12 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
		self.conv13 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
		self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv14 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
		self.conv15 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
		self.conv16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
		self.conv17 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
		self.conv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
		self.conv19 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
		self.conv20 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1, padding=0)
		self.conv21 = nn.Conv2d(in_channels=1280, out_channels=1024, kernel_size=3, stride=1, padding=1)
		self.conv22 = nn.Conv2d(in_channels=1024, out_channels=425, kernel_size=1, stride=1, padding=0)

	def reorg(self, x, num):
		# reorg => 38 * 38 * 64 => 19 * 19 * 256
		b, c, h, w = x.size()
		x = x.view(b, c, int(h/num), num, int(w/num), num).transpose(3,4).contiguous()
		x = x.view(b, c, int(h/num*w/num), num*num).transpose(2,3).contiguous()
		x = x.view(b, c, num*num, int(h/num), int(w/num)).transpose(1,2).contiguous()
		x = x.view(b, num*num*c, int(h/num), int(w/num))
		return x

	def concat(self, x1, x2):
		# print(x1.shape)
		# print(x2.shape)
		x = torch.cat((x1, x2), dim=1)
		return x

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.pool1(x)
		x = F.relu(self.conv2(x))
		x = self.pool2(x)
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = F.relu(self.conv5(x))
		x = self.pool3(x)
		x = F.relu(self.conv6(x))
		x = F.relu(self.conv7(x))
		x = F.relu(self.conv8(x))
		x = self.pool4(x)
		x = F.relu(self.conv9(x))
		x = F.relu(self.conv10(x))
		x = F.relu(self.conv11(x))
		x = F.relu(self.conv12(x))
		x = F.relu(self.conv13(x))
		route16 = x
		x = self.pool5(x)
		x = F.relu(self.conv14(x))
		x = F.relu(self.conv15(x))
		x = F.relu(self.conv16(x))
		x = F.relu(self.conv17(x))
		x = F.relu(self.conv18(x))
		x = F.relu(self.conv19(x))
		route24 = x
		# route [16]
		x = route16
		x = F.relu(self.conv20(x))
		# reorg /2
		x = self.reorg(x, 2)
		# route [27 24]
		x = self.concat(x, route24)
		x = F.relu(self.conv21(x))
		x = self.conv22(x)
		return x

net = Net()
# torch.save(net.state_dict(), 'models/yolov2')
net.load_state_dict(torch.load('models/yolov2'))
y = net(torch.ones(1, 3, 608, 608))
print(y.view(-1).detach().numpy()[:50])