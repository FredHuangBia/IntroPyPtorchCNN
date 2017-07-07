import torch

"defines modules, which are like functions, some has trainable parameters"
import torch.nn as nn

"defines functions, equivalent to non-trainable modules"
import torch.nn.functional as F

class myModel(nn.Module):

	"store all the trainable layers here"
	def __init__(self):
		nn.Module.__init__(self)
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(5,5), stride=(2,2), padding=(2,2))
		self.bn1 = nn.BatchNorm2d(8)
		self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3,3), stride=(2,2), padding=(1,1))
		self.bn2 = nn.BatchNorm2d(8)
		self.fc1 = nn.Linear(8*5*10, 1)


	"The define the computational flow of our model here"
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = F.max_pool2d(x, kernel_size=(2,2), stride=(2,2))
		x = F.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = F.max_pool2d(x, kernel_size=(2,2), stride=(2,2))
		x = F.relu(x)

		x = x.view(-1,8*5*10)
		x = F.dropout(x, p=0.3)
		x = self.fc1(x)

		return x


if __name__ == '__main__':
	example = myModel()
	ipt = torch.rand(10, 3, 80, 160)
	from torch.autograd import Variable # Defines Variable class, which is Tensor plus Gradient
	ipt = Variable(ipt)
	print(ipt)
	opt = example.forward(ipt)
	print(opt)