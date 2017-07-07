import torch
from tqdm import tqdm

"model and dataset are the codes we wrote just now"
from model import *
from dataset import *

"Defines Dataloader class, each Dataloader has a Dataset class"
from torch.utils.data.dataloader import *

"Defines Variable class, which is Tensor plus Gradient"
from torch.autograd import Variable

"Defines Optimizers, such as SGD, Adam, etc."
import torch.optim as optim


"Load Dataset"
dataset = myDataset()
trainLoader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)


"Load Model"
model = myModel()
model.train()


"Create Loss and Optimizer"
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)


"Train the model"
for numEpoch in range(20):
	print('==> strart epoch %d' %numEpoch)
	avgLoss = 0
	for i, (data, xml) in enumerate(tqdm(trainLoader)):
		data, xml = Variable(data), Variable(xml)
		optimizer.zero_grad()
		opt = model.forward(data)
		loss = criterion(opt, xml)
		loss.backward()
		optimizer.step()
		avgLoss = (avgLoss*i + loss.data[0])/(i+1)
	print('AvgLoss: %f \n' %avgLoss)

	"Save the trained model"
	torch.save(model, './trainedModel.pth')