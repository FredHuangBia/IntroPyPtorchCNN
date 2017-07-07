import os
import csv
from scipy import misc
import numpy as np

"defines torch.Tensor class and Tensor operations"
import torch 

"defines torch Dataset class"
from torch.utils.data.dataset import *


"inherit from torch Dataset class"
class myDataset(Dataset):
	def __init__(self):
		Dataset.__init__(self)
		self.csv = csv.reader(open('../data/driving_log.csv', 'r'))
		self.dataInfo = [line for line in self.csv]
		self.mean = -0.03369610860722366
		self.std = 0.13938239717663414


	"Return the number of data in our dataset"
	def __len__(self):
		return len(self.dataInfo)


	"Function to get data and label pair by given index, both are torch.Tensor class"
	def __getitem__(self, index):
		dataInfo = self.dataInfo[index]
		pieces = dataInfo[0].split('/')
		dataInfo[0] = os.path.join('../', 'data', pieces[-2], pieces[-1])
		img = misc.imread(dataInfo[0])
		img = misc.imresize(img, 0.5)
		img = np.asarray(img, dtype=np.uint8)
		img = np.float32(img/255)
		img = np.swapaxes(img, 0, 2)
		img = np.swapaxes(img, 1, 2)
		data = torch.from_numpy(img)

		xml = torch.zeros(1)
		xml[0] = (float(dataInfo[3])-self.mean)/self.std

		return data, xml


if __name__ == '__main__':
	example = myDataset()
	data, xml = example.__getitem__(100)
	print(data)
	print(xml)
	print(example.__len__())