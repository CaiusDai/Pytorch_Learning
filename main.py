import torch as t
import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch.utils.data.dataset as Dataset
from torch import nn
import matplotlib.pyplot as plt

device='cpu'
trainFile='train_q3.csv'
testFile='test_q3.csv'


class myDataset(Dataset.Dataset):
    def __init__(self,source_file):
        all_data=np.loadtxt(source_file,skiprows=1,delimiter=',')
        self.data=t.tensor(all_data[:,0]).unsqueeze(1).to(device)
        self.label=t.tensor(all_data[:,1]).unsqueeze(1).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if(t.is_tensor(index)):
            index=index.tolist(index)
        data=self.data[index]
        label=self.label[index]
        sample={'data':data,'target':label}
        return sample

train_Dataset=myDataset(trainFile)
test_Dataset=myDataset(testFile)
train_Loader=t.utils.data.DataLoader(train_Dataset,batch_size=len(train_Dataset),shuffle=False)
test_Loader=t.utils.data.DataLoader(test_Dataset,batch_size=len(test_Dataset),shuffle=False)

Q2_W1=t.tensor(t.tensor([[0.12, 0.26,  -0.15]]).t().tolist(), device='cpu', requires_grad=True)
Q2_W2=t.tensor([[0.11, 0.13, 0.07]], device='cpu', requires_grad=True)

class NeuralNetworkQ3(nn.Module):
    def __init__(self):
        super(NeuralNetworkQ3,self).__init__()
        self.layer1=nn.Linear(1,3,bias=False)
        self.layer1.weight=nn.Parameter(Q2_W1)
        self.outp=nn.Linear(3,1,bias=False)
        self.outp.weight=nn.Parameter(Q2_W2)
    def forward(self,input):
        out=t.relu(self.layer1(input))
        out=t.sigmoid(self.outp(out))
        return out

net=NeuralNetworkQ3().to(device)
print('test')

iteration=50
