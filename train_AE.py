import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader
from ori_AE import *
dir_path="data/cora/cora.content"
idx_features_labels = np.genfromtxt(dir_path, dtype=np.dtype(str))
features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
features = torch.FloatTensor(np.array(features.todense()))
num=features.shape[0]
train_set=features[0:2500,:]
test_set=features[2501:-1,:]
dataloader=DataLoader(train_set,50)

testloader=DataLoader(test_set,50)
train_number=len(train_set)
test_number=len(test_set)
epoch=100
l_rate=0.003
model=AE()
loss_function=nn.CrossEntropyLoss()
if torch.cuda.is_available():
    model=model.cuda()
    loss_function=loss_function.cuda()
optimizer=torch.optim.Adam(model.parameters(),lr=l_rate)

iteration=0

# Training stage
print("----------------training stage start----------------")
for i in range (epoch):
    print("--------This is {} epoch".format(i+1))
    for data in dataloader:
        fea=data
        if torch.cuda.is_available():
           fea=fea.cuda()
        prediction=model(fea)[0]
        loss = loss_function(prediction, fea)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iteration = iteration + 1
        if (iteration%50)==0:
            accuracy= (prediction==fea).sum()/(50*1433)
            result='{:.2%}'.format(accuracy)
            print("The validation accuracy is {} at {} iteration".format(result,iteration))
print("Training stage has finished,it has {} iteration".format(iteration))

# Testing stage
print("----------------Testing stage start----------------")
with torch.no_grad():
    accunum=0
    for data in testloader:
        fea=data
        if torch.cuda.is_available():
            fea=fea.cuda()
        prediction=model(fea)[0]
        accunum = accunum + (prediction== fea).sum()
    test_accuracy = accunum / (test_number*1433)
    test_accuracy = '{:.2%}'.format(test_accuracy)
    print("Testing stage end.For this model,test sets accuracy is {}".format(test_accuracy))
torch.save(model,"AE_MODEL")