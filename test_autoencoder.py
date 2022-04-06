import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

dir_path="data/cora/cora.content"
idx_features_labels = np.genfromtxt(dir_path, dtype=np.dtype(str))
features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
num=features.shape[0]
print(num)
features = torch.FloatTensor(np.array(features.todense()))
if torch.cuda.is_available():
    features=features.cuda()
dataloader=DataLoader(features,100)
model=torch.load("EnhanceAE_MODEL")
if torch.cuda.is_available():
    model=model.cuda()
print("----------------Testing stage start----------------")
with torch.no_grad():
    accunum=0
    for data in dataloader:
        fea=data
        if torch.cuda.is_available():
            fea=fea.cuda()
        prediction=model(fea)[0]
        accunum = accunum + (prediction== fea).sum()
    test_accuracy = accunum / (num*1433)
    test_accuracy = '{:.2%}'.format(test_accuracy)
    print("Testing stage end.For this CNN model,test sets accuracy is {}".format(test_accuracy))
latent=model(features)[1][0]

