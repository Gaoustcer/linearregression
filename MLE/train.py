from MLE.linearmodel import model
from Data.dataset import lineardataset
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./log/MLE')
N = 16
import numpy as np
from torch.optim import Adam
def _train():
    import torch
    import torch.nn as nn
    lossfunc = nn.MSELoss()
    data = lineardataset()
    weight = data.weight
    EPOCH = 128
    net = model()
    optimize = Adam(net.parameters(),lr = 0.001)
    from torch.utils.data import DataLoader
    loader = DataLoader(data,batch_size=64)
    from tqdm import tqdm
    for epoch in tqdm(range(EPOCH)):
        for feature,label in loader:
            feature = feature.to(torch.float32)
            optimize.zero_grad()
            predict = net(feature)
            label = label.to(torch.float32)
            # label = torch.from_numpy(label)
            loss = lossfunc(predict,label)
            loss.backward()
            optimize.step()
        testtensor = np.random.random([32,N])
        testgroundtruth = torch.from_numpy(testtensor.dot(weight))
        testgroundtruth = testgroundtruth.to(torch.float32)
        testpredict = net(testtensor)
        testloss = lossfunc(testpredict,testgroundtruth)
        writer.add_scalar('loss',testloss,epoch)
if __name__ == "__main__":
    _train()