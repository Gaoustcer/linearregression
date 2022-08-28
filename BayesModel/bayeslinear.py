import numpy as np

from Data.dataset import lineardataset
from numpy.random import multivariate_normal
EPOCH = 128
def _getmeancov():
    data = lineardataset()
    feature = data.feature
    label = data.label.squeeze(-1)
    weight = data.weight.squeeze(-1)

    # EPOCH = 1024
    # sigma = 1
    from torch.distributions import Normal

    N = 16
    

    identity = np.identity(N)
    w_conv = np.linalg.inv(feature.T.dot(feature) + identity)
    w_mean = w_conv.dot(feature.T.dot(label))
    print(w_conv.shape,w_mean.shape)
    return w_conv,w_mean,weight
def _train():
    sigma,mu,weight = _getmeancov()
    from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('./log/Bayes')
    for epoch in tqdm(range(EPOCH)):
        sample_weight = multivariate_normal(mu,sigma)
        testfeature = np.random.random((32,16))
        testgroundtruth = testfeature.dot(weight)
        testpredict = testfeature.dot(sample_weight)
        loss = sum((testgroundtruth - testpredict)**2)
        writer.add_scalar('loss',loss,epoch)
if __name__ == "__main__":
    _train()

