import numpy as np
from torch.distributions import Normal
norm = Normal(0,1)
Noise = 100
N = 16
Number_data = 1024 + 32
X = np.random.random((Number_data,N))
w = np.random.random((N,1))
noise = norm.sample((Number_data,1))/Noise
y = X.dot(w) + noise.numpy()
np.save('feature.npy',X)
np.save('label.npy',y)
np.save('weight.npy',w)