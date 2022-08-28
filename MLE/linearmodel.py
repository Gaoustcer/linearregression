import torch.nn as nn
N = 16

import torch
import numpy as np
class model(nn.Module):
    def __init__(self) -> None:
        super(model,self).__init__()
        self.net = nn.Linear(N,1)
    def forward(self,x):
        if isinstance(x,np.ndarray):
            x = torch.from_numpy(x).to(torch.float32)
        return self.net(x)

if __name__ == "__main__":
    m = model()
    tensorlist = []
    # tensor = m.parameters()[0]
    for m_ in m.parameters():
        tensorlist.append(m_)
        print(m_)
    print(type(m.parameters()))