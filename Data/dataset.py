from torch.utils.data import Dataset
import numpy as np
class lineardataset(Dataset):
    def __init__(self) -> None:
        super(lineardataset,self).__init__()
        self.feature = np.load('Data/feature.npy',allow_pickle=True)
        self.label = np.load('Data/label.npy',allow_pickle=True)
        self.weight = np.load('Data/weight.npy',allow_pickle=True)
    def __getitem__(self, index):
        return self.feature[index],self.label[index]
    
    def __len__(self):
        return 1024

