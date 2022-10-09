from torch.utils.data import Dataset
import numpy as np
class Time_Dataset(Dataset):
    def __init__(self,data,time):
        self.data=data
        self.time=time
        if len(self.data)!=len(self.time):
            raise ValueError(
                    "len(data)!=len(time) !!!!!!"
            )
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx].astype(np.float32),self.time[idx]  