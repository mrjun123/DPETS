import torch
import torch.utils.data as data
import numpy as np

class FreeTrend(data.Dataset):
    def __init__(self):
        self.data = []

    def empty(self):
        self.data = []
        
    def __len__(self) -> int:
        return len(self.data)
    
    def len(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y, a, y2, x2 = self.data[idx]
        # x, y = self.data[0]
        return torch.FloatTensor(x), torch.FloatTensor(y), torch.FloatTensor(a), torch.FloatTensor(y2), torch.FloatTensor(x2)

    def __repr__(self):
        return f'Dynamics Data Buffer with {len(self.data)} / {self.capacity} elements.\n'

    def get_x_all(self):
        x_all = []
        for i in range(len(self.data)):
            x, y = self.data[i]
            x_all.append(x)
        return x_all
    
    def get_x_y_all(self):
        x_all = []
        y_all = []
        a_all = []
        y2_all = []
        x2_all = []
        for i in range(len(self.data)):
            x, y, a, y2, x2 = self.data[i]
            x_all.append(x)
            y_all.append(y)
            a_all.append(a)
            y2_all.append(y2)
            x2_all.append(x2)
        return torch.tensor(x_all, dtype=torch.float32), torch.tensor(y_all, dtype=torch.float32), torch.tensor(a_all, dtype=torch.float32), torch.tensor(y2_all, dtype=torch.float32), torch.tensor(x2_all, dtype=torch.float32)

    def shuffle(self):
        pass
        # idxs = np.random.randint(self.x.shape[0], size=[self.model.num_nets, self.train_in.shape[0]])
    
    def push(self, x: np.ndarray, y: np.ndarray, a: np.ndarray, y2: np.ndarray, x2: np.ndarray):
        if x.ndim == 1:
            assert y.ndim == 1
            x = x[None, :]
            y = y[None, :]
            a = a[None, :]
            y2 = y2[None, :]
            x2 = x2[None, :]
        for i in range(x.shape[0]):
            self.data.append((x[i], y[i], a[i], y2[i], x2[i]))