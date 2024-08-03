# some utils related to v_information
import torch
from torch.utils.data import DataLoader, Dataset
# from FLamingo.core.utils.train_test_utils import infinite_dataloader
import numpy as np


class InfoDataset(Dataset):
    """
    A dataset class that can return the data, label, and v_information
    """
    def __init__(self, npz_file):
        # 加载数据
        with np.load(npz_file, allow_pickle=True) as data:
            self.data = data['data']  # 假设这是图像数据或者是 Shakespeare 数据
            self.targets = data['targets']  # 目标/标签
            self.pvi = data['pvi']  # 额外的pvi信息
        
        # 转换为适当的torch.Tensor
        # self.data = torch.tensor(self.data)     # 对于 Shakespeare 数据集，应当保留其数据类型
        self.data = torch.tensor(self.data)
        if self.data.dtype == torch.float64:
            self.data = self.data.to(torch.float32)
        self.targets = torch.Tensor(self.targets).type(torch.int64)
        self.pvi = torch.Tensor(self.pvi).type(torch.float32)
    
    def __len__(self):
        # 数据集的长度
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取索引为idx的数据、目标和pvi
        return self.data[idx], self.targets[idx], self.pvi[idx]
        
# def infinite_dataloader(dataloader):
#     """
#     Infinitely yield data from dataloader  
    
#     Args:
#         dataloader: DataLoader instance
#     Returns:
#         data: data from dataloader
#     """
#     while True:
#         for data in dataloader:
#             yield data
     
        
if __name__ == '__main__':
    # 测试 InfoDataset
    dataset = InfoDataset('../datasets/femnist/info/1.npz')
    data, target, pvi = dataset[0]
    print(len(dataset))
    print(data.shape, target, pvi)
    print(data.dtype, target.dtype, pvi.dtype)
    print(data.device, target.device, pvi.device)
    # 测试 DataLoader
    DL = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)
    print(len(DL))
    for data, target, pvi in DL:
        print(data.shape, target, pvi)
        print(data.dtype, target.dtype, pvi.dtype)
        print(data.device, target.device, pvi.device)
        break
    # # infinite dataloader
    # DL = DataLoader(dataset, batch_size=306, shuffle=True, drop_last=False)
    # iDL = infinite_dataloader(DL)
    # for i in range(9):
    #     d, t, p = next(iDL)
    #     print(d[0], t[0], p[0])
        