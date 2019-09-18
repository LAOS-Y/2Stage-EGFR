import torch

def myCollate(batch):
    if isinstance(batch[0], tuple):
        return [myCollate(list(i)) for i in zip(*batch)]
    
    if isinstance(batch[0], list):
        return batch
    
    if isinstance(batch[0], int):
        return torch.LongTensor(batch)
    
    if isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    
    if torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)    