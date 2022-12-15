from chrononet import ChronoNet
import torch
from torch.utils.data import DataLoader, TensorDataset
import math
from tqdm import tqdm
import mne

if __name__== "__main__":
    input = torch.randn(3,22,15000)

    ds_y = [i % 2 for i in range(15000)]
    ds_y = torch.as_tensor(ds_y)


    #train_ds = TensorDataset(input[::], ds_y[::])
    #val_ds = TensorDataset(ds_x[600:], ds_y[600:])

    model  = ChronoNet()
    #train_dl = DataLoader(train_ds, batch_size=input.shape[2])

    out= model(input)
    print(out.shape)

