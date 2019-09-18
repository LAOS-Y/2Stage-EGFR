import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data import NoduleDataset, myCollate
from model import DenseNet
from torch import optim
import matplotlib.pyplot as plt

trn_ds = NoduleDataset(ct_dir='data/dataset/EGFR/egfr_prep_result/',
                       bbox_csv_path='data/dataset/EGFR/predicted_bbox.csv',
                       label_csv_path='data/dataset/EGFR/train_simple.csv',
                       skip_missed_npy=True)

val_ds = NoduleDataset(ct_dir='data/dataset/EGFR/egfr_prep_result/',
                       bbox_csv_path='data/dataset/EGFR/predicted_bbox.csv',
                       label_csv_path='data/dataset/EGFR/val_simple.csv',
                       skip_missed_npy=True)

trn_dl = DataLoader(trn_ds, batch_size=32, collate_fn=myCollate)
val_dl = DataLoader(val_ds, batch_size=32, collate_fn=myCollate)

model = DenseNet().cuda()
loss_fn = nn.BCEWithLogitsLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trn_losses = []
val_losses = []

for i in range(300):
    
    for x_list, y in trn_dl:
        assert isinstance(x_list[0], list)
        assert isinstance(x_list[0][0], torch.Tensor)

        x_list = [i[0].unsqueeze(0) for i in x_list]
        x = torch.stack(x_list)

        x = x.cuda()
        y = y.float().cuda()

        logit = model(x)[:, 0]
        loss = loss_fn(logit, y)

        loss.backward()
        optimizer.step()

    trn_losses.append(loss.item())

    for x_list, y in val_dl:
        assert isinstance(x_list[0], list)
        assert isinstance(x_list[0][0], torch.Tensor)

        x_list = [i[0].unsqueeze(0) for i in x_list]
        x = torch.stack(x_list)

        x = x.cuda()
        y = y.float().cuda()

        with torch.no_grad():
            logit = model(x)[:, 0]
            loss = loss_fn(logit, y)

    val_losses.append(loss.item())

    plt.plot(trn_losses, color='r', label='TRN LOSS')
    plt.plot(val_losses, color='g', label='VAL LOSS')

    plt.legend()
    plt.savefig('data/log/log.png', dpi=200)
    plt.close('all')
    
    print("Epoch #{}: TRAIN LOSS: {} VAL LOSS: {}".format(i, round(trn_losses[-1], 3), round(val_losses[-1], 3)))
    
    torch.save(model.state_dict(), 'data/log/weight.ckpt')
