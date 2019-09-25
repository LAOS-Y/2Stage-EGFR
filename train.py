import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data import NoduleDataset, myCollate
from models import DenseNet
from torch import optim
import matplotlib.pyplot as plt

from model import generate_model

from setting import parseOpts

from sklearn import metrics
    
import os

def auc(label, pred):
    fpr, tpr, _ = metrics.roc_curve(y_true=label.cpu().numpy(),
                                    y_score=pred.cpu().numpy())
            
    return metrics.auc(fpr, tpr)

args = parseOpts()
print(args)

class sets:
    model='resnet'
    model_depth = 50

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
model = generate_model(sets)
model = model.cuda()

print("start loading params")

model.load_state_dict(torch.load(args.weight))

print("finish loading params")

model = nn.DataParallel(model)

print("Start Initializing Dataset")

trn_ds = NoduleDataset(ct_dir=args.ct_dir,
                       bbox_csv_path=args.bbox_csv,
                       label_csv_path=args.train_csv,
                       skip_missed_npy=True)

val_ds = NoduleDataset(ct_dir=args.ct_dir,
                       bbox_csv_path=args.bbox_csv,
                       label_csv_path=args.val_csv,
                       skip_missed_npy=True)

trn_dl = DataLoader(trn_ds, batch_size=args.batch_size, collate_fn=myCollate, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=myCollate, shuffle=True)

print("Finish Initializing Dataset")

# model = DenseNet().cuda()
loss_fn = nn.BCEWithLogitsLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

trn_losses = []
val_losses = []

trn_auc = []
val_auc = []

lrs = []

for i in range(100):
    loss_data = 0

    pred_data_lst = []
    label_data_lst = []

    for x_list, y in trn_dl:
        assert isinstance(x_list[0], list)
        assert isinstance(x_list[0][0], torch.Tensor)

        x_list = [i[0].unsqueeze(0) for i in x_list]
        x = torch.stack(x_list)

        x = x.cuda()
        y = y.float().cuda()

        logit = model(x)[:, 0]

        loss = loss_fn(logit, y)
        loss_data += loss.detach().item() * x.shape[0]

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        pred_data_lst.append(torch.sigmoid(logit).cpu().data)
        label_data_lst.append(y.cpu().data)
        
    pred = torch.cat(pred_data_lst)
    label = torch.cat(label_data_lst)
    trn_auc.append(auc(label, pred))
    
    trn_losses.append(loss_data / len(trn_ds))

    loss_data = 0

    pred_data_lst = []
    label_data_lst = []
    
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

            loss_data += loss.item() * x.shape[0]

        pred_data_lst.append(torch.sigmoid(logit).cpu().data)
        label_data_lst.append(y.cpu().data)

    pred = torch.cat(pred_data_lst)
    label = torch.cat(label_data_lst)
    val_auc.append(auc(label, pred))
    
    scheduler.step()

    val_losses.append(loss_data / len(val_ds))
    
    lrs.append(optimizer.param_groups[0]['lr'])

    plt.subplot(3,1,1)
    plt.plot(trn_losses, color='r', label='TRN LOSS')
    plt.plot(val_losses, color='g', label='VAL LOSS')
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.plot(trn_auc, color='r', label='TRN AUC')
    plt.plot(val_auc, color='g', label='VAL AUC')
    plt.legend()
    
    plt.subplot(3,1,3)
    plt.plot(lrs, color='b', label='lr')
    plt.legend()
    
    plt.savefig('data/log/freeze/log.png', dpi=200)
    plt.close('all')
    
    print("Epoch #{}: TRAIN LOSS: {} VAL LOSS: {}".format(i, round(trn_losses[-1], 3), round(val_losses[-1], 3)))
    
    if (i + 1) % 5 == 0:
        torch.save(model.module.state_dict(), 'data/log/freeze/weight_epoch#{}.ckpt'.format(i + 1))
