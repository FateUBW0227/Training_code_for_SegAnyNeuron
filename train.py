import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from MyUtil import GetLossOptimiLr
from dataset import get_dataset
from models.model import LoadModel
from net import Trainer


def Train():
    rootpath = r'.\Training_dataset\fMost_dataset1\With_features'
    trainTxt = "train.txt"
    valTxt = "val.txt"
    batch_size = 1
    imgSize = np.array([192, 192, 192], dtype=np.int32)
    feature_number = 7

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logPath = './logs/'  # log dir
    if not os.path.isdir(logPath): os.makedirs(logPath)
    logName = len(os.listdir(logPath))
    expName = 'exp%s' % str(logName).zfill(3)
    logAdd = './logs/' + expName
    while True:
        if os.path.isdir(logAdd):
            logName += 1
            logAdd = './logs/exp%s' % str(logName).zfill(3)
        else:
            break
    writer = SummaryWriter(logAdd)
    savePath = r'./ModelSave/%s' % expName
    name = {
        'raw_img_dir_name': 'raw',
        'label_dir_name': 'label',
        'feature_dir_name': 'feature_img',
    }

    train_dataset = get_dataset(rootpath, trainTxt, imgSize, feature_number, name)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataset = get_dataset(rootpath, valTxt, imgSize, feature_number, name)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    modelCfg = {
        "in_channels": 1 + feature_number,
        "out_channels": 1,
        "layer_order": "gcr",
        "f_maps": [16, 32, 64, 128, 256],
        "num_groups": 8,  #
        "final_sigmoid": True,
        "is_segmentation": True
    }

    model = LoadModel(modelCfg, type=feature_number)
    model.to(device)
    model.train(True)
    loss_criterion, optimizer, lr_scheduler, eval_metric = GetLossOptimiLr(model)
    eval_metric.to(device)

    netObj = Trainer(train_loader, val_loader, model, loss_criterion, optimizer, lr_scheduler, eval_metric,
                     modelPath=savePath, device=device, batchSize=batch_size)
    netObj.Train(turn=200, writer=writer)


if __name__ == "__main__":
    Train()
