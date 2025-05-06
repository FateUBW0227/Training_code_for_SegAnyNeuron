import torch, importlib
from loss import BCEDiceLoss, EvalScore, LSDLoss
from torch import nn as nn


def GetLossOptimiLr(model):
    optimizer_config = {
        'learning_rate': 0.0002,
        # weight decay
        'weight_decay': 0.00001
    }
    lr_config = {
        # reduce learning rate when evaluation metric plateaus
        'name': 'ReduceLROnPlateau',
        # use 'max' if eval_score_higher_is_better=True, 'min' otherwise
        'mode': 'max',
        # factor by which learning rate will be reduced
        'factor': 0.5,
        # number of *validation runs* with no improvement after which learning rate will be reduced
        'patience': 15
    }
    # 损失
    # alpha = 1
    # beta = 1
    # loss_criterion = BCEDiceLoss(alpha, beta)

    loss_criterion = LSDLoss()

    # 优化器
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config.get('weight_decay', 0)
    betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)

    # 学习率
    class_name = lr_config.pop('name')
    m = importlib.import_module('torch.optim.lr_scheduler')
    clazz = getattr(m, class_name)
    lr_config['optimizer'] = optimizer
    lr_scheduler = clazz(**lr_config)

    # 评估
    eval_metric = EvalScore()
    return loss_criterion, optimizer, lr_scheduler, eval_metric