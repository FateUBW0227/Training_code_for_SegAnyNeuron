import os
import random

import numpy as np
import tifffile
import torch




class TakeNotesLoss:
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.id = -1

    def update(self, value):
        self.sum += value
        self.count += 1

    def update2(self):
        tmp = self.sum / self.count
        self.sum = 0
        self.count = 0
        self.id += 1
        return tmp

class TakeNotesLoss2:
    def __init__(self):
        self.dice = 0
        self.focal = 0
        self.count = 0
        self.id = -1

    def update(self, tmp_dice, tmp_focal_loss):
        self.dice += tmp_dice
        self.focal += tmp_focal_loss
        self.count += 1

    def update2(self):
        tmp_dice = self.dice / self.count
        tmp_focal = self.focal / self.count
        self.dice = 0
        self.focal = 0
        self.count = 0
        self.id += 1
        return tmp_dice, tmp_focal


class Trainer:
    def __init__(self, datalodader, test_loader, model, loss_criterion, optimizer, lr_scheduler, eval_metric,
                 modelPath=None, device=torch.device('cpu'), batchSize=1):
        self.batchSize = batchSize
        self.device = device
        self.test_loader = test_loader
        self.dataloader = datalodader
        self.valCount = 50
        self.model = model
        self.model.to(self.device)
        #损失函数
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_metric = eval_metric
        self.modelPath = modelPath
        # if self.modelPath is not None:
        #     self.modelPath = './saved_models/'
        if not os.path.isdir(self.modelPath):
            os.makedirs(self.modelPath)

    def Train(self, turn=2, writer=None):
        train_small_losses = TakeNotesLoss2()
        train_big_losses = TakeNotesLoss2()
        evalVal = TakeNotesLoss()
        lastEvalVal = 0
        self.valCount = min(len(self.dataloader), self.valCount)
        iter_count = 0
        updated_turn = 0
        for t in range(turn+1):
            if t - updated_turn > 20:
                print("no update for 20 epochs, training done.")
                exit(-1)

            torch.cuda.empty_cache()
            torch.set_grad_enabled(True)
            self.model.train()
            for i, (img, mask, name) in enumerate(self.dataloader):
                if img.shape[0] != self.batchSize:
                    continue
                torch.cuda.empty_cache()
                img = img.to(self.device)
                mask = mask.to(self.device)
                seg = self.model(img)
                dice_loss, focal_loss = self.loss_criterion(seg, mask)
                loss = dice_loss + focal_loss
                train_small_losses.update(dice_loss.item(), focal_loss.item())
                train_big_losses.update(dice_loss.item(), focal_loss.item())
                self.optimizer.zero_grad()  # 梯度清零
                loss.backward()
                self.optimizer.step()  # 参数更新
                if (iter_count + 1) % 10 == 0:
                    tmp_dice, tmp_focal = train_small_losses.update2()
                    print('TRAIN [Epoch %d | %d] [Process %d | %d] [DiceLoss %.4f, FocalLoss %.4f, learning rate %.4f]'
                          % (t, turn, i, len(self.dataloader), tmp_dice, tmp_focal, self.optimizer.param_groups[0]['lr'])
                          )
                    writer.add_scalar('Loss/TrainSmallLoss', tmp_dice, tmp_focal, train_small_losses.id)

                if (iter_count + 1) % self.valCount == 0:
                    torch.cuda.empty_cache()
                    tmp_dice, tmp_focal = train_big_losses.update2()
                    writer.add_scalar('Loss/TrainBigLoss', tmp_dice, tmp_focal, train_big_losses.id)
                    self.model.eval()
                    with torch.no_grad():
                        for i, (img, mask, name) in enumerate(self.test_loader):
                            if img.shape[0] != self.batchSize:
                                continue
                            img = img.to(self.device)
                            mask = mask.to(self.device)
                            seg = self.model(img)
                            COUT = False
                            if i == 0:
                                COUT= True
                            eval = self.eval_metric(seg, mask, COUT=COUT)

                            evalVal.update(eval)
                        curEvalVal = evalVal.update2()
                        writer.add_scalar('Eval/EvalVal', curEvalVal, evalVal.id)
                        if curEvalVal > lastEvalVal:
                            print("save model")
                            updated_turn = t
                            torch.save({"state_dict": self.model.state_dict(), "param": self.optimizer}, os.path.join(self.modelPath,
                                            "betnet_%s.pth" % (str(t).zfill(5))))
                            lastEvalVal = curEvalVal
                        self.lr_scheduler.step(curEvalVal)
                        lr = self.optimizer.param_groups[0]['lr']
                        writer.add_scalar('TrainParam/lr', lr, evalVal.id)
                        print('VAL [Epoch %d | %d] [EvalVal; %.4f] [Lr: %f]' % (t, turn, curEvalVal, lr))
                    torch.cuda.empty_cache()
                    self.model.train()
                iter_count += 1
            torch.save({"state_dict": self.model.state_dict(), "param": self.optimizer}, os.path.join(self.modelPath, "%s_final.pth" % (str(t).zfill(5))))