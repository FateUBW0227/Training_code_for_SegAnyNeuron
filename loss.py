from pytorch3dunet.unet3d.losses import DiceLoss
from torch import nn
from focal_loss import BCEFocalLoss, SoftDiceLoss

class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""
    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)


class EvalScore(nn.Module):
    def __init__(self):
        super(EvalScore, self).__init__()
        self.maxThre = 3. / 255
        self.maxThre2 = 103.0 / 255
        self.l1_loss = nn.L1Loss(reduction='none')
        self.sigmod = nn.Sigmoid()
        self.dice = SoftDiceLoss()
        self.focal_loss = BCEFocalLoss(alpha=0.8)

    def forward(self, input, mask, COUT=False):
        mask2 = mask > self.maxThre
        mask2_sum = max(1, mask2.sum())
        mask3 = mask > self.maxThre2
        mask3_sum = max(1, mask3.sum())
        l1 = self.l1_loss(input, mask)
        # diceLoss = self.dice(input, mask)
        # bcedLoss = self.bced(input, mask)
        # if COUT:
        #     print(l1.mean(), l1[mask2].sum() / mask2_sum, l1[mask3].sum() / mask3_sum)
        # if COUT:
        #     print(diceLoss)
        l1Loss = l1.mean() * 0.5 + l1[mask2].sum() / mask2_sum + l1[mask3].sum() / mask3_sum
        loss_new = self.dice(input, mask) + self.focal_loss(input, mask) * 50.0
        # return 1 - l1Loss
        # return diceLoss
        return 1 - loss_new


class LSDLoss(nn.Module):
    def __init__(self):
        super(LSDLoss, self).__init__()
        self.maxThre = 3. / 255
        self.maxThre2 = 103.0 / 255
        self.bcelog_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
        self.sigmod = nn.Sigmoid()
        self.dice = SoftDiceLoss()
        self.focal_loss = BCEFocalLoss(alpha=0.8)
    def forward(self, input, mask, Train = True):
        mask2 = mask > self.maxThre
        mask2_sum = max(1, mask2.sum())
        mask3 = mask > self.maxThre2
        mask3_sum = max(1, mask3.sum())

        if Train:
            input = self.sigmod(input)
        l1 = self.l1_loss(input, mask)
        # l1Loss = l1.mean() * 0.5 + l1[mask2].sum() / mask2_sum + l1[mask3].sum() / mask3_sum
        dice_loss = self.dice(input, mask)
        focal_loss = self.focal_loss(input, mask) * 50.0
        # loss2 = 1 - dice_coeff(input, mask, reduce_batch_first=True)
        # return l1Loss, [l1Loss]
        return dice_loss, focal_loss
