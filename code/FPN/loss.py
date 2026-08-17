import torch
import torch.nn as nn

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred).view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1.0 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def total_loss(pred, target):
    """Downstream Supervised Fine-Tuning Loss."""
    return bce_loss(pred, target) + dice_loss(pred, target)


class UNITPathSSLLoss(nn.Module):
    """
    UNITPathSSL Multi-Task Pretraining Objective (Eq. 7):
    L = L_GAN(G, DG) + lambda1 * L_GAN(S, DS) + lambda2 * L_cyc(G, S) + lambda3 * L_seg(S)
    """
    def __init__(self, lambda1=2.0, lambda2=10.0, lambda3=2.0):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.l1_loss = nn.L1Loss()

    def forward(self, d_s_pred_fake, x_rec, real_mask_x, seg_loss_dict=None):
        # LSGAN Loss on Generator S
        loss_adv_s = 0.5 * torch.mean((d_s_pred_fake - 1.0) ** 2)

        # L1 Cycle Consistency Loss
        loss_cyc = self.l1_loss(x_rec, real_mask_x)

        # Instance Segmentation Guided Loss
        loss_isg = torch.tensor(0.0, device=x_rec.device)
        if seg_loss_dict is not None:
            loss_isg = sum(loss for loss in seg_loss_dict.values())

        total = (self.lambda1 * loss_adv_s) + (self.lambda2 * loss_cyc) + (self.lambda3 * loss_isg)
        return total, {"loss_adv_s": loss_adv_s.item(), "loss_cyc": loss_cyc.item(), "loss_isg": loss_isg.item()}