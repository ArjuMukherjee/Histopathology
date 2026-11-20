import torch
import torch.nn as nn

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred).view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1.0 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def total_loss(pred, target, mode="adaptive", bce_weight=0.5, dice_weight=0.5,
               min_bce_weight=0.2, max_bce_weight=0.8, sharpness=2.0, return_parts=False):
    """Downstream Supervised Fine-Tuning Loss. `mode` selects the formulation (for ablation):

    "sum"        : bce_weight * BCE + dice_weight * Dice            (original; unbounded)
    "normalized" : bce_weight * (1 - exp(-BCE)) + dice_weight * Dice (both terms in [0,1]; fixed weights)
    "adaptive"   : w * (1 - exp(-BCE)) + (1 - w) * Dice, where
                   w = clamp(1 - exp(-sharpness * BCE.detach()), min_bce_weight, max_bce_weight)
                   -> BCE dominates early (high BCE), Dice dominates late (low BCE)

    return_parts=True also returns a dict of detached components for logging.
    """
    bce = bce_loss(pred, target)
    dice = dice_loss(pred, target)

    if mode == "sum":
        w_bce, w_dice, bce_term = bce_weight, dice_weight, bce
    elif mode == "normalized":
        w_bce, w_dice, bce_term = bce_weight, dice_weight, 1.0 - torch.exp(-bce)
    elif mode == "adaptive":
        w_bce = torch.clamp(1.0 - torch.exp(-sharpness * bce.detach()), min_bce_weight, max_bce_weight)
        w_dice, bce_term = 1.0 - w_bce, 1.0 - torch.exp(-bce)
    else:
        raise ValueError(f"Unknown loss mode '{mode}'. Use 'sum', 'normalized' or 'adaptive'.")

    total = w_bce * bce_term + w_dice * dice
    if return_parts:
        return total, {"bce": bce.item(), "dice_loss": dice.item(),
                       "w_bce": float(w_bce), "w_dice": float(w_dice)}
    return total


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