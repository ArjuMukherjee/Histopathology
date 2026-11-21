import torch
import torch.nn as nn
import torch.nn.functional as F

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


def r1_penalty(d_out, real_in):
    """R1 regularization: E[ ||grad_x D(x)||^2 ] on REAL samples. `real_in` must have requires_grad=True."""
    grad, = torch.autograd.grad(outputs=d_out.sum(), inputs=real_in, create_graph=True)
    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()


class UNITPathSSLLoss(nn.Module):
    """
    UNITPathSSL pretraining objective (paper Eq. 1-3, 7):
        L = L_GAN(G, D_G) + lambda1 * L_GAN(S, D_S) + lambda2 * L_cyc(G, S) + lambda3 * L_seg(S)

    Eq. 1  L_GAN(G, D_G): non-saturating logistic GAN + R1 (gamma_g) on real images y
        D_G: -E[log D_G(y)] - E[log(1 - D_G(G(x)))] + gamma_g/2 * E[||grad D_G(y)||^2]
        G  : -E[log D_G(G(x))]
    Eq. 2  L_GAN(S, D_S): least-squares GAN + R1 (gamma_s) on real masks x
        D_S: 1/2 E[(D_S(x) - 1)^2] + 1/2 E[D_S(S(y))^2] + gamma_s/2 * E[||grad D_S(x)||^2]
        S  : 1/2 E[(D_S(S(y)) - 1)^2]                     <-- note: S applied to REAL tissue y
    Eq. 3  L_cyc(G, S) = E[||S(G(x)) - x||_1] + E[||G(S(y)) - y||_1]   (both directions)
    Eq. 6  L_seg(S)    = Mask R-CNN losses (RPN cls/reg + box cls/reg + mask). This project pretrains
                         S semantic-only, so lambda3 = 0 is used and seg_loss_dict is left None.

    All D_* outputs are raw logits (PatchGAN maps).
    """
    def __init__(self, lambda1=2.0, lambda2=10.0, lambda3=0.0, gamma_g=10.0, gamma_s=10.0):
        super().__init__()
        self.lambda1, self.lambda2, self.lambda3 = lambda1, lambda2, lambda3
        self.gamma_g, self.gamma_s = gamma_g, gamma_s
        self.l1_loss = nn.L1Loss()

    # ---------------- Discriminator objectives ----------------
    def d_g_loss(self, d_real_logits, d_fake_logits, real_y=None):
        """Eq. 1 (D_G). Pass real_y (requires_grad) to include the R1 term."""
        loss = F.softplus(-d_real_logits).mean() + F.softplus(d_fake_logits).mean()
        r1 = torch.tensor(0.0, device=d_real_logits.device)
        if real_y is not None and self.gamma_g > 0:
            r1 = r1_penalty(d_real_logits, real_y)
            loss = loss + 0.5 * self.gamma_g * r1
        return loss, {"d_g_adv": loss.item(), "d_g_r1": r1.item()}

    def d_s_loss(self, d_real_logits, d_fake_logits, real_x=None):
        """Eq. 2 (D_S). Pass real_x (requires_grad) to include the R1 term."""
        loss = 0.5 * ((d_real_logits - 1.0) ** 2).mean() + 0.5 * (d_fake_logits ** 2).mean()
        r1 = torch.tensor(0.0, device=d_real_logits.device)
        if real_x is not None and self.gamma_s > 0:
            r1 = r1_penalty(d_real_logits, real_x)
            loss = loss + 0.5 * self.gamma_s * r1
        return loss, {"d_s_adv": loss.item(), "d_s_r1": r1.item()}

    # ---------------- Generator (G + S) objective ----------------
    def forward(self, d_g_fake_logits, d_s_fake_logits, x_rec, real_x, y_rec, real_y, seg_loss_dict=None):
        """
        d_g_fake_logits : D_G(G(x))            -> Eq. 1 generator term (weight 1)
        d_s_fake_logits : D_S(S(y))            -> Eq. 2 generator term (lambda1)
        x_rec, real_x   : S(G(x)) vs x         -> forward cycle (lambda2)
        y_rec, real_y   : G(S(y)) vs y         -> backward cycle (lambda2)
        seg_loss_dict   : ISG Mask R-CNN losses -> lambda3 (optional)
        """
        loss_adv_g = F.softplus(-d_g_fake_logits).mean()                 # -log sigmoid(D_G(G(x)))
        loss_adv_s = 0.5 * ((d_s_fake_logits - 1.0) ** 2).mean()         # LSGAN on S
        loss_cyc_x = self.l1_loss(x_rec, real_x)                          # X -> Y -> X
        loss_cyc_y = self.l1_loss(y_rec, real_y)                          # Y -> X -> Y
        loss_cyc = loss_cyc_x + loss_cyc_y

        loss_isg = torch.tensor(0.0, device=x_rec.device)
        if seg_loss_dict is not None:
            loss_isg = sum(loss for loss in seg_loss_dict.values())

        total = loss_adv_g + self.lambda1 * loss_adv_s + self.lambda2 * loss_cyc + self.lambda3 * loss_isg
        return total, {
            "loss_adv_g": loss_adv_g.item(), "loss_adv_s": loss_adv_s.item(),
            "loss_cyc_x": loss_cyc_x.item(), "loss_cyc_y": loss_cyc_y.item(),
            "loss_isg": loss_isg.item(),
        }