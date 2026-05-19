from ModifiedFPN import ModifiedFPN
class FasterRCNN_FPN(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.fpn = ModifiedFPN()

        self.rpn = RPN()
        self.roi_head = ROIHead()

    def forward(self, x):
        C2, C3, C4, C5 = self.backbone(x)
        P2, P3, P4, P5 = self.fpn(C2, C3, C4, C5)

        proposals = self.rpn([P2, P3, P4, P5])
        return self.roi_head(proposals)