import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .process import intersection_union as iou

# ANCHORS = [
#     [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
#     [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
#     [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
# ]

# [x,y,h,w,p, cl1,... cln]
# [b,n_anchor, x,y,h,w, cl1,...cln]
class YoloLoss(nn.Module):
    """
        Calculation loss Yolo
        ---------------------------
        Output: (batch, num_anchors, grid_size, grid_size, num_classes, to, tx, ty, tw, th)
                tx, ty: is a x,y offset values of bboxes relative the anchor box
                tw, th: width, height
                to: confidence score

                Box localization: pred[..., sigmoid(1:3), torch.exp(3:5) * anchors_size]
                Box localizaition only compute cell that contains object

    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # constant signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    
    def forward(self, predict, target, scaled_anchor):

        # Identity which cells in target have objects/ no objects
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        # Box prediction confidence:
        anchors = anchors.reshape(1,3,1,1,2)
        box_preds = torch.cat([self.sigmoid(predict[..., 1:3]), \
                               torch.exp(predict[..., 3:5]) * anchors], dim = -1)

        # Iou with prediction have object and target (obj)
        ious = iou(box_preds[obj], target[..., 1:5][obj]).detach()

        # Confidence loss: No object loss
        noobj_loss = self.bce((predict[..., 0][noobj]), (target[..., 0][noobj]))

        obj_loss = self.mse(self.sigmoid(predict[..., 0][obj]), ious * target[..., 0][obj])

        # Localizaton loss:
        predict[..., 1:3] = self.sigmoid(predict[..., 1:3])
        target[..., 3:5] = torch.long(1e-6 + target[..., 3:5] / anchors)

        box_loss = self.mse(predict[..., 1:5][obj], target[..., 1:5][obj])

        # Classification loss:
        cls_loss = self.cross_entropy(predict[..., 5: ][obj], target[..., 5:][obj].long())

        total = self.lambda_box * box_loss + \
                self.lambda_noobj * noobj_loss + \
                self.lambda_obj * obj_loss + \
                self.lambda_class * cls_loss 

        return total