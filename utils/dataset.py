import numpy as np
import os
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from .process import intersection_union as iou

class YoloDataset(Dataset):
    """
        Config and create dataset
        -------------------------
        It is necessary to calculate IOU for the bounding box and the ANCHORs
        to select the best one.
        Fore each bbox, we will then assign it to the grid cell which contains
        its midpoint and decide which anchor is responsible for it by detrmining
        which anchor the bbox has higest IOU. We will for loop thw nine indices to assign
        the target to the best anchors.


    """
    def __init__(self,
                 csv_file,
                 img_dir,
                 label_dir,
                 anchors,
                 imgsz,
                 S=[13, 26, 52],
                 C=20,
                 transform=None):
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = imgsz
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.Tensor(box[2:4]), self.anchors, is_pred=False)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box

            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale

                S = self.S[scale_idx]
                
                # Compute cell which the bbox belongs to
                i, j = int(S * y), int(S * x)

                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    # set the probability to 1
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # Calculate the center of the bbox relative to the cell
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = (
                        width * S,
                        height * S
                    )
                    box_coordinates = torch.Tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1
                
        return image, tuple(targets)




