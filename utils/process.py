import numpy as np
import torch.nn as nn
import torch

def cell_transform_box(output, anchors, num_classes, input_shape):
    """
        Transform center coordinate to image coordinate
        ---------------------------------------------------------
        output yolo: Tensor [batch, num_anchors, grid size, grid size, to, tx, ty, tw, th, num_classes]

        Yolo predicts the center of each bounding box as an offset relative
        to the boundaries of a grid cell. To convert this to the actual img coordinates:
            bx = (sigmoid(x) + cx) * stride(x)
            by = (sigmoid(y) + cy) * stride(y)
        cx, cy: are the coordinates of the top-left corner of the grid cell 
                (e.g., for cell (3,4), cx =3, cy=4
        stride(x), stride(y): the size of each grid cell (width, height in pixels) 
        -------------------------------------------------
        Scaling width, height:
            bw = tw * img_w
            bh = th _ img_h
        -------------------------------------------------
        Calculating bounding box corner
            x1 = bx - (bw / 2)
            y1 = by - (bh / 2)
            x2 = bx + (bw / 2)
            y2 = by + (bh / 2)
        (x1,y1), (x2,y2) represent the top-left, bottom-right corners of bbox
    """

    grid_size = output.shape[1:3]
    num_anchors = len(anchors)
    
    # Reshape output to (batch_size, num_anchors, grid_size, grid_size, box_params)
    output = output.reshape(-1, num_anchors, grid_size[0], grid_size[1], num_classes + 5)
    
    box_xy = nn.Sigmoid(output[..., 1:3])  # bx, by (center coordinates)
    box_wh = torch.exp(output[..., 3:5]) * anchors  # bw, bh (scaled by anchor boxes)
    box_confidence = nn.Sigmoid(output[..., 0])  # Confidence score
    box_class_probs = nn.Sigmoid(output[..., 5:])  # Class probabilities
    
    # Convert box_xy from relative to grid to absolute pixel values
    grid_x, grid_y = np.meshgrid(np.arange(grid_size[0]), np.arange(grid_size[1]))
    grid_x = grid_x.reshape(-1, 1)
    grid_y = grid_y.reshape(-1, 1)
    box_xy = (box_xy + [grid_x, grid_y]) * (input_shape[0] / grid_size[0])
    
    # Convert box_wh to actual size
    box_wh = box_wh * input_shape
    
    # Calculate corner points
    box_min = box_xy - (box_wh / 2)
    box_max = box_xy + (box_wh / 2)
    
    return np.concatenate([box_min, box_max, box_confidence, box_class_probs], axis=-1)


def intersection_union(box1, box2, is_pred=True):
    """
    """
    if is_pred:
        box1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        box1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        box1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        box1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

        box2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        box2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        box2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        box2_y2 = box2[..., 3:4] + box2[..., 3:4] / 2

        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)

        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        box1_area = abs((box1_x2 - box1_x1) * (box1_y2, box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
        union = box1_area + box2_area - intersection

        epsilon = 1e-6
        iou_score = intersection / (union + epsilon)

        return iou_score
    
    else:
        # IOU score based on width and height of boundind box

        intersection_area = torch.min(box1[..., 0], box2[..., 0]) * \
                            torch.min(box1[..., 1], box2[..., 1])
        
        box1_area = box1[..., 0] * box1[..., 1]
        box2_area = box2[..., 0] * box2[..., 1]
        union_area = box1_area + box2_area - intersection_area

        iou_score = intersection_area / union_area

        return iou_score


# Non-maximum suppression function to remove overlapping bounding boxes 
def nms(bboxes, iou_threshold, threshold): 
    """
    """

	# Filter out bounding boxes with confidence below the threshold. 
    bboxes = [box for box in bboxes if box[1] > threshold] 

	# Sort the bounding boxes by confidence in descending order. 
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) 

	# Initialize the list of bounding boxes after non-maximum suppression. 
    bboxes_nms = [] 
    
    while bboxes: 
        # Get the first bounding box. 
        first_box = bboxes.pop(0) 

        # Iterate over the remaining bounding boxes. 
        for box in bboxes: 
            # If the bounding boxes do not overlap or if the first bounding box has 
            # a higher confidence, then add the second bounding box to the list of 
            # bounding boxes after non-maximum suppression. 
            if box[0] != first_box[0] or intersection_union( 
                torch.tensor(first_box[2:]), 
                torch.tensor(box[2:]), 
            ) < iou_threshold: 
                # Check if box is not in bboxes_nms 
                if box not in bboxes_nms: 
                    # Add box to bboxes_nms 
                    bboxes_nms.append(box) 

	# Return bounding boxes after non-maximum suppression. 
    return bboxes_nms
