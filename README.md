# :bomb: YOLOv3: Simple Architecture

_Knowledge provided by_ [AI VIET NAM](https://aivietnam.edu.vn/)

---

:tada: **YOLOv3 (You Only Look Once, Version 3)** is a real-time object detection algorithm that builds on the strengths of its predecessors (YOLOv1 and YOLOv2) with improvements in accuracy and speed. It is widely used in applications requiring fast and accurate object detection.

:tada: This project builds a training pipeline for the YOLOv3 model from scratch with the goal of understanding the model architecture.

---

## :rocket: Key Features of YOLOv3

**:collision: Backbone Network (Feature Extractor):**

- YOLOv3 uses Darknet-53, a deep convolutional neural network, as its backbone.
- **Darknet-53** has 53 convolutional layers with residual connections (similar to ResNet), which help in training deeper networks by addressing vanishing gradient issues.
- It achieves a good balance between speed and accuracy.
<div style="text-align:center">
<image src="./images/backbone.png", alt="darknet53">
</div>

**:collision: Multi-Scale Prediction:**

- YOLOv3 detects objects at three different scales, improving its ability to detect objects of varying sizes.
- Feature maps from the backbone at different resolutions are used for prediction:
  :zap: Large-scale feature maps for small objects.
  :zap: Medium-scale feature maps for medium-sized objects.
  :zap: Small-scale feature maps for large objects.

**:collision: Anchor Boxes:**

- YOLOv3 uses pre-defined anchor boxes (prior boxes) to predict bounding boxes.
- Each grid cell predicts three bounding boxes, and predictions are refined using offsets.

**:collision: Bounding Box Prediction:**

- Bounding boxes are parameterized using a sigmoid function for x, y coordinates (relative to the grid cell) and exponential functions for width and height (relative to anchor boxes).

<div style="text-align:center">
<image src="./images/bbox.png", alt="bbox">
</div>

**:collision: Class Prediction:**

- YOLOv3 adopts independent logistic regression for class prediction instead of the softmax function, allowing it to handle overlapping classes more effectively.

**:collision: Loss Function:**

- The loss function includes components for:
  Bounding box regression: Measures the accuracy of predicted bounding boxes.

**:collision: Confidence score:** Indicates how likely a box contains an object.

**:collision: Class probability:** Measures the classification accuracy of detected objects.

**:collision: Non-Maximum Suppression (NMS):** After predictions, NMS is applied to remove duplicate bounding boxes and keep only the most confident ones.
