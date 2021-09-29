# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn
import cv2
import numpy as np
import os

def visualize_feature_maps(feature_tensor):
    dst_path = "./feature_maps"
    therd_size = 256
    iter_range = feature_tensor.shape[0]
    for i in range(iter_range):

        feature = feature_tensor.cpu().data.numpy()
        feature_img = feature[i, :, :]
        feature_img = np.asarray(feature_img * 255).astype(np.uint8)
        # feature_img = np.clip(feature_img, 0, 255)
        feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_WINTER)
        if feature_img.shape[0] < therd_size:
            tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.png')
            tmp_img = feature_img.copy()
            tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
            # cv2.imwrite(tmp_file, tmp_img)

        dst_file = os.path.join(dst_path, str(i) + '.png')
        cv2.imwrite(dst_file, feature_img)


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        visualize_feature_maps(features[0][0])

        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses

        return detections
