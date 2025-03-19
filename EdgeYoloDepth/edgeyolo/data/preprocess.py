# import numpy as np
# import cv2
import torch.nn as nn


def preprocess(inputs, targets, input_size, depth_mode=False):
    _, _, h, w = inputs.shape
    scale_y = input_size[0] / h
    scale_x = input_size[1] / w
    if scale_x != 1 or scale_y != 1:
        if depth_mode:
            # TODO implement for depth mode
            raise NotImplementedError("Input images were not provided in expected dimesions (height or width) by dataloader. Automatic rescaling was attempted but is not yet implemented for depth mode.")
        inputs = nn.functional.interpolate(
            inputs, size=input_size, mode="bilinear", align_corners=False
        )
        # Updating every secound value (so only the bboxes) - this will fail in depth mode because depth info is addad in the end and would be wrongfully altered as ewll
        targets[..., 1::2] = targets[..., 1::2] * scale_x
        targets[..., 2::2] = targets[..., 2::2] * scale_y
    return inputs, targets
