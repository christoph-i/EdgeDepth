from typing import List

import cv2
import numpy as np

from EdgeYoloDepth.inference_runners.inference_runner_abc import InferenceRunner
from bbox import BoundingBox
from detection_image import DetectionImage
from EdgeYoloDepth.detect import Detector


class EdgeyoloPytorchDepthIr(InferenceRunner):
    nms_thres = 0.45
    conf_thres = 0.1
    fp16 = False

    def __init__(self,
                 model_file_path: str,
                 model_name: str,
                 excluded_classes=None,
                 input_shape=(384, 384),
                 use_cpu=True,
                 ):
        self.use_cpu = use_cpu
        self.detector = None
        self.excluded_classes = excluded_classes
        super().__init__(model_file_path, model_name, input_shape)


    def load_model(self):
        self.detector = Detector(
            weight_file=self.model_file_path,
            depth_mode=True,
            conf_thres=self.conf_thres,
            nms_thres=self.nms_thres,
            input_size=self.input_shape,
            fuse=True,
            fp16=self.fp16,
            use_decoder=False,
            cpu=self.use_cpu
        )


    def postprocess_result(self, result, img_height_orig, img_width_orig) -> DetectionImage:
        if result is None:
            return DetectionImage([])

        output_data = np.array(result.tolist())

        boxes_xyxy = output_data[:, :4]
        scores = output_data[:, 4] * output_data[:, 5]
        classes = output_data[:, 6].astype(int)
        depths = output_data[:, 7]

        # get relative bbox positions
        boxes_xyxy[:, 0] /= img_width_orig
        boxes_xyxy[:, 1] /= img_height_orig
        boxes_xyxy[:, 2] /= img_width_orig
        boxes_xyxy[:, 3] /= img_height_orig

        # account for aspect ratio (image 1:1 resizing)
        width_factor = img_width_orig / self.input_shape[0]
        height_factor = img_height_orig / self.input_shape[1]
        boxes_xyxy[:, 0] *= width_factor
        boxes_xyxy[:, 1] *= height_factor
        boxes_xyxy[:, 2] *= width_factor
        boxes_xyxy[:, 3] *= height_factor

        final_boxes = np.clip(boxes_xyxy, 0.0, 1.0)

        detection_image = DetectionImage([])
        for idx, bbox in enumerate(final_boxes):
            class_label = self.class_names[classes[idx]]
            if self.excluded_classes and class_label in self.excluded_classes:
                continue
            bbox = BoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], scores[idx],
                               classes[idx], self.class_names[classes[idx]])
            bbox.depth_in_mm_PRED = depths[idx]
            detection_image.bounding_boxes.append(bbox)

        return detection_image


    def run_batch_inference(self, images) -> List[DetectionImage]:
        height_orig, width_orig = images[0].shape[:2]
        images_resized = [cv2.resize(image, self.input_shape, interpolation=cv2.INTER_LINEAR).astype(np.uint8) for image in images]
        output_data = self.detector(images_resized, legacy=False)
        return [self.postprocess_result(frame_result, height_orig, width_orig) for frame_result in output_data]


    def run_single_inference(self, image) -> DetectionImage:
        # resize bevor handing imgs to detector - otherwise padding would be applied!
        height_orig, width_orig = image.shape[:2]
        image = cv2.resize(image, self.input_shape, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        output_data = self.detector([image], legacy=False)
        return self.postprocess_result(output_data[0], height_orig, width_orig)

