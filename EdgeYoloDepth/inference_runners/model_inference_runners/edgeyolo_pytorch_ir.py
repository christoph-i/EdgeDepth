import tensorflow as tf
import cv2
import numpy as np

from inference_runner_abc import InferenceRunner
from bbox import BoundingBox
from detection_image import DetectionImage
from EdgeYoloDepth.detect import Detector


class EdgeyoloPytorchIr(InferenceRunner):
    nms_thres = 0.45
    conf_thres = 0.1
    fp16 = False
    top_crop = 1 / 3

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
            depth_mode=False,
            conf_thres=self.conf_thres,
            nms_thres=self.nms_thres,
            input_size=self.input_shape,
            fuse=True,
            fp16=self.fp16,
            use_decoder=False,
            cpu=self.use_cpu
        )

    def run_single_inference(self, image) -> DetectionImage:
        image = cv2.resize(image, self.input_shape, interpolation=cv2.INTER_LINEAR).astype(np.uint8)

        output_data = self.detector([image], legacy=False)

        if output_data[0] is None:
            return DetectionImage([])

        output_data = np.array(output_data[0].tolist())

        input_height, input_width, _ = image.shape

        boxes_xyxy = output_data[:, :4]
        scores = output_data[:, 4] * output_data[:, 5]
        classes = output_data[:, 6].astype(int)

        # get relative bbox positions
        boxes_xyxy[:, 0] /= input_width
        boxes_xyxy[:, 1] /= input_height
        boxes_xyxy[:, 2] /= input_width
        boxes_xyxy[:, 3] /= input_height

        # account for aspect ratio (image 1:1 resizing)
        width_factor = input_width / self.input_shape[0]
        height_factor = input_height / self.input_shape[1]
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
            detection_image.bounding_boxes.append(BoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], scores[idx],
                                                              classes[idx], class_label))

        return detection_image