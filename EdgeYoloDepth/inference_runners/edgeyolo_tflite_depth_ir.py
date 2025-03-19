import tensorflow as tf
import cv2
import numpy as np

from inference_runner_abc import InferenceRunner
from bbox import BoundingBox
from detection_image import DetectionImage

class EdgeyoloTfliteDepthIr(InferenceRunner):

        def __init__(self,
                     model_file_path: str,
                     model_name: str,
                     input_shape=(384, 384)
                     ):
            super().__init__(model_file_path, model_name, input_shape)
            self.interpreter = None
            self.input_details = None
            self.output_details = None

            self.one_by_one_mode = True # TODO check if works -> if so remove it or keep it if useful..

            self.load_model()



        def load_model(self):
            self.interpreter = tf.lite.Interpreter(model_path=self.model_file_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

        # TODO check if bgr to rgb conversion is necessary or how its trained -> better results with rgb
        def preproc_image(self, img, input_size, swap=(2, 0, 1)):
            if len(img.shape) == 3:
                padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
            else:
                padded_img = np.ones(input_size, dtype=np.uint8) * 114

            r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
            if self.one_by_one_mode:
                padded_img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            else:
                resized_img = cv2.resize(
                    img,
                    (int(img.shape[1] * r), int(img.shape[0] * r)),
                    interpolation=cv2.INTER_LINEAR,
                ).astype(np.uint8)
                padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

            padded_img = padded_img.transpose(swap)
            padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
            return padded_img, r

        def nms(self, boxes, scores, nms_thr):
            """Single class NMS implemented in Numpy."""
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(ovr <= nms_thr)[0]
                order = order[inds + 1]

            return keep

        def multiclass_nms_class_agnostic(self, boxes, scores, depths, nms_thr, score_thr):
            """Multiclass NMS implemented in Numpy. Class-agnostic version."""
            cls_inds = scores.argmax(1)
            cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

            valid_score_mask = cls_scores > score_thr
            if valid_score_mask.sum() == 0:
                return None
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            valid_cls_inds = cls_inds[valid_score_mask]
            valid_depths = depths[valid_score_mask]
            keep = self.nms(valid_boxes, valid_scores, nms_thr)
            if keep:
                dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None], valid_depths[keep, None]], 1)
                 #   [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            return dets

        def run_single_inference(self, image) -> DetectionImage:
            image_preprocessed, ratio_padding_case = self.preproc_image(image, self.input_shape)
            input_data = np.expand_dims(np.array(image_preprocessed), axis=0)
            input_height, input_width, _ = image.shape

            # Perform the inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()

            # Get the output and print it
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            output_data = output_data[0]

            boxes = output_data[:, :4]
            scores = output_data[:, 4:5] * output_data[:, 5:-1]
            depths = output_data[:, -1]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
            if self.one_by_one_mode:
                width_factor = input_width / self.input_shape[0]
                height_factor = input_height / self.input_shape[1]
                boxes_xyxy[:, 0] *= width_factor
                boxes_xyxy[:, 1] *= height_factor
                boxes_xyxy[:, 2] *= width_factor
                boxes_xyxy[:, 3] *= height_factor
            else:
                boxes_xyxy /= ratio_padding_case
            dets = self.multiclass_nms_class_agnostic(boxes_xyxy, scores, depths, nms_thr=0.45, score_thr=0.004)
            if dets is not None:
                final_boxes, final_scores, final_cls_inds, final_depths = dets[:, :4], dets[:, 4], dets[:, 5], dets[:, 6]
            else:
                return DetectionImage([])
            final_cls_inds = final_cls_inds.astype(int)
            # get relative bbox positions
            final_boxes[:, 0] /= input_width
            final_boxes[:, 1] /= input_height
            final_boxes[:, 2] /= input_width
            final_boxes[:, 3] /= input_height
            final_boxes = np.clip(final_boxes, 0.0, 1.0)

            detection_image = DetectionImage([])
            for idx, bbox in enumerate(final_boxes):
                bbox = BoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], final_scores[idx],
                                                           final_cls_inds[idx], self.class_names[final_cls_inds[idx]])
                bbox.depth_in_mm_PRED = final_depths[idx]
                detection_image.bounding_boxes.append(bbox)

            return detection_image