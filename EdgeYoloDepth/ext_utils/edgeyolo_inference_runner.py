import numpy as np
import tensorflow as tf
import cv2

import CLASS_NAMES

class EdgeYoloInferenceRunner():

    def __init__(self,
                 tflite_model_file_path: str,
                 input_shape=(384, 384),
                 one_by_one_mode=True
                 ):
        self.tflite_model_file_path = tflite_model_file_path
        self.input_shape = input_shape
        self.one_by_one_mode = one_by_one_mode
        self.class_names = CLASS_NAMES.KITTI_CLASSES
        self._COLORS = np.array(
            [
                0.000, 0.447, 0.741,
                0.850, 0.325, 0.098,
                0.929, 0.694, 0.125,
                0.494, 0.184, 0.556,
                0.466, 0.674, 0.188,
                0.301, 0.745, 0.933,
                0.635, 0.078, 0.184,
                0.300, 0.300, 0.300,
                0.600, 0.600, 0.600,
                1.000, 0.000, 0.000,
                1.000, 0.500, 0.000,
                0.749, 0.749, 0.000,
                0.000, 1.000, 0.000,
                0.000, 0.000, 1.000,
                0.667, 0.000, 1.000,
                0.333, 0.333, 0.000,
                0.333, 0.667, 0.000,
                0.333, 1.000, 0.000,
                0.667, 0.333, 0.000,
                0.667, 0.667, 0.000,
                0.667, 1.000, 0.000,
                1.000, 0.333, 0.000,
                1.000, 0.667, 0.000,
                1.000, 1.000, 0.000,
                0.000, 0.333, 0.500,
                0.000, 0.667, 0.500,
                0.000, 1.000, 0.500,
                0.333, 0.000, 0.500,
                0.333, 0.333, 0.500,
                0.333, 0.667, 0.500,
                0.333, 1.000, 0.500,
                0.667, 0.000, 0.500,
                0.667, 0.333, 0.500,
                0.667, 0.667, 0.500,
                0.667, 1.000, 0.500,
                1.000, 0.000, 0.500,
                1.000, 0.333, 0.500,
                1.000, 0.667, 0.500,
                1.000, 1.000, 0.500,
                0.000, 0.333, 1.000,
                0.000, 0.667, 1.000,
                0.000, 1.000, 1.000,
                0.333, 0.000, 1.000,
                0.333, 0.333, 1.000,
                0.333, 0.667, 1.000,
                0.333, 1.000, 1.000,
                0.667, 0.000, 1.000,
                0.667, 0.333, 1.000,
                0.667, 0.667, 1.000,
                0.667, 1.000, 1.000,
                1.000, 0.000, 1.000,
                1.000, 0.333, 1.000,
                1.000, 0.667, 1.000,
                0.333, 0.000, 0.000,
                0.500, 0.000, 0.000,
                0.667, 0.000, 0.000,
                0.833, 0.000, 0.000,
                1.000, 0.000, 0.000,
                0.000, 0.167, 0.000,
                0.000, 0.333, 0.000,
                0.000, 0.500, 0.000,
                0.000, 0.667, 0.000,
                0.000, 0.833, 0.000,
                0.000, 1.000, 0.000,
                0.000, 0.000, 0.167,
                0.000, 0.000, 0.333,
                0.000, 0.000, 0.500,
                0.000, 0.000, 0.667,
                0.000, 0.000, 0.833,
                0.000, 0.000, 1.000,
                0.000, 0.000, 0.000,
                0.143, 0.143, 0.143,
                0.286, 0.286, 0.286,
                0.429, 0.429, 0.429,
                0.571, 0.571, 0.571,
                0.714, 0.714, 0.714,
                0.857, 0.857, 0.857,
                0.000, 0.447, 0.741,
                0.314, 0.717, 0.741,
                0.50, 0.5, 0
            ]
        ).astype(np.float32).reshape(-1, 3)
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.load_model()


    def load_model(self):
        self.interpreter = tf.lite.Interpreter(model_path=self.tflite_model_file_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def preproc(self, img, input_size, swap=(2, 0, 1)):
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

    def multiclass_nms_class_agnostic(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self.nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return dets

    def draw_boxes_on_image(self, image, boxes, scores, classes, conf_th):
        for box, score, cls in zip(boxes, scores, classes):
            if score < conf_th:
                continue
            # Extract coordinates
            x1, y1, x2, y2 = box

            # Draw rectangle (box)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Prepare text
            label = "{}: {:.2f}".format(cls, score)

            # Choose position for text
            y = y1 - 15 if y1 > 20 else y1 + 15

            # Put text (class and probability)
            cv2.putText(image, label, (int(x1), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image

    def vis_detections(self, img, boxes, scores, cls_ids, conf=0.5, class_names=None):

        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = (self._COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(self._COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            txt_bk_color = (self._COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

        return img


    def run_inference_return_dets(self, image, use_rel_bbox_coords=True):
        image_preprocessed, ratio_padding_case = self.preproc(image, self.input_shape)
        input_data = np.expand_dims(np.array(image_preprocessed), axis=0)
        input_height, input_width, _ = image.shape

        # Perform the inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Get the output and print it
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        output_data = output_data[0]

        boxes = output_data[:, :4]
        scores = output_data[:, 4:5] * output_data[:, 5:]

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
        dets = self.multiclass_nms_class_agnostic(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]

        if use_rel_bbox_coords:
            final_boxes[:, 0] /= input_width
            final_boxes[:, 1] /= input_height
            final_boxes[:, 2] /= input_width
            final_boxes[:, 3] /= input_height

        return final_boxes, final_scores, final_cls_inds


    def run_inference_return_vis(self, image_file_path: str, conf_th: float):
        image = cv2.imread(image_file_path)
        bboxes, scores, cls_inds = self.run_inference_return_dets(image, use_rel_bbox_coords=False)
        return self.vis_detections(image, bboxes, scores, cls_inds, conf=conf_th, class_names=self.class_names)