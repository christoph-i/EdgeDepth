'''
Generate depth predictions in the standardized format for the testset (or any other dataset).
Predicted and actual bounding boxes are matched based on IoU
some GT boxes will have no corresponding pred box - in this case pred value will be set as 0.0
'''
import os
import random
import sys
import cv2
from tqdm import tqdm
from typing import Callable
import numpy as np
from scipy import stats
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import CLASS_NAMES
from bbox import BoundingBox
from detection_image import DetectionImage

# from edgeyolo_tflite_depth_ir import EdgeyoloTfliteDepthIr
from edgeyolo_pytorch_depth_ir import EdgeyoloPytorchDepthIr
from inference_runner_abc import InferenceRunner
from read_write_DetectionImages_txt_files import load_train_images, load_test_images, save_depth_predictions, load_detection_images_from_ltrb_txt_files, load_test_images_kitti, load_train_images_kitti
from Evaluation_Framework.visualisation.render_DetectionImages import show_single_detection_images


model_path_pth_depth = r"/.../.../best.pth"
raw_img_dir = r"/.../..."
preds_output_dir = r"/.../..."

yolo_inference_runner = EdgeyoloPytorchDepthIr(model_path_pth_depth, "yolo-depth-pth", input_shape=(384, 384))


test_images_gt = load_test_images(include_gt=True)
# train_images_gt = load_train_images()

# test_images_gt = load_test_images_kitti(include_gt=True)


# train_preds_dir = os.path.join(preds_output_dir, "raw_train_preds")
test_preds_dir = os.path.join(preds_output_dir, "raw_preds_test")
os.makedirs(test_preds_dir, exist_ok=True)

final_preds_out_dir = os.path.join(preds_output_dir, "final_test_predictions")
os.makedirs(final_preds_out_dir, exist_ok=True)

# for dir in [train_preds_dir, test_preds_dir, final_preds_out_dir]:
    # os.makedirs(dir, exist_ok=True)


def yolo_prediction_for_detection_image(yolo_ir: InferenceRunner, detection_image: DetectionImage) -> DetectionImage:
    image_file_path = os.path.join(raw_img_dir, detection_image.filename)
    image = cv2.imread(image_file_path)
    pred_image = yolo_ir.run_single_inference(image)
    pred_image.filename = detection_image.filename
    return pred_image



###############  run this part once to get inference for all images, save the results to xtxt files, use these for next steps ######
# det_imgs_with_yolo_inference = []
# for detection_img in tqdm(train_images_gt):
    # det_imgs_with_yolo_inference.append(yolo_prediction_for_detection_image(yolo_inference_runner, detection_img))
# save_depth_predictions(det_imgs_with_yolo_inference, train_preds_dir)

det_imgs_with_yolo_inference = []
for detection_img in tqdm(test_images_gt):
    det_imgs_with_yolo_inference.append(yolo_prediction_for_detection_image(yolo_inference_runner, detection_img))
save_depth_predictions(det_imgs_with_yolo_inference, test_preds_dir)


############ calculate absolute preds, based on raw yolo preds from disk ###########

test_imgs_yolo_preds = load_detection_images_from_ltrb_txt_files(test_preds_dir, class_names=CLASS_NAMES.KITTI_CLASSES, img_file_suffix=".png")

def match_depth_predictions(yolo_depth_prediction: DetectionImage, gt_labels: DetectionImage, iou_th=0.5):
    # function matches yolo bboxes and annotated gt boxes based on IoU
    matched_bboxes_det_img = DetectionImage([])
    matched_bboxes_det_img.filename = gt_labels.filename

    for yolo_bbox in yolo_depth_prediction.bounding_boxes:
        for gt_bbox in gt_labels.bounding_boxes:
            iou = BoundingBox.calculate_iou(yolo_bbox, gt_bbox)
            if (yolo_bbox.class_id == gt_bbox.class_id) and (iou >= iou_th):
                matched_bbox = BoundingBox(
                    left=gt_bbox.left,
                    top=gt_bbox.top,
                    right=gt_bbox.right,
                    bottom=gt_bbox.bottom,
                    confidence=gt_bbox.confidence,
                    class_id=gt_bbox.class_id,
                    class_name=gt_bbox.class_name
                )
                matched_bbox.depth_in_mm_GT = gt_bbox.depth_in_mm_GT
                matched_bbox.depth_in_mm_PRED = yolo_bbox.depth_in_mm_PRED
                matched_bboxes_det_img.bounding_boxes.append(matched_bbox)
                break # Move to the next yolo_bbox after finding a match

    return matched_bboxes_det_img


# save testset predictions with absolute depth Preds
matched_testset_detections = []
for idx, test_image_gt in enumerate(tqdm(test_images_gt, desc="Creating testset predictions", unit="detection images")):
    test_image_pred = test_imgs_yolo_preds[idx]
    if test_image_gt.filename != test_image_pred.filename:
        raise ArithmeticError(
            "Pred and GT dection images do not match. If standard ordering does not match make sure to implement manual sorting of some kind.")
    matched_image = match_depth_predictions(test_image_pred, test_image_gt)
    for bounding_box in matched_image.bounding_boxes:
        # bounding_box.depth_in_mm_PRED = int(translation_function(float(bounding_box.depth_in_mm_PRED)))
        bounding_box.depth_in_mm_PRED = int(float(bounding_box.depth_in_mm_PRED) * 86.18 * 1000)   # kitti
        # bounding_box.depth_in_mm_PRED = int(float(bounding_box.depth_in_mm_PRED) * 41483)  # internal
        bounding_box.depth_in_mm_GT = int(float(bounding_box.depth_in_mm_GT) * 1000) # kitti
        # bounding_box.depth_in_mm_GT = int(bounding_box.depth_in_mm_GT) # internal
    matched_testset_detections.append(matched_image)


# show_single_detection_images(matched_testset_detections, r"/.../...", image_display_size=(1440, 810), conf_th=0.25, show_image_filename=True)

save_depth_predictions(matched_testset_detections, final_preds_out_dir)
