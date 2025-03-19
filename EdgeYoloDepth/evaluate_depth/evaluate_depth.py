import argparse
import os
import random
import sys
import warnings
from pathlib import Path
from typing import List, Tuple
from itertools import islice

import cv2
import numpy as np
from tqdm import tqdm

from bbox import BoundingBox
from detection_image import DetectionImage
from Evaluation_Framework.visualisation.render_DetectionImages import show_single_detection_images, add_text_to_image_top_right
from Evaluation_Framework.visualisation.bboxes_visualiser import vis_detections

from inference_runners.edgeyolo_pytorch_depth_ir import EdgeyoloPytorchDepthIr
#from inference_runners.inference_runner_abc import InferenceRunner
from gt_pred_set import GtPredSet
import eval_plots
from Evaluation_Framework.visualisation import render_DetectionImages
from eval_report import EvalReport
import CLASS_NAMES

from read_write_DetectionImages_txt_files import load_train_images, load_detection_images_from_ltrb_txt_files

# TODO safety check - if rotate or shift augs are used the qualitative output can not be used in the current implementation -> throw error if user tries to do both

CLASS_NAMES = CLASS_NAMES.KITTI_CLASSES

def get_args():
    parser = argparse.ArgumentParser("EdgeYOLO depth evaluate parser")
    parser.add_argument("-w", "--weights", type=str, default=r".../best.pth", help="weights")
    parser.add_argument("--model-name", type=str, default="edgeyolo-depth", help="Name for the specific model that is tested")
    parser.add_argument("-b", "--batch", type=int, default=8, help="batch size for each device")
    parser.add_argument("-i", "--input-size", type=int, default=384, help="image input size")

    parser.add_argument("--annotations-dir", type=str, help="dir containing annotation txt files with GT bounding boxes and and depths")
    parser.add_argument("--raw-imgs-dir", type=str, help="dir containing corresponding image files")
    parser.add_argument("--output-dir", type=str, help="dir where output (plots and excel file) will be saved")
    parser.add_argument('--save-preds', action='store_true', help='If set True the matched predictions and GT bboxes will be stored as txt files in a subdir of the output dir (format per line: class-id, x, y, w, h, depth-gt, depth-pred)')
    parser.add_argument('--save-visuals', type=float, default=0.0, help='If set True the percentage of images will be saved as visualisations with gt and pred annotations')

    parser.add_argument("--iou-th", type=float, default=0.5, help='IoU threshold for bounding box matching between GT bboxes and those predicted by the model (only matches above the threshold will be considered for depth evaluation')

    parser.add_argument("--aug-rl-shift-percent", type=float, default=0.0, help='augment by doing random left right shift of up to x percent - e.g. 0.25 = 25%')
    parser.add_argument("--aug-rotate-degrees", type=int, default=0, help='augment by doing random rotation around center up to x degree')
    parser.add_argument('--show-images-popup', action='store_true', help='Window will pop up showing the images - only for debug!')

    parser.add_argument("--device", type=int, nargs="+", default=[0], help="eval device")
    parser.add_argument('--fp16', action='store_true', help='half precision')

    # TODO auto denor and add kitti values as default 
    # TODO use kitti bins as default 
    parser.add_argument("--denorm-gt-depth-max", type=int, default=41483, help="Use only if GROUND TRUTH depth normalisation to 0.0-1.0 range was used - otherwise set as -1. Provide the largest metric values used for normalization (typically largest metric value in the dataset, abs depth will be computed by multiplying rel depth with the provided value)")
    parser.add_argument("--denorm-pred-depth-max", type=int, default=41483, help="Use only if PRED depth normalisation to 0.0-1.0 range was used - otherwise set as -1. Provide the largest metric values used for normalization (typically largest metric value in the dataset, abs depth will be computed by multiplying rel depth with the provided value)")
    parser.add_argument("--pred-conf-th", type=float, default=0.01, help="Consider only predictions with a confidence above the threshold")
    parser.add_argument("--pred-depth-th", type=int, default=27500, help="Consider only depth values below this threshold")
    parser.add_argument("--standard-bins", type=int, nargs="+", default=[0, 5000, 10000, 15000, 20000, 25000, 30000, 35000], help="Standard bins for bin-wise depth evaluation")
    return parser.parse_args()


####### read labels from disk #######

def clip_bbox(x, y, w, h):
    # Compute l, t, r, b
    l = x - w / 2.0
    t = y - h / 2.0
    r = x + w / 2.0
    b = y + h / 2.0
    
    # Validate 10% above/below
    coords = [("l", l), ("t", t), ("r", r), ("b", b)]
    for name, val in coords:
        if val < -0.1:
            raise ValueError(f"{name} = {val} is more than 10% below 0.0!")
        if val > 1.1:
            raise ValueError(f"{name} = {val} is more than 10% above 1.0!")
    
    # Now clip l, t, r, b into [0.0, 1.0]
    l = max(0.0, min(1.0, l))
    t = max(0.0, min(1.0, t))
    r = max(0.0, min(1.0, r))
    b = max(0.0, min(1.0, b))
    
    return l, t, r, b

def read_yolo_depth_file(filepath: str) -> DetectionImage:
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()
            detection_image = DetectionImage([])

            for line in lines:
                # Split the line into class_id, confidence, and bbox dimensions
                parts = line.strip().split(' ')
                if not len(parts) == 6:
                    raise ValueError(
                        f"Invalid annotation format in the txt file: {filepath}. Annotation will be skipped.")
                class_id, x, y, w, h, d = map(float, parts)
                l, t, r, b = clip_bbox(x, y, w, h)
                class_id = int(class_id)
                # l = x - w / 2.
                # t = y - h / 2.
                # r = x + w / 2.
                # b = y + h / 2.
                bounding_box = BoundingBox(l, t, r, b, 1.0, class_id, CLASS_NAMES[class_id])
                bounding_box.depth_in_mm_GT = d if args.denorm_gt_depth_max==-1 else int(d * args.denorm_gt_depth_max)
                detection_image.bounding_boxes.append(bounding_box)

            # detection_image.filename = Path(filepath).stem + ".jpg"
            detection_image.filename = Path(filepath).stem + ".png" # kitti
            return detection_image

    except FileNotFoundError:
        print(f"Annotation file '{filepath}' not found.")


def load_detection_images(annotations_dir: str, raw_imgs_dir: str) -> List[DetectionImage]:
    detection_images = []
    txt_files = [os.path.join(root, file) for root, dirs, files in os.walk(annotations_dir) for file in files if
                 file.endswith('.txt')]
    for txt_file in tqdm(txt_files, desc="Loading DetectionImages from txt files", unit="file"):
        detection_image = read_yolo_depth_file(txt_file)
        detection_image.image_dir = raw_imgs_dir
        detection_images.append(detection_image)
    return detection_images

def save_det_imgs_to_yolo_txt(detection_images: list[DetectionImage], output_dir: str, include_depth=True):
    print("Writing detection images in yolo format.")
    for detection_image in tqdm(detection_images):
        filename = os.path.splitext(detection_image.filename)[0]
        out_file_path = os.path.join(output_dir, filename + ".txt")
        with open(out_file_path, 'w') as file:
            for bounding_box in detection_image.bounding_boxes:
                dimensions_yolo_rel = bounding_box.get_dimensions_yolo_rel()
                x_y_w_h_string = f"{dimensions_yolo_rel[0]} {dimensions_yolo_rel[1]} {dimensions_yolo_rel[2]} {dimensions_yolo_rel[3]}"
                line = f"{bounding_box.get_class_id()} {bounding_box.confidence} {x_y_w_h_string}"
                if include_depth:
                    line = f"{line} {bounding_box.depth_in_mm_GT} {bounding_box.depth_in_mm_PRED}"
                file.write(f"{line}\n")

####### test augmentations #######

def aug_lr_shift(det_img: DetectionImage, image: np.ndarray, max_shift_percent: float) -> Tuple[DetectionImage, np.ndarray]:
    # shift the image randomly to the right or left up to the maximum defined by the max_shift_percent parameter
    # space to the right or left will be filled with black pixels
    # Bounding boxes are moved accordingly - if bboxes are completly off the screen due to shift remove them from the detection image

    h, w = image.shape[:2]

    # Compute random shift in [-max_shift, max_shift], in absolute integer pixels
    max_shift_px = int(max_shift_percent * w)
    shift_px = random.randint(-max_shift_px, max_shift_px)  # integer-based random shift

    if shift_px == 0.0 or shift_px == 0:
        # No shift, just return original
        return det_img, image

    # Create empty (black) output image
    shifted_image = np.zeros_like(image)

    # Decide the region of the original image that will map into the new image
    # and place it properly in "shifted_image"
    if shift_px > 0:
        # Positive shift: shift to the right
        # The left part of the new image is black
        new_left = shift_px
        old_left = 0
        old_right = w - shift_px
        shifted_image[:, new_left:] = image[:, old_left:old_right]
    else:
        # Negative shift: shift to the left
        shift_abs = abs(shift_px)
        new_left = 0
        old_left = shift_abs
        old_right = w
        shifted_image[:, new_left:w - shift_abs] = image[:, old_left:old_right]

    # Update bounding boxes
    i = 0
    while i < len(det_img.bounding_boxes):
        bbox = det_img.bounding_boxes[i]
        l_abs, t_abs, r_abs, b_abs = bbox.get_dimensions_ltrb_abs(w, h)

        # Shift horizontally
        l_abs_new = l_abs + shift_px
        r_abs_new = r_abs + shift_px

        # Check if the bounding box is fully off-screen
        if r_abs_new < 0 or l_abs_new >= w:
            # Remove the bounding box from the list
            det_img.bounding_boxes.pop(i)
        else:
            # Move to the next index only if no removal occurred
            i += 1

        # Clip to the image range [0, w)
        l_abs_clipped = max(0, min(w - 1, l_abs_new))
        r_abs_clipped = max(0, min(w - 1, r_abs_new))
        t_abs_clipped = max(0, min(h - 1, t_abs))
        b_abs_clipped = max(0, min(h - 1, b_abs))

        # Check if the bounding box is at least 2x2
        box_width = r_abs_clipped - l_abs_clipped
        box_height = b_abs_clipped - t_abs_clipped
        if box_width < 2 or box_height < 2:
            continue

        # Convert back to relative
        left_rel = l_abs_clipped / w
        right_rel = r_abs_clipped / w
        top_rel = t_abs_clipped / h
        bottom_rel = b_abs_clipped / h

        bbox.left = left_rel
        bbox.top = top_rel
        bbox.right = right_rel
        bbox.bottom = bottom_rel

    return det_img, shifted_image


def aug_rotate(det_img: DetectionImage, image: np.ndarray, max_degrees: int) -> Tuple[DetectionImage, np.ndarray]:
    # image is rotated around its center randomly either clockwise or conterclockwise up to the degree specified as a parameter
    # Bounding boxes are moved accordingly
    # potential empty areas in the corners are filled with black pixels

    h, w = image.shape[:2]
    if max_degrees == 0:
        # No rotation
        return det_img, image

    # Random angle in degrees, possibly negative
    angle = random.uniform(-max_degrees, max_degrees)

    # Compute rotation matrix for OpenCV
    # Center of rotation: (cx, cy)
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # Perform the rotation with black fill
    rotated_image = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)  # black
    )

    # Rotate bounding boxes
    i = 0
    while i < len(det_img.bounding_boxes):
        bbox = det_img.bounding_boxes[i]
        l_abs, t_abs, r_abs, b_abs = bbox.get_dimensions_ltrb_abs(w, h)

        # (x1, y1), (x2, y2), (x3, y3), (x4, y4) corners
        corners = np.array([
            [l_abs, t_abs],
            [r_abs, t_abs],
            [r_abs, b_abs],
            [l_abs, b_abs]
        ], dtype=np.float32)

        # Convert corners to (N×1×2) and apply affine transform
        corners_hom = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1)  # shape: (4, 3)
        M_full = np.vstack([M, [0, 0, 1]])  # make it 3×3 for matrix multiplication
        rotated_corners = (M_full @ corners_hom.T).T  # shape: (4, 3)
        # rotated_corners[:,0] and rotated_corners[:,1] are x and y
        xs = rotated_corners[:, 0]
        ys = rotated_corners[:, 1]

        # Find min/max
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        if max_x < 0 or min_x >= w or max_y < 0 or min_y >= h:
            # Fully outside
            det_img.bounding_boxes.pop(i)
        else:
            i += 1

        min_x_clipped = max(0, min(w - 1, min_x))
        max_x_clipped = max(0, min(w - 1, max_x))
        min_y_clipped = max(0, min(h - 1, min_y))
        max_y_clipped = max(0, min(h - 1, max_y))

        # Check if the bounding box is at least 2x2
        box_width = max_x_clipped - min_x_clipped
        box_height = max_y_clipped - min_y_clipped
        if box_width < 2 or box_height < 2:
            continue

        # Convert back to relative
        left_rel = min_x_clipped / w
        right_rel = max_x_clipped / w
        top_rel = min_y_clipped / h
        bottom_rel = max_y_clipped / h

        bbox.left = left_rel
        bbox.top = top_rel
        bbox.right = right_rel
        bbox.bottom = bottom_rel

    return det_img, rotated_image



def show_detection_image(det_img: DetectionImage, image: np.ndarray, image_display_size=(1920,1080)):
    rendered_image = vis_detections(image, det_img.bounding_boxes, conf_th=0.0,
                                                      show_conf=True, show_depth=True)
    rendered_image = add_text_to_image_top_right(rendered_image, det_img.filename)
    rendered_image = cv2.resize(rendered_image, image_display_size, interpolation=cv2.INTER_AREA)
    cv2.imshow('Image', rendered_image)
    key = cv2.waitKey(0)
    
def save_detection_image(det_img: DetectionImage, image: np.ndarray, save_dir: str) -> None:
    rendered_image = vis_detections(image, det_img.bounding_boxes, conf_th=0.0,
                                                      show_conf=True, show_depth=True)
    rendered_image = add_text_to_image_top_right(rendered_image, det_img.filename)
    out_path = os.path.join(save_dir, det_img.filename)
    cv2.imwrite(out_path, rendered_image)


####### run inference -> get predictions #######


def run_yolo_inference(args, detection_images_gt: List[DetectionImage]) -> Tuple[List[DetectionImage], List[DetectionImage]]:
    yolo_depth_inference_runner = EdgeyoloPytorchDepthIr(args.weights, args.model_name, input_shape=(args.input_size, args.input_size))
    yolo_depth_inference_runner.load_model()

    iterator = iter(detection_images_gt)
    total = len(detection_images_gt)
    
    random.seed(42)
    if args.save_visuals > 0.0:
        vis_save_dir_gt = os.path.join(args.output_dir, "visualisations_gt")
        vis_save_dir_pred = os.path.join(args.output_dir, "visualisations_pred")
        os.mkdir(vis_save_dir_gt)
        os.mkdir(vis_save_dir_pred)

    detection_images_pred = []
    with tqdm(total=total, desc="Running yolo model inference") as pbar:
        while det_imgs_batch := list(islice(iterator, args.batch)):
            images = [det_img.get_image(args.raw_imgs_dir) for det_img in det_imgs_batch]

            # do augmentations
            if args.aug_rl_shift_percent != 0:
                for i in range(len(det_imgs_batch)):
                    _, images[i] = aug_lr_shift(det_imgs_batch[i], images[i], args.aug_rl_shift_percent)

            if args.aug_rotate_degrees != 0:
                for i in range(len(det_imgs_batch)):
                    _, images[i] = aug_rotate(det_imgs_batch[i], images[i], args.aug_rotate_degrees)

            if args.show_images_popup:
                for i in range(len(det_imgs_batch)):
                    show_detection_image(det_imgs_batch[i], images[i])

            # run inference and add to PRED lists
            batch_preds = yolo_depth_inference_runner.run_batch_inference(images)
            detection_images_pred.extend(batch_preds)
            pbar.update(len(det_imgs_batch))
            
            if random.random() < args.save_visuals:
                for i in range(len(det_imgs_batch)):
                    save_detection_image(det_imgs_batch[i], images[i], vis_save_dir_gt)
                    pred_img = batch_preds[i]
                    pred_img.filename = det_imgs_batch[i].filename
                    save_detection_image(pred_img, images[i], vis_save_dir_pred)

    if args.denorm_pred_depth_max != -1:
        for det_img in detection_images_pred:
            for bbox in det_img.bounding_boxes:
                bbox.depth_in_mm_PRED = int(bbox.depth_in_mm_PRED * args.denorm_pred_depth_max)

    return detection_images_gt, detection_images_pred


####### match gt and pred bboxes #######

def match_bboxes(gt_labels: DetectionImage, yolo_prediction: DetectionImage, iou_th=0.5) -> DetectionImage:
    # function matches yolo bboxes and annotated gt boxes based on IoU
    matched_bboxes_det_img = DetectionImage([])
    matched_bboxes_det_img.filename = gt_labels.filename
    matched_bboxes_det_img.image_dir = gt_labels.image_dir

    for yolo_bbox in yolo_prediction.bounding_boxes:
        for gt_bbox in gt_labels.bounding_boxes:
            iou = BoundingBox.calculate_iou(yolo_bbox, gt_bbox)
            if (yolo_bbox.class_id == gt_bbox.class_id) and (iou >= iou_th):
                matched_bbox = BoundingBox(
                    left=gt_bbox.left,
                    top=gt_bbox.top,
                    right=gt_bbox.right,
                    bottom=gt_bbox.bottom,
                    confidence=yolo_bbox.confidence,
                    class_id=gt_bbox.class_id,
                    class_name=gt_bbox.class_name
                )
                matched_bbox.depth_in_mm_GT = gt_bbox.depth_in_mm_GT
                matched_bbox.depth_in_mm_PRED = yolo_bbox.depth_in_mm_PRED
                matched_bboxes_det_img.bounding_boxes.append(matched_bbox)
                break # Move to the next yolo_bbox after finding a match

    return matched_bboxes_det_img


def match_gt_pred_by_iou(gt_images: list[DetectionImage], pred_images: list[DetectionImage], iou_th: float) -> list[DetectionImage]:
    matched_images = []
    for gt_image, pred_image in zip(gt_images, pred_images):
        matched_images.append(match_bboxes(gt_image, pred_image, iou_th))
    return matched_images


####### calculate quantitative and qualitative metrics #######

def generate_eval_report(gt_pred_data: GtPredSet, report_save_dir: str):
    eval_report = EvalReport(gt_pred_data, report_save_dir)

    eval_report. \
        add_abs_relative_error_stddev_var(). \
        add_rmse_stddev_var(). \
        add_a1_threshold_accuracy_stddev_var(). \
        add_abs_relative_error_binned(bins=args.standard_bins). \
        add_rmse_binned(bins=args.standard_bins). \
        add_a1_threshold_binned(bins=args.standard_bins). \
        add_abs_error_binned(bins=args.standard_bins)

    # eval_report.\
    #     add_abs_relative_error().\
    #     add_rmse().\
    #     add_a1_threshold_accuracy().\
    #     add_abs_relative_error_binned(bins=standard_bins). \
    #     add_rmse_binned(bins=standard_bins). \
    #     add_a1_threshold_binned(bins=standard_bins). \
    #     add_abs_error_binned(bins=standard_bins)

    eval_report.write_to_excel("metrics-report.xlsx")


def generate_eval_plots(gt_pred_data: GtPredSet, save_dir: str):
    eval_plots.plot_relative_error(gt_pred_data, save_dir)
    eval_plots.plot_abs_relative_error_lines(gt_pred_data, save_dir)
    eval_plots.plot_abs_error_lines(gt_pred_data, save_dir)
    eval_plots.plot_gt_pred_values(gt_pred_data, save_dir)


def generate_qualitative_results(detection_images: list[DetectionImage], save_dir: str):
    random_dets_dir = os.path.join(save_dir, "vis_random_300_prediction")
    most_dets_dir = os.path.join(save_dir, "vis_most_boxes_300_prediction")
    os.mkdir(random_dets_dir)
    os.mkdir(most_dets_dir)
    render_DetectionImages.write_random_inference_results(detection_images, random_dets_dir,
                                                          args.raw_imgs_dir, max_imgs=300)
    render_DetectionImages.write_most_detections_inference_results(detection_images,
                                                                   most_dets_dir,
                                                                   args.raw_imgs_dir, max_imgs=300)



if __name__ == '__main__':
    args = get_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Made output dir - results will be saved to {args.output_dir}")

    test_images_gt = load_detection_images(args.annotations_dir, args.raw_imgs_dir)
    test_images_gt, test_images_pred = run_yolo_inference(args, test_images_gt)
    matched_images = match_gt_pred_by_iou(test_images_gt, test_images_pred, args.iou_th)

    if args.save_preds:
        save_preds_dir = os.path.join(args.output_dir, 'matched_predictions')
        os.makedirs(save_preds_dir)
        save_det_imgs_to_yolo_txt(matched_images, save_preds_dir)

    gt_pred_set = GtPredSet(matched_images, bbox_conf_th=args.pred_conf_th,
                            remove_gt_zero_values=True,
                            exclude_pred_zero_values=False,
                            pred_depth_threshold=args.pred_depth_th)
    print("Generating evaluation report with depth results")
    generate_eval_report(gt_pred_set, args.output_dir)
    # generate_eval_plots(gt_pred_set, args.output_dir) # not working for kitti yet

