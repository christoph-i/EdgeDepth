import os
import warnings
from pathlib import Path
from tqdm import tqdm

from detection_image import DetectionImage
from bbox import BoundingBox
import CLASS_NAMES
import LOCAL_PATHS

def write_yolo_results(detection_images: list[DetectionImage], output_dir: str, include_depth=True, incl_conf=True, incl_GT=True):
    print("Writing detection images in yolo format.")
    for detection_image in tqdm(detection_images):
        filename = os.path.splitext(detection_image.filename)[0]
        out_file_path = os.path.join(output_dir, filename + ".txt")
        with open(out_file_path, 'w') as file:
            for bounding_box in detection_image.bounding_boxes:
                dimensions_yolo_rel = bounding_box.get_dimensions_yolo_rel()
                x_y_w_h_string = f"{dimensions_yolo_rel[0]} {dimensions_yolo_rel[1]} {dimensions_yolo_rel[2]} {dimensions_yolo_rel[3]}"
                if not incl_conf:
                    line = f"{bounding_box.get_class_id()} {x_y_w_h_string}"
                else:    
                    line = f"{bounding_box.get_class_id()} {bounding_box.confidence} {x_y_w_h_string}"
                
                if include_depth:
                    if not incl_GT:
                        line = f"{line} {bounding_box.depth_in_mm_PRED}"
                    else:
                        line = f"{line} {bounding_box.depth_in_mm_GT} {bounding_box.depth_in_mm_PRED}"
                    
                file.write(f"{line}\n")


def write_ltrb_results(detection_images: list[DetectionImage], output_dir: str, include_depth=True, incl_conf=True, incl_GT=True):
    print("Writing detection images in ltrb format.")
    for detection_image in tqdm(detection_images):
        filename = os.path.splitext(detection_image.filename)[0]
        out_file_path = os.path.join(output_dir, filename + ".txt")
        with open(out_file_path, 'w') as file:
            for bounding_box in detection_image.bounding_boxes:
                dimensions_ltrb_rel = bounding_box.get_dimensions_ltrb_rel()
                ltrb_string = f"{dimensions_ltrb_rel[0]} {dimensions_ltrb_rel[1]} {dimensions_ltrb_rel[2]} {dimensions_ltrb_rel[3]}"
                if not incl_conf:
                    line = f"{bounding_box.get_class_id()} {ltrb_string}"
                else:    
                    line = f"{bounding_box.get_class_id()} {bounding_box.confidence} {ltrb_string}"
                if include_depth:
                    if not incl_GT:
                        line = f"{line} {bounding_box.depth_in_mm_PRED}"
                    else:
                        line = f"{line} {bounding_box.depth_in_mm_GT} {bounding_box.depth_in_mm_PRED}"
                file.write(f"{line}\n")


def read_ltrb_txt_file(ltrb_txt_file_path: str, include_gt=True, conf_th: float =None, class_names=CLASS_NAMES.REDUCED_CLASSES, img_file_suffix=".jpg") -> DetectionImage:
    try:
        with open(ltrb_txt_file_path, 'r') as file:
            lines = file.readlines()
            detection_image = DetectionImage([])

            for line in lines:
                # Split the line into class_id, confidence, and bbox dimensions
                parts = line.strip().split(' ')
                if len(parts) < 6 or len(parts) > 8:
                    warnings.warn(f"Invalid annotation format in the txt file: {ltrb_txt_file_path}. Annotation will be skipped.")
                    continue
                class_id, conf, l, t, r, b = map(float, parts[:6])
                if conf_th and conf < conf_th:
                    continue
                class_id = int(class_id)
                bounding_box = BoundingBox(l, t, r, b, conf, class_id, class_names[class_id])
                if len(parts) >= 7 and include_gt:
                    # has GT depth
                    if parts[6] == 'None':
                        bounding_box.depth_in_mm_GT = None
                    else:
                        bounding_box.depth_in_mm_GT = parts[6]
                if len(parts) == 8:
                    # has GT and PRED depth
                    if parts[7] == 'None':
                        bounding_box.depth_in_mm_PRED = None
                    else:
                        bounding_box.depth_in_mm_PRED = parts[7]
                detection_image.bounding_boxes.append(bounding_box)

            detection_image.filename = Path(ltrb_txt_file_path).stem + img_file_suffix
            return detection_image

    except FileNotFoundError:
        print(f"Annotation file '{ltrb_txt_file_path}' not found.")


def load_detection_images_from_ltrb_txt_files(ltrb_txt_files_dir: str, include_gt=True, conf_th: float =None, include_empty=True, class_names=None, img_file_suffix=".jpg") -> list[DetectionImage]:
    detection_images = []
    txt_files = [os.path.join(root, file) for root, dirs, files in os.walk(ltrb_txt_files_dir) for file in files if
                 file.endswith('.txt')]
    for txt_file in tqdm(txt_files, desc="Loading DetectionImages from txt files", unit="file"):
        if class_names is not None:
            detection_image = read_ltrb_txt_file(txt_file, include_gt, conf_th, class_names=class_names, img_file_suffix=img_file_suffix)
        else:
            detection_image = read_ltrb_txt_file(txt_file, include_gt, conf_th, img_file_suffix=img_file_suffix)
        if include_empty or len(detection_image.bounding_boxes) > 0:
            detection_images.append(detection_image)
    return detection_images


def load_depth_predictions_and_gt(pred_txt_files_dir: str, gt_txt_files_dir: str, conf_th=None, class_names=None, img_file_suffix=".jpg") -> list[DetectionImage]:
    '''
    Loads predictions from a depth estimation approach and ground truth for each bbox into the same [DetectionImage].
    Function is used to load data for evaluation (-> predictions_evaluator)
    :param pred_txt_files_dir: Dir with txt files containing the depth predictions from a depth estimation approach
    :param gt_txt_files_dir: Dir with txt files containing the ground truth depth
    :return: List of DetectionImages with both GT depth and predicted depth
    '''
    gt_images = load_detection_images_from_ltrb_txt_files(gt_txt_files_dir, include_gt=True, class_names=class_names, img_file_suffix=img_file_suffix)
    pred_images = load_detection_images_from_ltrb_txt_files(pred_txt_files_dir, include_gt=False, class_names=class_names, img_file_suffix=img_file_suffix)
    for idx, pred_image in enumerate(pred_images):
        gt_image = gt_images[idx]
        if gt_image.filename != pred_image.filename:
            warnings.warn(f"Image mismatch between GT data and prediction data. "
                          f"For filnames GT={gt_image.filename} and Pred={pred_image.filename}."
                          f"Nake sure the datasets are identical!")
            continue

        if len(pred_image.bounding_boxes) != len(gt_image.bounding_boxes):
            raise ValueError("Not all GT values have a matching prediction. Can't load gt and pred into same detection image since ambigous assignment of pred to gt values is error prone.")
        for idx_bbox, bbox_gt in enumerate(gt_image.bounding_boxes):
            bbox_gt.depth_in_mm_PRED = pred_image.bounding_boxes[idx_bbox].depth_in_mm_PRED
    return gt_images



def load_train_images(bbox_conf_th: float =None, include_empty=True) -> list[DetectionImage]:
    '''
    Load all train images as [DetectionImage]. These will contain GT depth.
    :param bbox_conf_th if not None only bboxes with confidence above the threshold will be loaded into the calibration set
    :return:
    '''
    return load_detection_images_from_ltrb_txt_files(LOCAL_PATHS.TRAIN_SET_PATH, include_gt=True, conf_th=bbox_conf_th, include_empty=include_empty)



def load_test_images(include_gt=False, bbox_conf_th: float =None) -> list[DetectionImage]:
    '''
        Load all test images as [DetectionImage]. These will NOT contain GT depth.
        While GT depth is present in the txt files this value is not loaded in the DetectionImages to
        :return:
        '''
    return load_detection_images_from_ltrb_txt_files(LOCAL_PATHS.TEST_SET_PATH, include_gt=include_gt, conf_th=bbox_conf_th)


def load_train_images_kitti(include_gt=True, include_empty=True) -> list[DetectionImage]:
    '''
    Load all KITTI train images as [DetectionImage]. These will contain GT depth.
    '''
    return load_detection_images_from_ltrb_txt_files(LOCAL_PATHS.KITTI_LTRB_TRAIN_PATH, include_gt=include_gt, conf_th=0.0, include_empty=include_empty, class_names=CLASS_NAMES.KITTI_CLASSES, img_file_suffix=".png")


def load_val_images_kitti(include_gt=True, include_empty=True) -> list[DetectionImage]:
    '''
    Load all KITTI val images as [DetectionImage]. These will contain GT depth.
    '''
    return load_detection_images_from_ltrb_txt_files(LOCAL_PATHS.KITTI_LTRB_VAL_PATH, include_gt=include_gt, conf_th=0.0, include_empty=include_empty,  class_names=CLASS_NAMES.KITTI_CLASSES, img_file_suffix=".png")


def load_test_images_kitti(include_gt=True, include_empty=True) -> list[DetectionImage]:
    '''
    Load all KITTI test images as [DetectionImage]. These will contain GT depth.
    '''
    return load_detection_images_from_ltrb_txt_files(LOCAL_PATHS.KITTI_LTRB_TEST_PATH, include_gt=include_gt, conf_th=0.0, include_empty=include_empty, class_names=CLASS_NAMES.KITTI_CLASSES, img_file_suffix=".png")




def save_depth_predictions(pred_detection_images: list[DetectionImage], save_dir: str, incl_conf=True, incl_GT=True, use_yolo_format=False):
    if use_yolo_format:
        write_yolo_results(pred_detection_images, save_dir, include_depth=True, incl_conf=incl_conf, incl_GT=incl_GT)
    else:
        write_ltrb_results(pred_detection_images, save_dir, include_depth=True, incl_conf=incl_conf, incl_GT=incl_GT)