import os
import warnings
from pathlib import Path
from typing import List
from matplotlib import cm
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

from bbox import BoundingBox
from detection_image import DetectionImage#
import CLASS_NAMES

CLASS_NAMES = CLASS_NAMES.KITTI_CLASSES

MATCHED_YOLO_TXT_DIR = r"..."
RAW_IMGS_DIR = r"..."
OUTPUT_DIR = r"..."

# th for number of bboxes per pixel to include it in visualisation
validity_threshold = 3

# either "abs" or "abs-rel"
error_type = "abs"

def read_matched_yolo_depth_file(filepath: str) -> DetectionImage:
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()
            detection_image = DetectionImage([])
            for line in lines:
                parts = line.strip().split(' ')
                if not len(parts) == 8:
                    raise ValueError(
                        f"Invalid annotation format in the txt file: {filepath}. Annotation will be skipped.")
                class_id, conf, x, y, w, h, d_gt, d_pred = map(float, parts)
                class_id = int(class_id)
                l = x - w / 2
                t = y - h / 2
                r = x + w / 2
                b = y + h / 2
                bounding_box = BoundingBox(l, t, r, b, conf, class_id, CLASS_NAMES[class_id])
                bounding_box.depth_in_mm_GT = d_gt
                bounding_box.depth_in_mm_PRED = d_pred
                detection_image.bounding_boxes.append(bounding_box)

            detection_image.filename = Path(filepath).stem + ".jpg"
            return detection_image

    except FileNotFoundError:
        print(f"Annotation file '{filepath}' not found.")


def load_matched_detection_images(txt_files_dir: str, raw_imgs_dir: str) -> List[DetectionImage]:
    detection_images = []
    txt_files = [os.path.join(root, file) for root, dirs, files in os.walk(txt_files_dir) for file in files if
                 file.endswith('.txt')]
    for txt_file in tqdm(txt_files, desc="Loading DetectionImages from txt files", unit="file"):
        detection_image = read_matched_yolo_depth_file(txt_file)
        detection_image.image_dir = raw_imgs_dir
        detection_images.append(detection_image)
    return detection_images


def generate_heatmaps(detection_images: List[DetectionImage], save_dir: str, separate_by_class=True, validity_threshold=0) -> None:
    """
    Generate heatmap style visualisations based on the error per pixel :param detection_images: :param save_dir:
    :param separate_by_class: :param validity_threshold: Number of objects that need to be at a pixel to consider it
    valid - if less objects are present at the pixel it will just be displayed in the background color :return:
    """
    if not detection_images:
        print("No detection images provided.")
        return

    # 1) Determine the shape of the first image, and check that all images share this shape
    first_img = detection_images[0].get_image()
    if first_img is None:
        raise ValueError("Failed to load the first image.")
    ref_height, ref_width = first_img.shape[:2]

    # 3) Dictionary: class_id -> (sum_error_array, count_array)
    classwise_arrays = {}

    # 4) Process each DetectionImage and update class-wise sums/counts
    for det_img in tqdm(detection_images, unit='images', desc="Generating heatmaps"):
        img = det_img.get_image()
        # Verify image dimension
        if img.shape[0] != ref_height or img.shape[1] != ref_width:
            raise ValueError("Not all images have the same dimensions.")

        for bbox in det_img.bounding_boxes:
            # Skip bounding boxes missing GT or PRED info
            if bbox.depth_in_mm_GT is None or bbox.depth_in_mm_PRED is None:
                warnings.warn("Object does not contain GT and/or PRED value - object will be ignored.")
                continue

            class_id = bbox.class_id
            # Allocate arrays for this class if not yet present
            if class_id not in classwise_arrays:
                sum_arr = np.zeros((ref_height, ref_width), dtype=np.float64)
                count_arr = np.zeros((ref_height, ref_width), dtype=np.float64)
                classwise_arrays[class_id] = (sum_arr, count_arr)

            sum_error_arr, count_arr = classwise_arrays[class_id]

            # Convert relative coords to absolute pixel coords
            left_abs, top_abs, right_abs, bottom_abs = bbox.get_dimensions_ltrb_abs(ref_width, ref_height)

            # Clip coordinates to valid range
            left_abs = max(0, min(ref_width, left_abs))
            right_abs = max(0, min(ref_width, right_abs))
            top_abs = max(0, min(ref_height, top_abs))
            bottom_abs = max(0, min(ref_height, bottom_abs))

            # Compute absolute relative error
            if bbox.depth_in_mm_GT != 0:
                if error_type == "abs-rel":
                    error_value = abs((bbox.depth_in_mm_PRED - bbox.depth_in_mm_GT) / bbox.depth_in_mm_GT)
                elif error_type == "abs":
                    error_value = abs(bbox.depth_in_mm_PRED - bbox.depth_in_mm_GT)
                else:
                    raise ValueError(f"Error type {error_type} is not recognized with a valid implementation.")
            else:
                # Ground truth is zero, can't compute relative error; skip
                warnings.warn("0 value in GT found, this is assumed to be an invalid value - object will be ignored.")
                continue

            # Update the sums and counts for the pixels within the bounding box
            sum_error_arr[top_abs:bottom_abs, left_abs:right_abs] += error_value
            count_arr[top_abs:bottom_abs, left_abs:right_abs] += 1.0

    # 5) Combine class-wise arrays into a single (joined) sum and count
    joined_sum_arr = np.zeros((ref_height, ref_width), dtype=np.float64)
    joined_count_arr = np.zeros((ref_height, ref_width), dtype=np.float64)
    for _, (sum_arr, cnt_arr) in classwise_arrays.items():
        joined_sum_arr += sum_arr
        joined_count_arr += cnt_arr

    # 6) Create output directory if necessary
    os.makedirs(save_dir, exist_ok=True)

    # 7) Define a helper function to compute and save a heatmap from sum_arr & count_arr
    def save_heatmap_arrays(sum_array, count_array, title_str, base_filename):
        """
        sum_array, count_array: shape (ref_height, ref_width)
        Plots a seaborn heatmap (PNG) and also saves a color-coded PNG (full resolution).
        """
        # Compute average and handle division by zero
        dive_zero_mask = (count_array > 0)
        threshold_mask = (count_array >= validity_threshold)
        valid_mask = np.logical_and(dive_zero_mask, threshold_mask)
        avg_error_arr = np.zeros_like(sum_array, dtype=np.float64)
        avg_error_arr[valid_mask] = sum_array[valid_mask] / count_array[valid_mask]

        # (A) Seaborn-based heatmap (downsampled to figure size):
        plt.figure(figsize=(18, 8))
        sns.heatmap(avg_error_arr, cmap="viridis", robust=True)
        plt.title(title_str)
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.savefig(os.path.join(save_dir, base_filename + ".png"), dpi=300)
        plt.close()

        # (B) Full-resolution color-coded PNG via the magma colormap
        #     (black for low values, red/yellow for high values) [[3],[4]].
        # Normalize array to [0..1] for colormap
        min_val = avg_error_arr[valid_mask].min() if np.any(valid_mask) else 0.0
        max_val = avg_error_arr[valid_mask].max() if np.any(valid_mask) else 0.0
        if np.isclose(min_val, max_val):
            # All zero or effectively constant -> fill with zeros
            norm_arr = np.zeros_like(avg_error_arr, dtype=np.float32)
        else:
            norm_arr = (avg_error_arr - min_val) / (max_val - min_val)
            norm_arr[~valid_mask] = 0.0  # set invalid region to 0.0 in the color scale

        magma_cmap = cm.get_cmap("magma")  # black at low, red/yellow at high [[3]]
        heatmap_rgba = magma_cmap(norm_arr)  # shape (H, W, 4)
        # Convert RGBA -> RGB (drop alpha), scale to 0..255
        heatmap_rgb = (heatmap_rgba[..., :3] * 255).astype(np.uint8)
        # If desired, we could invert to BGR for OpenCV, but we can save in RGB directly with cv2
        # cv2 defaults to BGR, so let's convert to BGR to match typical PNG output from OpenCV:
        heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)

        color_image_path = os.path.join(save_dir, base_filename + "_color.png")
        cv2.imwrite(color_image_path, heatmap_bgr)

    # 8) Always produce the joined (global) heatmap
    save_heatmap_arrays(
        joined_sum_arr,
        joined_count_arr,
        "Average Absolute Relative Error (All Classes)",
        "heatmap_all_classes"
    )

    # 9) Optionally produce separate heatmaps per class
    if separate_by_class:
        for class_id, (sum_arr, cnt_arr) in classwise_arrays.items():
            filename_base = f"heatmap_class_{class_id}"
            title_str = f"Average Absolute Relative Error - Class {class_id}"
            save_heatmap_arrays(sum_arr, cnt_arr, title_str, filename_base)

    print("Heatmap generation complete.")




if __name__ == "__main__":
    det_imgs = load_matched_detection_images(MATCHED_YOLO_TXT_DIR, RAW_IMGS_DIR)
    generate_heatmaps(det_imgs, OUTPUT_DIR, separate_by_class=True, validity_threshold=validity_threshold)
