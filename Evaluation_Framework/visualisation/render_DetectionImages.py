import copy
import random
import numpy as np
from tqdm import tqdm
import os
import cv2

from detection_image import DetectionImage
from Evaluation_Framework.visualisation import bboxes_visualiser


def render_and_write_detection_images(detection_images: list[DetectionImage], output_dir: str, raw_imgs_dir: str, max_imgs=None, conf_th=0.7):
    for detection_image in tqdm(detection_images[:max_imgs]):
        raw_image = cv2.imread(os.path.join(raw_imgs_dir, detection_image.filename))
        rendered_image = bboxes_visualiser.vis_detections(raw_image, detection_image.bounding_boxes, conf_th=conf_th, show_conf=True, show_depth=True)
        out_filepath = os.path.join(output_dir, detection_image.filename)
        cv2.imwrite(out_filepath, rendered_image)


def write_random_inference_results(detection_images: list[DetectionImage], output_dir: str, raw_imgs_dir: str, max_imgs=None, conf_th=0.7, random_seed=42):
    print("Writing random detection images until maximum is reached.")
    detection_images_shuffled = copy.deepcopy(detection_images)
    random.seed(random_seed)
    random.shuffle(detection_images_shuffled)
    render_and_write_detection_images(detection_images_shuffled, output_dir, raw_imgs_dir, max_imgs, conf_th)



def write_most_detections_inference_results(detection_images: list[DetectionImage], output_dir: str, raw_imgs_dir: str, max_imgs=None, conf_th=0.7):
    print("Writing detection images with most bboxes until maximum is reached.")
    detection_images_sorted = copy.deepcopy(detection_images)
    detection_images_sorted.sort(key=lambda x: len(x.bounding_boxes), reverse=True)
    render_and_write_detection_images(detection_images_sorted, output_dir, raw_imgs_dir, max_imgs, conf_th)

def add_text_to_image_top_right(image: np.ndarray, text: str) -> np.ndarray:
    blank_image = np.zeros_like(image)
    blank_image[0:image.shape[0], 0:image.shape[1], :] = image

    # Define the font and other properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)  # White color

    # Calculate the size ans position of the text
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_position = (blank_image.shape[1] - text_width - 10, text_height + 10)

    cv2.putText(blank_image, text, text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    return blank_image


def show_single_detection_images(detection_images: list[DetectionImage], raw_imgs_dir: str, image_display_size=(1920,1080), conf_th=0.7, show_image_filename=False, show_conf=True, show_depth=True):
    '''
    Opens a popup window displaying the provided detection images with all available data (boxes, depth etc)
    :return:
    '''
    for detection_image in detection_images:
        raw_image = cv2.imread(os.path.join(raw_imgs_dir, detection_image.filename))
        rendered_image = bboxes_visualiser.vis_detections(raw_image, detection_image.bounding_boxes, conf_th=conf_th, show_conf=show_conf, show_depth=show_depth)
        if show_image_filename and detection_image.filename is not None:
            rendered_image = add_text_to_image_top_right(rendered_image, detection_image.filename)
        rendered_image = cv2.resize(rendered_image, image_display_size, interpolation=cv2.INTER_AREA)
        cv2.imshow('Image', rendered_image)
        key = cv2.waitKey(0)
        if key == 27:  # 27 is the ASCII code for the Esc key
            break
    cv2.destroyAllWindows()