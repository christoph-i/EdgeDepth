import os
import warnings
import random
from typing import List

from tqdm import tqdm
import cv2
from abc import ABC

from detection_image import DetectionImage

import CLASS_NAMES

class InferenceRunner(ABC):

    def __init__(self,
                 model_file_path: str,
                 model_name: str,
                 input_shape=(384, 384)
                 ):
        self.model_file_path = model_file_path
        self.model_name = model_name
        self.input_shape = input_shape
        self.class_names = CLASS_NAMES.KITTI_CLASSES
        self.load_model()

    def load_model(self):
        pass

    def run_single_inference(self, image) -> DetectionImage:
        pass

    def run_batch_inference(self, images) -> List[DetectionImage]:
        pass

    def run_inference_for_dir_of_images(self, images_dir: str, max_imgs=None, shuffle_images=False, image_file_types=('.png', '.jpg', '.jpeg', '.gif', '.bmp')) -> list[DetectionImage]:
        if not os.path.exists(images_dir):
            warnings.warn(f"The directory '{images_dir}' does not exist.")
            return

        all_detection_images = []
        image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
        if shuffle_images:
            random.seed(42)
            random.shuffle(image_files)
        for idx, image_file in enumerate(tqdm(image_files)):
            if max_imgs is not None and idx > max_imgs:
                break
            if image_file.lower().endswith(image_file_types):
                image_path = os.path.join(images_dir, image_file)
                img = cv2.imread(image_path)
                if img is not None:
                    detection_image = self.run_single_inference(img)
                    detection_image.filename = image_file
                    all_detection_images.append(detection_image)
        return all_detection_images


