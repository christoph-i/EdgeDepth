from copy import deepcopy

from tqdm import tqdm

from detection_image import DetectionImage


class GtPredSet:

    def __init__(self,
                 detection_images: list[DetectionImage],
                 bbox_conf_th,
                 remove_gt_zero_values=True,
                 exclude_pred_zero_values=False,
                 pred_depth_threshold=None):
        if len(detection_images) == 0:
            raise ValueError("Can't create GTPredSet from empty detection images list")
        self.data_dict = self.transform_detection_images_to_dict(detection_images, bbox_conf_th, remove_gt_zero_values, exclude_pred_zero_values, pred_depth_threshold)
        self._check_data_integrity()

    def transform_detection_images_to_dict(self, detection_images: list[DetectionImage], remove_gt_zero_values=True, exclude_pred_zero_values=False,
                                           bbox_conf_th=0.5, pred_depth_threshold=None) -> dict:
        gt_pred_dict = {}
        for detection_image in tqdm(detection_images):
            for bbox in detection_image.bounding_boxes:
                if remove_gt_zero_values and int(bbox.depth_in_mm_GT) == 0:
                    continue
                if exclude_pred_zero_values and int(bbox.depth_in_mm_PRED) == 0:
                    continue
                if bbox_conf_th and bbox_conf_th < bbox.confidence:
                    continue
                if pred_depth_threshold and pred_depth_threshold < int(bbox.depth_in_mm_PRED):
                    continue
                if bbox.class_name not in gt_pred_dict:
                    gt_pred_dict[bbox.class_name] = []
                gt_pred_dict[bbox.class_name].append([int(bbox.depth_in_mm_GT), int(bbox.depth_in_mm_PRED)])
        return gt_pred_dict

    def _check_data_integrity(self):
        for object_class, measurements in self.data_dict.items():
            for gt, pred in measurements:
                if gt is None or pred is None:
                    raise ValueError(f"Missing ground truth or prediction for class {object_class}")

    def get_gt_pred_lists_for_class(self, class_name: str, gt_bin_min=None, gt_bin_max=None) -> (list, list):
        if (gt_bin_min is None) != (gt_bin_max is None):  # True if exactly one is None
            raise Exception("Passed only upper or lower bin limit but not both. Can't compute binned metric.")
        gt = []
        pred = []
        for gt_pred_pair in self.data_dict[class_name]:
            if (gt_bin_min is None and gt_bin_max is None) or (gt_pred_pair[0] > gt_bin_min and gt_pred_pair[0] <= gt_bin_max):
                gt.append(gt_pred_pair[0])
                pred.append(gt_pred_pair[1])
        return gt, pred

    def get_gt_pred_lists(self, specific_class: str = None, gt_bin_min=None, gt_bin_max=None) -> (list, list):
        if specific_class:
            return self.get_gt_pred_lists_for_class(specific_class, gt_bin_min, gt_bin_max)

        gt = []
        pred = []
        for class_name in self.get_class_names():
            gt_class, pred_class = self.get_gt_pred_lists_for_class(class_name, gt_bin_min, gt_bin_max)
            gt.extend(gt_class)
            pred.extend(pred_class)
        return gt, pred


    def get_class_names(self) -> list[str]:
        return self.data_dict.keys()