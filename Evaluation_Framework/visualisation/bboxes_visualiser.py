import copy

import numpy as np
import cv2

from bbox import BoundingBox

_COLORS = np.array(
            [
                0.301, 0.745, 0.933,
                0.635, 0.078, 0.184,
                0.000, 0.447, 0.741,
                0.850, 0.325, 0.098,
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


def vis_detections(img: np.ndarray, bboxes: list[BoundingBox], conf_th=0.7, alpha_area_box=False, alpha = 0.5, show_conf=False, show_depth=True, scale=0.50):
    for bbox in bboxes:
        cls_id = bbox.get_class_id()
        score = bbox.get_confidence()
        if score < conf_th:
            continue
        img_h, img_w, _ = img.shape
        left_rel, top_rel, right_rel, bottom_rel = bbox.get_dimensions_ltrb_rel()
        left_abs = int(left_rel * img_w)
        top_abs = int(top_rel * img_h)
        right_abs = int(right_rel * img_w)
        bottom_abs = int(bottom_rel * img_h)

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        if show_conf:
            text = '{}:{:.1f}%'.format(bbox.get_class_name(), score * 100)
        else:
            text = str(bbox.get_class_name())
        if show_depth:
            if bbox.depth_in_mm_GT is not None:
                text = text + f"|GT: {bbox.depth_in_mm_GT}"
            if bbox.depth_in_mm_PRED is not None:
                text = text + f"|Pred: {bbox.depth_in_mm_PRED}"
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, scale, 1)[0]
        if alpha_area_box:
            # rectangle_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            # rectangle_img[:, :, 2] = int(255 * alpha)
            # cv2.rectangle(rectangle_img, (left_abs, top_abs), (right_abs, bottom_abs), color, thickness=cv2.FILLED)
            # img = cv2.addWeighted(img, 1, rectangle_img, 1, 0)

            rectangle_mask = np.zeros_like(img)
            cv2.rectangle(rectangle_mask, (left_abs, top_abs), (right_abs, bottom_abs), color,
                          thickness=cv2.FILLED)

            # Blend the rectangle with the original image using the mask
            img = cv2.addWeighted(img, 1, rectangle_mask, alpha, 0)
        else:
            cv2.rectangle(img, (left_abs, top_abs), (right_abs, bottom_abs), color, 3)


        txt_bk_color = (_COLORS[cls_id] * 255 * scale).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (left_abs, top_abs + 1),
            (left_abs + txt_size[0] + 1, top_abs + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (left_abs, top_abs + txt_size[1]), font, scale, txt_color, thickness=1)

    return img

def top_left_corner_annotation_text(image, text):
    image_with_text = image.copy()

    # Define the font and text color
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)  # White color in BGR format

    # Get the size of the text
    text_size = cv2.getTextSize(text, font, 1, 2)[0]

    # Create a black background image for the text
    background = np.zeros((text_size[1] + 10, text_size[0] + 10, 3), dtype=np.uint8)

    # Put the text on the black background
    cv2.putText(background, text, (5, text_size[1] + 5), font, 1, text_color, 2, cv2.LINE_AA)

    # Overlay the text background on the original image
    image_with_text[0:text_size[1] + 10, 0:text_size[0] + 10] = background

    return image_with_text
