import random

import torch
from tqdm import tqdm
import numpy as np

from Evaluation_Framework.visualisation.render_DetectionImages import show_single_detection_images
from DisNet.disnet import DisNet, DisNetClassIds, DisNetSingleClass, DisNetSingleClassVehicleBottom, DisNetClassIdsInclXY
from read_write_DetectionImages_txt_files import load_train_images, load_test_images, save_depth_predictions, load_test_images_kitti
import LOCAL_PATHS
from detection_image import DetectionImage
from bbox import BoundingBox

from train_custom_dis_net import Args


OUTPUT_DIR = r"/.../..."

MODEL_FILEPATH = r"/.../.../best.pth"

# MODEL_FILEPATH_SIGNS = r"/.../.../best.pth"
# MODEL_FILEPATH_VEHICLES = r"/.../.../best.pth"

# MODEL_FILEPATH_VEHICLES_BOTTOM = r"/.../.../best.pth"



DEVICE = "cpu"

vehicle_prior = [1.60, 1.80, 4.00]  # “car” 160 cm, 180 cm and 400 cm -- from DisNet Paper
traffic_sign_prior = [0.5, 0.5, 0.01]
pedestrian_prior = [2.0, 0.5, 0.5]
bicycle_prior = [2.0, 0.5, 2.5]



def predict_depth_with_priors(bbox: BoundingBox, dis_net_model: DisNet, kitti_mode=False) -> float:
    x, y, w, h = bbox.get_dimensions_yolo_rel()
    diagonal = np.sqrt(h ** 2 + w ** 2)
    x = torch.tensor([h, w, diagonal])
    if kitti_mode:
        if bbox.class_name == "Car":
            x = torch.cat((x, torch.tensor(vehicle_prior)), dim=0).type(torch.float32)
        elif bbox.class_name == "Pedestrian":
            x = torch.cat((x, torch.tensor(pedestrian_prior)), dim=0).type(torch.float32)
        elif bbox.class_name == "Cyclist":
            x = torch.cat((x, torch.tensor(bicycle_prior)), dim=0).type(torch.float32)
        else:
            raise ValueError(f"Class name {bbox.class_name} not supportet.")
    else:
        if bbox.class_name == "vehicle":
            x = torch.cat((x, torch.tensor(vehicle_prior)), dim=0).type(torch.float32)
        elif bbox.class_name == "traffic_sign":
            x = torch.cat((x, torch.tensor(vehicle_prior)), dim=0).type(torch.float32)
        else:
            raise ValueError(f"Class name {bbox.class_name} not supportet.")

    with torch.no_grad():
        dist_pred = dis_net_model(x)

    return float(dist_pred)


def predict_depth_with_class_id(bbox: BoundingBox, dis_net_model: DisNet) -> float:
    x, y, w, h = bbox.get_dimensions_yolo_rel()
    diagonal = np.sqrt(h ** 2 + w ** 2)
    #TODO dynamic and elegant solution
    if bbox.class_name not in ["vehicle", "traffic_sign", "Car", "Pedestrian", "Cyclist"]:
        raise ValueError(f"Class name {bbox.class_name} not expected.")
    x = torch.tensor([h, w, diagonal, bbox.class_id]).type(torch.float32)

    with torch.no_grad():
        dist_pred = dis_net_model(x)

    return float(dist_pred)


def predict_depth_with_class_id_incl_xy_pos(bbox: BoundingBox, dis_net_model: DisNet) -> float:
    # input vector = [h_norm, w_norm, diagonal_norm, x_norm, y_norm, classes]
    x, y, w, h = bbox.get_dimensions_yolo_rel()
    diagonal = np.sqrt(h ** 2 + w ** 2)
    if bbox.class_name not in ["vehicle", "traffic_sign", "Car", "Pedestrian", "Cyclist"]:
        raise ValueError(f"Class name {bbox.class_name} not expected.")
    x = torch.tensor([h, w, diagonal, x, y, bbox.class_id]).type(torch.float32)
   
    with torch.no_grad():
        dist_pred = dis_net_model(x)

    return float(dist_pred)


def predict_depth_model_per_class(bbox: BoundingBox, dis_net_model: DisNet) -> float:
    x, y, w, h = bbox.get_dimensions_yolo_rel()
    diagonal = np.sqrt(h ** 2 + w ** 2)
    x = torch.tensor([h, w, diagonal]).type(torch.float32)
    with torch.no_grad():
        dist_pred = dis_net_model(x)
    return float(dist_pred)


def predict_depth_model_per_class_bottom_only(bbox: BoundingBox, dis_net_model: DisNet) -> float:
    _, _, _, b = bbox.get_dimensions_ltrb_rel()
    x = torch.tensor([b]).type(torch.float32)
    with torch.no_grad():
        dist_pred = dis_net_model(x)
    return float(dist_pred)



def calculate_depth_predictions_sigle_model_shape_priors(preds_output_dir: str, kitti_mode=False):
    model_args = Args()
    model = DisNet(model_args)
    model.load_w(MODEL_FILEPATH)
    model.eval()

    # test_images = load_test_images(include_gt=False)
    test_images = load_test_images_kitti(include_gt=True)
    
    # add predicted absolute depth to each detection image
    for test_image in tqdm(test_images, desc="Calculation Pred depth with DisNet", unit="detection images"):
        for bounding_box in test_image.bounding_boxes:
            if kitti_mode:
                bounding_box.depth_in_mm_PRED = int(predict_depth_with_priors(bounding_box, model, kitti_mode=kitti_mode) * 1000)
                bounding_box.depth_in_mm_GT = int(float(bounding_box.depth_in_mm_GT) * 1000)
            else:
                bounding_box.depth_in_mm_PRED = int(predict_depth_with_priors(bounding_box, model, kitti_mode=kitti_mode))

    # random.seed(42)
    # random.shuffle(test_images)
    # show_single_detection_images(test_images, LOCAL_PATHS., image_display_size=(1728, 972), conf_th=0.7)

    # worst_off = []
    # for test_image in tqdm(test_images):
    #     for bounding_box in test_image.bounding_boxes:
    #         if int(bounding_box.depth_in_mm_GT) != 0 and abs(bounding_box.depth_in_mm_PRED - int(bounding_box.depth_in_mm_GT)) > 15000:
    #             worst_off.append(test_image)
    #
    # show_single_detection_images(worst_off, LOCAL_PATHS., image_display_size=(1728,972), conf_th=0.5)

    save_depth_predictions(test_images, preds_output_dir)


def calculate_depth_predictions_sigle_model_with_class_ids(preds_output_dir: str, kitti_mode=False):
    model_args = Args()
    model = DisNetClassIds(model_args)
    model.load_w(MODEL_FILEPATH)
    model.eval()

    # test_images = load_test_images(include_gt=False)
    test_images = load_test_images_kitti(include_gt=True)

    # add predicted absolute depth to each detection image
    for test_image in tqdm(test_images, desc="Calculation Pred depth with DisNet", unit="detection images"):
        for bounding_box in test_image.bounding_boxes:
            if kitti_mode:
                bounding_box.depth_in_mm_PRED = int(predict_depth_with_class_id(bounding_box, model) * 1000)
                bounding_box.depth_in_mm_GT = int(float(bounding_box.depth_in_mm_GT) * 1000)
            else:
                bounding_box.depth_in_mm_PRED = int(predict_depth_with_class_id(bounding_box, model))

    save_depth_predictions(test_images, preds_output_dir)



def calculate_depth_predictions_sigle_model_with_class_ids_incl_xy_pos(preds_output_dir: str, kitti_mode=False):
    model_args = Args()
    model = DisNetClassIdsInclXY(model_args)
    model.load_w(MODEL_FILEPATH)
    model.eval()

    # test_images = load_test_images(include_gt=False)
    test_images = load_test_images_kitti(include_gt=True)
    
    # add predicted absolute depth to each detection image
    for test_image in tqdm(test_images, desc="Calculation Pred depth with DisNet", unit="detection images"):
        for bounding_box in test_image.bounding_boxes:
            if kitti_mode:
                bounding_box.depth_in_mm_PRED = int(predict_depth_with_class_id_incl_xy_pos(bounding_box, model) * 1000)
                bounding_box.depth_in_mm_GT = int(float(bounding_box.depth_in_mm_GT) * 1000)
            else:
                bounding_box.depth_in_mm_PRED = int(predict_depth_with_class_id_incl_xy_pos(bounding_box, model))

    save_depth_predictions(test_images, preds_output_dir)



def calculate_depth_predictions_model_per_class(preds_output_dir: str):
    model_args = Args()
    model_signs = DisNetSingleClass(model_args)
    model_signs.load_w(MODEL_FILEPATH_SIGNS)
    model_signs.eval()

    model_vehicles = DisNetSingleClass(model_args)
    model_vehicles.load_w(MODEL_FILEPATH_VEHICLES)
    model_vehicles.eval()

    test_images = load_test_images(include_gt=False)

    # add predicted absolute depth to each detection image
    for test_image in tqdm(test_images, desc="Calculation Pred depth with DisNet", unit="detection images"):
        for bounding_box in test_image.bounding_boxes:
            if bounding_box.class_name == "vehicle":
                bounding_box.depth_in_mm_PRED = int(predict_depth_model_per_class(bounding_box, model_vehicles))
            elif bounding_box.class_name == "traffic_sign":
                bounding_box.depth_in_mm_PRED = int(predict_depth_model_per_class(bounding_box, model_signs))
            else:
                raise ValueError(f"Class name {bounding_box.class_name} not supportet!")

    save_depth_predictions(test_images, preds_output_dir)




def calculate_depth_predictions_model_per_class_vehicles_bottom_only(preds_output_dir: str):
    model_args = Args()
    model_signs = DisNetSingleClass(model_args)
    model_signs.load_w(MODEL_FILEPATH_SIGNS)
    model_signs.eval()

    model_vehicles = DisNetSingleClassVehicleBottom(model_args)
    model_vehicles.load_w(MODEL_FILEPATH_VEHICLES_BOTTOM)
    model_vehicles.eval()

    test_images = load_test_images(include_gt=False)

    # add predicted absolute depth to each detection image
    for test_image in tqdm(test_images, desc="Calculation Pred depth with DisNet", unit="detection images"):
        for bounding_box in test_image.bounding_boxes:
            if bounding_box.class_name == "vehicle":
                bounding_box.depth_in_mm_PRED = int(predict_depth_model_per_class_bottom_only(bounding_box, model_vehicles))
            elif bounding_box.class_name == "traffic_sign":
                bounding_box.depth_in_mm_PRED = int(predict_depth_model_per_class(bounding_box, model_signs))
            else:
                raise ValueError(f"Class name {bounding_box.class_name} not supportet!")

    save_depth_predictions(test_images, preds_output_dir)




calculate_depth_predictions_sigle_model_shape_priors(OUTPUT_DIR, kitti_mode=True)
# calculate_depth_predictions_sigle_model_with_class_ids(OUTPUT_DIR, kitti_mode=True)
# calculate_depth_predictions_model_per_class(OUTPUT_DIR)
# calculate_depth_predictions_model_per_class_vehicles_bottom_only(OUTPUT_DIR)
# calculate_depth_predictions_sigle_model_with_class_ids_incl_xy_pos(OUTPUT_DIR, kitti_mode=True)

