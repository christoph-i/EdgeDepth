import random
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import Tuple, Any, List


from read_write_DetectionImages_txt_files import load_train_images, load_test_images, load_train_images_kitti, load_val_images_kitti

class InternalDataset(Dataset):

    PERCENT_TRAIN = 0.75

    def __init__(self, args, mode: str = 'train') -> None:

        # [height, width, depth] dims of object (or 3D Bbox)
        self.vehicle_prior = [1.60, 1.80, 4.00] # “car” 160 cm, 180 cm and 400 cm -- from DisNet Paper
        self.traffic_sign_prior = [0.5, 0.5, 0.01]

        full_train_val_dataset = load_train_images(include_empty=False)
        random.shuffle(full_train_val_dataset)
        split_index = int(len(full_train_val_dataset) * self.PERCENT_TRAIN)
        if mode == 'train':
            self.detection_images_data = full_train_val_dataset[:split_index]
        elif mode == 'test':
            self.detection_images_data = full_train_val_dataset[split_index:]
        else:
            raise ValueError(f'Dataset mode {mode} not implemented')

    def __getitem__(self, idx: int) -> tuple[Any, Tensor, Tensor]:
        h_norm = []
        w_norm = []
        diagonal_norm = []
        distances = []
        classes = []
        priors = []

        for bbox in self.detection_images_data[idx].bounding_boxes:
            x, y, w, h = bbox.get_dimensions_yolo_rel()
            h_norm.append(h)
            w_norm.append(w)
            distances.append(float(bbox.depth_in_mm_GT))
            classes.append(bbox.class_id)
            if bbox.class_name == "vehicle":
                priors.append(self.vehicle_prior)
            elif bbox.class_name == "traffic_sign":
                priors.append(self.traffic_sign_prior)
            else:
                raise ValueError(f'Bbox class of type {bbox.class_name} not supported in current DisNet dataset.')

        # Calculate the relative diagonal distances
        for h, w in zip(h_norm, w_norm):
            diagonal_norm.append(np.sqrt(h ** 2 + w ** 2))


        x = torch.tensor([h_norm, w_norm, diagonal_norm]).T
        # concat person prior
        x = torch.cat((x, torch.tensor(priors)), dim=1).type(torch.float32)
        # x = torch.cat((x, self.person_prior.repeat(x.shape[0], 1)), dim=1).type(torch.float32)

        distances = torch.tensor(distances)
        classes = torch.tensor(classes)

        return x, distances, classes

    def __len__(self) -> int:
        return len(self.detection_images_data)



class InternalDatasetSingleClass(InternalDataset):

    PERCENT_TRAIN = 0.75

    def __init__(self, args, mode: str = 'train') -> None:
        super().__init__(args, mode)
        self.active_class = args.single_class_active_class

    def __getitem__(self, idx: int) -> tuple[Any, Tensor, Tensor]:
        h_norm = []
        w_norm = []
        diagonal_norm = []
        distances = []
        classes = []

        for bbox in self.detection_images_data[idx].bounding_boxes:
            if bbox.class_name != self.active_class:
                continue
            x, y, w, h = bbox.get_dimensions_yolo_rel()
            h_norm.append(h)
            w_norm.append(w)
            distances.append(float(bbox.depth_in_mm_GT))
            classes.append(bbox.class_id)

        # Calculate the relative diagonal distances
        for h, w in zip(h_norm, w_norm):
            diagonal_norm.append(np.sqrt(h ** 2 + w ** 2))


        x = torch.tensor([h_norm, w_norm, diagonal_norm]).T.type(torch.float32)
        # concat person prior
        # x = torch.cat((x, torch.tensor(priors)), dim=1).type(torch.float32)
        # x = torch.cat((x, self.person_prior.repeat(x.shape[0], 1)), dim=1).type(torch.float32)

        distances = torch.tensor(distances)
        classes = torch.tensor(classes)

        return x, distances, classes




class InternalDatasetSingleClassVehicleBottom(InternalDataset):

    PERCENT_TRAIN = 0.75

    def __init__(self, args, mode: str = 'train') -> None:
        super().__init__(args, mode)
        self.active_class = args.single_class_active_class

    def __getitem__(self, idx: int) -> tuple[Any, Tensor, Tensor]:
        b_norm = []
        distances = []
        classes = []

        for bbox in self.detection_images_data[idx].bounding_boxes:
            if bbox.class_name != self.active_class:
                continue
            _, _, _, b = bbox.get_dimensions_ltrb_rel()
            b_norm.append(b)
            distances.append(float(bbox.depth_in_mm_GT))
            classes.append(bbox.class_id)

        x = torch.tensor([b_norm]).T.type(torch.float32)

        distances = torch.tensor(distances)
        classes = torch.tensor(classes)

        return x, distances, classes



class InternalDatasetClassIds(InternalDataset):

    PERCENT_TRAIN = 0.75

    def __init__(self, args, mode: str = 'train') -> None:
        super().__init__(args, mode)

    def __getitem__(self, idx: int) -> tuple[Any, Tensor, Tensor]:
        h_norm = []
        w_norm = []
        diagonal_norm = []
        distances = []
        classes = []

        for bbox in self.detection_images_data[idx].bounding_boxes:
            x, y, w, h = bbox.get_dimensions_yolo_rel()
            h_norm.append(h)
            w_norm.append(w)
            distances.append(float(bbox.depth_in_mm_GT))
            classes.append(bbox.class_id)

        # Calculate the relative diagonal distances
        for h, w in zip(h_norm, w_norm):
            diagonal_norm.append(np.sqrt(h ** 2 + w ** 2))

        x = torch.tensor([h_norm, w_norm, diagonal_norm, classes]).T.type(torch.float32)

        distances = torch.tensor(distances)
        classes = torch.tensor(classes)

        return x, distances, classes





class InternalDatasetClassIdsInclXY(InternalDataset):

    PERCENT_TRAIN = 0.75

    def __init__(self, args, mode: str = 'train') -> None:
        super().__init__(args, mode)

    def __getitem__(self, idx: int) -> tuple[Any, Tensor, Tensor]:
        h_norm = []
        w_norm = []
        diagonal_norm = []
        x_norm = []
        y_norm = []
        distances = []
        classes = []

        for bbox in self.detection_images_data[idx].bounding_boxes:
            x, y, w, h = bbox.get_dimensions_yolo_rel()
            h_norm.append(h)
            w_norm.append(w)
            x_norm.append(x)
            y_norm.append(y)
            distances.append(float(bbox.depth_in_mm_GT))
            classes.append(bbox.class_id)

        # Calculate the relative diagonal distances
        for h, w in zip(h_norm, w_norm):
            diagonal_norm.append(np.sqrt(h ** 2 + w ** 2))

        x = torch.tensor([h_norm, w_norm, diagonal_norm, x_norm, y_norm, classes]).T.type(torch.float32)

        distances = torch.tensor(distances)
        classes = torch.tensor(classes)

        return x, distances, classes






class MockDataset(Dataset):
    def __init__(self, conf, mode: str = 'train') -> None:
        self.person_prior = torch.tensor([1.75, 0.55, 0.30]).unsqueeze(0)

    def __getitem__(self, idx: int) -> tuple[Any, Tensor, Tensor]:
        # Fill with mock data
        h_norm = [random.uniform(0.1, 0.9) for _ in range(10)]
        w_norm = [random.uniform(0.1, 0.9) for _ in range(10)]
        distances = [h + w for h, w in zip(h_norm, w_norm)]
        d_norm = [h + w for h, w in zip(h_norm, w_norm)]

        x = torch.tensor([h_norm, w_norm, d_norm]).T
        # concat person prior
        x = torch.cat((x, self.person_prior.repeat(x.shape[0], 1)), dim=1).type(torch.float32)

        distances = torch.tensor(distances)
        classes = torch.zeros_like(distances)

        return x, distances, classes

    def __len__(self) -> int:
        return 100
    
    
    

class KittiDataset(Dataset):

    def __init__(self, args, mode: str = 'train') -> None:
        # self.vehicle_prior = torch.tensor([1.75, 0.55, 0.30]).unsqueeze(0)
        # self.traffic_sign_prior = torch.tensor([1.75, 0.55, 0.30]).unsqueeze(0)

        # [height, width, depth] dims of object (or 3D Bbox)
        self.vehicle_prior = [1.60, 1.80, 4.00] # “car” 160 cm, 180 cm and 400 cm -- aus DisNet Paper
        self.pedestrian_prior = [2.0, 0.5, 0.5]
        self.bicycle_prior = [2.0, 0.5, 2.5]
        

        train_dataset = load_train_images_kitti(include_empty=False)
        val_dataset = load_val_images_kitti(include_empty=False)
        
        random.shuffle(train_dataset)
        random.shuffle(val_dataset)

        if mode == 'train':
            self.detection_images_data = train_dataset
        elif mode == 'test':
            self.detection_images_data = val_dataset
        else:
            raise ValueError(f'Dataset mode {mode} not implemented')

    def __getitem__(self, idx: int) -> tuple[Any, Tensor, Tensor]:
        h_norm = []
        w_norm = []
        diagonal_norm = []
        distances = []
        classes = []
        priors = []

        for bbox in self.detection_images_data[idx].bounding_boxes:
            x, y, w, h = bbox.get_dimensions_yolo_rel()
            h_norm.append(h)
            w_norm.append(w)
            distances.append(float(bbox.depth_in_mm_GT))
            classes.append(bbox.class_id)
            if bbox.class_name == "Car":
                priors.append(self.vehicle_prior)
            elif bbox.class_name == "Pedestrian":
                priors.append(self.pedestrian_prior)
            elif bbox.class_name == "Cyclist":
                priors.append(self.bicycle_prior)
            else:
                raise ValueError(f'Bbox class of type {bbox.class_name} not supported in current DisNet dataset.')

        # Calculate the relative diagonal distances
        for h, w in zip(h_norm, w_norm):
            diagonal_norm.append(np.sqrt(h ** 2 + w ** 2))


        x = torch.tensor([h_norm, w_norm, diagonal_norm]).T
        # concat person prior
        x = torch.cat((x, torch.tensor(priors)), dim=1).type(torch.float32)
        # x = torch.cat((x, self.person_prior.repeat(x.shape[0], 1)), dim=1).type(torch.float32)

        distances = torch.tensor(distances)
        classes = torch.tensor(classes)

        return x, distances, classes

    def __len__(self) -> int:
        return len(self.detection_images_data)
    
    
    
class KittiDatasetClassIds(KittiDataset):

    PERCENT_TRAIN = 0.75

    def __init__(self, args, mode: str = 'train') -> None:
        super().__init__(args, mode)

    def __getitem__(self, idx: int) -> tuple[Any, Tensor, Tensor]:
        h_norm = []
        w_norm = []
        diagonal_norm = []
        distances = []
        classes = []

        for bbox in self.detection_images_data[idx].bounding_boxes:
            x, y, w, h = bbox.get_dimensions_yolo_rel()
            h_norm.append(h)
            w_norm.append(w)
            distances.append(float(bbox.depth_in_mm_GT))
            classes.append(bbox.class_id)

        # Calculate the relative diagonal distances
        for h, w in zip(h_norm, w_norm):
            diagonal_norm.append(np.sqrt(h ** 2 + w ** 2))

        x = torch.tensor([h_norm, w_norm, diagonal_norm, classes]).T.type(torch.float32)

        distances = torch.tensor(distances)
        classes = torch.tensor(classes)

        return x, distances, classes
    
    

class KittiDatasetClassIdsInclXY(KittiDataset):

    def __init__(self, args, mode: str = 'train') -> None:
        super().__init__(args, mode)

    def __getitem__(self, idx: int) -> tuple[Any, Tensor, Tensor]:
        h_norm = []
        w_norm = []
        diagonal_norm = []
        x_norm = []
        y_norm = []
        distances = []
        classes = []

        for bbox in self.detection_images_data[idx].bounding_boxes:
            x, y, w, h = bbox.get_dimensions_yolo_rel()
            h_norm.append(h)
            w_norm.append(w)
            x_norm.append(x)
            y_norm.append(y)
            distances.append(float(bbox.depth_in_mm_GT))
            classes.append(bbox.class_id)

        # Calculate the relative diagonal distances
        for h, w in zip(h_norm, w_norm):
            diagonal_norm.append(np.sqrt(h ** 2 + w ** 2))

        x = torch.tensor([h_norm, w_norm, diagonal_norm, x_norm, y_norm, classes]).T.type(torch.float32)

        distances = torch.tensor(distances)
        classes = torch.tensor(classes)

        return x, distances, classes