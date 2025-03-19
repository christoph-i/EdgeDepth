import torch
import torch.nn as nn

from torch.utils.data import Dataset

from base_trainer import Trainer
from disnet_trainer import DisNetTrainer
from dataset import MockDataset, InternalDataset, InternalDatasetSingleClass, InternalDatasetClassIds, InternalDatasetSingleClassVehicleBottom, InternalDatasetClassIdsInclXY, KittiDataset, KittiDatasetClassIds, KittiDatasetClassIdsInclXY

class DisNet(nn.Module):
    TEMPORAL = False

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.distance_estimator = nn.Sequential(
            nn.Linear(6, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 1), )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.distance_estimator(x).squeeze(-1)

    def get_loss_fun(self) -> nn.Module:
        return nn.L1Loss()

    def get_trainer(self) -> DisNetTrainer:
        return DisNetTrainer

    def get_dataset(self, args) -> Dataset:
        if args.dataset == 'mock':
            return MockDataset
        elif args.dataset == 'kitti':
            return KittiDataset
        else:
            return InternalDataset
    
    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

    def save_full(self, path: str):
        torch.save(self, path[:-4] + "_full.pth")


    def load_w(self, path: str, strict: bool = True) -> None:
        state_dict = torch.load(path, map_location='cpu')

        self.load_state_dict(state_dict['weights'], strict=strict)

        #self.ds_stats = state_dict['ds_stats']

    def save_w(self, path: str, cnf) -> None:
        state_dict = {
            'weights': self.state_dict(),
            #'ds_stats': self.ds_stats,
            'cnf': cnf
        }
        torch.save(state_dict, path)



class DisNetClassIds(DisNet):
    TEMPORAL = False

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.distance_estimator = nn.Sequential(
            nn.Linear(4, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 1), )

    def get_dataset(self, args) -> Dataset:
        if args.dataset == 'kitti':
            return KittiDatasetClassIds
        else:
            return InternalDatasetClassIds




class DisNetClassIdsInclXY(DisNet):
    TEMPORAL = False

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.distance_estimator = nn.Sequential(
            nn.Linear(6, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 1), )

    def get_dataset(self, args) -> Dataset:
        if args.dataset == 'kitti':
            return KittiDatasetClassIdsInclXY
        else:
            return InternalDatasetClassIdsInclXY




class DisNetSingleClass(DisNet):
    TEMPORAL = False

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.distance_estimator = nn.Sequential(
            nn.Linear(3, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 1), )

    def get_dataset(self, args) -> Dataset:
        return InternalDatasetSingleClass


# Designed for classes like car where the most important information is given by the position of the bbox bottom
# So only one input - the bottom position - is used
class DisNetSingleClassVehicleBottom(DisNet):
    TEMPORAL = False

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.distance_estimator = nn.Sequential(
            nn.Linear(1, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 100),
            nn.SELU(inplace=True),
            nn.Linear(100, 1), )

    def get_dataset(self, args) -> Dataset:
        return InternalDatasetSingleClassVehicleBottom


