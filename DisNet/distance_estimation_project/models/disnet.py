import torch
import torch.nn as nn
from DisNet.distance_estimation_project.trainers.base_trainer import Trainer
from DisNet.distance_estimation_project.dataset.naive_features import DisNetDataset, DisNetDatasetCustom
from DisNet.distance_estimation_project.models.base_model import BaseLifter
from DisNet.distance_estimation_project.trainers.trainer_mlp import TrainerMLP
from torch.utils.data import Dataset


class DisNet(BaseLifter):
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

    def get_trainer(self) -> Trainer:
        return TrainerMLP

    def get_dataset(self, args) -> Dataset:
        # return DisNetDataset
        return DisNetDatasetCustom

    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

    def save_full(self, path: str):
        torch.save(self, path[:-4] + "_full.pth")
