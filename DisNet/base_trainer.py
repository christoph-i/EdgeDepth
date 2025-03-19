import argparse
import os
from typing import Tuple
from pathlib import Path
import numpy as np

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

from distance_estimation_project.utils.utils_scripts import fill_missing_args, init_wandb


class Trainer:
    def __init__(self, model: nn.Module, args: argparse.Namespace) -> None:
        self.cnf = args

        self.model = model

        # self.model.ds_stats = args.ds_stats  # TODO: remove this - bzw. wie relevant ist das - wird im Code teils noch genutzt....

        if torch.cuda.device_count() > 1 and self.cnf.num_gpus > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
        elif "cuda" in args.device:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            print(f"Using single GPU ({gpu_name})")
        else:
            print("Using CPU")

        if self.cnf.wandb and self.cnf.rank == 0:
            init_wandb(self.cnf, project_name='Lifter2.0', model_to_watch=self.model, resume=self.cnf.resume)

        if args.optimizer == 'adam':
            self.optimizer = optim.AdamW(params=self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            self.optimizer = optim.SGD(params=self.model.parameters(), lr=args.lr, momentum=0.9)

        # init logging stuffs
        self.log_path = args.exp_log_path

        # starting values
        self.epoch = 0
        self.best_test_error_mean = None
        self.best_test_rmse_linear = None
        self.patience = args.max_patience

        training_set, test_set = self.get_dataset(args)
        self.scheduler = None
        if not self.cnf.test_only:

            if args.scheduler == 'plateau':
                self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer, mode='min',
                                                   factor=0.5, patience=10, verbose=True)
            elif args.scheduler == 'cosine':
                t0 = len(self.train_loader) // args.accumulation_steps
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, t0,
                                                                                      eta_min=1e-6)
        # possibly load checkpoint
        self.model = self.model.to(args.device)
        if args.checkpoint:
            self.load_ck(args.checkpoint)
        elif self.cnf.test_only:
            self.load_best_model()
        else:
            self.load_ck()

        if args.distributed:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                                   output_device=args.local_rank,
                                                                   find_unused_parameters=True)

    def load_ck(self, ck_path=None) -> None:
        """
        load training checkpoint
        """
        prev_cnf = self.cnf
        if ck_path is None and not self.cnf.resume:
            return
        if ck_path and not Path(ck_path).exists():
            raise FileNotFoundError(f'checkpoint not found at {ck_path}')
        ck_path = self.log_path / 'training.ck' if ck_path is None else ck_path
        ck = torch.load(Path(ck_path), map_location=self.cnf.device)
        print(f'[loading checkpoint {ck_path}]')
        weights = ck['weights'] if 'weights' in ck else ck['model']
        state_dict = {k.replace("module.", ""): v for k, v in weights.items()}
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

        if self.cnf.resume:
            self.epoch = ck['epoch']
            self.optimizer.load_state_dict(ck['optimizer'])
            self.best_test_error_mean = ck['best_test_error_mean']
            self.patience = ck['patience']
            self.scheduler.load_state_dict(ck['scheduler']) if self.scheduler else None
            self.cnf = ck.get('cnf', self.cnf)
            self.cnf.epochs = prev_cnf.epochs

        from DisNet.distance_estimation_project.main import parse_args
        self.cnf = fill_missing_args(self.cnf, parse_args_fn=parse_args)

    def load_best_model(self, ck_path=None) -> None:
        """
        load best model
        """
        if ck_path is None:
            ck_path = self.log_path / 'best.pth'
        if not ck_path.exists():
            raise FileNotFoundError(f'best model not found at {ck_path}')
        ck = torch.load(ck_path)
        print(f'[loading best model {ck_path}]')
        state_dict = {k.replace("module.", ""): v for k, v in ck['weights'].items()}
        # check if self.model is dataparallel
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

    def save_ck(self) -> None:
        """
        save training checkpoint
        """
        # fix for dataparallel
        state_dict = {k.replace("module.", ""): v for k, v in self.model.state_dict().items()}
        ck = {
            'epoch': self.epoch,
            'model': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'best_test_error_mean': self.best_test_error_mean,
            'patience': self.patience,
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'cnf': self.cnf
        }
        torch.save(ck, os.path.join(self.log_path, 'training.ck'))
        print(f'[saving checkpoint to {os.path.join(self.log_path, "training.ck")}]')

    def save_results(self, results: dict, filename: str) -> None:
        """
        save results to json file
        """
        np.savez(Path(self.log_path, filename), **results)

    def train(self) -> None:
        raise NotImplementedError

    def test(self) -> None:
        raise NotImplementedError

    def infer(self) -> None:
        raise NotImplementedError

    def get_dataset(self, args: argparse.Namespace) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError

    def run(self) -> None:
        """
        start model training procedure (train > test > checkpoint > repeat)
        """
        if self.cnf.test_only:
            with torch.inference_mode():
                self.test()
            return

        for _ in range(self.epoch, self.cnf.epochs):
            self.train()

            with torch.inference_mode():
                self.test()

            self.epoch += 1

            if not self.cnf.use_debug_dataset:
                self.save_ck()

            if (self.patience <= 0) and (self.cnf.max_patience > 0):
                break

        print(f'\n[DONE].')
