import argparse
import os
from typing import Tuple
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn

from base_trainer import Trainer

from distance_estimation_project.utils.metrics import get_metrics, print_metrics

def custom_collate(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    # vis = [item[2] for item in batch]
    cls = [item[2] for item in batch]
    return torch.cat(x, dim=0), torch.cat(y, dim=0), torch.cat(cls, dim=0)




class DisNetTrainer(Trainer):
    def __init__(self, model: nn.Module, args: argparse.Namespace) -> None:
        super().__init__(model, args)

        self.optimizer = self.model.get_optimizer()

        self.device = args.device

    def load_w(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print(f"Loaded checkpoint - lr: {self.optimizer.param_groups[0]['lr']:.5f}")

    def save_w(self, path: str):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, path)

    def train(self):
        self.model.train()
        tot_loss = 0.0
        acc = 0.0
        n_samples = 0
        tot_err = 0.0
        loss_fun = self.model.get_loss_fun()
        with tqdm(total=len(self.train_loader), desc=f'Epoch {self.epoch:3d} Train') as pbar:
            for x, y, _ in self.train_loader:
                if len(x) == 0:
                    continue
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)
                loss = loss_fun(y_pred, y)
                loss.backward()
                self.optimizer.step()
                if self.cnf.scheduler == 'cosine':
                    self.scheduler.step()
                tot_loss += loss.item()
                error = torch.abs(y_pred - y)
                tot_err += error.sum().item()
                acc += (error < 1).sum().item()
                n_samples += y.shape[0]
                pbar.update(1)
                pbar.set_postfix({'ep_loss': tot_loss / n_samples, 'ep_acc': acc / n_samples})

    def infer(self, x):
        self.model.eval()
        x = x.to(self.device)
        y_pred = self.model(x)
        print(y_pred)

    def test(self):
        self.model.eval()
        loss_fun = self.model.get_loss_fun()
        with torch.no_grad(), tqdm(total=len(self.test_loader), desc=f'Epoch {self.epoch:3d}  Test') as pbar:
            tot_loss = 0.0
            acc = 0.0
            n_samples = 0
            tot_err = 0.0
            all_true = []
            all_pred = []
            all_vis = []
            all_class = []
            for x, y, _class in self.test_loader:
                if len(x) == 0:
                    continue
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)
                loss = loss_fun(y_pred, y)
                tot_loss += loss.item()
                error = torch.abs(y_pred - y)
                tot_err += error.sum().item()
                acc += (error < 1).sum().item()
                n_samples += y.shape[0]
                all_true += y.cpu().numpy().tolist()
                all_pred += y_pred.cpu().numpy().tolist()
                # all_vis += vis.cpu().numpy().tolist()
                all_class += _class.cpu().numpy().tolist()
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item(), 'acc': acc / n_samples})
            mean_error = tot_err / n_samples

            # save best model
            if self.best_test_error_mean is None or mean_error < self.best_test_error_mean:
                self.best_test_error_mean = mean_error
                self.patience = self.cnf.max_patience
                self.model.save_w(os.path.join(self.log_path, 'best.pth'), self.cnf)
            else:
                self.patience -= 1

            metrics = get_metrics(y_true=all_true, y_pred=all_pred, y_visibilities=None, y_classes=all_class,
                                  long_range=self.cnf.long_range)

            if self.epoch % 10 == 0:
                print_metrics(metrics)

    def get_dataset(self, args) -> Tuple[Dataset, Dataset]:
        training_set = None
        ds = self.model.get_dataset(args)
        if not self.cnf.test_only:
            training_set = ds(args)
            self.train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.n_workers, collate_fn=custom_collate)

        test_set = ds(args, mode='test')
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.n_workers, collate_fn=custom_collate)

        return training_set, test_set

