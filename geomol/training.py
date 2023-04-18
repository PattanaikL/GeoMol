import math
from tqdm import tqdm
from typing import List, Union
from argparse import Namespace
import numpy as np

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn as nn


def train(model, loader, optimizer, device, scheduler, logger, epoch, writer):
    model.train()
    loss_all, loss_log = 0, 0
    a, b, c, d, e = 0, 0, 0, 0, 0

    for i, data in tqdm(enumerate(loader), total=len(loader)):
        data = data.to(device)
        optimizer.zero_grad()

        result = model(data) if epoch > 1 else model(data, ignore_neighbors=True)
        result.backward()

        # clip the gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)

        if logger:
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            logger.info(f'Parameter Norm: {pnorm}\t Gradient Norm: {gnorm}\t Loss: {result.item()}')

        optimizer.step()
        if isinstance(scheduler, NoamLR):
            scheduler.step()
        loss_all += result.item()
        loss_log += result.item()

        a += model.one_hop_loss_write.item()
        b += model.two_hop_loss_write.item()
        c += model.angle_loss_write.item()
        d += model.dihedral_loss_write.item()
        e += model.three_hop_loss_write.item()

        if i % 100 == 99:
            writer.add_scalar('Training/one_hop_loss', a / 100, epoch * len(loader) + i)
            writer.add_scalar('Training/two_hop_loss', b / 100, epoch * len(loader) + i)
            writer.add_scalar('Training/angle_loss', c / 100, epoch * len(loader) + i)
            writer.add_scalar('Training/dihedral_loss', d / 100, epoch * len(loader) + i)
            writer.add_scalar('Training/three_hop_loss', e / 100, epoch * len(loader) + i)
            writer.add_scalar('Training/total_loss', loss_log / 100, epoch * len(loader) + i)
            a, b, c, d, e, loss_log = 0, 0, 0, 0, 0, 0

            if optimizer.param_groups:
                writer.add_scalar('Training/learning_rate', optimizer.param_groups[0]['lr'], epoch * len(loader) + i)
            else:
                for j in range(len(optimizer.optimizers)):
                    writer.add_scalar(f'Training/learning_rate_{j}', optimizer.optimizers[j].param_groups[0]['lr'],
                                      epoch * len(loader) + i)

    return loss_all / len(loader)


@torch.no_grad()
def test(model, loader, device, epoch, writer):
    model.eval()
    loss_all = 0
    a, b, c, d, e = 0, 0, 0, 0, 0

    for i, data in tqdm(enumerate(loader), total=len(loader)):

        data = data.to(device)
        result = model(data)
        loss_all += result.item()

        a += model.one_hop_loss_write.item()
        b += model.two_hop_loss_write.item()
        c += model.angle_loss_write.item()
        d += model.dihedral_loss_write.item()
        e += model.three_hop_loss_write.item()

    writer.add_scalar('Validation/one_hop_loss', a / len(loader), epoch * len(loader) + len(loader))
    writer.add_scalar('Validation/two_hop_loss', b / len(loader), epoch * len(loader) + len(loader))
    writer.add_scalar('Validation/angle_loss', c / len(loader), epoch * len(loader) + len(loader))
    writer.add_scalar('Validation/dihedral_loss', d / len(loader), epoch * len(loader) + len(loader))
    writer.add_scalar('Validation/three_hop_loss', e / len(loader), epoch * len(loader) + len(loader))
    writer.add_scalar('Validation/total_loss', loss_all / len(loader), epoch * len(loader) + len(loader))

    return loss_all / len(loader)


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):
        """
        Initializes the learning rate scheduler.

        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after warmup_epochs).
        :param final_lr: The final learning rate (achieved after total_epochs).
        """
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Gets a list of the current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
        If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]


def build_lr_scheduler(optimizer: Optimizer, args: Namespace, train_data_size: int) -> _LRScheduler:
    """
    Builds a learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: Arguments.
    :param train_data_size: The size of the training dataset.
    :return: An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=[args.n_epochs],
        steps_per_epoch=train_data_size // args.batch_size,
        init_lr=[args.lr / 10],
        max_lr=[args.lr],
        final_lr=[args.lr / 100]
    )


def compute_pnorm(model: nn.Module) -> float:
    """Computes the norm of the parameters of a model."""
    return math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))


def compute_gnorm(model: nn.Module) -> float:
    """Computes the norm of the gradients of a model."""
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))
