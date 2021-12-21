import logging
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import seaborn as sns
import yaml
import torch

from model.training import build_lr_scheduler


sns.set_style('whitegrid', {'axes.edgecolor': '.2'})
sns.set('poster', rc={"xtick.bottom": True, "ytick.left": True,
                      'axes.edgecolor': '.2',
                      "font.weight": 'bold',
                      "axes.titleweight": 'bold',
                      'axes.labelweight': 'bold'})
sns.color_palette('husl')
local_modules = ['gnn', 'encoder', 'coord_pred', 'd_mlp']

class Standardizer:
    """Z-score standardization"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, rev=False):
        if rev:
            return (x * self.std) + self.mean
        return (x - self.mean) / self.std


def create_logger(name: str, log_dir: str = None) -> logging.Logger:
    """
    Creates a logger with a stream handler and file handler.

    :param name: The name of the logger.
    :param log_dir: The directory in which to save the logs.
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Set logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fh = logging.FileHandler(os.path.join(log_dir, name + '.log'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    return logger


def dict_to_str(dictionary: dict,
                level: int = 0,
                ) -> str:
    """
    A helper function to log dictionaries in a pretty way.

    Args:
        dictionary (dict): A general python dictionary.
        level (int): A recursion level counter, sets the visual indentation.

    Returns:
        str: A text representation for the dictionary.
    """
    message = ''
    for key, value in dictionary.items():
        if isinstance(value, dict):
            message += ' ' * level * 2 + str(key) + ':\n' + dict_to_str(value, level + 1)
        else:
            message += ' ' * level * 2 + str(key) + ': ' + str(value) + '\n'
    return message


def string_representer(dumper, data):
    """
    Add a custom string representer to use block literals for multiline strings.
    """
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data, style='|')
    return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data)


def save_yaml_file(path: str,
                   content: list or dict,
                   ) -> None:
    """
    Save a YAML file (usually an input / restart file, but also conformers file)

    Args:
        path (str): The YAML file path to save.
        content (list, dict): The content to save.
    """
    if not isinstance(path, str):
        raise InputError(f'path must be a string, got {path} which is a {type(path)}')
    yaml.add_representer(str, string_representer)
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)


def plot_train_val_loss(log_file):
    """
    Plots the training and validation loss by parsing the log file.

    :param log_file: The path to the log file created during training.
    """
    train_loss = []
    val_loss = []
    with open(log_file) as f:
        lines = f.readlines()
        for line in lines:
            if ': Training Loss' in line:
                train_loss.append(float(line.split(' ')[-1].rstrip()))
            if ': Validation Loss' in line:
                val_loss.append(float(line.split(' ')[-1].rstrip()))

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(np.arange(len(train_loss)), train_loss, label='Train Loss')
    ax.plot(np.arange(len(val_loss)), val_loss, label='Val Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    fig.savefig(os.path.join(os.path.dirname(log_file), 'train_val_loss.pdf'), bbox_inches='tight')


class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op
        self.param_groups = None

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


class MultipleScheduler(object):
    def __init__(self, opts, scheduling_fn):
        self.schedulers = [scheduling_fn(op) for op in opts.optimizers]

    def step(self, val_loss):
        for sch in self.schedulers:
            sch.step(val_loss)


def get_optimizer_and_scheduler(args, model, train_data_size):
    if args.optimizer == 'adam':
        if args.separate_opts:
            local_params = [p for n, p in list(model.named_parameters()) if n.split('.')[0] in local_modules]
            pair_params = [p for n, p in list(model.named_parameters()) if n.split('.')[0] not in local_modules]
            local_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, local_params), lr=args.lr)
            pair_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, pair_params), lr=args.lr)
            optimizer = MultipleOptimizer(local_optimizer, pair_optimizer)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        raise NotImplementedError("Optimizer not implemented.")

    if args.scheduler == 'plateau':
        if args.separate_opts:
            scheduling_fn = lambda opt: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.7,
                                                                                   patience=5, min_lr=args.lr / 100)
            scheduler = MultipleScheduler(optimizer, scheduling_fn)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7,
                                                                   patience=5, min_lr=args.lr/100)
    elif args.scheduler == 'noam':
        scheduler = build_lr_scheduler(optimizer=optimizer, args=args, train_data_size=train_data_size)
    else:
        scheduler = None

    return optimizer, scheduler
