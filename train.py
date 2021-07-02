from argparse import ArgumentParser
import math
import os
import yaml
import torch
import numpy as np
import random

from model.model import GeoMol
from model.training import train, test, NoamLR
from utils import create_logger, dict_to_str, plot_train_val_loss, save_yaml_file, get_optimizer_and_scheduler
from model.featurization import construct_loader
from model.parsing import parse_train_args, set_hyperparams

from torch.utils.tensorboard import SummaryWriter
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
# torch.multiprocessing.set_sharing_strategy('file_system')

# add training args
args = parse_train_args()

logger = create_logger('train', args.log_dir)
logger.info('Arguments are...')
for arg in vars(args):
    logger.info(f'{arg}: {getattr(args, arg)}')

# seeds
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# construct loader and set device
train_loader, val_loader = construct_loader(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# build model
if args.restart_dir:
    with open(f'{args.restart_dir}/model_parameters.yml') as f:
        model_parameters = yaml.full_load(f)
    model = GeoMol(**model_parameters).to(device)
    state_dict = torch.load(f'{args.restart_dir}/best_model.pt', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=True)

else:
    hyperparams = set_hyperparams(args)
    model_parameters = {'hyperparams': hyperparams,
                        'num_node_features': train_loader.dataset.num_node_features,
                        'num_edge_features': train_loader.dataset.num_edge_features}
    model = GeoMol(**model_parameters).to(device)

# get optimizer and scheduler
optimizer, scheduler = get_optimizer_and_scheduler(args, model, len(train_loader.dataset))

# record parameters
logger.info(f'\nModel parameters are:\n{dict_to_str(model_parameters)}\n')
yaml_file_name = os.path.join(args.log_dir, 'model_parameters.yml')
save_yaml_file(yaml_file_name, model_parameters)

# instantiate summary writer
writer = SummaryWriter(args.log_dir)

best_val_loss = math.inf
best_epoch = 0

logger.info("Starting training...")
for epoch in range(1, args.n_epochs):
    train_loss = train(model, train_loader, optimizer, device, scheduler, logger if args.verbose else None, epoch, writer)
    logger.info("Epoch {}: Training Loss {}".format(epoch, train_loss))

    val_loss = test(model, val_loader, device, epoch, writer)
    logger.info("Epoch {}: Validation Loss {}".format(epoch, val_loss))
    if scheduler and not isinstance(scheduler, NoamLR):
        scheduler.step(val_loss)

    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pt'))
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, os.path.join(args.log_dir, 'last_model.pt'))

logger.info("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))

log_file = os.path.join(args.log_dir, 'train.log')
plot_train_val_loss(log_file)
