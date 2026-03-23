import os
import random
import numpy as np
import torch
import yaml
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from neural_nets import model_factory
from datasets import dataset_factory
from train_utils.train_utils import main_training_loop, test, parse_args
from adversity.transforms import apply_augmentation
from torch.utils.tensorboard import SummaryWriter
from ConfigSpace import Configuration, ConfigurationSpace
# from smac import HyperparameterOptimizationFacade, Scenario
# from smac import RunHistory
import pandas as pd
import pdb

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)




def main(hyperparam_config=None, seed=42):
    args = parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    if hyperparam_config:
        config["dataset"]["kwargs"].update(hyperparam_config)

    # torch.cuda.set_device(config['device'])
    model_name = config['model']['name']
    dataset_name = config['dataset']['name']
    save_path = f'models/{model_name}_{dataset_name}.pth'
    train_dataset = dataset_factory[config['dataset']['name']](
                    **config['dataset']['kwargs'], mode="train", 
                    transforms=apply_augmentation)
    print("Got here")
    train_loader = DataLoader(train_dataset, 
                              batch_size=config['dataset']['batch_size'], 
                              shuffle=True)
    print("train:", train_dataset)

    print('total batches:', len(train_loader))
    DEVICE = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() else "cpu")
    net = model_factory[model_name](**config['model']['kwargs']).to(torch.double).to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        mode='min', factor=0.5, patience=3)
    # Initialize TensorBoard writer
    writer = SummaryWriter()
    final_ep_loss = main_training_loop(train_loader, net, optimizer, scheduler, 
                       writer=writer, save_path=save_path,
                    num_epochs=config["num_epochs"], device=DEVICE, log_interval=2, config=config)
    # Close the writer
    writer.close()
    if not hyperparam_config:
        test_loader = DataLoader(test_dataset,
            batch_size=config['dataset']['batch_size'], 
            shuffle=True)
        test_dataset = dataset_factory[config['dataset']['name']](
            **config['dataset']['kwargs'], mode="test_full", 
            transforms=apply_augmentation) # test on full image

        mIOU, gdice = test(test_loader, net, save_path=save_path, 
                        num_classes=config['model']['kwargs']['output_channels'])
        print(f"mIOU: {mIOU}, gdice: {gdice}")
    else:
        mIOU, gdice = test(train_loader, net, save_path=save_path, 
                        num_classes=config['model']['kwargs']['output_channels'])
        return 1 - gdice
main()


# def get_best_params():
#     configspace = ConfigurationSpace({"conductivity": (0.0, 1.0),
#                                       "window_size": [2, 3, 4, 5, 6]
#                                       })
#     configspace = ConfigurationSpace({
#                                         "alpha": (0.0, 1.0),
#                                       "gamma": [2.0, 3.0, 4.0]
#                                       })
#     scenario = Scenario(configspace,
#                         name="get_loss_urban_best", 
#                         deterministic=True, n_trials=10)
#     smac = HyperparameterOptimizationFacade(scenario, main)
#     incumbent = smac.optimize()
#     
#     # Let's calculate the cost of the incumbent
#     incumbent_cost = smac.validate(incumbent)
#     print(f"Incumbent cost: {incumbent_cost}")
  

# get_best_params()
# main()
