import argparse
from linecache import cache
import yaml
from addict import Dict
import wandb
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from lib.trainer import Trainer
from lib.dataset import stratifiadDataset
from lib.transforms import ToTensor, Rescale

if __name__ == "__main__":

    '''Creating data parser for train.py'''
    trainparser = argparse.ArgumentParser(description='[StratifIAD] Parameters for training', allow_abbrev=False)
    trainparser.add_argument('-c','--config-file', type=str, default='configs/default_config_train.yaml', help='Config file for the experiment')

    args = trainparser.parse_args()

    conf = Dict(yaml.safe_load(open(args.config_file, "r")))
    
    wandb.init(project="stratifIAD", entity="gabrieljg")
    wandb.config.update(conf)

    torch.backends.cudnn.benchmark = True

    data_dir = conf.dataset.data_dir
    train_file = conf.dataset.train
    dev_file = conf.dataset.dev
    normalization = conf.dataset.normalization
    cache_data = conf.dataset.cache_data

    wandb.run.name = dev_file.replace('dev','train_dev').split('.')[0].split('/')[-1]
    print(f'RUN: {wandb.run.name}')

    train_dataset = stratifiadDataset(meta_data=train_file,
                                root_dir=data_dir, 
                                normalization=normalization,
                                cache_data=cache_data,
                                transform=transforms.Compose([
                                Rescale(128), ToTensor()]))
    dev_dataset = stratifiadDataset(meta_data=dev_file,
                                root_dir=data_dir,
                                normalization=normalization,
                                cache_data=cache_data,
                                transform=transforms.Compose([
                                Rescale(128), ToTensor()]))

    train_dataloader = DataLoader(train_dataset, batch_size=conf.train_par.batch_size,
                            shuffle=True, num_workers=conf.train_par.workers, pin_memory=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=conf.train_par.batch_size,
                            shuffle=True, num_workers=conf.train_par.workers, pin_memory=True)

    loaders = {'train': train_dataloader, 'val': dev_dataloader}

    trainer = Trainer(model_opts=conf.model_opts, train_par=conf.train_par, loaders=loaders)
    trainer.train(conf.train_par.epochs)