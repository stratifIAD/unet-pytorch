import argparse
import yaml
from addict import Dict
import wandb
import os

from lib import 

if __name__ == "__main__":

    '''Creating data parser for train.py'''
    trainparser = argparse.ArgumentParser(description='[StratifIAD] Parameters for training', allow_abbrev=False)
    trainparser.add_argument('-c','--config-file', 
                                type=str, default='../configs/default_config_train.yaml', 
                                help='Config file for the experiment')

    args = trainparser.parse_args()

    conf = Dict(yaml.safe_load(open(args.config_file, "r")))
    
    wandb.init(project="stratifIAD", entity="gabrieljg")
    wandb.config.update(conf)

    train_file = os.path.join(args.split_dir, 'train_df.csv')
    dev_file = os.path.join(args.split_dir, 'dev_df.csv')

    train_dataset = CGIARDataset(meta_data=train_file,
                                root_dir=args.data_dir,
                                transform=transforms.Compose([
                                Rescale(512), ToTensor()]))
    dev_dataset = CGIARDataset(meta_data=dev_file,
                              root_dir=args.data_dir,
                              transform=transforms.Compose([
                              Rescale(512), ToTensor()]))

    if args.load_limit == -1:
      sampler, shuffle = None, True
    else:
      sampler, shuffle = SubsetRandomSampler(range(args.load_limit)), False

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=shuffle, num_workers=1, sampler=sampler)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size,
                            shuffle=shuffle, num_workers=1, sampler=sampler)

    loaders = {'train': train_dataloader, 'val': dev_dataloader}

    trainer = Trainer(model_opts=conf.model_opts, loaders=loaders)
    trainer.train(args.epochs)