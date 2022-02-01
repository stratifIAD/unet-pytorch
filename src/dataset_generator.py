import argparse
import yaml
from addict import Dict
from lib import utils

'''
What we will need to input:
- input folder to WSI
- outputfolder for the dataset (create one if not found)
- parameters for the creation of the dataset:
    - patchsize
'''

if __name__ == '__main__':
    
    '''Creating data parser for data_generator.py'''
    datagenparser = argparse.ArgumentParser(description='[StratifIAD] Parameters for dataset generator', allow_abbrev=False)
    datagenparser.add_argument('-c','--config-file', 
                                type=str, default='../configs/default_config_dataset.yaml', 
                                help='Config file for the experiment')

    args = datagenparser.parse_args()

    conf = Dict(yaml.safe_load(open(args.config_file, "r")))
    print(conf.datasetargs)







