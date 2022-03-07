import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import tqdm

if __name__ == "__main__":
    '''
    Script that divides data for cross-validation. 
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, help="This is the folder where you store the patches.", required=True)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--num_exp", type=int, help="This is the number of the experiment. The division of data is different from each experiment", required=True)

    args = parser.parse_args()

    df = pd.DataFrame({'wsi': os.listdir(args.data_dir)})

    output_dir = 'data/experiment_'+f'{args.num_exp:03}/'
    os.makedirs(output_dir, exist_ok=True)

    trainfile = f'{output_dir}train_00.csv'
        
    print(f'Saving train/dev dataset #00')
    kf = KFold(n_splits = args.num_folds, shuffle = False, random_state = None)
    cv = 0
    for train_index, test_index in kf.split(df):
        print(f'Saving train/dev dataset #00 --> CV {cv}')

        trainfile = f'{output_dir}train_00_cv_{cv:02}.csv'
        devfile = f'{output_dir}dev_00_cv_{cv:02}.csv'

        train, dev = df.iloc[train_index], df.iloc[test_index]
            
        train.to_csv(trainfile, index=False)
        dev.to_csv(devfile, index=False)
        cv += 1