import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import tqdm

if __name__ == "__main__":
    '''
    Script that divides data into N groups. Each group will form a fold for the 
    cross-validation (leave-one-fold-out) and cross-testing. To divide the data into N groups, 
    we first shuffle the data. 
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, help="This is the folder where you store the patches.", required=True)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--num_exp", type=int, help="This is the number of the experiment. The division of data is different from each experiment", required=True)

    args = parser.parse_args()

    WSIs = pd.DataFrame({'wsi': os.listdir(args.data_dir)})
    wsiFolds = np.array_split(WSIs.sample(frac=1), args.num_folds) # Shuffle and split

    output_dir = 'data/experiment_'+f'{args.num_exp:03}/'
    os.makedirs(output_dir, exist_ok=True)

    assert len(wsiFolds) == args.num_folds, f'Consider a different number of folds so number of WSI is divisible.'

    for i in range(args.num_folds):
        # Save one of the folds as test dataset --> used for cross testing
        testfile = f'{output_dir}test_{i:02}.csv'
        print(f'Saving test fold #{i} to {testfile}')
        wsiFolds[i].to_csv(testfile, index=False)

        # Join the remaining folds and do cross validation
        df = pd.concat([x for j, x in enumerate(wsiFolds) if j != i])
        trainfile = f'{output_dir}train_{i:02}.csv'
        
        print(f'Saving train/dev dataset #{i}')
        kf = KFold(n_splits = args.num_folds - 1, shuffle = False, random_state = None)
        cv = 0
        for train_index, test_index in kf.split(df):
            print(f'Saving train/dev dataset #{i} --> CV {cv}')

            trainfile = f'{output_dir}train_{i:02}_cv_{cv:02}.csv'
            devfile = f'{output_dir}dev_{i:02}_cv_{cv:02}.csv'

            train, dev = df.iloc[train_index], df.iloc[test_index]
            
            train.to_csv(trainfile, index=False)
            dev.to_csv(devfile, index=False)
            cv += 1
    