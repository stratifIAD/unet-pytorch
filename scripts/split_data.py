import os
import argparse
import numpy as np
import pandas as pd

if __name__ == "__main__":
    '''
    [StratifIAD] Script that divides data into N groups. Each group will form a fold for the 
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

    for i in range(len(wsiFolds)):
        filename = f'{output_dir}fold_{i:02}.csv'
        print(f'Saving fold #{i} to {filename}')
        wsiFolds[i].to_csv(filename, index=False)
    
    
    # num_wsi = len(WSIs)
    # slides_per_fold = num_wsi/args.num_groups
    # train_perc = 1 - (slides_per_fold/num_wsi)
    # train_wsi = WSIs.sample(frac=train_perc)
    # test_wsi = WSIs.drop(train_wsi.index)

    
    