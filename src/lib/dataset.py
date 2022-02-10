import os
import numpy as np
import pandas as pd
import torch
import glob
from torch.utils.data import Dataset
from PIL import Image


class stratifiadDataset(Dataset):
    """ StratifIAD Dataset for plaques and tangle segmentation/classification """

    def __init__(self, meta_data, root_dir, normalization, transform=None):
        """
        Args:
            root_dir (string): Directory with all the WSI. Each WSI has 4 folders:
                - macenko: all patches with macenko normalization.
                - masks: all masks from patches.
                - patches: original patches without normalization.
                - vahadane: all patches with vahadane normalization.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        df = pd.read_csv(meta_data)
        self.wsi_list = df.wsi.to_list()
        self.root_dir = root_dir
        self.transform = transform
        self.normalization = normalization

        if self.normalization == 'macenko':
            self.imgs = np.concatenate([glob.glob(os.path.join(self.root_dir, self.wsi_list[i], 'macenko','*.png')) for i in range(len(self.wsi_list))])
        elif self.normalization == 'vahadane':
            self.imgs = np.concatenate([glob.glob(os.path.join(self.root_dir, self.wsi_list[i], 'vahadane','*.png')) for i in range(len(self.wsi_list))])
        else:
            print(f'[ERROR] Normalization method is not recognized. Change the configuration file.')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        '''
        Load patch and mask arrays:
        - images come from a single WSI.
        - mask --> 1 for plaques, 0 for background.
        '''

        image = Image.open(self.imgs[idx])
        gt_img = Image.open(self.imgs[idx].replace(self.normalization,'masks').replace('patch','mask')).convert('1')

        sample = {'image': np.array(image), 'gt': np.array(gt_img)}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample['image'], sample['gt']
    
    '''scalar 
    def find_max(im_pths):
    
    
    minimo_pixel=[]
    maximo_pixel=[]
    size=len(im_pths) 

    for i in im_pths:
        img = np.load(str(i))
           
        minimo_pixel.append(np.min(img))
        maximo_pixel.append(np.max(img))

    return   np.min(minimo_pixel),np.max(maximo_pixel), size
        

    '''
