import os

import numpy as np
import pandas as pd
import tifffile as tiff

import torch
from torch.utils.data import Dataset

class CGIARDataset(Dataset):
    """ CGIAR Dataset for satellite segmentation/classification """

    def __init__(self, meta_data, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        df = pd.read_csv(meta_data)
        self.patch_list = df.input_img.to_list()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        '''
        Load patch and mask array:
        - image comes from GeoTIFF format --> no need to transpose it.
        - mask is one-hot-encoded --> 1 for water, 0 for background.
        '''
        patch_img = tiff.imread(os.path.join(self.root_dir,'images/',self.patch_list[idx]))
        # patch_img = np.load(os.path.join(self.root_dir,'images/',self.patch_list[idx]+'.npy'))
        # patch_img = patch_img.transpose(1,2,0)
        gt_img = tiff.imread(os.path.join(self.root_dir,'masks/',self.patch_list[idx].replace('.tif','_mask.tif')))

        sample = {'image': patch_img, 'gt': gt_img}

        if self.transform:
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
