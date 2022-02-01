from __future__ import print_function, division
import argparse

from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import GRSSDataset
from src.transforms import ToTensor, Rescale

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="dfc2021_dse_train/Train")
args = parser.parse_args()

transformed_GRSS_dataset = GRSSDataset(meta_data='data/train_df.csv',
									root_dir=args.data_dir,
                                    transform=transforms.Compose([
                                    Rescale(512),
                                    ToTensor()
                                    ]))

dataloader = DataLoader(transformed_GRSS_dataset, batch_size=4,
                        shuffle=True, num_workers=4)


#############################################################
# To test if GRSSDataset and Dataloader is working properly #
#############################################################
for i_batch, sample_batched in enumerate(dataloader):
	images, gt = sample_batched
	print(i_batch, images.size(), gt.size())