# A class to train models
from tqdm import tqdm

import torch
import wandb

from models.unet import Unet
from models.conv_net import ConvNet

class Trainer:
    def __init__(self, model_opts, loaders):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_model(model_opts)
        self.loaders = loaders
        self.op = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def set_model(self, model_opts):
        model_def = globals()[model_opts.name]
        self.model = model_def(**model_opts.args)
        self.multi_cls = True if model_opts.args.outchannels > 1 else False
        wandb.watch(self.model)
        self.model.to(self.device)

    def get_loss(self, y_hat, y):
        if self.multi_cls:
            self.loss_f = torch.nn.CrossEntropyLoss()
        else:
            self.loss_f = torch.nn.BCEWithLogitsLoss()
        return self.loss_f(y_hat, y)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for img, mask in tqdm(train_loader):
            self.op.zero_grad()
            img, mask = img.to(self.device, dtype=torch.float), mask.to(self.device)
            pred = self.model(img)
            loss = self.get_loss(pred, mask)
            total_loss += loss
            loss.backward()
            self.op.step()
        # Ignore last in loader
        # TODO 
        return total_loss / len(train_loader)
    

    def test(self, test_loader):
        self.model.eval()  
        total_loss = 0
        for img, mask in tqdm(test_loader):
            img, mask = img.to(self.device, dtype=torch.float), mask.to(self.device)
            pred = self.model(img)
            loss = self.loss_f(pred, mask)
            total_loss += loss
        return total_loss / len(test_loader)

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch(self.loaders['train'])
            test_loss = self.test(self.loaders['val'])
            print(f'Epoch {epoch}/{epochs}: training loss = {train_loss}, test loss = {test_loss}')
            wandb.log({"loss/train": train_loss, "loss/dev": test_loss}, step=epoch)


    def predict(self):
        pass