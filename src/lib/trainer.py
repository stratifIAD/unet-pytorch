# A class to train models
from tqdm import tqdm

import torch
import wandb

from .models.unet import Unet
from .models.conv_net import ConvNet
from . import utils

class Trainer:
    def __init__(self, model_opts, train_par, loaders):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_model(model_opts)
        self.loaders = loaders
        self.op = torch.optim.Adam(self.model.parameters(), lr=train_par.lr)
        self.eval_threshold = train_par.eval_threshold
        self.patience = train_par.patience
        self.early_stopping_flag = train_par.early_stopping_flag
        self.results_model_filename = train_par.results_model_filename

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
        self.model.train
        total_loss = 0
        for img, mask in tqdm(train_loader):
            img, mask = img.to(self.device, dtype=torch.float), mask.to(self.device, dtype=torch.float)
            pred = self.model(img)
            loss = self.get_loss(pred, mask)
            total_loss += loss
            loss.backward()
            self.op.step()
            self.op.zero_grad()

        return total_loss / len(train_loader)
    
    def test(self, test_loader):
        self.model.eval()  
        total_loss = 0
        dice_score = 0
        for img, mask in tqdm(test_loader):
            img, mask = img.to(self.device, dtype=torch.float), mask.to(self.device, dtype=torch.float)
            with torch.no_grad():
                pred_mask = self.model(img)
                loss = self.loss_f(pred_mask, mask)
                total_loss += loss
                self.op.zero_grad()

                pred = torch.sigmoid(pred_mask)
                pred = (pred > self.eval_threshold).float()
                dice_score += utils.dice_coeff_batch(pred, mask).item()

        return total_loss / len(test_loader), dice_score / len(test_loader) 

    def train(self, epochs):
        early_stopping = utils.EarlyStopping(patience=self.patience, verbose=True, path=self.results_model_filename)
        for epoch in range(epochs):
            train_loss = self.train_epoch(self.loaders['train'])
            test_loss, test_dice = self.test(self.loaders['val'])
            print(f'Epoch {epoch}/{epochs}: training loss = {train_loss}, test loss = {test_loss}, test dice = {test_dice}')
            wandb.log({"loss/train": train_loss, "loss/dev": test_loss, "dice/dev": test_dice}, step=epoch)

            # Adding early stopping according to the evolution of the validation loss
            if self.early_stopping_flag:
                early_stopping(test_loss, self.model)
                if early_stopping.early_stop:
                    print(f'Early stopping')
                    break

        self.model.load_state_dict(torch.load(self.results_model_filename))

    def predict(self):
        pass