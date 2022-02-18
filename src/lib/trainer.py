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
        total_tp , total_fp, total_tn, total_fn = 0, 0, 0, 0
        total_precision, total_recall, total_accuracy, total_f1 = 0, 0, 0, 0

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

                tp, fp, tn, fn, precision, recall, accuracy, f1 = utils.confusion_matrix(pred, mask)
                total_tp += tp
                total_fp += fp
                total_tn += tn
                total_fn += fn

                total_precision += precision
                total_recall += recall
                total_accuracy += accuracy
                total_f1 += f1

        return total_loss/len(test_loader), dice_score/len(test_loader), total_tp/len(test_loader), \
                total_fp/len(test_loader), total_tn/len(test_loader), total_fn/len(test_loader), total_precision/len(test_loader), \
                total_recall/len(test_loader), total_accuracy/len(test_loader), total_f1/len(test_loader)

    def train(self, epochs):
        early_stopping = utils.EarlyStopping(patience=self.patience, verbose=True, path=self.results_model_filename)
        for epoch in range(epochs):
            train_loss = self.train_epoch(self.loaders['train'])
            test_loss, test_dice, test_tp, test_fp, test_tn, test_fn, test_precision, test_recall, test_accuracy, test_f1 = self.test(self.loaders['val'])
            print(f'Epoch {epoch}/{epochs}: training loss = {train_loss}, test loss = {test_loss}, test dice = {test_dice}, \
                    test precision = {test_precision}, test recall = {test_recall}, test accuracy = {test_accuracy}, test f1 = {test_f1}')
            wandb.log({"loss/train": train_loss, "loss/val": test_loss, "val_metrics/dice": test_dice, "val_metrics/f1": test_f1, \
                        "val_metrics/precision": test_precision, "val_metrics/recall": test_recall, "val_metrics/accuracy": test_accuracy, \
                        "val_metrics/tp": test_tp, "val_metrics/fp": test_fp, "val_metrics/tn": test_tn, "val_metrics/fn": test_fn}, step=epoch)

            # Adding early stopping according to the evolution of the validation loss
            if self.early_stopping_flag:
                early_stopping(test_loss, self.model)
                if early_stopping.early_stop:
                    print(f'Early stopping')
                    break

        self.model.load_state_dict(torch.load(self.results_model_filename))

    def predict(self):
        pass