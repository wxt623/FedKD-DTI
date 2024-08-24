import numpy as np
import torch
from datetime import datetime


""" 以aupr为指标的早停 """
class EarlyStopping:

    def __init__(self, patience= 5, delta= 0.0001, verbose= False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        self.accompany_loss = np.Inf
        self.accompany_accuracy = 0
        self.accompany_precision = 0
        self.accompany_recall = 0
        self.accompany_f1 = 0
        self.accompany_auc_roc = 0
        self.best_auc_pr = 0
        self.best_model_parameter = np.Inf

    def __call__(self, loss, accuracy, precision, recall, f1, auc_roc, auc_pr, model):
        if self.best_score is None:
            self.best_score = auc_pr
            self.save_checkpoint(loss, accuracy, precision, recall, f1, auc_roc, auc_pr, model)
        elif auc_pr < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = auc_pr
            self.counter = 0
            self.save_checkpoint(loss, accuracy, precision, recall, f1, auc_roc, auc_pr, model)

    def save_checkpoint(self, loss, accuracy, precision, recall, f1, auc_roc, auc_pr, model):
        if self.verbose:
            print(f'aupr increased ({self.best_auc_pr:.5f} --> {auc_pr:.5f}).  Saving model ...')
        self.accompany_loss = loss
        self.accompany_accuracy = accuracy
        self.accompany_precision = precision
        self.accompany_recall = recall
        self.accompany_f1 = f1
        self.accompany_auc_roc = auc_roc
        self.best_auc_pr = auc_pr
        self.best_model_parameter = model.state_dict()


