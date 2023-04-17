import logging
import torch
import time
from abc import ABCMeta, abstractmethod
import torch.nn as nn
import os
import itertools
import time
import numpy as np # linear algebra
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score
import pandas as pd
import matplotlib as plt
from utils import plot_Confusion
'''
This is the worker for sharing the local weights.
'''

class Meter:
    def __init__(self, n_classes=5):
        self.metrics = {}
        self.confusion = torch.zeros((n_classes, n_classes))

    def update_train(self, x, y, loss):
        x = np.argmax(x.detach().cpu().numpy(), axis=1)
        y = y.detach().cpu().numpy()
        self.metrics['loss'] += loss
        self.metrics['accuracy'] += accuracy_score(x, y)
        self.metrics['f1'] += f1_score(x, y, average='macro')
        self.metrics['precision'] += precision_score(x, y, average='macro', zero_division=1)
        self.metrics['recall'] += recall_score(x, y, average='macro', zero_division=1)

        self._compute_cm(x, y)
    def update_val(self, x, y):
        x = np.argmax(x.detach().cpu().numpy(), axis=1)
        y = y.detach().cpu().numpy()
        self.metrics['accuracy'] += accuracy_score(x, y)
        self.metrics['f1'] += f1_score(x, y, average='macro')
        self.metrics['precision'] += precision_score(x, y, average='macro', zero_division=1)
        self.metrics['recall'] += recall_score(x, y, average='macro', zero_division=1)

        self._compute_cm(x, y)
    def _compute_cm(self, x, y):
        for prob, target in zip(x, y):
            if prob == target:
                self.confusion[target][target] += 1
            else:
                self.confusion[target][prob] += 1

    def init_metrics(self):
        self.metrics['loss'] = 0
        self.metrics['accuracy'] = 0
        self.metrics['f1'] = 0
        self.metrics['precision'] = 0
        self.metrics['recall'] = 0

    def get_metrics(self):
        return self.metrics

    def get_confusion_matrix(self):
        return self.confusion
class WorkerBase(metaclass=ABCMeta):
    def __init__(self, model, train_iter,  test_iter, args, optimizer, device):
        self.model = model

        self.train_iter = train_iter
        self.test_iter = test_iter
        self.args = args
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.train_df_logs = pd.DataFrame()
        self.val_df_logs = pd.DataFrame()

        self.device = device
        self._level_length = None
        self._weights_len = 0
        self._weights = None

    def get_weights(self):
        """ getting weights """
        return self._weights

    def set_weights(self, weights):
        """ setting weights """
        self._weights = weights

    def upgrade(self):
        """ Use the processed weights to update the model """

        idx = 0
        for param in self.model.parameters():
            tmp = self._weights[self._level_length[idx]:self._level_length[idx + 1]]
            weights_re = torch.tensor(tmp, device=self.device)
            weights_re = weights_re.view(param.data.size())

            param.data = weights_re
            idx += 1

    @abstractmethod
    def update(self):
        pass


    @property
    def gnn_train(self):  # This function is for local train one epoch using local dataset on client
        """ General local training methods """
        self.model.train()

        meter = Meter()
        meter.init_metrics()
        for i, (data, target) in enumerate(self.train_iter):
            data = data.to(self.device)
            target  = target.to(self.device)  # num x feat
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            meter.update_train(output, target, loss.item())


# test
        self._weights = []
        self._level_length = [0]

        for param in self.model.parameters():
            self._level_length.append(param.data.numel() + self._level_length[-1])
            self._weights += param.data.view(-1).cpu().numpy().tolist()
        self._weights_len = len(self._weights)

        metrics = meter.get_metrics()
        metrics = {k: v / i for k, v in metrics.items()}
        df_logs = pd.DataFrame([metrics])
        df_logs.to_csv(self.args.eda_save_path+'Train_Fed_model-{}_metric.csv'.format(self.args.model))
        confusion_matrix = meter.get_confusion_matrix()

        # show logs
        print('Train: {}: {}, {}: {}, {}: {}, {}: {}, {}: {}'
              .format(*(x for kv in metrics.items() for x in kv))
              )

        title = "Train_Fed_model-{}".format(self.args.model)
        plot_Confusion(confusion_matrix,self.args,title)
        _, val_df = self.gnn_evaluate()
        self.train_df_logs = pd.concat([self.train_df_logs, df_logs], axis=0)
        train_df = self.train_df_logs
        return metrics, train_df, val_df

    def gnn_evaluate(self):
        """ General local training methods """
        meter = Meter()
        meter.init_metrics()
        self.model.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(self.test_iter ):
                data = data.to(self.device)
                target  = target.to(self.device)  # num x feat
                output = self.model(data)
                meter.update_val(output, target)

        metrics = meter.get_metrics()
        metrics = {k: v / i for k, v in metrics.items()}
        df_logs = pd.DataFrame([metrics])
        confusion_matrix = meter.get_confusion_matrix()

        # show logs
        print('Val: {}: {}, {}: {}, {}: {}, {}: {}, {}: {}'
              .format(*(x for kv in metrics.items() for x in kv))
              )
        title = "Val_Fed_model-{}".format(self.args.model)
        plot_Confusion(confusion_matrix,self.args,title)
        df_logs.to_csv(self.args.eda_save_path + 'Val_Fed_model-{}_metric.csv'.format(self.args.model))
        self.val_df_logs = pd.concat([self.val_df_logs, df_logs], axis=0)
        val_df = self.val_df_logs
        return metrics, val_df


def train(model_id,model,train_iter,device,optimizer,criterion,args,train_df_logs):
    model.train()
    meter = Meter()
    meter.init_metrics()
    for i, (data, target) in enumerate(train_iter):
        data = data.to(device)
        target  = target.to(device)  # num x feat
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        meter.update_train(output, target, loss.item())

    metrics = meter.get_metrics()
    metrics = {k: v / i for k, v in metrics.items()}
    df_logs = pd.DataFrame([metrics])
    df_logs.to_csv(args.eda_save_path+'Train_Fed_model-{}_model-id-{}_metric.csv'.format(args.model,model_id))
    confusion_matrix = meter.get_confusion_matrix()

    # show logs
    print("Model: {} \n".format(model_id))
    print('Train: {}: {}, {}: {}, {}: {}, {}: {}, {}: {}'
          .format(*(x for kv in metrics.items() for x in kv))
          )

    title = "Train_Fed_model-{}-id-{}".format(args.model,model_id)
    plot_Confusion(confusion_matrix,args,title)

    train_df_logs = pd.concat([train_df_logs, df_logs], axis=0)
    train_df = train_df_logs
    return meter, train_df

def evaluate(model_id,model,test_iter,device,criterion,args,val_df_logs):
    """ General local training methods """
    meter = Meter()
    meter.init_metrics()
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_iter):
            data = data.to(device)
            target  = target.to(device)  # num x feat
            output = model(data)
            loss = criterion(output, target)
            meter.update_train(output, target, loss.item())

    metrics = meter.get_metrics()
    metrics = {k: v / i for k, v in metrics.items()}
    df_logs = pd.DataFrame([metrics])
    confusion_matrix = meter.get_confusion_matrix()

    # show logs
    print("Model: {} \n".format(model_id))
    print('Val: {}: {}, {}: {}, {}: {}, {}: {}, {}: {}'
          .format(*(x for kv in metrics.items() for x in kv))
          )
    title = "Val_Fed_model-{}-id".format(args.model,model_id)
    plot_Confusion(confusion_matrix,args,title)
    df_logs.to_csv(args.eda_save_path + 'Val_Fed_model-{}_model-id-{}_metric.csv'.format(args.model,model_id))
    val_df_logs = pd.concat([val_df_logs, df_logs], axis=0)
    val_df = val_df_logs
    return metrics, val_df