import os
import itertools
import time
import random

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      StepLR,
                                      ExponentialLR)

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score
from utils import  plot_labels,plot_ECG_signical, args_parser
import argparse



class Config:
    csv_path = ''
    seed = 2021
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    attn_state_path = '../input/mitbih-with-synthetic/attn.pth'
    lstm_state_path = '../input/mitbih-with-synthetic/lstm.pth'
    cnn_state_path = '../input/mitbih-with-synthetic/cnn.pth'

    attn_logs = '../input/mitbih-with-synthetic/attn.csv'
    lstm_logs = '../input/mitbih-with-synthetic/lstm.csv'
    cnn_logs = '../input/mitbih-with-synthetic/cnn.csv'

    train_csv_path = './data/mitbih_train.csv'
    test_csv_path = './data/mitbih_test.csv'


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


config = Config()
seed_everything(config.seed)

args = args_parser()

df_train = pd.read_csv(args.train_path)
df_test = pd.read_csv(args.test_path)
df_train.dropna(axis=0,how='all')
df_test.dropna(axis=0,how='all')
df_train.rename(columns={'187': 'label'}, inplace=True)
#df_test.rename(columns={'186': 'label'}, inplace=True)

print(df_train)
#print("test data",df_train)
print(df_train['label'].value_counts())
#print("test label count",df_test['label'].value_counts())
plot_labels(df_train,args, "train_data")
plot_ECG_signical(df_train,args)
#plot_labels(df_test,args, "test_data")