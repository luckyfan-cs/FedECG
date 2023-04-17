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
import argparse

def args_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # federated arguments
    parser.add_argument('--num_workers', type=int, default=5, help="number of clients in total")
    parser.add_argument('--num_sample_submodels', type=int, default=5, help="number of sampled model")
    parser.add_argument('--model', type=str, default="RNNAttentionModel", help="model")
    parser.add_argument('--batch_size', type=int, default=32, help="local batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--epochs', type=int, default=100, help="training epochs")
    parser.add_argument('--seed', type=int, default=10, help="seed")
    # data
    parser.add_argument('--train_path', type=str, default='./data/train.csv', help="path for test data")
    parser.add_argument('--test_path', type=str, default='./data/test.csv', help="path for train data")
    parser.add_argument('--eda_save_path', type=str, default='./eda_figs/', help="path for train data")
    parser.add_argument('--log_save_path', type=str, default='./logs/', help="path for train data")
    parser.add_argument('--device_id', type=int, default= 0,
                        help='device id')
    args = parser.parse_args()
    return args
id_to_label = {
    0: "Normal",
    1: "Artial Premature",
    2: "Premature ventricular contraction",
    3: "Fusion of ventricular and normal",
    4: "Fusion of paced and normal"
}
def plot_labels(df,args, title):

    percentages = [count / df.shape[0] * 100 for count in df['label'].value_counts()]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(
        x=df['label'],
        ax=ax,
        palette="bright",
        order=df['label'].value_counts().index
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15);

    for percentage, count, p in zip(
            percentages,
            df['label'].value_counts(sort=True).values,
            ax.patches):
        percentage = f'{np.round(percentage, 2)}%'
        x = p.get_x() + p.get_width() / 2 - 0.4
        y = p.get_y() + p.get_height()
        ax.annotate(str(percentage) + " / " + str(count), (x, y), fontsize=12, fontweight='bold')

    plt.savefig(args.eda_save_path+ '{}_data_dist.pdf'.format(title), bbox_inches='tight')
    # plt.savefig('data_dist.svg', facecolor='w', edgecolor='w', format='svg',
    #             transparent=False, bbox_inches='tight', pad_inches=0.1)

def plot_ECG_signical(df,args):
    N = 5

    samples = [df.loc[df['label'] == cls].sample(N) for cls in range(N)]
    titles = [id_to_label[cls] for cls in range(5)]

    with plt.style.context("seaborn-white"):

        for i in range(5):
            fig, ax = plt.subplots(1, 1, figsize=(20, 7))
            #ax = axs.flat[i]
            ax.plot(samples[i].values[:, 1:-1].transpose())
            ax.set_title(titles[i])
            # plt.ylabel("Amplitude")

            plt.tight_layout()
            #plt.suptitle("ECG Signals", fontsize=20, y=1.05, weight="bold")
            plt.savefig(args.eda_save_path+"signals_{}_class.pdf".format(i), bbox_inches='tight')
            plt.clf()

def plot_Confusion(confusion_matrix,args,title):
    fig, ax = plt.subplots(figsize=(5, 5))
    cm_ = ax.imshow(confusion_matrix, cmap='hot')
    ax.set_title('Confusion matrix', fontsize=15)
    ax.set_xlabel('Actual', fontsize=13)
    ax.set_ylabel('Predicted', fontsize=13)
    plt.colorbar(cm_)
    plt.savefig(args.eda_save_path + '{}_Confusion_matrix.pdf'.format(title), bbox_inches='tight')
