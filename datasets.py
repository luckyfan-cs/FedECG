import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold
class ECGDataset(Dataset):

    def __init__(self, df):
        self.df = df
        self.data_columns = self.df.columns[1:-1].tolist()
        feature_dim = len(df.loc[df.index[0]]) -1
        self.re_index = list(self.df.index)
    def __getitem__(self, idx):

        signal = self.df.loc[self.re_index[idx], self.data_columns].astype('float32')
        #signal = self.df.loc[self.df.index[idx], self.df.columns[self.col_idxs]].astype('float32')
        signal = torch.FloatTensor([signal.values])
        target = torch.LongTensor(np.array(self.df.loc[self.re_index[idx], 'label']))
        return signal, target

    def __len__(self):
        return len(self.df)

def get_dataloader(args):
    '''
    Dataset and DataLoader.
    Parameters:
        pahse: training or validation phase.
        batch_size: data per iteration.
    Returns:
        data generator
    '''
    df = pd.read_csv(args.train_path)
    df.dropna(axis=0, how='all')
    df.rename(columns={'187': 'label'}, inplace=True)


    skf = StratifiedKFold(n_splits=args.num_workers, shuffle=True, random_state=args.seed)

    train_dataloaders = []
    test_dataloaders = []
    for fold, (train_index, test_index) in enumerate(skf.split(df, df["label"])):
        fold_index = np.concatenate((train_index, test_index))
        #print("test index", test_index)
        df_data = df.iloc[fold_index]
        df = pd.DataFrame(df_data)
        train_df, val_df = train_test_split(
            df, test_size=0.20, random_state=args.seed)
        #print(train_df)
        train_dataset = ECGDataset(train_df)
        val_dataset = ECGDataset(val_df)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size)
        test_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size)
        train_dataloaders.append(train_dataloader)
        test_dataloaders.append(test_dataloader)
    return train_dataloaders, test_dataloaders