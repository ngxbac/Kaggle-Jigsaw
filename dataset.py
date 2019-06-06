import pandas as pd
import numpy as np
import os
import gc
from torch.utils.data import Dataset, DataLoader
import torch


class JigsawDataset(Dataset):
    def __init__(self, X, X_meta, y, y_aux):
        self.X, self.X_meta, self.y, self.y_aux = X, X_meta, y, y_aux
        self.X_weight = self.y[:, 1:2]
        self.y = self.y[:, :1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.long)
        x_weight = self.X_weight[idx].astype(np.float32)
        x_meta = self.X_meta[idx].astype(np.float32)
        y = self.y[idx].astype(np.float32)
        y_aux = self.y_aux[idx].astype(np.float32)

        return {
            "X": x,
            "X_weight": x_weight,
            "X_meta": x_meta,
            "y": y,
            "y_aux": y_aux
        }


class JigsawDatasetLSTM(Dataset):
    def __init__(self, X, X_meta, X_length, y, y_aux):
        self.X, self.X_meta, self.y, self.y_aux = X, X_meta, y, y_aux
        self.X_weight = self.y[:, 1:2]
        self.y = self.y[:, :1]
        self.X_length = X_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.long)
        x_weight = self.X_weight[idx].astype(np.float32)
        x_meta = self.X_meta[idx].astype(np.float32)
        x_length = self.X_length[idx].astype(np.float32)
        y = self.y[idx].astype(np.float32)
        y_aux = self.y_aux[idx].astype(np.float32)

        return {
            "X": x,
            "X_weight": x_weight,
            "X_meta": x_meta,
            "length": x_length,
            "y": y,
            "y_aux": y_aux
        }



def get_data_loaders(Config):
    # Load data
    X = np.load(os.path.join(Config.features, 'sequence_train.npy'))
    X_meta = np.load(os.path.join(Config.features, 'meta_features_train.npy'))
    y = np.load(os.path.join(Config.features, 'y_train.npy'))
    y_aux = np.load(os.path.join(Config.features, 'y_train_aux.npy'))
    loss_weight = np.load(os.path.join(Config.features, 'loss_weight.npy'))
    loss_weight = float(loss_weight)

    # print(y.shape)
    # print(y_aux.shape)

    df = pd.read_csv(os.path.join(Config.data_dir, 'train.csv'))

    np.random.seed(10)
    indexs = np.random.permutation(X.shape[0])
    n_train = int(Config.train_percent * len(indexs))
    n_valid = int(Config.valid_percent * len(indexs))

    train_indexs = indexs[:n_train]
    X_train = X[train_indexs]
    X_train_meta = X_meta[train_indexs]
    y_train = y[train_indexs]
    y_train_aux = y_aux[train_indexs]

    valid_indexs = indexs[-n_valid:]
    X_valid = X[valid_indexs]
    X_valid_meta = X_meta[valid_indexs]
    y_valid = y[valid_indexs]
    y_valid_aux = y_aux[valid_indexs]
    valid_df = df.iloc[valid_indexs]

    del X, X_meta, y, y_aux
    gc.collect()

    train_dataset = JigsawDataset(
        X=X_train,
        X_meta=X_train_meta,
        y=y_train,
        y_aux=y_train_aux
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=8
    )

    valid_dataset = JigsawDataset(
        X=X_valid,
        X_meta=X_valid_meta,
        y=y_valid,
        y_aux=y_valid_aux
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )

    return train_loader, valid_loader, valid_df, loss_weight


from torch.utils.data.dataloader import default_collate
class SequenceBucketCollator():
    def __init__(self, choose_length, maxlen):
        self.choose_length = choose_length
        self.maxlen = maxlen

    def __call__(self, batch):
        batch = default_collate(batch)
        lengths = batch['length']
        length = self.choose_length(lengths).long()
        mask = torch.arange(start=self.maxlen, end=0, step=-1) < length
        batch['X'] = batch['X'][:, mask]
        return batch


def get_data_lstm_loaders(Config):
    # Load data
    X = np.load(os.path.join(Config.features, 'X_train.npy'))
    X_meta = np.load(os.path.join(Config.features, 'meta_features_train.npy'))
    lengths = np.load(os.path.join(Config.features, 'train_lengths.npy'))
    y = np.load(os.path.join(Config.features, 'y_train.npy'))
    y_aux = np.load(os.path.join(Config.features, 'y_train_aux.npy'))
    loss_weight = np.load(os.path.join(Config.features, 'loss_weight.npy'))
    loss_weight = float(loss_weight)

    df = pd.read_csv(os.path.join(Config.data_dir, 'train.csv'))

    np.random.seed(10)
    indexs = np.random.permutation(X.shape[0])
    n_train = int(Config.train_percent * len(indexs))
    n_valid = int(Config.valid_percent * len(indexs))

    train_indexs = indexs[:n_train]
    X_train = X[train_indexs]
    X_train_meta = X_meta[train_indexs]
    X_train_length = lengths[train_indexs]
    y_train = y[train_indexs]
    y_train_aux = y_aux[train_indexs]

    valid_indexs = indexs[-n_valid:]
    X_valid = X[valid_indexs]
    X_valid_meta = X_meta[valid_indexs]
    X_valid_length = lengths[valid_indexs]
    y_valid = y[valid_indexs]
    y_valid_aux = y_aux[valid_indexs]
    valid_df = df.iloc[valid_indexs]

    del X, X_meta, y, y_aux
    gc.collect()

    train_collate = SequenceBucketCollator(
        choose_length=lambda lenghts: lenghts.max(),
        maxlen=Config.max_sequence_length
    )

    train_dataset = JigsawDatasetLSTM(
        X=X_train,
        X_meta=X_train_meta,
        X_length=X_train_length,
        y=y_train,
        y_aux=y_train_aux
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=train_collate
    )

    valid_dataset = JigsawDatasetLSTM(
        X=X_valid,
        X_meta=X_valid_meta,
        X_length=X_valid_length,
        y=y_valid,
        y_aux=y_valid_aux
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=train_collate
    )

    return train_loader, valid_loader, valid_df, loss_weight
