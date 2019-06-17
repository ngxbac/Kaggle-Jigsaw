import pandas as pd
import numpy as np
import os
import gc
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
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


class LenMatchBatchSampler(BatchSampler):
    def __iter__(self):

        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            # import pdb
            # pdb.set_trace()
            count_zeros = np.sum(self.sampler.data_source[idx]["X"] == 0)
            count_zeros = int(count_zeros / 128)
            if len(buckets[count_zeros]) == 0:  buckets[count_zeros] = []

            buckets[count_zeros].append(idx)

            if len(buckets[count_zeros]) == self.batch_size:
                batch = list(buckets[count_zeros])
                yield batch
                yielded += 1
                buckets[count_zeros] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

        assert len(self) == yielded, "produced an inccorect number of batches. expected %i, but yielded %i" %(len(self), yielded)


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

    np.random.seed(Config.seed)
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

    ran_sampler = RandomSampler(train_dataset)
    len_sampler = LenMatchBatchSampler(ran_sampler, batch_size=Config.batch_size, drop_last=False)

    train_loader = DataLoader(
        train_dataset,
        # batch_sampler=len_sampler,
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

    ran_sampler = RandomSampler(valid_dataset)
    len_sampler = LenMatchBatchSampler(ran_sampler, batch_size=Config.batch_size, drop_last=False)

    valid_loader = DataLoader(
        valid_dataset,
        # batch_sampler=len_sampler,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, valid_loader, valid_df, loss_weight


def get_kfold_data_loaders(Config):
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

    np.random.seed(Config.seed)

    train_indexs = np.load(f'./kfold/train_{Config.fold}.npy')
    X_train = X[train_indexs]
    X_train_meta = X_meta[train_indexs]
    y_train = y[train_indexs]
    y_train_aux = y_aux[train_indexs]

    valid_indexs = np.load(f'./kfold/valid_{Config.fold}.npy')
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
        num_workers=8
    )

    return train_loader, valid_loader, valid_df, loss_weight

