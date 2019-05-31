import pandas as pd
import numpy as np
import os
import click

import torch
import torch.nn as nn
from apex import amp
from tqdm import *

from config import Config
from models import BertForTokenClassificationMultiOutput
from pytorch_pretrained_bert import BertAdam
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from utils import *

# torch.cuda.set_device(3)
device = torch.device('cuda')


def custom_loss_BCE(data, targets, loss_weight):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:,1:2])(data[:,:1],targets[:,:1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:,1:],targets[:,2:])
    return (bce_loss_1 * loss_weight) + bce_loss_2


def main():

    # Load data
    X = np.load(os.path.join(Config.working_dir, 'sequence_train.npy'))
    X_meta = np.load(os.path.join(Config.working_dir, 'meta_features_train.npy'))
    y = np.load(os.path.join(Config.working_dir, 'y_train.npy'))
    y_aux = np.load(os.path.join(Config.working_dir, 'y_train_aux.npy'))
    loss_weight = np.load(os.path.join(Config.working_dir, 'loss_weight.npy'))
    loss_weight = float(loss_weight)

    df = pd.read_csv(os.path.join(Config.data_dir, 'train.csv'))
    print(loss_weight)

    np.random.seed(10)
    indexs = np.random.permutation(X.shape[0])
    n_train = int(0.95 * len(indexs))
    n_valid = len(indexs) - n_train

    train_indexs = indexs[:n_train]
    X_train = X[train_indexs]
    X_train_meta = X_meta[train_indexs]
    y_train = y[train_indexs]
    y_train_aux = y_aux[train_indexs]

    valid_indexs = indexs[n_train:]
    X_valid = X[valid_indexs]
    X_valid_meta = X_meta[valid_indexs]
    y_valid = y[valid_indexs]
    y_valid_aux = y_aux[valid_indexs]
    valid_df = df.iloc[valid_indexs]

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.long),
        torch.tensor(X_train_meta, dtype=torch.float),
        torch.tensor(np.hstack([y_train, y_train_aux]), dtype=torch.float)
    )

    np.random.seed(Config.seed)
    torch.manual_seed(Config.seed)
    torch.cuda.manual_seed(Config.seed)
    torch.backends.cudnn.deterministic = True

    model = BertForTokenClassificationMultiOutput.from_pretrained(
        Config.working_dir,
        cache_dir=None,
        num_aux_labels=Config.n_aux_targets
    )

    model.zero_grad()
    model = model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(Config.epochs * len(train_dataset) / Config.batch_size / Config.accumulation_steps)

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=Config.lr,
                         warmup=0.1,
                         t_total=num_train_optimization_steps)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    model = nn.DataParallel(model)
    # model = model.train()

    tq = tqdm(range(Config.epochs))
    for epoch in tq:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
        avg_loss = 0.
        avg_accuracy = 0.
        lossf = None
        tk0 = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for i, (x_batch, added_fts, y_batch) in tk0:
            optimizer.zero_grad()
            all_y_pred = model(
                x_batch.to(device),
                f=added_fts.to(device),
                attention_mask=(x_batch > 0).to(device),
                labels=None
            )
            y_pred = all_y_pred[:, 0]

            loss = custom_loss_BCE(all_y_pred, y_batch.to(device), loss_weight)
            loss /= Config.accumulation_steps

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            if (i + 1) % Config.accumulation_steps == 0:  # Wait for several backward steps
                optimizer.step()  # Now we can do an optimizer step
                optimizer.zero_grad()
            if lossf:
                lossf = 0.98 * lossf + 0.02 * loss.item()
            else:
                lossf = loss.item()
            tk0.set_postfix(loss=lossf)

            avg_loss += loss.item() / len(train_loader)

            avg_accuracy += torch.mean(
                ((torch.sigmoid(y_pred) > 0.5) == (y_batch[:, 0] > 0.5).to(device)).to(torch.float)).item() / len(
                train_loader)

        tq.set_postfix(avg_loss=avg_loss, avg_accuracy=avg_accuracy)

        os.makedirs(Config.checkpoint, exist_ok=True)
        output_model_file = os.path.join(Config.checkpoint, f"11layer_features_{epoch}.bin")
        torch.save(model.state_dict(), output_model_file)

    # model.load_state_dict(torch.load(output_model_file))

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    valid_preds = np.zeros((len(X_valid)))
    valid = torch.utils.data.TensorDataset(torch.tensor(X_valid, dtype=torch.long),
                                           torch.tensor(X_valid_meta, dtype=torch.float))

    valid_loader = torch.utils.data.DataLoader(valid, batch_size=32, shuffle=False, num_workers=4)

    tk0 = tqdm(valid_loader, total=len(valid_loader))
    for i, (x_batch, added_fts) in enumerate(tk0):
        pred = model(x_batch.to(device), f=added_fts.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
        valid_preds[i * 32: (i + 1) * 32] = pred[:, 0].detach().cpu().squeeze().numpy()

    np.save('valid_pred.npy', valid_preds)

    MODEL_NAME = 'quora_multitarget'
    identity_valid = valid_df[Config.identity_columns].copy()
    predict_valid = torch.sigmoid(torch.tensor(valid_preds)).numpy()
    total_score = scoring_valid(predict_valid, identity_valid, y_valid[:, 0],  model_name=MODEL_NAME, save_output = True)
    print(total_score)


if __name__ == '__main__':
    main()