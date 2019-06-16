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
from config import *

# torch.cuda.set_device(3)
device = torch.device('cuda')


if __name__ == '__main__':

    for fold in range(5):
        seed = 2411 + fold
        depth = 11
        maxlen = 300
        batch_size = 32
        accumulation_steps = 4

        config.seed = seed
        config.max_sequence_length = maxlen
        config.batch_size = batch_size
        config.accumulation_steps = accumulation_steps
        config.bert_weight = f"../bert_weight/uncased_L-{depth}_H-768_A-12/"
        config.features = f"../bert_features_{maxlen}/"
        config.experiment = f"{depth}layers"
        config.checkpoint = f"{config.logdir}/{config.today}/kfold/fold_{fold}/{config.experiment}_" \
                            f"{config.batch_size}bs_{config.accumulation_steps}accum_{config.seed}seed_{config.max_sequence_length}/"

        print_config(config)

        X = np.load(os.path.join(config.features, 'sequence_train.npy'))
        X_meta = np.load(os.path.join(config.features, 'meta_features_train.npy'))

        valid_indexs = np.load(f'./kfold/valid_{fold}.npy')
        X = X[valid_indexs]
        X_meta = X_meta[valid_indexs]

        model = BertForTokenClassificationMultiOutput.from_pretrained(
            config.bert_weight,
            cache_dir=None,
            num_aux_labels=config.n_aux_targets
        )

        state_dict = torch.load(os.path.join(config.checkpoint, "checkpoints/best.pth"))["model_state_dict"]
        model.load_state_dict(state_dict)
        model = model.to(device)

        model = torch.nn.DataParallel(model)

        for param in model.parameters():
            param.requires_grad = False

        model.eval()
        valid_preds = []
        valid = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.long),
                                               torch.tensor(X_meta, dtype=torch.float))

        valid_loader = torch.utils.data.DataLoader(valid, batch_size=128, shuffle=False, num_workers=8)

        with torch.no_grad():
            tk0 = tqdm(valid_loader, total=len(valid_loader))
            for i, (x_batch, added_fts) in enumerate(tk0):
                pred = model(x_batch.to(device), features=added_fts.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
                pred[:, 0] = torch.sigmoid(pred[:, 0])
                valid_preds.append(pred.detach().cpu().squeeze().numpy())

        valid_preds = np.concatenate(valid_preds, axis=0)
        os.makedirs(f'./oofs/{config.experiment}_{config.batch_size}bs_{config.accumulation_steps}accum_{config.max_sequence_length}/', exist_ok=True)
        np.save(f'./oofs/{config.experiment}_{config.batch_size}bs_{config.accumulation_steps}accum_{config.max_sequence_length}/fold_{fold}.npy', valid_preds)
