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


if __name__ == '__main__':
    X = np.load(os.path.join(Config.working_dir, 'sequence_test.npy'))
    X_meta = np.load(os.path.join(Config.working_dir, 'meta_features_test.npy'))
    test_df = pd.read_csv(os.path.join(Config.data_dir, "test.csv"))

    model = BertForTokenClassificationMultiOutput.from_pretrained(
        Config.working_dir,
        cache_dir=None,
        num_aux_labels=Config.n_aux_targets
    )

    state_dict = torch.load(os.path.join(Config.checkpoint, "checkpoints/best.pth"))["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    valid_preds = np.zeros((len(X)))
    valid = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.long),
                                           torch.tensor(X_meta, dtype=torch.float))

    valid_loader = torch.utils.data.DataLoader(valid, batch_size=32, shuffle=False, num_workers=4)

    tk0 = tqdm(valid_loader, total=len(valid_loader))
    for i, (x_batch, added_fts) in enumerate(tk0):
        pred = model(x_batch.to(device), features=added_fts.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
        valid_preds[i * 32: (i + 1) * 32] = pred[:, 0].detach().cpu().squeeze().numpy()

    valid_preds = torch.sigmoid(torch.tensor(valid_preds)).numpy().ravel()
    submission = pd.DataFrame.from_dict({
        'id': test_df['id'],
        'prediction': valid_preds
    })
    submission.to_csv('submission_2epoch_openai_adam.csv', index=False)
