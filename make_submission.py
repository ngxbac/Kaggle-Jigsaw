import pandas as pd
import numpy as np
import os
import click

import torch
import torch.nn as nn
from apex import amp
from tqdm import *

from config import Config
from models import BertForTokenClassificationMultiOutput, GPT2ClassificationMultioutput
from pytorch_pretrained_bert import BertAdam
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from utils import *
from config import *

# torch.cuda.set_device(3)
device = torch.device('cuda')


if __name__ == '__main__':

    seed = 6037
    depth = 24
    maxlen = 300
    batch_size = 32
    accumulation_steps = 4
    model_name = "bert"

    config.seed = seed
    config.max_sequence_length = maxlen
    config.batch_size = batch_size
    config.accumulation_steps = accumulation_steps
    if depth != 24:
        config.bert_weight = f"../bert_weight/uncased_L-{depth}_H-768_A-12/"
    else:
        config.bert_weight = f"../bert_weight/uncased_L-{depth}_H-1024_A-16/"
    if model_name == 'bert':
        config.features = f"../bert_features_{maxlen}/"
    else:
        config.features = f"../features_{maxlen}_gpt/"
    config.experiment = f"{depth}layers"
    config.checkpoint = f"{config.logdir}/{config.today}/{model_name}_{config.experiment}_" \
                        f"{config.batch_size}bs_{config.accumulation_steps}accum_{config.seed}seed_{config.max_sequence_length}/"

    print_config(config)

    X = np.load(os.path.join(config.features, 'sequence_test.npy'))
    X_meta = np.load(os.path.join(config.features, 'meta_features_test.npy'))
    test_df = pd.read_csv(os.path.join(config.data_dir, "test.csv"))

    # Model and optimizer
    if model_name == 'bert':
        print("BERT MODEL")
        model = BertForTokenClassificationMultiOutput.from_pretrained(
            config.bert_weight,
            cache_dir=None,
            num_aux_labels=config.n_aux_targets
        )
    elif model_name == 'gpt2':
        print("GPT2 MODEL")
        model = GPT2ClassificationMultioutput.from_pretrained(
            config.gpt2_weight,
            cache_dir=None,
            num_aux_labels=config.n_aux_targets
        )
    else:
        raise ("Model is not implemented")

    state_dict = torch.load(os.path.join(config.checkpoint, "checkpoints/best.pth"))["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model = model.to(device)

    model = torch.nn.DataParallel(model)

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
    os.makedirs(f'./submission/{config.today}/', exist_ok=True)
    submission.to_csv(f'./submission/{config.today}/{model_name}_{config.experiment}_{config.batch_size}bs_{config.accumulation_steps}accum_{config.seed}seed_{config.max_sequence_length}.csv', index=False)
