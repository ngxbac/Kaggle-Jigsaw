from models import BertForTokenClassificationMultiOutput
from pytorch_pretrained_bert import BertAdam
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from utils import *
import gc

from dataset import get_data_loaders
import torch
from losses import CustomLoss
from callbacks import *
from runner import ModelRunner

np.random.seed(Config.seed)
torch.manual_seed(Config.seed)
torch.cuda.manual_seed(Config.seed)
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # Data loaders
    train_loader, valid_loader, valid_df, loss_weight = get_data_loaders(Config)
    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    # Criterion
    criterion = CustomLoss(loss_weight)

    # Model and optimizer
    model = BertForTokenClassificationMultiOutput.from_pretrained(
        Config.working_dir,
        cache_dir=None,
        num_aux_labels=Config.n_aux_targets
    )
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_optimization_steps = int(Config.epochs * len(train_loader.dataset) / Config.batch_size / Config.accumulation_steps)
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=Config.lr,
        warmup=0.1,
        t_total=num_train_optimization_steps
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    identity_valid = valid_df[Config.identity_columns].copy()
    target_valid = valid_df.target.values
    auc_callback = AucCallback(
        identity=identity_valid,
        target=target_valid
    )

    # model runner
    runner = ModelRunner()

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        main_metric='auc',
        minimize_metric=False,
        logdir=Config.checkpoint,
        num_epochs=Config.epochs,
        verbose=True,
        fp16={"opt_level": "O1"},
        callbacks=[auc_callback]
    )
