from models import NeuralNet
from pytorch_pretrained_bert import BertAdam
from torch.optim import Adam
from dataset import get_data_lstm_loaders
import torch
from losses import CustomLoss
from callbacks import *
from runner import ModelRunner
import pickle

np.random.seed(ConfigLSTM.seed)
torch.manual_seed(ConfigLSTM.seed)
torch.cuda.manual_seed(ConfigLSTM.seed)
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # Data loaders
    train_loader, valid_loader, valid_df, loss_weight = get_data_lstm_loaders(ConfigLSTM)
    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    # Load embedding
    with open(f'{ConfigLSTM.features}/embedding_matrix.pkl', 'rb') as f:
        embedding_matrix = pickle.load(f)

    # Criterion
    criterion = CustomLoss(loss_weight)

    # Model and optimizer
    model = NeuralNet(
        embedding_matrix=embedding_matrix,
        num_aux_targets=ConfigLSTM.n_aux_targets,
        max_features=ConfigLSTM.max_sequence_length,
        LSTM_UNITS=ConfigLSTM.LSTM_UNITS,
        DENSE_HIDDEN_UNITS=ConfigLSTM.DENSE_HIDDEN_UNITS
    )

    params = model.parameters()

    master_params = [p for p in params if p.requires_grad]
    optimizer = Adam(master_params, lr=ConfigLSTM.lr, weight_decay=0.0001)
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
        # fp16={"opt_level": "O1"},
        callbacks=[auc_callback]
    )
