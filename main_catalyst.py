from models import BertForTokenClassificationMultiOutput, GPT2ClassificationMultioutput
from pytorch_pretrained_bert import BertAdam, OpenAIAdam
import click

from dataset import get_data_loaders
from losses import CustomLoss
from callbacks import *
from runner import ModelRunner

from config import *


@click.group()
def cli():

    print("Training bert")


@cli.command()
@click.option('--seed', type=int)
@click.option('--depth', type=int)
@click.option('--maxlen', type=int)
@click.option('--batch_size', type=int)
@click.option('--accumulation_steps', type=int)
@click.option('--model_name', type=str, default='bert')
def train(
    seed,
    depth,
    maxlen,
    batch_size,
    accumulation_steps,
    model_name
):

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

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    # Data loaders
    train_loader, valid_loader, valid_df, loss_weight = get_data_loaders(config)
    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    # Criterion
    criterion = CustomLoss(loss_weight)

    # Model and optimizer
    if model_name == 'bert':
        print("BERT MODEL")
        model = BertForTokenClassificationMultiOutput.from_pretrained(
            config.bert_weight,
            cache_dir=None,
            num_aux_labels=config.n_aux_targets
        )

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        num_train_optimization_steps = np.ceil(
            len(train_loader.dataset) / config.batch_size / config.accumulation_steps) * config.epochs
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=config.lr,
            warmup=0.01,
            t_total=num_train_optimization_steps
        )

    elif model_name == 'gpt2':
        print("GPT2 MODEL")
        model = GPT2ClassificationMultioutput.from_pretrained(
            config.gpt2_weight,
            cache_dir=None,
            num_aux_labels=config.n_aux_targets
        )

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        num_train_optimization_steps = np.ceil(
            len(train_loader.dataset) / config.batch_size / config.accumulation_steps) * config.epochs
        optimizer = OpenAIAdam(
            optimizer_grouped_parameters,
            lr=config.lr,
            warmup=0.01,
            t_total=num_train_optimization_steps
        )
    else:
        raise ("Model is not implemented")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    identity_valid = valid_df[config.identity_columns].copy()
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
        logdir=config.checkpoint,
        num_epochs=config.epochs,
        verbose=True,
        fp16={"opt_level": "O1"},
        callbacks=[auc_callback]
    )


if __name__ == '__main__':
    cli()

