from models import *
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
    elif model_name == 'gpt2':
        config.features = f"../features_{maxlen}_gpt/"
    else:
        config.features = f"../features_{maxlen}_xlnet/"
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
        model = BertForTokenClassificationMultiOutput2.from_pretrained(
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
    elif model_name == 'xlnet':
        model = XLNetWithMultiOutput.from_pretrained(
            config.xlnet_weight,
            clf_dropout=0.4, n_class=6
            # num_aux_labels=config.n_aux_targets
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
    model = model.cuda()

    from apex import amp
    model, optimizer = amp.initialize(
        model, optimizer, opt_level="O1")

    # if distributed_rank > -1:
    # from apex.parallel import DistributedDataParallel
    # model = DistributedDataParallel(model)
    model = torch.nn.DataParallel(model)

    if config.resume:
        checkpoint = torch.load(config.checkpoint + "/checkpoints/best.pth")
        import pdb
        pdb.set_trace()
        new_state_dict = {}
        old_state_dict = checkpoint['model_state_dict']
        for k, v in old_state_dict.items():
            new_state_dict["module." + k] = v
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        criterion.load_state_dict(checkpoint['criterion_state_dict'])
        print("!!! Loaded checkpoint ", config.checkpoint + "/checkpoints/best.pth")

    identity_valid = valid_df[config.identity_columns].copy()
    target_valid = valid_df.target.values
    auc_callback = AucCallback(
        identity=identity_valid,
        target=target_valid
    )

    checkpoint_callback = IterationCheckpointCallback(
        save_n_last=2000,
        num_iters=10000,
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
        callbacks=[auc_callback, checkpoint_callback]
    )


if __name__ == '__main__':
    cli()

