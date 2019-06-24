import click
import shutil
import os
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import convert_xlnet_checkpoint_to_pytorch


@click.group()
def cli():
    print("Convert BERT weight from tensorflow to pytorch")


@cli.command()
@click.option('--model_path', type=str)
@click.option('--output_path', type=str)
def convert_tf_to_pytorch(
    model_path,
    output_path
):
    os.makedirs(output_path, exist_ok=True)
    convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
        os.path.join(model_path, 'bert_model.ckpt'),
        os.path.join(model_path, 'bert_config.json'),
        os.path.join(output_path, 'pytorch_model.bin')
    )

    shutil.copyfile(
        os.path.join(model_path, 'bert_config.json'),
        os.path.join(output_path, 'bert_config.json')
    )


@cli.command()
@click.option('--model_path', type=str)
@click.option('--output_path', type=str)
def convert_xl_to_pytorch(
    model_path,
    output_path
):
    os.makedirs(output_path, exist_ok=True)
    convert_xlnet_checkpoint_to_pytorch.convert_xlnet_checkpoint_to_pytorch(
        os.path.join(model_path, 'xlnet_model.ckpt'),
        os.path.join(model_path, 'xlnet_config.json'),
        output_path
    )

    CONFIG = {
        "attn_type": "bi",
        "bi_data": False,
        "clamp_len": -1,
        "d_head": 64,
        "d_inner": 4096,
        "d_model": 1024,
        "dropatt": 0.1,
        "dropout": 0.1,
        "ff_activation": "gelu",
        "init": "normal",
        "init_range": 0.1,
        "init_std": 0.02,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "mem_len": None,
        "n_head": 16,
        "n_layer": 24,
        "n_token": 32000,
        "reuse_len": None,
        "same_length": True,
        "untie_r": True
    }
    import json
    with open(f"{output_path}/xlnet_config.json", "w") as f:
        json.dump(CONFIG, f)

    # shutil.copyfile(
    #     os.path.join(model_path, 'bert_config.json'),
    #     os.path.join(output_path, 'bert_config.json')
    # )


if __name__ == '__main__':
    cli()
