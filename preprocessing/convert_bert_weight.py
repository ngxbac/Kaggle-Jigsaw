import click
import shutil
import os
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch


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


if __name__ == '__main__':
    cli()
