import numpy as np
import pandas as pd
import click
import os
from tqdm import *
from pytorch_pretrained_bert import BertTokenizer


@click.group()
def cli():
    print("Extract data to BERT format")


def convert_lines(example, max_seq_length, tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)


@cli.command()
@click.option('--model_path', type=str)
@click.option('--csv_file', type=str)
@click.option('--dataset', type=str)
@click.option('--max_sequence_length', type=int)
@click.option('--output_path', type=str)
def extract_data(
    model_path,
    csv_file,
    dataset,
    max_sequence_length,
    output_path,
):
    os.makedirs(output_path, exist_ok=True)
    df = pd.read_csv(csv_file)
    tokenizer = BertTokenizer.from_pretrained(model_path, cache_dir=None, do_lower_case=True)

    # Make sure all comment_text values are strings
    df['comment_text'] = df['comment_text'].astype(str)

    sequences = convert_lines(df["comment_text"].fillna("DUMMY_VALUE"), max_sequence_length, tokenizer)
    np.save(os.path.join(output_path, f'sequence_{dataset}.npy'), sequences)


if __name__ == '__main__':
    cli()
