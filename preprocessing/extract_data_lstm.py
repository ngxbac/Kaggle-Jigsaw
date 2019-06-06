import numpy as np
import pandas as pd
import click
import os
from tqdm import *
import pickle
from sklearn.preprocessing import StandardScaler
from keras.preprocessing import text, sequence
from pytorch_pretrained_bert import BertTokenizer
import gc
from preprocessing.meta import *
tqdm.pandas()

from nltk.tokenize.treebank import TreebankWordTokenizer
nltk_tokenizer = TreebankWordTokenizer()


isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}
remove_dict = {ord(c):f'' for c in symbols_to_delete}


@click.group()
def cli():
    print("Extract data to BERT format")


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path,'rb') as f:
        emb_arr = pickle.load(f)
    return emb_arr


def build_matrix(word_index, path, max_features):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((max_features + 1, 300))
    unknown_words = []

    for word, i in word_index.items():
        if i <= max_features:
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                try:
                    embedding_matrix[i] = embedding_index[word.lower()]
                except KeyError:
                    try:
                        embedding_matrix[i] = embedding_index[word.title()]
                    except KeyError:
                        unknown_words.append(word)
    return embedding_matrix, unknown_words


def handle_punctuation(x):
    x = x.translate(remove_dict)
    x = x.translate(isolate_dict)
    return x

def handle_contractions(x):
    x = nltk_tokenizer.tokenize(x)
    return x

def fix_quote(x):
    x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]
    x = ' '.join(x)
    return x

def remove_spaces(x):
    for sp in spaces:
        x = x.replace(sp, ' ')
    return x

def correct_contraction(x, dic):
    for word in dic.keys():
        if word in x:
            x = x.replace(word, dic[word])
    return x

def preprocess(x):
    x = handle_punctuation(x)
    x = handle_contractions(x)
    x = fix_quote(x)
    x = correct_contraction(x, contraction_mapping)
    return x


epsilon = 1e-6
def add_features(df):
    df['comment_text'] = df['comment_text'].astype(str)
    df['total_length'] = df['comment_text'].apply(len)
    df['capitals'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals']) / (float(row['total_length']) + epsilon), axis=1)
    df['num_exclamation_marks'] = df['comment_text'].apply(lambda comment: comment.count('!'))
    df['num_question_marks'] = df['comment_text'].apply(lambda comment: comment.count('?'))
    df['num_punctuation'] = df['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))
    df['num_symbols'] = df['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))
    df['num_words'] = df['comment_text'].apply(lambda comment: len(comment.split()))
    df['num_unique_words'] = df['comment_text'].apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
    df['num_smilies'] = df['comment_text'].apply(lambda comment: sum(comment.count(w) for w in good_emoji))
    return df


@cli.command()
@click.option('--data_dir', type=str)
@click.option('--output_path', type=str)
@click.option('--crawl_embedding_path', type=str)
@click.option('--glove_embedding_path', type=str)
def extract_data(
    data_dir,
    output_path,
    crawl_embedding_path,
    glove_embedding_path
):
    os.makedirs(output_path, exist_ok=True)

    # Load csv datasets
    print("Load dataset...")
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    print("Remove spaces...")
    train['comment_text'] = train['comment_text'].astype(str).apply(lambda x: remove_spaces(x))
    test['comment_text'] = test['comment_text'].astype(str).apply(lambda x: remove_spaces(x))

    print("Add features...")
    # Add meta features
    train = add_features(train)
    test = add_features(test)

    feature_cols = [
        'total_length', 'capitals', 'caps_vs_length', 'num_exclamation_marks', 'num_question_marks',
        'num_punctuation', 'num_words', 'num_unique_words', 'words_vs_unique', 'num_smilies', 'num_symbols'
    ]
    aux_cols = [
        'target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat'
    ]

    # Meta features
    train_features = train[feature_cols].fillna(0)
    test_features = test[feature_cols].fillna(0)

    ss = StandardScaler()
    ss.fit(train_features)
    train_features = np.array(ss.transform(train_features), dtype=np.float32)
    test_features = np.array(ss.transform(test_features), dtype=np.float32)

    print("Save features...")
    # Save meta features
    np.save(f'{output_path}/meta_features_train.npy', train_features)
    np.save(f'{output_path}/meta_features_test.npy', test_features)

    # Save train ss for test
    with open(os.path.join(output_path, f'ss_train.pkl'), 'wb') as f:
        pickle.dump(ss, f)

    # Processing text
    print("Preprocessing text...")
    x_train = train['comment_text'].progress_apply(lambda x: preprocess(x))
    x_test = test['comment_text'].progress_apply(lambda x: preprocess(x))

    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
    ]
    # Overall
    weights = np.ones((len(x_train),)) / 4
    # Subgroup
    weights += (train[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 4
    # Background Positive, Subgroup Negative
    weights += (((train['target'].values >= 0.5).astype(bool).astype(np.int) +
                 (train[identity_columns].fillna(0).values < 0.5).sum(axis=1).astype(bool).astype(
                     np.int)) > 1).astype(bool).astype(np.int) / 4
    # Background Negative, Subgroup Positive
    weights += (((train['target'].values < 0.5).astype(bool).astype(np.int) +
                 (train[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(
                     np.int)) > 1).astype(bool).astype(np.int) / 4
    loss_weight = 1.0 / weights.mean()
    np.save(os.path.join(output_path, f'loss_weight.npy'), loss_weight)

    # Stack y_train with weights
    y_train_aux = train[aux_cols].values
    y_train = np.vstack([(train['target'].values >= 0.5).astype(np.int), weights]).T
    # Save y and y_aux
    print("Save target and aux target...")
    np.save(os.path.join(output_path, f'y_train_aux.npy'), y_train_aux)
    np.save(os.path.join(output_path, f'y_train.npy'), y_train)

    max_features = None
    tokenizer = text.Tokenizer(num_words=max_features, filters='', lower=False)

    tokenizer.fit_on_texts(list(x_train))
    max_features = max_features or len(tokenizer.word_index) + 1
    print(max_features)

    print("Get embedding...")
    crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, crawl_embedding_path, max_features)
    print('n unknown words (crawl): ', len(unknown_words_crawl))

    glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, glove_embedding_path, max_features)
    print('n unknown words (glove): ', len(unknown_words_glove))

    embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)

    print("Dump embedding...")
    pickle.dump(glove_matrix, open(f'{output_path}/glove.pkl', 'wb'), -1)
    pickle.dump(crawl_matrix, open(f'{output_path}/crawl.pkl', 'wb'), -1)
    pickle.dump(embedding_matrix, open(f'{output_path}/embedding_matrix.pkl', 'wb'), -1)

    del crawl_matrix
    del glove_matrix
    gc.collect()

    print("Text to sequences...")
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    train_lengths = np.array([len(x) for x in x_train])
    test_lengths = np.array([len(x) for x in x_test])

    maxlen = 300
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    print("Save train and test embedding...")

    np.save(os.path.join(output_path, f'X_train.npy'), x_train)
    np.save(os.path.join(output_path, f'X_test.npy'), x_test)

    np.save(os.path.join(output_path, f'train_lengths.npy'), train_lengths)
    np.save(os.path.join(output_path, f'test_lengths.npy'), test_lengths)


if __name__ == '__main__':
    cli()