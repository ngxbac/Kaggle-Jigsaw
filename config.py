
class Config:
    logdir = "/raid/bac/kaggle/logs/jigsaw/"
    experiment = 'test'
    max_sequence_length = 220
    lr = 2e-5
    batch_size = 34
    accumulation_steps = 8
    train_percent = 0.01
    valid_percent = 0.01

    seed = 12345
    epochs = 2
    data_dir = "/raid/data/kaggle/jigsaw/"
    toxicity_column = 'target'
    bert_weight = "../bert_weight/uncased_L-11_H-768_A-12/"
    features = '../meta/'
    checkpoint = f"{logdir}/{experiment}_{batch_size}bs_{accumulation_steps}accum_{train_percent * 100}train_{valid_percent * 100}/"

    aux_columns = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
    n_aux_targets = len(aux_columns)

    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
    ]
