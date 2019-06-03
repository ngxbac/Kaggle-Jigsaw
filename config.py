
class Config:
    logdir = "/raid/bac/kaggle/logs/jigsaw/"
    experiment = 'openai_adam'
    max_sequence_length = 220
    lr = 2e-5
    batch_size = 64
    accumulation_steps = 1
    train_percent = 0.95
    valid_percent = 0.05

    seed = 12345
    epochs = 2
    data_dir = "/raid/data/kaggle/jigsaw/"
    toxicity_column = 'target'
    working_dir = '../meta/'
    checkpoint = f"{logdir}/{experiment}_{epochs}epoch_{train_percent * 100}train_{valid_percent * 100}/"

    aux_columns = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
    n_aux_targets = len(aux_columns)

    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
    ]
