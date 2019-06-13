
class Config:

    def __init__(self):
        self.today = "190612"
        self.logdir = "/raid/bac/kaggle/logs/jigsaw/"
        self.experiment = ""
        self.max_sequence_length = 220
        self.lr = 2e-5
        self.batch_size = 32
        self.accumulation_steps = 4
        self.train_percent = 1.0
        self.valid_percent = 0.05
        self.seed = 94285
        self.epochs = 2
        self.data_dir = "/raid/data/kaggle/jigsaw/"
        self.toxicity_column = 'target'
        self.bert_weight = "../bert_weight/uncased_L-11_H-768_A-12/"
        self.gpt2_weight = "../gpt2_weight/"
        self.features = '../meta/'
        self.checkpoint = ""
        self.aux_columns = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
        self.n_aux_targets = len(self.aux_columns)
        self.identity_columns = [
            'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
            'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
        ]

        self.use_bug = False


def print_config(Config):
    print("\n")
    print("*" * 50)
    print(f"seed: {Config.seed}")
    print(f"Feature dir: {Config.features}")
    print(f"bert_weight: {Config.bert_weight}")
    print(f"gpt2_weight: {Config.gpt2_weight}")
    print(f"max_sequence_length: {Config.max_sequence_length}")
    print(f"batch_size: {Config.batch_size}")
    print(f"accumulation_steps: {Config.accumulation_steps}")
    print("\n")
    print("*" * 50)
    print(f"Today: {Config.today}")
    print(f"Experiment: {Config.experiment}")
    print(f"checkpoint: {Config.checkpoint}")


config = Config()