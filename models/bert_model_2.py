import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertForTokenClassification, BertModel


class BertForTokenClassificationMultiOutput2(BertPreTrainedModel):
    def __init__(self, config, num_aux_labels):
        super(BertForTokenClassificationMultiOutput2, self).__init__(config)
        self.num_labels = num_aux_labels
        self.bert = BertModel(config)

        added_hidden_size = 256
        # self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.added_linear = nn.Linear(11, added_hidden_size)
        self.added_dropout = nn.Dropout(0.1)

        n_dense_units = config.hidden_size + added_hidden_size
        self.linear1 = nn.Linear(n_dense_units, n_dense_units)
        self.dropout = nn.Dropout(0.4)

        self.out = nn.Linear(n_dense_units, 1)
        self.aux_out = nn.Linear(n_dense_units, num_aux_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, features, token_type_ids=None, attention_mask=None, labels=None):
        seq, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)

        h12 = seq[-1][:, 0].reshape((-1, 1, 768))
        h11 = seq[-2][:, 0].reshape((-1, 1, 768))
        h10 = seq[-3][:, 0].reshape((-1, 1, 768))
        h9 = seq[-4][:, 0].reshape((-1, 1, 768))

        all_h = torch.cat([h9, h10, h11, h12], 1)
        # print(all_h.shape)

        max_pool = torch.mean(all_h, 1)
        # print(max_pool.shape)

        # branch1 = F.relu(self.linear1(pooled_output))
        # output1 = pooled_output + branch1
        add_ft_branches = self.added_dropout(F.relu(self.added_linear(features)))

        h_conc = torch.cat((max_pool, add_ft_branches), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))

        output1 = self.dropout(h_conc_linear1)

        out = self.out(output1)
        aux_out = self.aux_out(output1)

        return torch.cat([out, aux_out], 1)