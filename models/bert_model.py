import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertForTokenClassification, BertModel


class BertForTokenClassificationMultiOutput(BertPreTrainedModel):
    def __init__(self, config, num_aux_labels):
        super(BertForTokenClassificationMultiOutput, self).__init__(config)
        self.num_labels = num_aux_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        added_hidden_size = 64
        # self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.added_linear = nn.Linear(11, added_hidden_size)
        self.added_dropout = nn.Dropout(0.1)

        self.out = nn.Linear(config.hidden_size + added_hidden_size, 1)
        self.aux_out = nn.Linear(config.hidden_size + added_hidden_size, num_aux_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, features, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)

        # branch1 = F.relu(self.linear1(pooled_output))
        # output1 = pooled_output + branch1
        added_fts = self.added_dropout(F.relu(self.added_linear(features)))

        output1 = torch.cat([pooled_output, added_fts], 1)

        out = self.out(output1)
        aux_out = self.aux_out(output1)

        return torch.cat([out, aux_out], 1)


# class BertForTokenClassificationMultiOutput(BertPreTrainedModel):
#     def __init__(self, config, num_aux_labels):
#         super(BertForTokenClassificationMultiOutput, self).__init__(config)
#         self.num_labels = num_aux_labels
#         self.bert = BertModel(config)
#
#         added_hidden_size = 256
#         # self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
#         self.added_linear = nn.Linear(11, added_hidden_size)
#         self.added_dropout = nn.Dropout(0.1)
#
#         n_dense_units = 2 * config.hidden_size + added_hidden_size
#         self.linear1 = nn.Linear(n_dense_units, n_dense_units)
#         self.dropout = nn.Dropout(0.4)
#
#         self.out = nn.Linear(n_dense_units, 1)
#         self.aux_out = nn.Linear(n_dense_units, num_aux_labels)
#
#         self.apply(self.init_bert_weights)
#
#     def forward(self, input_ids, features, token_type_ids=None, attention_mask=None, labels=None):
#         seq, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#
#         # global average pooling
#         avg_pool = torch.mean(seq, 1)
#         # global max pooling
#         max_pool, _ = torch.max(seq, 1)
#
#         # branch1 = F.relu(self.linear1(pooled_output))
#         # output1 = pooled_output + branch1
#         add_ft_branches = self.added_dropout(F.relu(self.added_linear(features)))
#
#         h_conc = torch.cat((max_pool, avg_pool, add_ft_branches), 1)
#         h_conc_linear1 = F.relu(self.linear1(h_conc))
#
#         output1 = self.dropout(h_conc_linear1)
#
#         out = self.out(output1)
#         aux_out = self.aux_out(output1)
#
#         return torch.cat([out, aux_out], 1)
