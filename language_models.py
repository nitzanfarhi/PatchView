from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaModel, RobertaTokenizer


class PoolerClassificationHead(RobertaClassificationHead):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, pooler_type="cls", pooler_dropout=0.3):
        super().__init__(config)
        self.pooler_type = pooler_type
        self.dropout = nn.Dropout(p=pooler_dropout)

    def forward(self, features, attention_mask=None):
        if self.pooler_type == "cls":
            features = features[:, 0]
        elif self.pooler_type == "avg" and attention_mask is not None:
            features = (features * attention_mask.unsqueeze(-1)
                        ).sum(axis=-2) / attention_mask.sum(axis=-1).unsqueeze(-1)
        else:
            raise NotImplementedError
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaClassificationModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(RobertaClassificationModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.num_labels = config.num_labels
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args

    def forward(self, input_ids=None, labels=None):

        attention_mask = input_ids.ne(1)
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class CustomRoberta(nn.Module):
    def __init__(self, model, args):
        super(CustomRoberta, self).__init__()
        self.encoder = model
        # New layers:
        self.linear1 = nn.Linear(768, 256)
        # 3 is the number of classes in this example
        self.linear2 = nn.Linear(256, 2)
        self.args = args

    def forward(self, input_ids=None, labels=None):
        attention_mask = input_ids.ne(1)

        res = self.encoder(input_ids, attention_mask=attention_mask)

        # extract the 1st token's embeddings
        linear1_output = self.linear1(res[0][:, 0, :].view(-1, 768))
        linear2_output = self.linear2(linear1_output)
        return linear2_output


class RobertaClass(torch.nn.Module):
    def __init__(self, l1, args):
        super(RobertaClass, self).__init__()
        self.encoder = l1
        self.hidden_size = self.encoder.config.hidden_size
        self.pre_classifier = torch.nn.Linear(
            self.hidden_size, self.hidden_size)
        self.dropout = torch.nn.Dropout(args.dropout)
        self.relu = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(self.hidden_size, 2)
        self.args = args

    def forward(self, input_ids, labels=None):
        attention_mask = input_ids.ne(1)
        output_1 = self.encoder(input_ids=input_ids,
                                attention_mask=attention_mask)

        if self.args.pooler_type == "cls":
            hidden_state = output_1[1]
        elif self.args.pooler_type == "avg":
            hidden_state = (output_1[0] * attention_mask.unsqueeze(-1)
                            ).sum(axis=-2) / attention_mask.sum(axis=-1).unsqueeze(-1)

        pooler = self.pre_classifier(hidden_state)
        pooler = self.relu(pooler)
        pooler = self.dropout(pooler)
        logits = self.classifier(pooler)
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0]+1e-10)*labels + \
                torch.log((1-prob)[:, 0]+1e-10)*(1-labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob
