from transformers import RobertaModel, RobertaTokenizer
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
import logging
import torch
import torch.nn as nn

from transformers import (BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'roberta_classification': (RobertaConfig, RobertaModel, RobertaTokenizer)
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


class LSTM(nn.Module):

    def __init__(self, args, xshape1, xshape2):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(xshape2, 100, batch_first=True)
        self.lstm2 = nn.LSTM(100, 50, batch_first=True)
        self.lstm3 = nn.LSTM(50, 25, batch_first=True)
        self.lstm4 = nn.LSTM(25, 12, batch_first=True)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc = nn.Linear(12, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, labels):
        x, (h, c) = self.lstm1(x)
        x, (h, c) = self.lstm2(x)
        x, (h, c) = self.lstm3(x)
        x, (h, c) = self.lstm4(x)
        x = self.fc(x[:, -1])
        x = self.sigmoid(x)

        if labels is None:
            return x
        labels = labels.float()
        loss = torch.log(x[:, 0]+1e-10)*labels + \
            torch.log((1-x)[:, 0]+1e-10)*(1-labels)
        loss = -loss.mean()

        return loss, x


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
        if self.args.source_model == "Multi" and not self.args.return_class:
            return pooler
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


class Model(nn.Module):
    def __init__(self, encoder, config, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.args = args

    def forward(self, input_ids=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        logits = outputs
        if self.args.source_model == "Multi" and not self.args.return_class:
            return logits

        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0]+1e-10)*labels + \
                torch.log((1-prob)[:, 0]+1e-10)*(1-labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


class MultiModel(nn.Module):
    def __init__(self, code_model, message_model, events_model, args):
        super(MultiModel, self).__init__()
        self.code_model = code_model
        self.message_model = message_model
        self.events_model = events_model
        self.args = args
        self.dropout = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(args.hidden_size * 3, 2)

    def forward(self, data, labels=None):
        code, message, events = data
        code = self.code_model(code)
        message = self.message_model(message)
        events = self.events_model(events)
        x = torch.stack([code, message, events], dim=1)
        x = x.reshape(code.shape[0], -1)
        x = self.dropout(x)
        logits = self.classifier(x)
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0]+1e-10)*labels + \
                torch.log((1-prob)[:, 0]+1e-10)*(1-labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


class Conv1DTune(nn.Module):
    def __init__(self, args, xshape1, xshape2, l1=1024, l2=256, l3=256, l4=64):
        super(Conv1DTune, self).__init__()
        self.args = args
        self.xshape1 = xshape1
        self.xshape2 = xshape2

        self.conv = nn.Conv1d(xshape1, l2, kernel_size=2)
        self.batchnorm1 = nn.BatchNorm1d(l2)
        self.batchnorm2 = nn.BatchNorm1d(l1)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc1 = nn.Linear(l2 * ((self.xshape2 // 2)), l1)
        if not args.return_class:
            self.fc2 = nn.Linear(l1, args.hidden_size)
        else:
            self.fc2 = nn.Linear(l1, l3)
        self.fc3 = nn.Linear(l3, l4)
        self.fc4 = nn.Linear(l4, 2)
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, labels=None):
        x = self.activation(self.conv(x))
        x = self.batchnorm1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.batchnorm2(x)

        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        if self.args.source_model == "Multi" and not self.args.return_class:
            # Returning for the multimodel to
            return x
        x = self.dropout(x)
        x = self.activation(self.fc3(x))

        x = self.dropout(x)
        x = self.sigmoid(self.fc4(x))

        if labels is None:
            return x
        labels = labels.float()
        loss = torch.log(x[:, 0]+1e-10)*labels + \
            torch.log((1-x)[:, 0]+1e-10)*(1-labels)
        loss = -loss.mean()

        return loss, x


def get_events_model(args):
    xshape1 = args.xshape1
    xshape2 = args.xshape2
    events_config = {'l1': 64, 'l2': 64, 'l3': 64, 'l4': 64}
    if args.events_model_type == "conv1d":
        model = Conv1DTune(args,
                           xshape1, xshape2, l1=events_config["l1"], l2=events_config["l2"], l3=events_config["l3"], l4=events_config["l4"])
    elif args.events_model_type == "lstm":
        logger.warning(f"shapes are {xshape1}, {xshape2}")
        model = LSTM(args, xshape1, xshape2)
    elif args.events_model_type == "gru":
        raise NotImplementedError
    else:
        raise NotImplementedError

    model = model.to(args.device)
    return model


def get_multi_model(args):
    code_model = get_code_model(args)
    args.hidden_size = code_model.encoder.config.hidden_size

    message_model = get_message_model(args)

    events_model = get_events_model(args)
    if args.multi_model_type == "multiv1":
        model = MultiModel(code_model, message_model, events_model, args)
    elif args.multi_model_type == "multiv2":
        model = MultiModel(code_model, message_model, events_model, args)
    else:
        model = MultiModel(code_model, message_model, events_model, args)

    return model


def get_model(args, dataset, tokenizer):
    if args.source_model == "Code" or args.source_model == "Message":
        model = get_code_model(args)
        model.encoder.resize_token_embeddings(len(tokenizer))

    elif args.source_model == "Message":
        model = get_message_model(args)
        model.encoder.resize_token_embeddings(len(tokenizer))

    elif args.source_model == "Events":
        model = get_events_model(args, dataset)

    elif args.source_model == "Multi":
        model = get_multi_model(args)
    return model


def get_message_model(args):
    config_class, model_class, _ = MODEL_CLASSES[args.message_model_type]
    config = config_class.from_pretrained(args.message_model_name,
                                          cache_dir=args.model_cache_dir if args.model_cache_dir else None, num_labels=2)
    config.num_labels = 2
    config.hidden_dropout_prob = args.dropout
    config.classifier_dropout = args.dropout
    if args.message_model_name:
        model = model_class.from_pretrained(args.message_model_name,
                                            from_tf=bool(
                                                '.ckpt' in args.message_model_name),
                                            config=config,
                                            cache_dir=args.model_cache_dir if args.model_cache_dir else None)
    else:
        model = model_class(config)

    if args.message_model_type == "roberta_classification":
        config.hidden_dropout_prob = args.dropout
        config.attention_probs_dropout_prob = args.dropout
        model = RobertaClass(model, args)
        logger.warning("Using RobertaClass")
    else:
        model = Model(model, config, args)
    return model


def get_code_model(args):
    config_class, model_class, _ = MODEL_CLASSES[args.code_model_type]
    config = config_class.from_pretrained(args.code_model_name,
                                          cache_dir=args.model_cache_dir if args.model_cache_dir else None, num_labels=2)
    config.num_labels = 2
    config.hidden_dropout_prob = args.dropout
    config.classifier_dropout = args.dropout
    if args.code_model_name:
        model = model_class.from_pretrained(args.code_model_name,
                                            from_tf=bool(
                                                '.ckpt' in args.code_model_name),
                                            config=config,
                                            cache_dir=args.model_cache_dir if args.model_cache_dir else None)
    else:
        model = model_class(config)

    if args.code_model_type == "roberta_classification":
        config.hidden_dropout_prob = args.dropout
        config.attention_probs_dropout_prob = args.dropout
        model = RobertaClass(model, args)
        logger.warning("Using RobertaClass")
    else:
        model = Model(model, config, args)
    return model
