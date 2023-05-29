import logging
logger = logging.getLogger(__name__)

from transformers import RobertaModel, RobertaTokenizer
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
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


class RecurrentModels(nn.Module):
    def __init__(self, args, xshape1, xshape2, l1=1024, l2=256, l3=256, l4=64):
        super(RecurrentModels, self).__init__()
        if args.events_model_type == "lstm":
            self.model_type = nn.LSTM
        elif args.events_model_type == "gru":
            self.model_type = nn.GRU
        else:
            raise NotImplementedError
        self.args = args
        self.layer1 = self.model_type(xshape2, l1, batch_first=True)
        self.layer2 = self.model_type(l1, l2, batch_first=True)
        self.layer3 = self.model_type(l2, l3, batch_first=True)
        self.layer4 = self.model_type(l3, l4, batch_first=True)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc = nn.Linear(l4, 2)
        self.activation = self.args.activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, labels):
        x, (h, c) = self.layer1(x)
        x = self.activation(x)
        x, (h, c) = self.layer2(x)
        x = self.activation(x)
        x, (h, c) = self.layer3(x)
        x = self.activation(x)
        x, (h, c) = self.layer4(x)
        x = self.activation(x)
        x = self.fc(x[:, -1])
        x = self.sigmoid(x)

        if labels is None:
            return x
        labels = labels.float()
        loss = torch.log(x[:, 0]+1e-10)*labels + \
            torch.log((1-x)[:, 0]+1e-10)*(1-labels)
        loss = -loss.mean()

        return loss, x


class RobertaClass(torch.nn.Module):
    def __init__(self, l1, args):
        super(RobertaClass, self).__init__()
        self.encoder = l1
        self.args = args
        self.hidden_size = self.encoder.config.hidden_size
        self.dropout = torch.nn.Dropout(args.dropout)
        self.activation = args.activation
        self.linear1 = torch.nn.Linear(self.hidden_size, self.args.hidden_size)
        self.linear2 = torch.nn.Linear(self.args.hidden_size, 2)
        self.args = args

    def forward(self, input_ids, labels=None):
        attention_mask = input_ids.ne(1)
        sequence_output, pooled_output = self.encoder(input_ids=input_ids,
                                attention_mask=attention_mask)

        if self.args.pooler_type == "cls":
          pooler = self.linear1(sequence_output[:,0,:].view(-1,768)) ## extract the 1st token's embeddings

        elif self.args.pooler_type == "avg":
            raise NotImplementedError

        pooler = self.activation(pooler)
        pooler = self.dropout(pooler)

        if self.args.source_model == "Multi" and not self.args.return_class:
            return pooler
        
        logits = self.linear2(pooler)
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
        self.classifier1 = nn.Linear(args.hidden_size * 3, 64)
        self.classifier2 = nn.Linear(args.hidden_size * 64, args.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, data, labels=None):
        code, message, events = data
        code = self.code_model(code)
        message = self.message_model(message)
        events = self.events_model(events)
        x = torch.stack([code, message, events], dim=1)
        x = x.reshape(code.shape[0], -1)
        x = self.classifier1(x)
        x = self.activation(x)

        x = self.classifier2(x)
        x = self.activation(x)
        prob = torch.sigmoid(x)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0]+1e-10)*labels + \
                torch.log((1-prob)[:, 0]+1e-10)*(1-labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


class Conv1D(nn.Module):
    def __init__(self, args, xshape1, xshape2, l1=1024, l2=256, l3=256, l4=64):
        super(Conv1D, self).__init__()
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

        self.activation = self.args.activation
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
    if args.events_model_type == "conv1d":
        model = Conv1D(args,
                           xshape1, xshape2, l1=args.event_l1, l2=args.event_l2, l3=args.event_l3, l4=args.event_l4)
    elif args.events_model_type == "lstm" or args.events_model_type == "gru":
        logger.warning(f"shapes are {xshape1}, {xshape2}")
        model = RecurrentModels(args, xshape1, xshape2, l1=args.event_l1, l2=args.event_l2, l3=args.event_l3, l4=args.event_l4)
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
    if args.source_model == "Code":
        model = get_code_model(args)
        model.encoder.resize_token_embeddings(len(tokenizer))

    elif args.source_model == "Message":
        model = get_message_model(args)
        model.encoder.resize_token_embeddings(len(tokenizer))

    elif args.source_model == "Events":
        model = get_events_model(args)

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
        raise NotImplementedError
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
        raise NotImplementedError
    return model
