import logging
import os

import wandb
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
    def __init__(self, args, xshape1, xshape2, l1=1024, l2=256, l3=256):
        super(RecurrentModels, self).__init__()
        if args.events_model_type == "lstm":
            self.model_type = nn.LSTM
        elif args.events_model_type == "gru":
            self.model_type = nn.GRU
        else:
            raise NotImplementedError
        self.args = args

        self.num_classes = 2
        self.num_layers = 2
        self.hidden_size = l1

        self.l3 = l3
        self.num_layers = 1
        self.bidirectional = self.args.event_bidirectional == 1 

        self.layer1 = self.model_type(input_size = xshape2, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        # self.layer2 = self.model_type(l1, l2, batch_first=True)
        # self.layer3 = self.model_type(l2, l3, batch_first=True)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc = nn.Linear(l1, 2)
        # self.activation = self.args.activation
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, labels=None):
        h_0 = torch.zeros(self.num_layers * (1+self.bidirectional), x.size(0), self.hidden_size).to(self.args.device)
        _, h_out = self.layer1(x, h_0)
        if self.bidirectional:
            h_out  = h_out.mean(dim=0)
        h_out = h_out.view(-1, self.hidden_size)
        h_out = self.dropout(h_out)

        out = self.fc(h_out)

        x = self.sigmoid(out)

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

        if args.source_model == "Message":
            self.activation = self.args.message_activation
        else:
            self.activation =  self.args.code_activation


        self.linear1 = torch.nn.Linear(self.hidden_size, self.args.hidden_size)
        self.linear2 = torch.nn.Linear(self.args.hidden_size, 2)
        self.args = args

    def forward(self, input_ids, labels=None):
        attention_mask = input_ids.ne(1)
        outputs  = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state

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
        if args.cut_layers:
            # todo fix
            self.classifier1 = nn.Linear(self.events_model.cut_layer_last_dim + 2*768, self.args.multi_model_hidden_size_1)
            self.classifier2 = nn.Linear(self.args.multi_model_hidden_size_1, self.args.multi_model_hidden_size_2)
            self.classifier3 = nn.Linear(self.args.multi_model_hidden_size_2, 2)
        else:
            # 6 = 2 + 2 + 2
            self.classifier1 = nn.Linear(6, 2)
        self.activation = nn.Tanh()

    def forward(self, data, labels=None):
        code, message, events = data
        code = self.code_model(code)
        message = self.message_model(message)
        events = self.events_model(events)
        x = torch.cat([code, message, events], dim=1)
        x = x.reshape(code.shape[0], -1)
        if self.args.cut_layers:
            x = self.classifier1(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.classifier2(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.classifier3(x)

        else:
            x = self.classifier1(x)
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
    def __init__(self, args, xshape1, xshape2, l1=1024, l2=256, l3=64):
        super(Conv1D, self).__init__()
        self.args = args
        self.xshape1 = xshape1
        self.xshape2 = xshape2
        # todo this is not correct!
        self.conv1d = nn.Conv1d(xshape1, l1, kernel_size=2 )
        self.max_pooling = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(l1*((self.xshape2-1)//2), l2)
        self.dense2 = nn.Linear(l2, l3)
        self.dense3 = nn.Linear(l3, 2)
        self.dropout = nn.Dropout(p=args.dropout)
        self.activation = self.args.event_activation
        self.sigmoid = nn.Sigmoid()
        if self.args.cut_layers:
            self.cut_layer_last_dim = l3

    def cut_layers(self):
        pass

    def forward(self, x, labels=None):
        x = self.conv1d(x)
        x = self.activation(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.activation(x)
        
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.activation(x)
        if self.args.cut_layers:
            return x

        x = self.dense3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.sigmoid(x)

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
                           xshape1, xshape2, l1=args.event_l1, l2=args.event_l2, l3=args.event_l3)
    elif args.events_model_type == "lstm" or args.events_model_type == "gru":
        logger.warning(f"shapes are {xshape1}, {xshape2}")
        model = RecurrentModels(args, xshape1, xshape2, l1=args.event_l1, l2=args.event_l2, l3=args.event_l3)
    else:
        raise NotImplementedError

    model = model.to(args.device)
    return model


def get_multi_model(args, message_tokenizer = None, code_tokenizer = None):
    code_model = get_code_model(args)
    code_model.encoder.resize_token_embeddings(len(code_tokenizer))
    args.hidden_size = code_model.encoder.config.hidden_size
    if args.multi_code_model_artifact:
        initialize_model_from_wandb(args, code_model, args.multi_code_model_artifact)

        
    events_model = get_events_model(args)
    if args.multi_events_model_artifact:
        initialize_model_from_wandb(args, events_model, args.multi_events_model_artifact)

    
    message_model = get_message_model(args)
    if args.multi_message_model_artifact:
        initialize_model_from_wandb(args, message_model, args.multi_message_model_artifact)

    if args.cut_layers:
        code_model.cut_layers()
        events_model.cut_layers()
        message_model.cut_layers()

    if args.multi_model_type == "multiv1":
        model = MultiModel(code_model, message_model, events_model, args)
    else:
        raise NotImplementedError
        

    return model

def initialize_model_from_wandb(args, model, artifact_path):
    artifact = wandb.use_artifact(artifact_path, type="model")
    artifact_dir = artifact.download()
    assert len(os.listdir(artifact_dir)) == 1, "More than one model in the artifact"

    model_path = os.path.join(artifact_dir,os.listdir(artifact_dir)[0])
    model.load_state_dict(torch.load(model_path))
    if args.freeze_submodel_layers:
        for param in model.parameters():
            param.requires_grad = False

def get_model(args, message_tokenizer=None, code_tokenizer=None):
    if args.source_model == "Code":
        model = get_code_model(args)
        model.encoder.resize_token_embeddings(len(code_tokenizer))
        if args.cut_layers:
            model.cut_layers()

    elif args.source_model == "Message":
        model = get_message_model(args)
        if args.cut_layers:
            model.cut_layers()

    elif args.source_model == "Events":
        model = get_events_model(args)
        if args.cut_layers:
            model.cut_layers()

    elif args.source_model == "Multi":
        model = get_multi_model(args, message_tokenizer=message_tokenizer, code_tokenizer=code_tokenizer)

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
    elif args.message_model_type == "roberta":
        model = XGlueModel(model, args)
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
        model = XGlueModel(model, args)
    return model



class CustomBERTModel(nn.Module):
    def __init__(self, args):
        super(CustomBERTModel, self).__init__()
        self.bert = RobertaModel.from_pretrained(args.message_model_name)
        self.args = args
        ### New layers:
        self.linear1 = nn.Linear(768, self.args.message_l1)
        self.linear2 = nn.Linear(self.args.message_l1, self.args.message_l2)
        self.linear3 = nn.Linear(self.args.message_l2, self.args.message_l3)
        self.linear4 = nn.Linear(self.args.message_l3, self.args.message_l4)
        self.linear2 = nn.Linear(self.args.message_l4, 2) ## 3 is the number of classes in this example
        self.args = args
        self.sigmoid = nn.Sigmoid()
        if args.source_model == "Message":
            self.activation = self.args.message_activation
        else:
            self.activation =  self.args.code_activation

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, input_ids, labels=None):
        attention_mask = input_ids.ne(1)
        sequence_output, pooled_output = self.bert(input_ids, attention_mask=attention_mask)

        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        x = self.linear1(sequence_output[:,0,:].view(-1,768)) ## extract the 1st token's embeddings
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.linear2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.linear3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.linear4(x)

        if self.args.source_model == "Multi" and not self.args.return_class:
            # Returning for the multimodel to
            return x

        x = self.sigmoid(x)

        if labels is None:
            return x
        labels = labels.float()
        loss = torch.log(x[:, 0]+1e-10)*labels + \
        torch.log((1-x)[:, 0]+1e-10)*(1-labels)
        loss = -loss.mean()

        return loss, x
    

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class XGlueModel(nn.Module):   
    def __init__(self, encoder,args):
        super(XGlueModel, self).__init__()
        self.encoder = encoder
        self.args=args

    def cut_layers(self):
        self.encoder.classifier.out_proj = Identity()
    
        
    def forward(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        if self.args.cut_layers:
            return outputs
        logits=outputs
        prob=torch.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob
      
        