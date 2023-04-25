from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
import torch
import torch.nn as nn
import torch.nn.functional as F

class PoolerClassificationHead(RobertaClassificationHead):
    """Head for sentence-level classification tasks."""
    def __init__(self, config, pooler_type="cls"):
        super().__init__(config)
        self.pooler_type = pooler_type

    def forward(self, features, attention_mask=None):
        if self.pooler_type == "cls":
            features = features[:, 0]
        elif self.pooler_type == "avg" and attention_mask is not None:
            features = (features * attention_mask.unsqueeze(-1)).sum(axis=-2) / attention_mask.sum(axis=-1).unsqueeze(-1)
        else:
            raise NotImplementedError
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    


       
class RobertaClassificationModel(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(RobertaClassificationModel, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=PoolerClassificationHead(config, pooler_type=args.pooler_type)
        self.args=args
    
        
    def forward(self, input_ids=None,labels=None): 
        input_ids=input_ids.view(-1,self.args.block_size)
        attention_mask=input_ids.ne(1)
        outputs = self.encoder(input_ids= input_ids,attention_mask=attention_mask)[0]
        logits=self.classifier(outputs, attention_mask=attention_mask)
        prob=F.softmax(logits)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob
        


