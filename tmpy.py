from message_create_dataset import MessageDataset
from transformers import RobertaTokenizer
import argparse

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
args = argparse.Namespace()
args.language = "python"
args.max_input_length = 512
b = MessageDataset(tokenizer, args, 'train')
print(b.final_list)