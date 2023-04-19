import pickle
from misc import GeneralDataset
from code_training import get_commit_from_repo, handle_commit
from misc import safe_mkdir
import os
from tqdm import tqdm
import torch

COMMITS_PATH = r"C:\Users\nitzan\local\analyzeCVE\data_collection\data\commits"

def handle_commit_message(message, tokenizer, args):
    token_arr = tokenizer.encode(message, add_special_tokens=True)
    if len(token_arr) > args.block_size:
        token_arr = token_arr[:args.block_size]
    else:
        token_arr = token_arr + [tokenizer.pad_token_id] * (args.block_size - len(token_arr))
    return token_arr

class MessageDataset(GeneralDataset):
    def __init__(self, tokenizer, args, phase, csv_list_dir=r"C:\Users\nitzan\local\analyzeCVE"):
        
        safe_mkdir("languages_cache")
        self.tokenizer = tokenizer
        self.args = args
        self.cache = args.cache
        self.counter = 0
        self.current_path = f"languages_cache\\message_{phase}.json"
        self.final_list = []


        if phase == 'train':
            with open(os.path.join(csv_list_dir,"train_details.pickle"), 'rb') as f:
                self.csv_list = pickle.load(f)
        elif phase == 'val':
            with open(os.path.join(csv_list_dir,"validation_details.pickle"), 'rb') as f:
                self.csv_list = pickle.load(f)
        elif phase == 'test':
            with open(os.path.join(csv_list_dir,"test_details.pickle"), 'rb') as f:
                self.csv_list = pickle.load(f)
        else:
            raise ValueError(f"Unknown phase: {phase}")

        
        if self.cache and os.path.exists(self.current_path):
            self.final_list = torch.load(self.current_path)
        else:
            self.create_final_list()
            torch.save(self.final_list, self.current_path)

    def create_final_list(self):
        counter = 0
        for repo, _, label, cur_hash in tqdm(self.csv_list[:30], leave=False):
            cur_hash = cur_hash.values[0]
            if cur_hash == '':
                counter+=1
                continue
            commit = get_commit_from_repo(
                os.path.join(COMMITS_PATH, repo), cur_hash)

            try:
                token_arr = handle_commit_message(
                    commit.msg,
                    self.tokenizer,
                    self.args)

                if len(token_arr) > 512:
                    assert False
                self.final_list.append((torch.tensor(token_arr), int(label)))
            except Exception as e:
                print(e)
                continue


    def __len__(self):
        return len(self.final_list)

    def __getitem__(self, i):
        source_ids, label = self.final_list[i]
        return source_ids, torch.tensor(int(label))
