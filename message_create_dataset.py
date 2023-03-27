from datasets import GeneralDataset
from code_training import get_commit_from_repo, handle_commit
from misc import safe_mkdir
import os
from tqdm import tqdm
import torch


def handle_commit_message(message, tokenizer, args):
    token_arr = tokenizer.encode(message, add_special_tokens=True)
    if len(token_arr) > args.max_input_length:
        token_arr = token_arr[:args.max_input_length]
    else:
        token_arr = token_arr + [tokenizer.pad_token_id] * (args.max_input_length - len(token_arr))
    return token_arr

class MessageDataset(GeneralDataset):
    def __init__(self, tokenizer, args, phase, cache = True, csv_list_dir=r"C:\Users\nitzan\local\analyzeCVE"):
        
        safe_mkdir("languages_cache")
        self.tokenizer = tokenizer
        self.args = args
        self.cache = cache
        self.language = args.language
        self.counter = 0
        self.current_path = f"languages_cache\\message_{phase}.json"
        self.final_list = []


        # if phase == 'train':
        #     with open(os.path.join(csv_list_dir,"train_details.pickle"), 'rb') as f:
        #         self.csv_list = pickle.load(f)
        # elif phase == 'val':
        #     with open(os.path.join(csv_list_dir,"validation_details.pickle"), 'rb') as f:
        #         self.csv_list = pickle.load(f)
        # elif phase == 'test':
        #     with open(os.path.join(csv_list_dir,"test_details.pickle"), 'rb') as f:
        #         self.csv_list = pickle.load(f)
        # else:
        #     raise ValueError(f"Unknown phase: {phase}")

        # if os.path.exists(self.current_path) and self.cache:
        #     self.final_list = torch.load(self.current_path)
        #     return
        
        self.csv_list = [("RpcInvestigator", "a", 1, "03abf7480e19a238aa750bdbcd7bd793ff25a90a"), ("RpcInvestigator", "a", 0, "83e10b8d2d9a9fa2a4586c5f51c8e0f45f9c7a0f")]
        self.create_final_list()

        torch.save(self.final_list, self.current_path)

    def create_final_list(self):
        counter = 0
        for repo, _, label, cur_hash in tqdm(self.csv_list[:]):
            # cur_hash = cur_hash.values[0]

            if cur_hash == '':
                counter+=1
                continue
            commit = get_commit_from_repo(
                os.path.join(r"C:\Users\nitza\source\repos", repo), cur_hash)
            


            try:
                token_arr = handle_commit_message(
                    commit.msg,
                    self.tokenizer,
                    self.args)

                self.final_list.append((token_arr, int(label)))
            except Exception as e:
                print(e)
                continue


    def __len__(self):
        return len(self.final_list)

    def __getitem__(self, i):
        source_ids, label = self.final_list[i]
        return source_ids, torch.tensor(int(label))
