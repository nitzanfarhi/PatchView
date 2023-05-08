import pickle
import torch

from torch.utils.data import Dataset, DataLoader
from data_utils import extract_window, fix_repo_idx, fix_repo_shape, get_event_window
from misc import Set, add_metadata, GeneralDataset
from tqdm import tqdm
import json
import pandas as pd
import os

from data_creation import gh_cve_dir

csv_list_dir=r"C:\Users\nitzan\local\analyzeCVE"

class EventsDataset(GeneralDataset):
    def __init__(self, args, set_name, cache=True, backs = 10, input_path = r"C:\Users\nitzan\local\analyzeCVE\data_collection\data"):
        self.backs = backs
        self.args = args
        self.x_set = []
        self.y_set = []
        self.info = []
        self.current_path = f"cache_data\\events\\events_{set_name}.json"
        self.cache = not args.recreate_cache
        with open(os.path.join(args.cache_dir,f"orchestrator_{set_name}.json"), 'r') as f:
            self.hash_list = json.load(f)
        
        if self.cache and os.path.exists(self.current_path):
            x_set, y_set = torch.load(self.current_path)
            self.x_set = x_set
            self.y_set = y_set
        else:
            self.create_list_of_hashes(input_path )
            torch.save((self.x_set, self.y_set), self.current_path)


    def create_list_of_hashes(self, input_path):
        repo_dict = {}
        with open(os.path.join(self.args.cache_dir,"events","repo_metadata.json"), 'r') as f:
            all_metadata = json.load(f)
        for repo in tqdm(self.hash_list, leave=False):
            for mhash, label in self.hash_list[repo]:
                try:
                    if mhash == "":
                        continue
                    repo_name = repo.replace("/", "_")
                    if repo_name not in repo_dict:
                        try:
                            cur_repo = pd.read_parquet(f"{input_path}\{gh_cve_dir}\{repo_name}.parquet")
                            cur_repo = fix_repo_idx(cur_repo)
                            cur_repo = fix_repo_shape(cur_repo) 
                            repo_dict[repo_name] = cur_repo
                        except FileNotFoundError:
                            print(f"File not found: {repo_name}")
                            continue
                    else:
                        cur_repo = repo_dict[repo_name]


                    wanted_row = cur_repo.index[cur_repo['Hash'] == mhash].tolist()
                    if len(wanted_row) == 0:
                        continue
                    assert len(wanted_row) == 1, "Hash is not unique"             
                    wanted_row = wanted_row[0]
                    self.info.append((repo_name,mhash))
                    event_window = get_event_window(cur_repo,wanted_row, backs=self.backs)
                    event_window = add_metadata(input_path, all_metadata, event_window, repo_name)
                    event_window = event_window.drop(["Hash","Vuln"], axis = 1)
                    event_window = event_window.fillna(0)
                    self.x_set.append(event_window.values)
                    self.y_set.append(label)
                except KeyError as e:
                    print(e)


    def __len__(self):
        return len(self.x_set)

    def __getitem__(self, idx):
        item = self.x_set[idx]
        label = self.y_set[idx]
        item = torch.from_numpy(item.astype(float)).float()
        return item, label



def create_datasets(DatasetClass, orchestrator_location=r"C:\Users\nitzan\local\analyzeCVE", cache=True, **kwargs):
    res = []
    for set_name in ["train", "validation", "test"]:
        with open(os.path.join(orchestrator_location,f"{set_name}_details.pickle"), "rb") as f:
            cur_set = pickle.load(f)
        if cache and os.path.exists(f"{set_name}_dataset.pkl"):
            cur_set = pickle.load(open(f"{set_name}_dataset.pkl", "rb"))
        else:
            cur_set = DatasetClass(cur_set, set_name, cache=cache, **kwargs)
            pickle.dump(cur_set, open(f"{set_name}_dataset.pkl", "wb"))
        res.append(cur_set)
    return res
        
