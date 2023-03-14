import torch

from torch.utils.data import Dataset, DataLoader
from dataset_utils import extract_window, fix_repo_idx, fix_repo_shape, get_event_window
from misc import Set, add_metadata
from tqdm import tqdm
import json
import pandas as pd
import os

from events_create_dataset import gh_cve_dir, repo_metadata_filename



class EventsDataset(Dataset):
    def __init__(self, hash_list, backs = 10, input_path = r"C:\Users\nitzan\local\analyzeCVE\data_collection\data"):
        self.hash_list = hash_list
        self.backs = backs
        self.x_set = []
        self.y_set = []
        self.info = []

        self.create_list_of_hashes(hash_list, input_path)

    def create_list_of_hashes(self, hash_list, input_path):
        all_metadata = json.load(open(os.path.join(input_path,repo_metadata_filename), 'r'))
        for repo in tqdm(hash_list):
            repo_name = repo.replace("/", "_")
            try:
                cur_repo = pd.read_parquet(f"{input_path}\{gh_cve_dir}\{repo_name}.parquet")
            except FileNotFoundError:
                continue
            # cur_repo = add_metadata(input_path, all_metadata, cur_repo)
            # cur_repo = fix_repo_shape(cur_repo)
            cur_repo = fix_repo_idx(cur_repo)

            
            for hash, label in hash_list[repo]:
                try: 
                    wanted_row = cur_repo.index[cur_repo['Hash'] == hash].tolist()
                    if len(wanted_row) == 0:
                        continue
                    assert len(wanted_row) == 1, "Hash is not unique"             
                    wanted_row = wanted_row[0]
                    self.info.append((repo_name,wanted_row))
                    event_window = get_event_window(cur_repo,wanted_row, backs=self.backs)
                    event_window = add_metadata(input_path, all_metadata, event_window)
                    event_window = fix_repo_shape(event_window)
                    event_window = event_window.drop(["Hash","Vuln"], axis = 1)
                    event_window = event_window.fillna(0)
                    self.x_set.append(event_window.values)
                    self.y_set.append(label)
                except Exception as e:
                    print(e)
                    continue



    def __len__(self):
        return len(self.hash_list)

    def __getitem__(self, idx):
        item = self.x_set[idx]
        label = self.y_set[idx]
        item = torch.from_numpy(item).float()
        return item, label


class Codes(Dataset):
    pass

class CommitMessages(Dataset):
    pass


def create_datasets(DatasetClass, **kwargs):
    with open("orchestrator_training.json", "r") as f:
        orchestrator_train = json.load(f)

    with open("orchestrator_validation.json", "r") as f:
        orchestrator_validation = json.load(f)

    with open("orchestrator_testing.json", "r") as f:
        orchestrator_test = json.load(f)

    train_dataset = DatasetClass(orchestrator_train, **kwargs)
    validation_dateset = DatasetClass(orchestrator_validation, **kwargs)
    test_dataset = DatasetClass(orchestrator_test, **kwargs)

    return train_dataset, validation_dateset, test_dataset