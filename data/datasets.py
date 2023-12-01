from data.misc import add_metadata
from torch.utils.data import Dataset, DataLoader
from data.data_utils import fix_repo_idx, fix_repo_shape, get_event_window
from tqdm import tqdm
import torch
import pandas as pd
import pickle
import os
import json
import difflib
import logging
logger = logging.getLogger(__name__)


# files bigger than that will be ignored as they are probably binaries / not code
MAXIMAL_FILE_SIZE = 100000

INS_TOKEN = '[INS]'
DEL_TOKEN = '[DEL]'
REP_BEFORE_TOKEN = '[RBT]'
REP_AFTER_TOKEN = '[RAT]'

ADD_TOKEN = '[ADD]'
DELETE_TOKEN = '[DEL]'


class TextDataset(Dataset):

    def __init__(self, tokenizer, args, all_json, keys, embedding_type):
        logger.warning(f"Loading dataset")
        self.tokenizer = tokenizer
        self.args = args
        self.cache = not args.recreate_cache
        self.counter = 0
        self.embedding_type = embedding_type
        self.final_list_labels = []
        self.final_list_tensors = []
        self.final_commit_info = []
        self.keys = keys
        self.all_json = all_json
        self.commit_path = os.path.join(
            args.cache_dir, "code", f"commits.json")
        self.final_cache_list = os.path.join(
            args.cache_dir, "code", f"{self.embedding_type}_final_list.pickle")
        self.positive_label_counter = 0
        self.negative_label_counter = 0
        self.commit_repos_path = args.commit_repos_path

        self.csv_list = keys

        if self.embedding_type == 'simple_with_tokens':
            logger.warning(
                f"Tokenizer size before adding tokens: {len(self.tokenizer)}")
            self.tokenizer.add_special_tokens(
                {'additional_special_tokens': [ADD_TOKEN, DEL_TOKEN]})
        elif self.embedding_type == "sum":
            self.tokenizer.add_special_tokens({'additional_special_tokens': [
                                              REP_BEFORE_TOKEN, REP_AFTER_TOKEN, INS_TOKEN, DEL_TOKEN]})

        self.commit_list = self.get_commits()
        self.commit_list = sorted(self.commit_list, key=lambda x: x['hash'])
        logger.warning(f"Number of commits: {len(self.commit_list)}")
        self.create_final_list()



    def get_commits(self):
        result = []
        positives = 0
        negatives = 0
        if os.path.exists(self.commit_path):
            logger.warning("Get Commits from cache")
            with open(self.commit_path, 'rb') as f:
                return pickle.load(f)
        else:
            logger.warning("Get Commits from repos")
            for commit in (pbar := tqdm(list(self.keys)[:], leave=False)):
                try:
                    if commit == "":
                        assert False, "shouldnt be empty hashes here"
                        continue

                    repo = self.all_json[commit]["repo"]
                    label = self.all_json[commit]["label"]
                    result.append(self.prepare_dict(
                        repo.replace("/", "_"), label, commit))
                    if label == 1:
                        positives += 1
                    else:
                        negatives += 1

                    pbar.set_description(
                        f"Repo - {repo} - Positives: {positives}, Negatives: {negatives}")
                except Exception as e:
                    print(e)
                    continue

            with open(self.commit_path, 'wb') as f:
                pickle.dump(result, f)

            logger.warning(f"Positives: {positives}, Negatives: {negatives}")
            return result

    def add_code_data_to_dict(self, file):
        cur_dict = {}
        before = ""
        after = ""
        if file.content_before is not None:
            try:
                before = file.content_before.decode('utf-8')
            except UnicodeDecodeError:
                return None
        else:
            before = ""

        if file.content is not None:
            try:
                after = file.content.decode('utf-8')
            except UnicodeDecodeError:
                return None
        if "." not in file.filename:
            return None
        if len(after) > MAXIMAL_FILE_SIZE or len(before) > MAXIMAL_FILE_SIZE:
            return None

        filetype = file.filename.split(".")[-1].lower()
        cur_dict["filetype"] = filetype
        cur_dict["filename"] = file.filename
        cur_dict["content"] = after
        cur_dict["before_content"] = before
        cur_dict["added"] = file.diff_parsed["added"]
        cur_dict["deleted"] = file.diff_parsed["deleted"]
        return cur_dict

    def prepare_dict(self, repo, label, cur_hash):
        commit = get_commit_from_repo(
            os.path.join(self.commit_repos_path, repo), cur_hash)
        final_dict = {}
        final_dict["name"] = commit.project_name
        final_dict["hash"] = commit.hash
        final_dict["files"] = []
        final_dict["source"] = []
        final_dict["label"] = label
        final_dict["repo"] = repo
        final_dict["message"] = commit.msg
        try:
            for file in commit.modified_files:
                cur_dict = self.add_code_data_to_dict(file)
                if cur_dict is not None:
                    final_dict["files"].append(cur_dict)
        except Exception as e:
            print(e)
            final_dict["files"] = []
            return final_dict
        return final_dict

    def create_final_list(self):
        import gc
        if os.path.exists(self.final_cache_list) and self.cache:
            logger.warning("Get final list from cache")
            gc.disable()
            with open(self.final_cache_list, 'rb') as f:
                cached_list = torch.load(f)
                self.final_commit_info = cached_list["final_commit_info"]
                self.final_list_tensors = cached_list["final_list_tensors"]
                self.final_list_labels = cached_list["final_list_labels"]
            gc.enable()
            return

        logger.warning("Create final list")
        for commit in (pbar := tqdm(self.commit_list[:], leave=False)):
            token_arr_lst = handle_commit(
                commit,
                self.tokenizer,
                self.args,
                embedding_type=self.embedding_type)

            for token_arr in token_arr_lst:
                if token_arr is not None:
                    self.final_list_tensors.append(
                        torch.tensor(token_arr['input_ids']))
                    self.final_commit_info.append(commit)
                    self.final_list_labels.append(commit['label'])

            pbar.set_description(
                f"Current Project: {commit['repo']}")

        with open(self.final_cache_list, 'wb') as f:
            torch.save({"final_commit_info": self.final_commit_info,
                        "final_list_tensors": self.final_list_tensors,
                        "final_list_labels": self.final_list_labels}, f)

    def __len__(self):
        return len(self.final_list_tensors)

    def __getitem__(self, i):
        return self.final_list_tensors[i], self.final_list_labels[i]


class MyConcatDataset(torch.utils.data.Dataset):
    def __init__(self, args, code_dataset=None, message_dataset=None, events_dataset=None):
        self.args = args
        self.code_dataset = code_dataset
        self.message_dataset = message_dataset
        self.events_dataset = events_dataset

        self.merged_dataset = []
        self.merged_labels = []
        self.final_commit_info = []

        self.code_hash_list = None
        self.message_hash_list = None
        self.events_hash_list = None

        if code_dataset is not None:
            self.code_hash_list = [x["hash"] for x in code_dataset.final_commit_info]
        if message_dataset is not None:
            self.message_hash_list = [x["hash"]
                                  for x in message_dataset.final_commit_info]
        if events_dataset is not None:
            self.events_hash_list = [x["hash"]
                                 for x in events_dataset.final_commit_info]
        
        self.is_train = True
     
    def set_hashes(self, hash_list, is_train = True):
        # if is_train:
        #     self.train_merged_dataset = []
        #     self.train_merged_labels = []
        #     self.train_final_commit_info = []
        # else:
        #     self.val_merged_dataset = []
        #     self.val_merged_labels = []
        #     self.val_final_commit_info = []
        pos_cur_merged_dataset = []
        pos_cur_merged_labels = []
        pos_cur_final_commit_info = []
        neg_cur_merged_dataset = []
        neg_cur_merged_labels = []
        neg_cur_final_commit_info = []


        # ugly but easy
        labels_counter = [0,0]
        for hash in tqdm(hash_list):
            try:
                labels = []
                infos = []
                if self.code_hash_list is None:
                    cur_code = {}
                else:
                    code_idx = self.code_hash_list.index(hash)
                    cur_code = self.code_dataset[code_idx][0]
                    labels.append(self.code_dataset[code_idx][1])
                    infos.append(self.code_dataset.final_commit_info[code_idx])

                if self.message_hash_list is None:
                    cur_message = {}
                else:
                    message_idx = self.message_hash_list.index(hash)
                    cur_message = self.message_dataset[message_idx][0]
                    labels.append(self.message_dataset[message_idx][1])
                    infos.append(self.message_dataset.final_commit_info[message_idx])

                if self.events_hash_list is None:
                    cur_events = {}
                else:
                    events_idx = self.events_hash_list.index(hash)
                    cur_events = self.events_dataset[events_idx][0]
                    labels.append(self.events_dataset[events_idx][1])
                    infos.append(self.events_dataset.final_commit_info[events_idx])

                assert labels.count(labels[0]) == len(labels)
                labels_counter[labels[0]] += 1

                if labels[0] == 1:
                    pos_cur_merged_dataset.append((cur_code, cur_message, cur_events ))
                    pos_cur_merged_labels.append(labels[0])
                    pos_cur_final_commit_info.append(infos[0])
                else:
                    neg_cur_merged_dataset.append((cur_code, cur_message, cur_events ))
                    neg_cur_merged_labels.append(labels[0])
                    neg_cur_final_commit_info.append(infos[0])

            except ValueError:
                continue
        

        min_idxs = min(len(pos_cur_merged_dataset), len(neg_cur_merged_dataset))
        pos_cur_merged_dataset = pos_cur_merged_dataset[:min_idxs]
        pos_cur_merged_labels = pos_cur_merged_labels[:min_idxs]
        pos_cur_final_commit_info = pos_cur_final_commit_info[:min_idxs]

        neg_cur_merged_dataset = neg_cur_merged_dataset[: int(min_idxs * self.args.balance_factor)]
        neg_cur_merged_labels = neg_cur_merged_labels[: int(min_idxs * self.args.balance_factor)]
        neg_cur_final_commit_info = neg_cur_final_commit_info[: int(min_idxs * self.args.balance_factor)]

        cur_merged_dataset = pos_cur_merged_dataset + neg_cur_merged_dataset
        cur_merged_labels = pos_cur_merged_labels + neg_cur_merged_labels
        cur_final_commit_info = pos_cur_final_commit_info + neg_cur_final_commit_info

        if is_train:
            self.train_merged_dataset = cur_merged_dataset
            self.train_merged_labels = cur_merged_labels
            self.train_final_commit_info = cur_final_commit_info
            mlen = len(self.train_merged_dataset)
            name = 'Train'
        else:
            self.val_merged_dataset = cur_merged_dataset
            self.val_merged_labels = cur_merged_labels
            self.val_final_commit_info = cur_final_commit_info
            mlen = len(self.val_merged_dataset)
            name = 'Val'

        logger.warning(f"Merged {name} is {mlen}")



    def __getitem__(self, i):
        if self.is_train:
            return self.train_merged_dataset[i], self.train_merged_labels[i]
        else:
            return self.val_merged_dataset[i], self.val_merged_labels[i]


    def get_info(self, i):
        if self.is_train:
            return self.train_final_commit_info[i]
        else:
            return self.val_final_commit_info[i]

    def __len__(self):
        if self.is_train:
            return len(self.train_merged_dataset)
        else:
            return len(self.val_merged_dataset)



def get_commit_from_repo(cur_repo, hash):
    from pydriller import Repository
    return next(Repository(cur_repo, single=hash).traverse_commits())


def convert_to_ids_and_pad(source_tokens, tokenizer, args):
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    if len(source_ids) > args.block_size:
        return None
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return torch.tensor(source_ids)


def embed_file(file, tokenizer, args):
    before, after = "", ""
    if file["before_content"] is not None:
        before = file["before_content"]
    else:
        before = ""

    if file["content"] is not None:
        after = file["content"]

    if len(before) > MAXIMAL_FILE_SIZE or len(after) > MAXIMAL_FILE_SIZE or len(before) == 0 or len(after) == 0:
        return None

    operation_list = []
    opcodes = difflib.SequenceMatcher(a=before, b=after).get_opcodes()

    logger.warning(
        f"Size Before - {len(before)}, Size After - {len(after)}, Opcode Number -  {len(opcodes)}")
    for opp, a1, a2, b1, b2 in opcodes:
        if opp == 'equal':
            continue

        elif opp == 'replace':
            res = REP_BEFORE_TOKEN+" " + \
                before[a1:a2]+" "+REP_AFTER_TOKEN+" "+after[b1:b2]
            if not args.code_merge_file:
                res = tokenizer(res, truncation=True,
                                padding='max_length', max_length=args.block_size)
            operation_list.append(res)

        elif opp == 'insert':
            res = INS_TOKEN + " " + after[b1:b2]
            if not args.code_merge_file:
                res = tokenizer(res, truncation=True,
                                padding='max_length', max_length=args.block_size)
            operation_list.append(res)

        elif opp == 'delete':
            res = DEL_TOKEN + " " + before[a1:a2]
            if not args.code_merge_file:
                res = tokenizer(res, truncation=True,
                                padding='max_length', max_length=args.block_size)
            operation_list.append(res)

        else:
            raise ValueError(f"Unknown operation: {opp}")

    return operation_list


def get_line_comment(language):
    from code_utils import ext_to_comment

    if language in ext_to_comment:
        return ext_to_comment[language]+" "
    else:
        return "// "


def handle_commit(commit, tokenizer, args, embedding_type='concat'):
    res = []
    file_counter = 0 
    for file in commit["files"]:
        if len("".join(['added'])+"".join(['deleted'])) > args.block_size:
            continue
        
        if embedding_type == 'sum':
            embed_file_res = embed_file(file, tokenizer, args)
            if embed_file_res is not None:
                res += embed_file_res

        elif embedding_type == 'simple':
            added = [diff[1] for diff in file['added']]
            deleted = [diff[1] for diff in file['deleted']]

            if args.code_merge_file:
                res.append((added, deleted))
            else:
                file_res = " ".join(added+deleted)
                file_res = tokenizer(file_res, truncation=True,
                                     padding='max_length', max_length=args.block_size)
                res.append(file_res)

        elif embedding_type == 'simple_with_tokens':
            added = [ADD_TOKEN]+[diff[1]
                                 for diff in file['added']]+[tokenizer.sep_token]
            deleted = [DEL_TOKEN]+[diff[1] for diff in file['deleted']]

            if args.code_merge_file:
                res.append((added, deleted))
            else: 
                file_res = " ".join(added+deleted)
                file_res = tokenizer(file_res, truncation=True,
                                    padding='max_length', max_length=args.block_size)
                res.append(file_res)

        elif embedding_type == 'simple_with_comments':
            added = [diff[1] for diff in file['added']]
            deleted = [get_line_comment(file["filetype"])+diff[1]
                       for diff in file['deleted']]
            
            if args.code_merge_file:
                res.append((added, deleted))
            else: 
                file_res = " \n ".join(added+deleted)
                file_res = tokenizer(file_res, truncation=True,
                                    padding='max_length', max_length=args.block_size)
                res.append(file_res)

        elif embedding_type == 'commit_message':
            file_res = tokenizer(commit["message"], truncation=True,
                                 padding='max_length', max_length=args.block_size)
            res.append(file_res)

        file_counter +=1
        
        # Probably at this point we have too many changes and we should stop
        if file_counter > args.block_size and args.code_merge_file:
            break


    if args.code_merge_file and res != []:
        if embedding_type == 'sum':
            res = tokenizer(" ".join(res), truncation=True,
                padding='max_length', max_length=args.block_size)
            return [res] 
        else:
            added_lst = []
            deleted_lst = []
            for added, deleted in res:
                added_lst += added
                deleted_lst += deleted
            file_res = " ".join(added_lst+deleted_lst)
            file_res = tokenizer(file_res, truncation=True,
                                padding='max_length', max_length=args.block_size)
            return [file_res]

    return res


class EventsDataset(Dataset):
    def __init__(self, args, all_json, keys, balance=False):
        self.args = args
        self.before_backs = args.event_window_size_before
        self.after_backs = args.event_window_size_after
        self.final_list_tensors = []
        self.final_list_labels = []
        self.final_commit_info = []

        self.current_path = os.path.join(
            args.cache_dir, "events", f"events_{self.before_backs}_{self.after_backs}.json")
        self.timezones_path = os.path.join(
            args.cache_dir, "events", "timezones")
        self.cache = not args.recreate_cache
        self.hash_list = keys

        if self.cache and os.path.exists(self.current_path):
            logger.warning(f"Loading from cache - {self.current_path}")
            final_list_tensors, final_list_labels, final_commit_info = torch.load(
                self.current_path)
            self.final_list_tensors = final_list_tensors
            self.final_list_labels = final_list_labels
            self.final_commit_info = final_commit_info

        else:
            logger.warning(f"Creating from scratch - {self.current_path}")
            self.create_list_of_hashes(all_json)
            torch.save((self.final_list_tensors, self.final_list_labels,
                       self.final_commit_info), self.current_path)


    def create_list_of_hashes(self, all_json):
        repo_dict = {}
        with open(os.path.join(self.args.cache_dir, "events", "repo_metadata.json"), 'r') as f:
            all_metadata = json.load(f)
        for mhash in tqdm(list(self.hash_list)[:], leave=False):
            try:
                if mhash == "":
                    continue
                repo = all_json[mhash]["repo"]
                label = all_json[mhash]["label"]
                repo_name = repo.replace("/", "_")
                if repo_name not in repo_dict:
                    try:
                        cur_repo = pd.read_parquet(os.path.join(
                            self.args.cache_dir, "events", "gh_cve_proccessed", f"{repo_name}.parquet"))
                        cur_repo = fix_repo_idx(cur_repo)
                        cur_repo = fix_repo_shape(cur_repo)
                        repo_dict[repo_name] = cur_repo
                    except FileNotFoundError:
                        print(f"File not found: {repo_name}")
                        continue
                else:
                    cur_repo = repo_dict[repo_name]

                wanted_row = cur_repo.index[cur_repo['Hash'] == mhash].tolist(
                )
                if len(wanted_row) == 0:
                    continue
                assert len(wanted_row) == 1, "Hash is not unique"
                wanted_row = wanted_row[0]
                self.cur_repo_column_names = cur_repo.columns
                event_window = get_event_window(
                    cur_repo, wanted_row, before_backs=self.before_backs, after_backs=self.after_backs )
                event_window = add_metadata(
                    self.timezones_path, all_metadata, event_window, repo_name)
                event_window = event_window.drop(["Hash", "Vuln"], axis=1)
                event_window = event_window.fillna(0)
                self.final_list_tensors.append(event_window.values)
                self.final_list_labels.append(label)
                self.final_commit_info.append(
                    {"name": repo_name, "hash": mhash, "label": label})

            except KeyError as e:
                print(e)

  

    def __len__(self):
        return len(self.final_list_tensors)

    def __getitem__(self, idx):
        item = self.final_list_tensors[idx]
        label = self.final_list_labels[idx]
        item = torch.from_numpy(item.astype(float)).float()
        return item, label


def create_datasets(DatasetClass, orchestrator_location=r"cache_data/orc", cache=True, **kwargs):
    res = []
    for set_name in ["train", "validation", "test"]:
        with open(os.path.join(orchestrator_location, f"{set_name}_details.pickle"), "rb") as f:
            cur_set = pickle.load(f)
        if cache and os.path.exists(f"{set_name}_dataset.pkl"):
            cur_set = pickle.load(open(f"{set_name}_dataset.pkl", "rb"))
        else:
            cur_set = DatasetClass(cur_set, set_name, cache=cache, **kwargs)
            pickle.dump(cur_set, open(f"{set_name}_dataset.pkl", "wb"))
        res.append(cur_set)
    return res
