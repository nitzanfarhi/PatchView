from misc import add_metadata
from torch.utils.data import Dataset, DataLoader
from data_utils import fix_repo_idx, fix_repo_shape, get_event_window
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

    def __init__(self, tokenizer, args, all_json, keys, embedding_type, balance=False):
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
        if balance == True:
            self.balance_data()

    def balance_data(self):
        pos_idxs = []
        neg_idxs = []

        for i, label in enumerate(self.final_list_labels):
            if label == 1:
                pos_idxs.append(i)
            else:
                neg_idxs.append(i)
        min_idxs = min(len(pos_idxs), len(neg_idxs))
        pos_idxs = pos_idxs[:min_idxs]
        neg_idxs = neg_idxs[:min_idxs]

        tmp_final_list_tensors = []
        tmp_final_list_labels = []
        tmp_final_commit_info = []
        for i in range(len(self.final_list_labels)):
            if i in pos_idxs or i in neg_idxs:
                tmp_final_list_tensors.append(self.final_list_tensors[i])
                tmp_final_list_labels.append(self.final_list_labels[i])
                tmp_final_commit_info.append(self.final_commit_info[i])

        self.final_list_tensors = tmp_final_list_tensors
        self.final_list_labels = tmp_final_list_labels
        self.final_commit_info = tmp_final_commit_info

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
        if os.path.exists(self.final_cache_list) and self.cache:
            logger.warning("Get final list from cache")
            with open(self.final_cache_list, 'rb') as f:
                cached_list = torch.load(f)
                self.final_commit_info = cached_list["final_commit_info"]
                self.final_list_tensors = cached_list["final_list_tensors"]
                self.final_list_labels = cached_list["final_list_labels"]
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
    def __init__(self, code_dataset, message_dataset, events_dataset):
        self.code_dataset = code_dataset
        self.message_dataset = message_dataset
        self.events_dataset = events_dataset

        self.merged_dataset = []
        self.merged_labels = []
        self.final_commit_info = []
        code_counter = 0
        message_counter = 0
        events_counter = 0
        logger.warning(f"Number of code samples: {len(code_dataset)}")
        logger.warning(f"Number of message samples: {len(message_dataset)}")
        logger.warning(f"Number of events samples: {len(events_dataset)}")

        in_counter = 0
        code_hash_list = [x["hash"] for x in code_dataset.final_commit_info]
        message_hash_list = [x["hash"]
                             for x in message_dataset.final_commit_info]
        events_hash_list = [x["hash"]
                            for x in events_dataset.final_commit_info]
        assert sorted(code_hash_list) == code_hash_list
        assert sorted(message_hash_list) == message_hash_list
        assert sorted(events_hash_list) == events_hash_list
        while code_counter < len(code_dataset) and message_counter < len(message_dataset) and events_counter < len(events_dataset):
            code_commit = code_dataset.final_commit_info[code_counter]
            message_commit = message_dataset.final_commit_info[message_counter]
            events_commit = events_dataset.final_commit_info[events_counter]

            if code_commit["hash"] == message_commit["hash"] == events_commit["hash"]:
                self.merged_dataset.append(
                    (code_dataset[code_counter][0], message_dataset[message_counter][0], events_dataset[events_counter][0]))
                self.merged_labels.append(code_dataset[code_counter][1])
                self.final_commit_info.append(
                    code_dataset.final_commit_info[code_counter])
                in_counter += 1
                # print(in_counter)
                code_counter += 1
                message_counter += 1
                events_counter += 1
            elif code_commit["hash"] < message_commit["hash"]:
                code_counter += 1
            elif message_commit["hash"] < events_commit["hash"]:
                message_counter += 1
            else:
                events_counter += 1

        logger.warning(
            "Number of merged samples before balancing: "+str(len(self.merged_dataset)))
        self.balance_dataset()
        logger.warning("Number of merged samples after balancing: " +
                       str(len(self.merged_dataset)))

    def balance_dataset(self):
        pos_idxs = []
        neg_idxs = []

        for i, label in enumerate(self.merged_labels):
            if label == 1:
                pos_idxs.append(i)
            else:
                neg_idxs.append(i)
        min_idxs = min(len(pos_idxs), len(neg_idxs))
        pos_idxs = pos_idxs[:min_idxs]
        neg_idxs = neg_idxs[:min_idxs]

        tmp_merged_dataset = []
        tmp_merged_labels = []
        tmp_final_commit_info = []
        for i in range(len(self.merged_labels)):
            if i in pos_idxs or i in neg_idxs:
                tmp_merged_dataset.append(self.merged_dataset[i])
                tmp_merged_labels.append(self.merged_labels[i])
                tmp_final_commit_info.append(self.final_commit_info[i])

        self.merged_dataset = tmp_merged_dataset
        self.merged_labels = tmp_merged_labels
        self.final_commit_info = tmp_final_commit_info

    def __getitem__(self, i):
        return self.merged_dataset[i], self.merged_labels[i]

    def get_info(self, i):
        return self.merged_info[i]

    def __len__(self):
        return len(self.merged_dataset)


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
            res = tokenizer(res, truncation=True,
                            padding='max_length', max_length=args.block_size)
            operation_list.append(res)

        elif opp == 'insert':
            res = INS_TOKEN + " " + after[b1:b2]
            res = tokenizer(res, truncation=True,
                            padding='max_length', max_length=args.block_size)
            operation_list.append(res)

        elif opp == 'delete':
            res = DEL_TOKEN + " " + before[a1:a2]
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
    for file in commit["files"]:
        if embedding_type == 'sum':
            embed_file_res = embed_file(file, tokenizer, args)
            if embed_file_res is not None:
                for embed in embed_file_res:
                    res.append(embed)

        elif embedding_type == 'simple':
            added = [diff[1] for diff in file['added']]
            deleted = [diff[1] for diff in file['deleted']]
            if not args.code_merge_file:
                file_res = " ".join(added+deleted)
                file_res = tokenizer(file_res, truncation=True,
                                     padding='max_length', max_length=args.block_size)
                res.append(file_res)
            else:
                res.append((added, deleted))

        elif embedding_type == 'simple_with_tokens':
            added = [ADD_TOKEN]+[diff[1]
                                 for diff in file['added']]+[tokenizer.sep_token]
            deleted = [DEL_TOKEN]+[diff[1] for diff in file['deleted']]
            file_res = " ".join(added+deleted)
            file_res = tokenizer(file_res, truncation=True,
                                 padding='max_length', max_length=args.block_size)
            res.append(file_res)

        elif embedding_type == 'simple_with_comments':
            added = [diff[1] for diff in file['added']]
            deleted = [get_line_comment(file["filetype"])+diff[1]
                       for diff in file['deleted']]

            file_res = " \n ".join(added+deleted)

            file_res = tokenizer(file_res, truncation=True,
                                 padding='max_length', max_length=args.block_size)
            res.append(file_res)

        elif embedding_type == 'commit_message':
            file_res = tokenizer(commit["message"], truncation=True,
                                 padding='max_length', max_length=args.block_size)
            res.append(file_res)
            break

    if args.code_merge_file and res != []:
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
        self.backs = args.event_window_size
        self.final_list_tensors = []
        self.final_list_labels = []
        self.final_commit_info = []

        self.current_path = os.path.join(
            args.cache_dir, "events", f"events.json")
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

        if balance == True:
            self.balance_data()

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
                event_window = get_event_window(
                    cur_repo, wanted_row, backs=self.backs)
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

    def balance_data(self):
        pos_idxs = []
        neg_idxs = []

        for i, label in enumerate(self.final_list_labels):
            if label == 1:
                pos_idxs.append(i)
            else:
                neg_idxs.append(i)
        min_idxs = min(len(pos_idxs), len(neg_idxs))
        pos_idxs = pos_idxs[:min_idxs]
        neg_idxs = neg_idxs[:min_idxs * self.args.balance_factor]

        tmp_final_list_tensors = []
        tmp_final_list_labels = []
        tmp_final_commit_info = []
        for i in range(len(self.final_list_labels)):
            if i in pos_idxs or i in neg_idxs:
                tmp_final_list_tensors.append(self.final_list_tensors[i])
                tmp_final_list_labels.append(self.final_list_labels[i])
                tmp_final_commit_info.append(self.final_commit_info[i])

        self.final_list_tensors = tmp_final_list_tensors
        self.final_list_labels = tmp_final_list_labels
        self.final_commit_info = tmp_final_commit_info

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
