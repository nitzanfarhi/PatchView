import json
import os
import random
import subprocess
import git
import traceback
from tqdm import tqdm
from pydriller import Repository

SEED = 0x1337

repo_commits_location = r"cache_data\repo_commits.json"
COMMIT_REPO_PATH = r"D:\multisource\commits"
MAXIMAL_FILE_SIZE = 100000


TRAIN_RATE = 1
VALIDATION_RATE = 0
TEST_RATE = 0


def get_benign_commits(repo, security_commits):
    cur_repo = COMMIT_REPO_PATH + "\\" + repo.replace("/", "_")
    mall = subprocess.run("git rev-list --all --since=2015", stdout=subprocess.PIPE, cwd = cur_repo)
    mall = mall.stdout.decode('utf-8').split("\n")
    random.shuffle(mall)
    for commit in mall:
        if commit in security_commits:
            continue

        # pydriller_commit = get_commit_from_repo(cur_repo, commit)
        # if len(pydriller_commit.modified_files) == 0:
        #     continue

        yield commit
    return


def add_code_data_to_dict(file):
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


def get_commit_from_repo(cur_repo, hash):
    from pydriller import Repository
    res = next(Repository(cur_repo, single=hash).traverse_commits())
    return res


def prepare_dict(repo, commit, label):
    try:
        commit = get_commit_from_repo(
            os.path.join(COMMIT_REPO_PATH, repo.replace("/", "_")), commit)
        final_dict = {}
        final_dict["name"] = commit.project_name
        final_dict["hash"] = commit.hash
        final_dict["files"] = []
        final_dict["source"] = []
        final_dict["label"] = label
        final_dict["repo"] = repo
        final_dict["message"] = commit.msg
        if len(commit.modified_files) == 0:
            raise CommitNotFound
        for file in commit.modified_files:
            cur_dict = add_code_data_to_dict(file)
            if cur_dict is not None:
                final_dict["files"].append(cur_dict)
    except Exception:
        raise CommitNotFound

    return final_dict


def main():
    random.seed(SEED)

    with open(r"C:\Users\nitzan\local\analyzeCVE\last_train.json", "r") as mfile:
        train_details = json.load(mfile)
    with open(r"C:\Users\nitzan\local\analyzeCVE\last_val.json", "r") as mfile:
        val_details = json.load(mfile)
    with open(r"C:\Users\nitzan\local\analyzeCVE\last_test.json", "r") as mfile:
        test_details = json.load(mfile)
    mall = {}
    for row in train_details:
        mall[row[3]] = {"label": row[2], "repo": row[0]}
    for row in val_details:
        mall[row[3]] = {"label": row[2], "repo": row[0]}
    for row in test_details:
        mall[row[3]] = {"label": row[2], "repo": row[0]}

    new_mall = {}
    repo_set = dict()
    counter = 0
    for a, b in tqdm(mall.items()):
        counter += 1
        new_mall[a] = b
        if b['repo'] not in repo_set:
            commit = get_benign_commits(b['repo'], mall.keys())
            repo_set[b['repo']] = commit
        for _ in range(2):
            try:
                commit_hash = next(repo_set[b['repo']])
                if commit_hash in new_mall:
                    print(f"Already found {commit_hash}-{b['repo']}-{0}")
                    continue
                new_mall[commit_hash] = {"label": 0, "repo": b['repo']}
            except StopIteration:
                continue

    with open(os.path.join("cache_data", "orc", "orchestrator.json"), "w") as f:
        json.dump(new_mall, f, indent=4)


class CommitNotFound(Exception):
    """Raised when the commit is not found"""
    pass


def split_randomly(data):
    commit_dict = {}
    data_keys = list(data)[:]

    all_commits = []
    for repo in tqdm(data_keys):
        print(repo)
        positive_repo_counter = 0
        negative_repo_counter = 0
        for commit in data[repo]:
            if commit == "":
                continue
            try:
                commit_dict[commit] = prepare_dict(repo, commit, 1)
                positive_repo_counter += 1
            except CommitNotFound:
                print(commit)
                continue

        commit = get_benign_commits(repo, data[repo])
        while negative_repo_counter < positive_repo_counter:
            try:
                current_commit = next(commit)
            except StopIteration:
                break

            commit_hash = current_commit.hexsha
            try:
                commit_dict[commit_hash] = prepare_dict(repo, commit_hash, 0)
                negative_repo_counter += 1
            except CommitNotFound:
                continue

    random.shuffle(all_commits)

    return commit_dict


def split_by_repos(data):
    data_keys = list(data)[:]
    # random.shuffle(data_keys)

    num_of_repos = len(data_keys)
    num_of_repos_training = int(num_of_repos * TRAIN_RATE)
    num_of_repos_validation = int(num_of_repos * VALIDATION_RATE)
    num_of_repos_testing = int(num_of_repos * TEST_RATE)

    training_keys = data_keys[:num_of_repos_training]
    validation_keys = data_keys[num_of_repos_training:
                                num_of_repos_training + num_of_repos_validation]
    testing_keys = data_keys[num_of_repos_training + num_of_repos_validation:]

    print(f"Training size: {num_of_repos_training}")
    print(f"Validation size: {num_of_repos_validation}")
    print(f"Testing size: {num_of_repos_testing}")

    training_dict = {}
    validation_dict = {}
    testing_dict = {}

    for dict, keys in zip([training_dict, validation_dict, testing_dict], [training_keys, validation_keys, testing_keys]):
        for repo in tqdm(keys):
            try:
                dict[repo] = []
                for commit in data[repo]:
                    if commit == "":
                        continue
                    dict[repo].append((commit, 1))

                for commit in get_benign_commits(repo, data[repo]):
                    dict[repo].append((commit, 0))
            except git.exc.NoSuchPathError as e:
                print(e)
            except ValueError as e:
                print(e)
    return training_dict, validation_dict, testing_dict


if __name__ == "__main__":
    main()
