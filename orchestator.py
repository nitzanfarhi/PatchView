import json
import random
import git
import traceback
from tqdm import tqdm
SEED = 0x1337

repo_commits_location = r"cache_data\repo_commits.json"
repo_commit_code_location = r"D:\multisource\commits"


TRAIN_RATE = 0.8
VALIDATION_RATE = 0.1
TEST_RATE = 0.1


def get_benign_commits(repo, security_commits):
    num_of_patch_commits = len(security_commits)
    number_of_retrieved_commits = 0
    repo = git.Repo(repo_commit_code_location + "\\" + repo.replace("/", "_"))
    all_commit_list = list(repo.iter_commits())
    random.shuffle(all_commit_list)
    for commit in all_commit_list:
        if number_of_retrieved_commits >= num_of_patch_commits:
            break
        if commit not in security_commits:
            number_of_retrieved_commits += 1
            yield commit.hexsha
    return


def main():
    random.seed(SEED)
    should_split_by_repos = False
    with open(r"cache_data\repo_commits.json", "r") as f:
        data = json.load(f)

    if should_split_by_repos:
        training_dict, validation_dict, testing_dict = split_by_repos(data)

    else:
        training_dict, validation_dict, testing_dict = split_randomly(data)

    with open("cache_data/orchestrator_training.json", "w") as f:
        json.dump(training_dict, f, indent=4)
    with open("cache_data/orchestrator_validation.json", "w") as f:
        json.dump(validation_dict, f, indent=4)
    with open("cache_data/orchestrator_testing.json", "w") as f:
        json.dump(testing_dict, f, indent=4)


def split_randomly(data):
    training_dict, validation_dict, testing_dict = {}, {}, {}
    data_keys = list(data)[:]

    all_commits = []
    for repo in tqdm(data_keys):
        try:
            all_commits.extend((repo, commit, 1) for commit in data[repo] if commit != "")
            all_commits.extend((repo, commit, 0) for commit in get_benign_commits(repo, data[repo]))
        except Exception as e:
            print(f"Failed to get commits for repo {repo}")
            traceback.print_exc()

    random.shuffle(all_commits)
    # split the data 
    num_of_commits = len(all_commits)
    num_of_commits_training = int(num_of_commits * TRAIN_RATE)
    num_of_commits_validation = int(num_of_commits * VALIDATION_RATE)

    training_list = all_commits[:num_of_commits_training]
    validation_list = all_commits[num_of_commits_training:
                                 num_of_commits_training + num_of_commits_validation]
    testing_list = all_commits[num_of_commits_training + num_of_commits_validation:]

    for commit_list,commit_dict in zip((training_list, validation_list, testing_list),(training_dict, validation_dict, testing_dict)):
        for commit in commit_list:
            repo, commit, label = commit
            if repo not in commit_dict:
                commit_dict[repo] = []
            commit_dict[repo].append((commit, label))

    return training_dict, validation_dict, testing_dict


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
