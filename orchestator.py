import json
import random
import git
import traceback
from tqdm import tqdm
SEED = 0x1337

repo_commits_location = r"C:\Users\nitzan\local\analyzeCVE\data_collection\data\repo_commits.json"
repo_commit_code_location = r"C:\Users\nitzan\local\analyzeCVE\data_collection\data\commits"


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

    with open(r"C:\Users\nitzan\local\analyzeCVE\data_collection\data\repo_commits.json", "r") as f:
        data = json.load(f)

    data_keys = list(data)[:]
    random.shuffle(data_keys)

    training = []
    validation = []
    testing = []

    num_of_repos = len(data_keys)
    num_of_repos_training = int(num_of_repos * 0.7)
    num_of_repos_validation = int(num_of_repos * 0.2)
    num_of_repos_testing = int(num_of_repos * 0.1)

    training_keys = data_keys[:num_of_repos_training]
    validation_keys = data_keys[num_of_repos_training:num_of_repos_training + num_of_repos_validation]
    testing_keys = data_keys[num_of_repos_training + num_of_repos_validation:]

    print(f"Training size: {num_of_repos_training}")
    print(f"Validation size: {num_of_repos_validation}")
    print(f"Testing size: {num_of_repos_testing}")

    training_dict = {}
    validation_dict = {}
    testing_dict = {}

    for dict, keys in zip([training_dict, validation_dict, testing_dict],[training_keys, validation_keys, testing_keys]):
        for repo in tqdm(keys):
            try:
                dict[repo] = []
                for commit in data[repo]:
                    if commit == "":
                        continue
                    dict[repo].append((commit,1))

                for commit in get_benign_commits(repo, data[repo]):
                    dict[repo].append((commit,0))
            except git.exc.NoSuchPathError as e:
                print(e)
            except ValueError as e:
                print(e)


    with open("orchestrator_training.json","w") as f:
        json.dump(training_dict, f, indent=4)
    with open("orchestrator_validation.json","w") as f:
        json.dump(validation_dict, f, indent=4)
    with open("orchestrator_testing.json","w") as f:
        json.dump(testing_dict, f, indent=4)


if __name__ == "__main__":
    main()