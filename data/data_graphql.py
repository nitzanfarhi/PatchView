import csv
import itertools
import logging
import time
import pathlib
import collections

import requests
import os

from data.misc import safe_mkdir


class RepoNotFoundError(BaseException):
    pass


OUTPUT_DIRNAME = "graphql"

# copied from https://github.com/n0vad3v/get-profile-data-of-repo-stargazers-graphql
try:
    
    token = pathlib.Path(r"C:\secrets\github_token.txt").read_text()
except FileNotFoundError:
    token = pathlib.Path("/storage/nitzan/code/github_token.txt").read_text()

headers = {"Authorization": "token " + token.replace("\n","")}

generalQL = """
{{
  repository(name: "{0}", owner: "{1}") {{
    {2}(first: 100 {3}) {{	
          totalCount
          pageInfo {{
            endCursor
            hasPreviousPage
            startCursor
          }}
          edges {{
            cursor
            node {{
              createdAt
            }}
          }}
    }}
  }}
}}

"""

stargazer_query = """
{{
  repository(name: "{0}", owner: "{1}") {{
    stargazers(first: 100 {2}) {{	
        totalCount
        pageInfo {{
        endCursor
        hasPreviousPage
        startCursor
      }}
      edges {{
        starredAt
      }}
    }}
  }}
}}
"""

# todo check if we can find commits from other branches
commits_ql = """
{{
  repository(name: "{0}",owner: "{1}") {{
    object(expression: "{2}") {{
      ... on Commit {{
        history (first:100 {3}){{
          totalCount
          pageInfo{{
            endCursor
          }}
          nodes {{
            committedDate
            deletions
            additions
            oid
          }}
          pageInfo {{
            endCursor
          }}
        }}
      }}
    }}
  }}
}}
"""

branches_ql = """
{{
  repository(owner: "{0}", name: "{1}") {{
    refs(first: 50, refPrefix:"refs/heads/") {{
      nodes {{
        name
      }}
    }}
  }}
}}

"""

repo_meta_data = """
{{
  repository(owner: "{0}", name: "{1}") {{
    owner {{
      
      ... on User {{
        company
        isEmployee
        isHireable
        isSiteAdmin
        isGitHubStar
        isSponsoringViewer
        isCampusExpert
        isDeveloperProgramMember
      }}
      ... on Organization {{
        
        isVerified        
      }}
    }}
    isInOrganization
    createdAt
    diskUsage
    hasIssuesEnabled
    hasWikiEnabled
    isMirror
    isSecurityPolicyEnabled
    fundingLinks {{
      platform
    }}
    primaryLanguage {{
      name
    }}
    languages(first: 100) {{
      edges {{
        node {{
          name
        }}
      }}
    }}
  }}
}}

"""

attrib_list = [
    "vulnerabilityAlerts",
    "forks",
    "issues",
    "pullRequests",
    "releases",
    "stargazers",
]


def run_query(query):
    """sends a query to the github graphql api and returns the result as json"""
    counter = 0
    while True:
        request = requests.post(
            "https://api.github.com/graphql", json={"query": query}, headers=headers
        )
        if request.status_code == 200:
            return request.json()
        elif request.status_code == 502:
            raise RuntimeError(
                f"Query failed to run by returning code of {request.status_code}. {request}"
            )

        else:
            request_json = request.json()
            if "errors" in request_json and (
                "timeout" in request_json["errors"][0]["message"]
                or request_json["errors"]["type"] == "RATE_LIMITED"
            ):
                print("Waiting for an hour")
                print(request, request_json)
                counter += 1
                if counter < 6:
                    time.sleep(60 * 60)
                    continue
                break

            raise RuntimeError(
                f"Query failed to run by returning code of {request.status_code}. {query}"
            )


def flatten(d, parent_key="", sep="_"):
    """flatten a nested dict"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_commit_metadata(owner, repo):
    """get commit metadata from all branches"""
    res = run_query(repo_meta_data.format(owner, repo))
    if "data" not in res:
      print(res)
    if not res["data"]["repository"]:
        return None
    res = flatten(res["data"]["repository"])
    res["languages_edges"] = list(
        map(lambda lang: lang["node"]["name"], res["languages_edges"])
    )

    return res


def get_all_commits(owner, repo):
    """
    Get all commits from all branches
    """
    branch_lst = run_query(branches_ql.format(owner, repo))
    branch_lst = [
        res["name"] for res in branch_lst["data"]["repository"]["refs"]["nodes"]
    ]
    commit_date, additions, deletions, oids = [], [], [], []
    final_lst = []
    if "master" in branch_lst:
        final_lst.append("master")
    if "main" in branch_lst:
        final_lst.append("main")

    for branch in branch_lst:
        print(f"\t\t{branch}")
        cur_commit_date, cur_additions, cur_deletions, cur_oids = get_commits(
            owner, repo, branch
        )
        commit_date += cur_commit_date
        additions += cur_additions
        deletions += cur_deletions
        oids += cur_oids
    return commit_date, additions, deletions, oids


def get_commits(owner, repo, branch):
    """Get commits from a branch"""
    endCursor = ""  # Start from begining
    this_query = commits_ql.format(repo, owner, branch, endCursor)
    commit_date, additions, deletions, oid = [], [], [], []

    result = run_query(this_query)  # Execute the query
    if "data" in result and result["data"]["repository"]["object"] is not None:
        total_count = result["data"]["repository"]["object"]["history"]["totalCount"]
        for _ in range(0, total_count, 100):
            endCursor = result["data"]["repository"]["object"]["history"]["pageInfo"][
                "endCursor"
            ]
            for val in result["data"]["repository"]["object"]["history"]["nodes"]:
                if val is not None:
                    commit_date.append(val["committedDate"])
                    additions.append(val["additions"])
                    deletions.append(val["deletions"])
                    oid.append(val["oid"])

            result = run_query(
                commits_ql.format(repo, owner, branch, 'after:"{0}"'.format(endCursor))
            )
            if "data" not in result:
                print("Error3", result)
                break
    else:
        print("Error4", result)

    return additions, deletions, commit_date, oid


def get_stargazers(owner, repo):
    """
    Get all the stargazers of a repo
    """
    endCursor = ""  # Start from begining
    this_query = stargazer_query.format(repo, owner, endCursor)
    has_next_page = True
    staredAt = []
    result = run_query(this_query)  # Execute the query
    if "data" in result:
        total_count = result["data"]["repository"]["stargazers"]["totalCount"]
        for _ in range(0, total_count, 100):
            endCursor = result["data"]["repository"]["stargazers"]["pageInfo"][
                "endCursor"
            ]
            staredAt.extend(
                val["starredAt"]
                for val in result["data"]["repository"]["stargazers"]["edges"]
            )

            result = run_query(
                stargazer_query.format(repo, owner, 'after:"{0}"'.format(endCursor))
            )
            if "data" not in result:
                raise RuntimeError(f"result {result} does not contain data")
    else:
        logging.error(result)
        raise RuntimeError(
            f"Query failed to run by returning code of {result}. {this_query}"
        )
    return staredAt


def get_attribute(owner, repo, attribute):
    endCursor = ""  # Start from begining
    this_query = generalQL.format(repo, owner, attribute, endCursor)
    dates = []
    result = run_query(this_query)  # Execute the query
    if "data" in result:
        total_count = result["data"]["repository"][attribute]["totalCount"]
        for _ in range(0, total_count, 100):
            endCursor = result["data"]["repository"][attribute]["pageInfo"]["endCursor"]
            dates.extend(
                val["node"]["createdAt"]
                for val in result["data"]["repository"][attribute]["edges"]
            )

            result = run_query(
                generalQL.format(
                    repo, owner, attribute, 'after:"{0}"'.format(endCursor)
                )
            )
            if "data" not in result:
                break

    else:
        logging.error("Attribute acquire error:", result)
    return dates


def get_repo(output_dir, repo):
    safe_mkdir(os.path.join(output_dir, OUTPUT_DIRNAME))

    owner = repo.split("/")[0]
    repo = repo.split("/")[1]

    logging.warning(f"Getting repo {repo} from {owner}")
    res_dict = {}
    for attribute in attrib_list:
        logging.debug("\t" + attribute)
        if attribute == "stargazers":
            res_dict[attribute] = get_stargazers(owner, repo)
        elif attribute == "commits":
            (
                res_dict["additions"],
                res_dict["deletions"],
                res_dict["commit_date"],
                res_dict["oid"],
            ) = get_all_commits(owner, repo)
        else:
            res_dict[attribute] = get_attribute(owner, repo, attribute)

    with open(
        os.path.join(output_dir, OUTPUT_DIRNAME, f"{owner}_{repo}.csv"), "w", newline=""
    ) as outfile:
        writer = csv.writer(outfile)
        writer.writerow(res_dict.keys())
        writer.writerows(itertools.zip_longest(*res_dict.values()))


def get_date_for_commit(repo, commit):
    owner = repo.split("/")[0]
    repo = repo.split("/")[1]
    ql_query = """
    {{
      repository(owner: "{0}", name: "{1}") {{
        object(expression: "{2}") {{
          ... on Commit {{
            committedDate
          }}
        }}
      }}
    }}""".format(
        owner, repo, commit
    )
    result = run_query(ql_query)
    if "errors" in result:
        print("ERROR1", ql_query, result)
        raise RepoNotFoundError()
    if "data" in result and result["data"]["repository"]["object"] is not None:
        return result["data"]["repository"]["object"]["committedDate"]
    print("ERROR2", ql_query, result)
    raise RepoNotFoundError()


def get_date_for_alternate_proj_commit(proj_name, commit_hash):
    owner = proj_name.split("/")[0]
    repo = proj_name.split("/")[1]
    query = """{{
          search(query: "{0}", type: REPOSITORY, first: 100) {{
            repositoryCount
            edges {{
              node {{
                ... on Repository {{
                  nameWithOwner
                  name
                }}
              }}
            }}
          }}
        }}
    
    """

    result = run_query(query.format(repo))
    if "data" not in result:
        return None, None
    for res in result["data"]["search"]["edges"]:
        cur_repo = res["node"]["nameWithOwner"]
        if res["node"]["name"] != repo:
            continue
        url = "http://www.github.com/{0}/commit/{1}".format(cur_repo, commit_hash)
        f = requests.get(url)
        print(url, f.status_code)
        if f.status_code == 200:
            try:
                return cur_repo, get_date_for_commit(cur_repo, commit_hash)
            except RepoNotFoundError:
                pass

    return None, None
