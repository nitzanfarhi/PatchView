import pandas as pd
import math

pldb = pd.read_csv("https://pldb.com/pldb.csv", low_memory=False)

ext_to_comment = {}
for i , row in pldb.iterrows():
    if type(row["fileExtensions"]) != str or type(row["lineCommentToken"]) != str:
        continue

    try:
        if " " in row.fileExtensions:
            for ext in row.fileExtensions.split(" "):
                ext_to_comment[ext] = row.lineCommentToken
        else:
            ext_to_comment[row.fileExtensions] = row.lineCommentToken
    except Exception as e:
        print(i)
        raise e
