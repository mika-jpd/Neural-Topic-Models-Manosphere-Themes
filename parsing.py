import pandas as pd

# for chunk in pd.read_json("ndjson/reddit.ndjson", lines=True, chunksize=10000):
#     df = chunk.reset_index()
#     for index, row in df.iterrows():
#         text = row["text_post"].replace("\n", "")
#         with open(f"subreddits/{row['subreddit']}.txt", "a") as f:
#             f.write(f"{text}\n")


with open("subreddits/ForeverUnwanted.txt", "r") as f:
    with open("subreddits/ForeverUnwanted2.txt", "w") as f2:
        for line in f:
            words = line.split(" ")
            if len(words) > 128: 
                continue
            elif len(words) < 20:
                continue
            if "[removed]" not in line and "[deleted]" not in line:
                f2.write(line)

# with open("subreddits/depression.txt", "r") as f:
#     print("Original file has:", len(f.readlines()))

# with open("subreddits/depression2.txt", "r") as f:
#     print("New file has:", len(f.readlines()))
            

    
