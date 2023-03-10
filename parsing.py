import pandas as pd
import re
import csv

# for chunk in pd.read_json("ndjson/reddit.ndjson", lines=True, chunksize=10000):
#     df = chunk.reset_index()
#     for index, row in df.iterrows():
#         text = row["text_post"].replace("\n", "")
#         with open(f"subreddits/{row['subreddit']}.txt", "a") as f:
#             f.write(f"{text}\n")


# with open("subreddits/Braincels.txt", "r") as f:
#     with open("subreddits/Braincels2.txt", "w") as f2:
#         for line in f:
#             words = line.split(" ")
#             if len(words) > 128: 
#                 continue
#             if len(words) < 20:
#                 continue

#             line = re.sub(r'http\S+', '', line)
#             if "[removed]" not in line and "[deleted]" not in line:
#                 f2.write(line)


with open("csv/gymcels.csv") as f:
    with open("csv/gymcels2.csv", "w") as f2:
        reader = csv.reader(f)
        writer = csv.writer(f2)
        for row in reader:
            text_post = row[5]
            date_post = row[1]
            words = text_post.split(" ")

            if "[deleted]" in text_post or "[removed]" in text_post:
                continue
            if len(words) > 300 or len(words) < 20:
                continue
            
            text_post = re.sub(r'http\S+', '', text_post)
            text_post = text_post.encode('ascii',errors='ignore').decode('ascii').strip().replace(r'@\w+', '')

            writer.writerow([text_post, date_post])
            




# with open("subreddits/depression.txt", "r") as f:
#     print("Original file has:", len(f.readlines()))

# with open("subreddits/depression2.txt", "r") as f:
#     print("New file has:", len(f.readlines()))
            

    
