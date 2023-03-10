import argparse
import signal
import pandas as pd
import os
from csv import writer
from tqdm import tqdm

subreddits = []  # contains list of created subreddit.csv files
file = []
row_number = 0

stop = False

def handler(sig, frame):
    global stop
    stop = True

signal.signal(signal.SIGINT, handler)

cols = ['author', 'date_post', 'id_post', 'number_post', 'subreddit', 'text_post', 'thread']

parser = argparse.ArgumentParser()
parser.add_argument('--input_ndjson', type=str, default="E:\\School\\U-of-E-Schoolwork\\MLP\\data\\ndjson\\reddit.ndjson")
parser.add_argument('--output_dir', type=str, default="E:\\School\\U-of-E-Schoolwork\\MLP\\data\\csv")
parser.add_argument('--array_dir', type=str, default='C:\\Users\\mikad\\PycharmProjects\\MLP-Manosphere')
parser.add_argument('--chunksize', type=int, default=500)
parser.add_argument('--total', type=int, default=29482387)
parser.add_argument('--startpoint', type=int, default=0)
args = parser.parse_args()

def append_csv(values: list, index:int) -> None:
    f = file[index]
    writer_obj = writer(f)
    writer_obj.writerow(values)

def write_csv(values: list) -> None:
    subreddits.append(values[4])

    df = pd.DataFrame(data=[values], columns=cols)
    df.to_csv(os.path.join(args.output_dir, f"{values[4]}.csv"), index=None)
    f = open(os.path.join(args.output_dir, f"{values[4]}.csv"), 'a', encoding='utf-8', newline='')
    file.append(f)

def fun_csv(series: list) -> None:
    sr = series[4]
    if sr in subreddits:
        idx = subreddits.index(sr)
        append_csv(series, idx)
    else:
        write_csv(series)

def save_information():
    pass

with pd.read_json(
        path_or_buf=args.input_ndjson,
        lines=True,
        chunksize=args.chunksize) as reader:

    # cycle through the iterator
    for chunk in tqdm(reader, total=round((args.total/args.chunksize) + 0.5)):
        if row_number*args.chunksize > args.startpoint:
            if not stop:
                chunk.apply(fun_csv, axis=1, raw=True)
                row_number += 1
            else:
                print(f'Manually ended program with idx = {row_number*args.chunksize}')
                print('Closing files')
                for f in file:
                    f.close()
        else:
            continue
