import os

import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
import argparse
import string
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
import json
import re
import pandas as pd
import time

nltk.download('stopwords')
nltk.download('punkt')
string.punctuation

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--min_df', type=int, default=20)
parser.add_argument('--chunksize', type=int, default=50)
args = parser.parse_args()

texts = list()
stopwords_list = stopwords.words('english')

num = 0

rm_web = 0
low = 0
rm_punc = 0
rm_sp = 0
rm_tok = 0
rm_sw = 0
join = 0


def process(values: pd.Series):
    global rm_web, low, rm_punc, rm_sp, rm_tok, rm_sw, join, num

    text = values['text_post']

    tic = time.perf_counter()
    # remove websites
    input_str = re.sub(r'http\S+', '', text)
    toc = time.perf_counter()
    rm_web += toc - tic

    tic = time.perf_counter()
    # lower caps
    input_str = input_str.lower()
    toc = time.perf_counter()
    low += toc - tic

    tic = time.perf_counter()
    # remove punctuation
    input_str = input_str.translate(
        str.maketrans("", "", string.punctuation)
    )
    toc = time.perf_counter()
    rm_punc += toc - tic

    tic = time.perf_counter()
    # remove trailing spaces
    input_str = input_str.strip()
    toc = time.perf_counter()
    rm_sp += toc - tic

    tic = time.perf_counter()
    # tokenize
    input_str = wordpunct_tokenize(input_str)
    toc = time.perf_counter()
    rm_tok += toc - tic

    tic = time.perf_counter()
    # remove stop words
    input_str_w_sw = [word for word in input_str if not word in stopwords_list]
    toc = time.perf_counter()
    rm_sw += toc - tic

    # filter out short texts
    if len(input_str_w_sw) < 5:
        pass
    else:
        tic = time.perf_counter()
        input_str = " ".join(input_str_w_sw)
        toc = time.perf_counter()
        join += toc - tic
        texts.append(input_str)
    num += 1


with pd.read_csv(args.data_path, chunksize=args.chunksize) as reader:
    for chunk in tqdm(reader):
        if num > 5:
            break
        else:
            chunk.apply(process, axis=1, raw=False)

print('vectorizing')

vectorizer = CountVectorizer(min_df=args.min_df)
bow_matrix = vectorizer.fit_transform(texts).toarray()

idx = np.where(bow_matrix.sum(axis=-1) > 0)
bow_matrix = bow_matrix[idx]

vocab = vectorizer.get_feature_names_out()

print(vocab)

print("===>saving files")

os.makedirs(os.path.join(args.output_dir, f'{args.name}'), exist_ok=True)

scipy.sparse.save_npz(os.path.join(args.output_dir, f'{args.name}', f'bow_matrix.npz'),
                      scipy.sparse.csr_matrix(bow_matrix))
with open(os.path.join(args.output_dir, f'{args.name}', f'vocab.txt'), 'w') as file:
    for line in vocab:
        file.write(line + '\n')

print('===>done.')

print("rm_web:", rm_web * 100000 / num,
      "\nlow:", low * 100000 / num,
      "\nrm_punc:", rm_punc * 100000 / num,
      "\nrm_sp:", rm_sp * 100000 / num,
      "\nrm_tok:", rm_tok * 100000 / num,
      "\nrm_sw:", rm_sw * 100000 / num,
      "\njoin:", join * 100000 / num)
