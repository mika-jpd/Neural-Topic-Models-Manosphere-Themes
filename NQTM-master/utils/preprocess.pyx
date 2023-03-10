import os
import cython
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

cdef list texts = []
cdef list stopwords_list = stopwords.words('english')

cdef void process(text):
    # remove websites
    input_str = re.sub(r'http\S+', '', text)


    # lower caps
    input_str = input_str.lower()

    # remove punctuation
    input_str = input_str.translate(
        str.maketrans("", "", string.punctuation)
    )

    # remove trailing spaces
    input_str = input_str.strip()

    # tokenize
    input_str = wordpunct_tokenize(input_str)

    # remove stop words
    input_str_w_sw = [word for word in input_str if not word in stopwords_list]

    # filter out short texts
    if len(input_str_w_sw) < args.min_df:
        pass
    else:
        input_str = " ".join(input_str_w_sw)
        texts.append(input_str)

cdef void run():
    with pd.read_csv(args.data_path, chunksize=args.chunksize) as reader:
        for chunk in tqdm(reader):
            chunk.apply(lambda x: process(x['text_post']), axis=1, raw=False)

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