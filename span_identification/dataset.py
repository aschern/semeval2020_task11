# coding=utf-8
import glob
import os
from shutil import copyfile, rmtree
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_data(data_folder, propaganda_techniques_file):
    file_list = glob.glob(os.path.join(data_folder, "*.txt"))
    articles_content, articles_id = ([], [])
    for filename in sorted(file_list):
        with open(filename, "r", encoding="utf-8") as f:
            articles_content.append(f.read())
            articles_id.append(os.path.basename(filename).split(".")[0][7:])

    with open(propaganda_techniques_file, "r") as f:
        propaganda_techniques_names = [line.rstrip() for line in f.readlines()]
    
    return articles_content, articles_id, propaganda_techniques_names


def read_predictions_from_file(filename):
    articles_id, gold_spans = ([], [])
    with open(filename, "r") as f:
        for row in f.readlines():
            article_id, gold_span_start, gold_span_end = row.rstrip().split("\t")
            articles_id.append(article_id)
            gold_spans.append(tuple(int(el) for el in [gold_span_start, gold_span_end]))
    return articles_id, gold_spans


def group_spans_by_article_ids(span_list):
    data = {}
    for el in span_list:
        article_id, span = el[0], el[1]
        data.setdefault(article_id, [])
        data[article_id].append(span)
    return data


def get_train_dev_files(articles_id, articles_content, nlp, labels_path, train_file, dev_file, split_by_ids=True, 
                     dev_size=0.3, random_state=42):
    articles_content_dict = dict(zip(articles_id, articles_content))
    articles_id, gold_spans = read_predictions_from_file(labels_path)
    span_list = list(zip(articles_id, gold_spans))
    
    if split_by_ids:
        data = group_spans_by_article_ids(span_list)
        train_ids, dev_ids = train_test_split(np.unique(articles_id), test_size=dev_size, random_state=random_state)
        train_data = sorted([(key, value) for (key, value) in data.items() if key in train_ids])
        dev_data = sorted([(key, value) for (key, value) in data.items() if key in dev_ids])
    else:
        span_list_train, span_list_test = train_test_split(span_list, test_size=dev_size, random_state=random_state)
        train_data = sorted(group_spans_by_article_ids(span_list_train).items())
        dev_data = sorted(group_spans_by_article_ids(span_list_train).items())
        train_ids = [example[0] for example in train_data]
        dev_ids = [example[0] for example in dev_data]
    
    create_BIO_labeled(train_file, train_data, articles_content_dict, nlp)
    create_BIO_labeled(dev_file, dev_data, articles_content_dict, nlp)
    
    return train_ids, dev_ids
    
                    
def get_test_file(file, articles_id, articles_content, nlp):
    create_BIO_unlabeled(file, articles_id, articles_content, nlp)
    

def token_label_from_spans(pos, spans):
    for el in spans:
        if el[0] <= int(pos) < el[1]:
            return "PROP"
    return 'O'

                    
def create_BIO_labeled(file, data, articles_content_dict, nlp):
    prev_label = 'O'
    with open(file, 'w') as f:
        for article_id, spans in tqdm(data):
            text = articles_content_dict[article_id]
            tokens = [(token.idx, token.text) for token in nlp(text)]
            idx = np.array(tokens)[:,0]
            tokens = np.array(tokens)[:,1]
            prev_tok = '\n'
            
            for i in range(len(tokens)):
                tok = tokens[i].replace('\n', ' ').replace('\t', ' ').strip()
                if len(tok) != 0 and repr(tok) != repr('\ufeff') and repr(tok) != repr('\u200f'):
                    tok = tokens[i].strip().replace('\n', ' ').replace('\t', ' ')
                    label =  token_label_from_spans(idx[i], spans)
                    if label != 'O':
                        if prev_label != 'O':
                            label = 'I-' + 'PROP'
                        else:
                            label = 'B-' + 'PROP'
                    f.write(tok + '\t' + label + '\n')
                    prev_label = label
                    prev_tok = tok
                else:
                    if prev_tok != '\n':
                        f.write('\n')
                        prev_tok = '\n'
                    prev_label = 'O'

                    
def create_BIO_unlabeled(file, articles_id, articles_content, nlp):
    prev_label = 'O'
    with open(file, 'w') as f:
        for article_id, text in tqdm(zip(articles_id, articles_content)):
            tokens = [(token.idx, token.text) for token in nlp(text)]
            idx = np.array(tokens)[:,0]
            tokens = np.array(tokens)[:,1]
            prev_tok = '\n'
            
            for i in range(len(tokens)):
                tok = tokens[i].replace('\n', ' ').replace('\t', ' ').strip()
                if len(tok) != 0 and repr(tok) != repr('\ufeff') and repr(tok) != repr('\u200f'):
                    tok = tokens[i].strip().replace('\n', ' ').replace('\t', ' ')
                    label = 'O'
                    f.write(tok + '\t' + label + '\n')
                    prev_label = label
                    prev_tok = tok
                else:
                    if prev_tok != '\n':
                        f.write('\n')
                        prev_tok = '\n'
                    prev_label = 'O'

                    
def create_subfolder(subfolder, source_folder, articles_id):
    if os.path.exists(subfolder):
        rmtree(subfolder)
    os.makedirs(subfolder)
    for article_id in articles_id:
        file = 'article' + str(article_id) + '.txt'
        copyfile(os.path.join(source_folder, file), os.path.join(subfolder, file))
