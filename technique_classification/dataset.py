import codecs
import glob
import os
import numpy as np
import pandas as pd
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.model_selection import train_test_split


def read_articles_from_file_list(folder_name, file_pattern="*.txt"):
    file_list = glob.glob(os.path.join(folder_name, file_pattern))
    articles = {}
    article_id_list, sentence_id_list, sentence_list = ([], [], [])
    for filename in sorted(file_list):
        article_id = os.path.basename(filename).split(".")[0][7:]
        with codecs.open(filename, "r", encoding="utf8") as f:
            articles[article_id] = f.read()
    return articles


def read_predictions_from_file(filename):
    articles_id, span_starts, span_ends, gold_labels = ([], [], [], [])
    with open(filename, "r") as f:
        for row in f.readlines():
            article_id, gold_label, span_start, span_end = row.rstrip().split("\t")
            articles_id.append(article_id)
            gold_labels.append(gold_label)
            span_starts.append(span_start)
            span_ends.append(span_end)
    return articles_id, span_starts, span_ends, gold_labels


def load_data(data_folder, labels_file):
    articles = read_articles_from_file_list(data_folder)
    ref_articles_id, ref_span_starts, ref_span_ends, labels = read_predictions_from_file(labels_file)
    return articles, ref_articles_id, ref_span_starts, ref_span_ends, labels


def sents_token_bounds(text):
    sents_starts = []
    for start, end in PunktSentenceTokenizer().span_tokenize(text):
        sents_starts.append(start)
    sents_starts.append(100000)
    return np.array(sents_starts)


def clear(text):
    return text.strip().replace('\t', ' ').replace('\n', ' ')


def get_context(article, span_start, span_end):
    bounds = sents_token_bounds(article)
    context_start = bounds[np.where(bounds <= span_start)[0][-1]]
    context_end = bounds[np.where(bounds >= span_end)[0][0]]
    return clear(article[context_start:context_end])


def balance_pandas(data):
    lst = [data]
    max_size = data['label'].value_counts().max()
    for class_index, group in data.groupby('label'):
        lst.append(group.sample(max_size - len(group), replace=True))
    return pd.concat(lst)


def dataset_to_pandas(articles, ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels):
    data = pd.DataFrame.from_dict({'article_id': ref_articles_id, 
              'article': [articles[id] for id in ref_articles_id], 
              'span_start': np.array(ref_span_starts).astype(int), 
              'span_end': np.array(ref_span_ends).astype(int),
              'label': train_gold_labels
             })
    data['span'] = data.apply(lambda x: clear(x['article'][x['span_start']:x['span_end']]), axis=1)
    data['context'] = data.apply(lambda x: get_context(x['article'], x['span_start'], x['span_end']), axis=1)
    return data[['article_id', 'span_start', 'span_end', 'span', 'context', 'label']]


def get_train_dev_files(articles, ref_articles_id, ref_span_starts, ref_span_ends, labels, train_file, dev_file,
                     split_by_ids=False, dev_size=0.3, random_state=40, balance=False, shuffle=True):
    data = dataset_to_pandas(articles, ref_articles_id, ref_span_starts, ref_span_ends, labels)
    if split_by_ids:
        train_ids, dev_ids = train_test_split(data.article_id.unique(), test_size=dev_size, random_state=random_state)
        train = data[data.article_id.isin(train_ids)]
        dev = data[data.article_id.isin(dev_ids)]
    else:
        train, dev = train_test_split(data, test_size=dev_size, random_state=random_state)
        
    if balance:
        train = balance_pandas(train)
    if shuffle:
        train = train.sample(frac=1).reset_index(drop=True)
    
    save_dataset(train, train_file)
    save_dataset(dev, dev_file)

    
def get_test_file(articles, ref_articles_id, ref_span_starts, ref_span_ends, labels, test_file):
    test = dataset_to_pandas(articles, ref_articles_id, ref_span_starts, ref_span_ends, labels)
    save_dataset(test, test_file)
    

def save_dataset(data, file_path):
    data.to_csv(file_path, sep='\t', index=False)
