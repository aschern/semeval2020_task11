# coding=utf-8
import numpy as np
from unidecode import unidecode
import string
import nltk
from nltk.corpus import stopwords


def merge_spans(spans, articles_id, articles_content):
    res = dict()
    articles_content_dict = dict(zip(articles_id, articles_content))
    for article_id in spans:
        article = articles_content_dict[article_id]
        res[article_id] = []
        mask = np.zeros(len(article))
        for span in spans[article_id]:
            mask[span[0]: span[1]] = 1
        start = -1
        length = 0
        for i in range(len(mask)):
            if mask[i] == 0:
                if start != -1:
                    res[article_id].append((start, start + length))
                    start = -1
                    length = 0
            if mask[i] == 1:
                if start == -1:
                    start = i
                    length = 1
                else:
                    length += 1     
    return res


def correct_spans(spans, articles_id, articles_content):
    stop_words = set(stopwords.words('english'))
    res = dict()
    articles_content_dict = dict(zip(articles_id, articles_content))
    for article_id in spans:
        article = articles_content_dict[article_id]
        res[article_id] = []
        mask = np.zeros(len(article))
        for span in spans[article_id]:
            mask[span[0]: span[1] + 1] = 1
        start = -1
        length = 0
        for i in range(len(mask)):
            if mask[i] == 0:
                if start != -1:
                    end = start + length
                    
                    if unidecode(article[start - 1]) == '"':
                        start -= 1
                    else:
                        while not article[start].isalnum():
                            start += 1
                    if unidecode(article[end]) == '"':
                        end += 1
                    
                    if unidecode(article[end - 1]) != '"':
                        while not article[end - 1].isalnum():
                            end -= 1
                    if end - start > 1:
                        if article[start: end].lower() not in stop_words:
                            res[article_id].append((start, end))
                    '''
                    while article[end - 1].isspace():
                        end -= 1
                    if end > start:
                        res[article_id].append((start, end))
                    '''
                    start = -1
                    length = 0
            
            if mask[i] == 1:
                if start == -1:
                    start = i
                    length = 1
                else:
                    length += 1
        
        if start != -1:
            if unidecode(article[start - 1]) == '"':
                start -= 1
                length += 1
            if unidecode(article[start + length]) == '"':
                length += 1
            if unidecode(article[start + length - 1]) != '"':
                while not article[start + length - 1].isalnum():
                    length -= 1
            if length > 0:
                res[article_id].append((start, start + length))
    return res # merge_spans(res, articles_id, articles_content)


def get_spans_from_file(file, articles_id, articles_content, nlp):
    pred_spans = dict()
    with open(file, 'r') as f:
        for article_id, text in zip(articles_id, articles_content):
            pred_spans.setdefault(article_id, [])
            tokens = [(token.idx, token.text) for token in nlp(text)]
            idx = np.array(tokens)[:,0]
            tokens = np.array(tokens)[:,1]
            tokens = [token.strip().replace('\n', ' ').replace('\t', ' ') for token in tokens]
            
            i = 0
            start = -1
            for i in range(len(tokens)):
                tok = tokens[i]
                if len(tok) != 0 and repr(tok) != repr('\ufeff') and repr(tok) != repr('\u200f'):
                    token, label = f.readline().split('\t')
                    label = label.strip()
                    if label == 'B-PROP' or (label == 'I-PROP' and start == -1):
                        if start != -1:
                            pred_spans[article_id].append((start, int(idx[i - 1]) + len(tokens[i - 1])))
                        start = int(idx[i])                        
                    if label == 'O':
                        if start != -1:
                            pred_spans[article_id].append((start, int(idx[i - 1]) + len(tokens[i - 1])))
                        start = -1
                    assert token == tok
                    assert tok == text[int(idx[i]): int(idx[i]) + len(tok)]
                    prev_label = label
                    prev_tok = tok
                else:
                    if prev_tok != '\n':
                        f.readline()
                        prev_tok = '\n'
                    prev_label = 'O'
    
    return correct_spans(pred_spans, articles_id, articles_content)


def get_submission_format(predicted_labels_files, articles_id, articles_content, nlp, output_file):    
    agg_result = dict()
    for file in predicted_labels_files:
        result = get_spans_from_file(file, articles_id, articles_content, nlp)
        for el in result:
            agg_result[el] = agg_result.get(el, []) + result[el]
    agg_result = merge_spans(agg_result, articles_id, articles_content)
    
    with open(output_file, "w") as fout:
        for article_id, spans in agg_result.items():
            for span in spans:
                fout.write("%s\t%s\t%s\n" % (article_id, span[0], span[1]))
