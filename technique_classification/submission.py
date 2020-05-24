# coding=utf-8
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict
from sklearn.utils.extmath import softmax
from sklearn.metrics import accuracy_score, f1_score
from nltk.corpus import stopwords
import string
import pickle
import os
from unidecode import unidecode
from joblib import dump, load


def get_insides(data):  
    insides = defaultdict(dict)
    spans_coords = list(zip(data['span_start'].values, data['span_end'].values))
    labels = data['label'].values
    article_ids = data['article_id'].values
    for i in range(len(spans_coords)):
        for j in range(i):
            if article_ids[i] == article_ids[j]:
                if spans_coords[i][0] >= spans_coords[j][0] and spans_coords[i][1] <= spans_coords[j][1]:
                    if spans_coords[i][0] != spans_coords[j][0] or spans_coords[i][1] != spans_coords[j][1]:
                        insides[labels[i]][labels[j]] = insides[labels[i]].get(labels[j], 0) + 1
                if spans_coords[j][0] >= spans_coords[i][0] and spans_coords[j][1] <= spans_coords[i][1]:
                    if spans_coords[j][0] != spans_coords[i][0] or spans_coords[j][1] != spans_coords[i][1]:
                        insides[labels[j]][labels[i]] = insides[labels[j]].get(labels[i], 0) + 1
    return insides


def correct_preds_for_insides(preds, spans_coords, logits, insides, mapping, inverse_mapping):
    for i in range(len(preds)):
        for j in range(len(preds)):
            if spans_coords[j][0] >= spans_coords[i][0] and spans_coords[j][1] <= spans_coords[i][1]:
                if spans_coords[j][0] != spans_coords[i][0] or spans_coords[j][1] != spans_coords[i][1]:
                    def_i = preds[i]
                    def_j = preds[j]
                    log = softmax([logits[i]])[0]
                    login = softmax([logits[j]])[0]
                    def_prob_i = log[inverse_mapping[preds[i]]]
                    def_prob_j = login[inverse_mapping[preds[j]]]
                    while preds[j] not in insides.get(preds[i], []):
                        if log[inverse_mapping[preds[i]]] > login[inverse_mapping[preds[j]]]:
                            values = np.sort(login)[-2:]
                            if values[1] / (values[0] + 1e-6) > 1.4:
                                preds[i] = def_i
                                preds[j] = def_j
                                break
                            login[inverse_mapping[preds[j]]] = 0
                            preds[j] = mapping[np.argmax(login)]
                        else:
                            values = np.sort(log)[-2:]
                            if values[1] / (values[0] + 1e-6) > 1.4:
                                preds[i] = def_i
                                preds[j] = def_j
                                break
                            log[inverse_mapping[preds[i]]] = 0
                            preds[i] = mapping[np.argmax(log)]
    return preds

                            
def stem_spans(spans):
    ps = PorterStemmer()
    res = []
    for el in spans:
        result = " ".join(ps.stem(word) for word in word_tokenize(el.lower()))
        if len(result) > 0:
            res.append(result)
    return res


def get_train_instances(data, data_dir, save=True):
    train_instances = dict()
    stemmed_spans = stem_spans(data.span.values)
    labels = data.label.values
    for i in range(len(stemmed_spans)):
        if labels[i] != 'Repetition':
            span = stemmed_spans[i]
            train_instances.setdefault(span, set())
            train_instances[span].add(labels[i])
    if save:
        with open(os.path.join(data_dir, 'train_instances_train'), 'wb') as f:
            pickle.dump(train_instances, f)
    return train_instances
                            
    
def postprocess(x, mapping, inverse_mapping, insides, stop_words, ps, train_instances):
    spans_coords = list(zip(x['span_start'].values, x['span_end'].values))
    spans_source = x['span'].values
    spans_text = [' '.join([ps.stem(word) for word in word_tokenize(span.lower())]) for span in spans_source]
    spans = [' '.join([ps.stem(word) for word in word_tokenize(unidecode(span.lower())) 
                       if word not in stop_words and word not in string.punctuation]) for span in spans_source]
    
    counts = dict()
    for i in range(len(spans)):
        counts.setdefault(spans[i], set())
        counts[spans[i]].add(spans_coords[i][0])
    for el in counts:
        counts[el] = len(counts[el])
        
    preds = x['pred'].values
    logits = [np.array(log.split(), dtype=np.float32) for log in x['logits']]
    for i in range(len(preds)):
        log = logits[i]
        
        if counts[spans[i]] >= 3 or (counts[spans[i]] >= 2 and logits[i][inverse_mapping["Repetition"]] > 0.001):
            log[inverse_mapping["Repetition"]] = 100
        
        if counts[spans[i]] == 1 and (logits[i][inverse_mapping["Repetition"]] < 0.99 or len(spans[i].split()) <= 1):
            log[inverse_mapping["Repetition"]] = 0
        
        for prediction in train_instances.get(spans_text[i], set()):
            log[inverse_mapping[prediction]] += 0.5
        if spans_source[i].startswith('#'):
            log[inverse_mapping['Slogans']] = 20
         
        
        prev_same = []
        for j in range(i):
            if spans_coords[j][0] == spans_coords[i][0] and spans_coords[j][1] == spans_coords[i][1]:
                prev_same.append(j)
        if len(prev_same) > 0:
            for prediction in preds[prev_same]:
                log[inverse_mapping[prediction]] = 0
        
        logits[i] = log
        preds[i] = mapping[np.argmax(log)]
        
    x["pred"] = correct_preds_for_insides(preds, spans_coords, logits, insides, mapping, inverse_mapping)
    #x["pred"] = preds
    return x


def postprocess_predictions(predictions_logits, data, insides, train_instances):
    mapping = {i: el for i, el in enumerate(
        ['Appeal_to_Authority', 'Doubt', 'Repetition', 'Appeal_to_fear-prejudice', 'Slogans', 'Black-and-White_Fallacy',
         'Loaded_Language', 'Flag-Waving', 'Name_Calling,Labeling', 'Whataboutism,Straw_Men,Red_Herring', 
         'Causal_Oversimplification', 'Exaggeration,Minimisation', 'Bandwagon,Reductio_ad_hitlerum', 
         'Thought-terminating_Cliches']
    )}
    inverse_mapping = {b: a for (a, b) in mapping.items()}
    
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    
    predictions = np.argmax(predictions_logits, axis=1)
    data['pred'] = [mapping[p] for p in predictions]
    data['logits'] = [' '.join(np.array(log, dtype=str)) for log in predictions_logits]
    data = data.groupby('article_id', as_index=False).apply(postprocess, mapping, inverse_mapping, insides,
                                                            stop_words, ps, train_instances)
    return np.array(data["pred"].values)


def softmax_with_temperature(z, T): 
    z = z / T 
    max_z = np.max(z, axis=1).reshape(-1, 1) 
    exp_z = np.exp(z - max_z)
    return exp_z / np.sum(exp_z, axis=1).reshape(-1, 1)


def create_submission_file(predicted_logits_files, train_file_path, dev_file_path, test_file_path, 
                        article_ids, span_starts, span_ends, output_file, weights=None, data_dir=None, agg_model=None): 
    data_train = pd.read_csv(train_file_path, sep='\t')
    data_eval = pd.read_csv(dev_file_path, sep='\t')
    #data_train = pd.concat([data_train, data_eval], ignore_index=True)
    
    insides = get_insides(data_train)
    train_instances = get_train_instances(data_train, data_dir)
    
    data = pd.read_csv(test_file_path, sep='\t')
    
    if weights is None:
        weights = [1. / len(predicted_logits_files) for _ in range(len(predicted_logits_files))]
    assert len(weights) == len(predicted_logits_files)
    
    predictions_logits = None
    predictions_logits_list = []
    for file, weight in zip(predicted_logits_files, weights):
        with open(file, 'rb') as f:
            logits = pickle.load(f)
            if predictions_logits is None:
                predictions_logits = float(weight) * softmax_with_temperature(logits, 1)
            else:
                predictions_logits += float(weight) * softmax_with_temperature(logits, 1)
            if agg_model is not None:
                predictions_logits_list.append(logits)
    
    predictions = postprocess_predictions(predictions_logits, data, insides, train_instances)
    
    if agg_model is not None:
        clf = load(agg_model)
        predictions_sklearn_agg = clf.predict(np.concatenate(predictions_logits_list, axis=1))
        predictions_sklearn_agg[predictions_sklearn_agg == 'Repetition'] = predictions[predictions_sklearn_agg == 'Repetition']
        predictions_sklearn_agg[predictions == 'Repetition'] = 'Repetition'
        predictions = predictions_sklearn_agg
    
    with open(output_file, "w") as fout:
        for article_id, prediction, span_start, span_end in zip(article_ids, predictions, span_starts, span_ends):
            fout.write("%s\t%s\t%s\t%s\n" % (article_id, prediction, span_start, span_end))

            
def load_result(file):
    result = defaultdict(dict)
    with open(file, "r") as f:
        for line in f:
            article_id, prediction, spl, spr = line.split('\t')
            result[article_id].setdefault(prediction, [])
            result[article_id][prediction].append([int(spl), int(spr)])
    return result


def read_ground_truth(gt_file_path, label_names):
    ground_truth = []
    with open(gt_file_path, "r") as f:
        for line in f:
            gold_label = line.split('\t')[-1].strip()
            if gold_label in label_names:
                ground_truth.append(gold_label)
    return ground_truth

            
def eval_submission(result_file_path, gt_file_path):
    predictions = []
    with open(result_file_path, "r") as f:
        for line in f:
            prediction = line.split('\t')[1].strip()
            predictions.append(prediction)
    
    label_names = sorted(['Appeal_to_Authority', 'Doubt', 'Repetition', 'Appeal_to_fear-prejudice', 'Slogans',
                          'Black-and-White_Fallacy', 'Loaded_Language', 'Flag-Waving', 'Name_Calling,Labeling', 
                          'Whataboutism,Straw_Men,Red_Herring', 'Causal_Oversimplification', 'Exaggeration,Minimisation', 
                          'Bandwagon,Reductio_ad_hitlerum', 'Thought-terminating_Cliches'])
    ground_truth = read_ground_truth(gt_file_path, label_names)
    
    acc = accuracy_score(ground_truth, predictions)
    f1 = list(zip(label_names, f1_score(ground_truth, predictions, average=None, labels=label_names)))
    return acc, f1
