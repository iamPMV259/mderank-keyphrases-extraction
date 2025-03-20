#coding=utf-8
import re
import codecs
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Dataset
from transformers import T5Tokenizer
from src.keyphrase_extractor import extract_keyphrases

import nltk
from nltk.corpus import stopwords

from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm



def get_long_data(file_path="data/nus/nus_test.json"):
    """ Load file.jsonl ."""
    data = {}
    labels = {}
    with codecs.open(file_path, 'r', 'utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl['keywords'].lower().split(";")
                abstract = jsonl['abstract']
                fulltxt = jsonl['fulltext']
                doc = ' '.join([abstract, fulltxt])
                doc = re.sub('\. ', ' . ', doc)
                doc = re.sub(', ', ' , ', doc)
                doc = doc.replace('\n', ' ')
                data[jsonl['name']] = doc
                labels[jsonl['name']] = keywords
            except:
                raise ValueError
    return data,labels


def get_duc2001_data(file_path="data/DUC2001"):
    pattern = re.compile(r'<TEXT>(.*?)</TEXT>', re.S)
    data = {}
    labels = {}
    for dirname, dirnames, filenames in os.walk(file_path):
        for fname in filenames:
            if (fname == "annotations.txt"):
                # left, right = fname.split('.')
                infile = os.path.join(dirname, fname)
                f = open(infile,'rb')
                text = f.read().decode('utf8')
                lines = text.splitlines()
                for line in lines:
                    left, right = line.split("@")
                    d = right.split(";")[:-1]
                    l = left
                    labels[l] = d
                f.close()
            else:
                infile = os.path.join(dirname, fname)
                f = open(infile,'rb')
                text = f.read().decode('utf8')
                text = re.findall(pattern, text)[0]
                data[fname] = text
    return data,labels

def get_inspec_data(file_path="data/Inspec"):

    data={}
    labels={}
    for dirname, dirnames, filenames in os.walk(file_path):
        for fname in filenames:
            left, right = fname.split('.')
            if (right == "abstr"):
                infile = os.path.join(dirname, fname)
                f=open(infile)
                text=f.read()
                text = text.replace("%", '')
                data[left]=text
            if (right == "uncontr"):
                infile = os.path.join(dirname, fname)
                f=open(infile)
                text=f.read()
                text = text.replace("\n\t", ' ')
                text=text.replace("\n",' ')
                label=text.split("; ")
                labels[left]=label
    return data,labels

def get_semeval2017_data(data_path="data/SemEval2017/docsutf8",labels_path="data/SemEval2017/keys"):

    data={}
    labels={}
    for dirname, dirnames, filenames in os.walk(data_path):
        for fname in filenames:
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            # f = open(infile, 'rb')
            # text = f.read().decode('utf8')
            with codecs.open(infile, "r", "utf-8") as fi:
                text = fi.read()
                text = text.replace("%", '')
            data[left] = text.lower()
            # f.close()
    for dirname, dirnames, filenames in os.walk(labels_path):
        for fname in filenames:
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            f = open(infile, 'rb')
            text = f.read().decode('utf8')
            text = text.strip()
            ls=text.splitlines()
            labels[left] = ls
            f.close()
    return data,labels

def get_short_data(file_path="data/krapivin/kravipin_test.json"):
    """ Load file.jsonl ."""
    data = {}
    labels = {}
    with codecs.open(file_path, 'r', 'utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl['keywords'].lower().split(";")
                abstract = jsonl['abstract']
                doc =abstract
                doc = re.sub('\. ', ' . ', doc)
                doc = re.sub(', ', ' , ', doc)
                doc = doc.replace('\n', ' ')
                doc = doc.replace('\t', ' ')
                data[i] = doc
                labels[i] = keywords
            except:
                raise ValueError
    return data,labels

def get_krapivin_data(file_path="data/krapivin/krapivin_test.json"):
    return get_short_data(file_path)

def get_nus_data(file_path="data/nus/nus_test.json"):
    return get_long_data(file_path)

def get_semeval2010_data(file_path="data/SemEval2010/semeval_test.json"):
    return get_short_data(file_path)

def get_dataset_data(dataset_name):
    if dataset_name == "duc2001":
        return get_duc2001_data()
    elif dataset_name == "inspec":
        return get_inspec_data()
    elif dataset_name == "krapivin":
        return get_krapivin_data()
    elif dataset_name == "nus":
        return get_nus_data()
    elif dataset_name == "semeval2010":
        return get_semeval2010_data()
    elif dataset_name == "sameval2017":
        return get_semeval2017_data()
    
def calculate_f1(predicted, ground_truth, k)->float:
    """
    Calculate precision, recall, and F1@K.
    
    Parameters:
      predicted (list): List of predicted keyphrases.
      ground_truth (list): List of ground truth keyphrases.
      k (int): The cutoff for evaluation.
    
    Returns:
      tuple: precision, recall, and F1 score.
    """
    predicted_top_k = predicted[:k]
    common = set(predicted_top_k) & set(ground_truth)
    precision = len(common) * 1.0 / k if k > 0 else 0
    recall = len(common) * 1.0 / len(ground_truth) if ground_truth else 0
    f1 = 0
    if precision + recall > 0:
       f1 = 200.0 * precision * recall / (precision + recall)
    # return precision, recall, f1
    return f1


def print_to_json(data_name, k, score):
    """
    Print the evaluation results to a JSON file.
    
    Parameters:
      data_name (str): The name of the dataset.
      k (int): The cutoff for evaluation.
      score (list): The list of evaluation results.
    """
    average_score = sum(score) / len(score) if score else 0
    result = {
        "dataset": data_name,
        "top_k": k,
        "average_score": average_score,}
    with open(f"results/{data_name}_{k}.json", "w") as outfile:
        json.dump(result, outfile)


if __name__ == "__main__":
    dataset = ['duc2001', 'inspec', 'krapivin', 'nus', 'semeval2010', 'sameval2017']
    # dataset = ['krapivin']

    for data_name in dataset:
        data,labels = get_dataset_data(data_name) 
        # print(data_name)
        # print(data)
        # print(labels)
        # for i in range(10):
        #     print("========")     
      
        for k in [5, 10, 15]:
            score = []
            for id in data:
                keyphrases = extract_keyphrases(data[id], top_k=k)
                f1 = calculate_f1(keyphrases, labels[id], k)
                score.append(f1)
                # print(id, "({k})", " ==> ", f1)
            print_to_json(data_name, k, score)
        
        # for id in data:
        #     keyphrases = extract_keyphrases(data[id], top_k=5)
        #     f1 = calculate_f1(keyphrases, labels[id], 5)
        #     print(id, " ==> ", f1)
        #     print(keyphrases)
        #     print(labels[id])
        #     print()