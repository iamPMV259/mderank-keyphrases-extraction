#coding=utf-8
import re
import codecs
import json
import os
import sys
import aiofiles
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Dataset
from transformers import T5Tokenizer
from src.keyphrase_extractor import extract_keyphrases

import nltk
from nltk.corpus import stopwords

from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm
from nltk.stem import PorterStemmer
nltk.download('punkt')

def clean_labels(labels):
    clean_labels = {}
    for id in labels:
        label = labels[id]
        clean_label = []
        for kp in label:
            if kp.find(";") != -1:
                left, right = kp.split(";")
                clean_label.append(left)
                clean_label.append(right)
            else:
                clean_label.append(kp)
        clean_labels[id] = clean_label        
    return clean_labels

# --------------------
# Asynchronous file reading functions
# --------------------

async def get_long_data(file_path="data/nus/nus_test.json"):
    """ Load file.jsonl asynchronously """
    data = {}
    labels = {}
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
        json_text = await f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl['keywords'].lower().split(";")
                doc = jsonl['abstract']
                doc = re.sub(r'\. ', ' . ', doc)
                doc = re.sub(r', ', ' , ', doc)
                doc = doc.replace('\n', ' ')
                data[jsonl['name']] = doc
                labels[jsonl['name']] = keywords
            except Exception as e:
                raise ValueError(f"Error processing line {i}: {e}")
    labels = clean_labels(labels)
    return data, labels

async def get_duc2001_data(file_path="data/DUC2001"):
    pattern = re.compile(r'<TEXT>(.*?)</TEXT>', re.S)
    data = {}
    labels = {}
    for dirname, _, filenames in os.walk(file_path):
        for fname in filenames:
            if fname == "annotations.txt":
                infile = os.path.join(dirname, fname)
                async with aiofiles.open(infile, 'rb') as f:
                    text = await f.read()
                    text = text.decode('utf8')
                lines = text.splitlines()
                for line in lines:
                    left, right = line.split("@")
                    d = right.split(";")[:-1]
                    l = left
                    labels[l] = d
            else:
                infile = os.path.join(dirname, fname)
                async with aiofiles.open(infile, 'rb') as f:
                    text = await f.read()
                    text = text.decode('utf8')
                found = re.findall(pattern, text)
                if found:
                    data[fname] = found[0]
    labels = clean_labels(labels)
    return data, labels

async def get_inspec_data(file_path="data/Inspec"):
    data = {}
    labels = {}
    for dirname, _, filenames in os.walk(file_path):
        for fname in filenames:
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            if right == "abstr":
                async with aiofiles.open(infile, 'r', encoding='utf-8') as f:
                    text = await f.read()
                    text = text.replace("%", '')
                    data[left] = text
            elif right == "uncontr":
                async with aiofiles.open(infile, 'r', encoding='utf-8') as f:
                    text = await f.read()
                    text = text.replace("\n\t", ' ')
                    text = text.replace("\n", ' ')
                    label = text.split("; ")
                    labels[left] = label
    labels = clean_labels(labels)
    return data, labels

async def get_semeval2017_data(data_path="data/SemEval2017/docsutf8", labels_path="data/SemEval2017/keys"):
    data = {}
    labels = {}
    for dirname, _, filenames in os.walk(data_path):
        for fname in filenames:
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            async with aiofiles.open(infile, "r", encoding="utf-8") as fi:
                text = await fi.read()
                text = text.replace("%", '')
            data[left] = text.lower()
    for dirname, _, filenames in os.walk(labels_path):
        for fname in filenames:
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            async with aiofiles.open(infile, 'r', encoding='utf-8') as f:
                text = await f.read()
                text = text.strip()
                ls = text.splitlines()
                labels[left] = ls
    labels = clean_labels(labels)
    return data, labels

async def get_short_data(file_path="data/krapivin/kravipin_test.json"):
    data = {}
    labels = {}
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
        json_text = await f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl['keywords'].lower().split(";")
                abstract = jsonl['abstract']
                doc = abstract
                doc = re.sub(r'\. ', ' . ', doc)
                doc = re.sub(r', ', ' , ', doc)
                doc = doc.replace('\n', ' ')
                doc = doc.replace('\t', ' ')
                data[i] = doc
                labels[i] = keywords
            except Exception as e:
                raise ValueError(f"Error processing line {i}: {e}")
    labels = clean_labels(labels)
    return data, labels

# Simple functions that call the defined functions above
async def get_krapivin_data(file_path="data/krapivin/krapivin_test.json"):
    return await get_short_data(file_path)

async def get_nus_data(file_path="data/nus/nus_test.json"):
    return await get_long_data(file_path)

async def get_semeval2010_data(file_path="data/SemEval2010/semeval_test.json"):
    return await get_short_data(file_path)

async def get_dataset_data(dataset_name):
    if dataset_name == "duc2001":
        return await get_duc2001_data()
    elif dataset_name == "inspec":
        return await get_inspec_data()
    elif dataset_name == "krapivin":
        return await get_krapivin_data()
    elif dataset_name == "nus":
        return await get_nus_data()
    elif dataset_name == "semeval2010":
        return await get_semeval2010_data()
    elif dataset_name == "sameval2017":
        return await get_semeval2017_data()

# F1 score calculation function (not async because it's just computation)
def calculate_f1(keyphrases: list, ground_truth: list) -> float:
    """
    Calculate F1 score for predicted keyphrases compared to actual keyphrases.

    Args:
        keyphrases (list): List of predicted keyphrases.
        ground_truth (list): List of actual keyphrases (labels[id]).

    Returns:
        float: F1 score as a percentage.
    """
    # Find common keyphrases between predictions and actuals
    common = set(keyphrases) & set(ground_truth)
    
    # Calculate precision and recall
    precision = len(common) / len(keyphrases) if keyphrases else 0
    recall = len(common) / len(ground_truth) if ground_truth else 0
    
    # If no values, F1 = 0
    if precision + recall == 0:
        return 0.0
    
    # Calculate F1 score and convert to percentage
    f1 = 2 * precision * recall / (precision + recall)
    return f1 * 100

# Function to stem a phrase
def stem_phrase(phrase: str) -> str:
    """
    Convert a phrase to its stemmed form.
    Split the phrase into words, stem each word, and join them back in the original order.
    
    Args:
        phrase (str): Input phrase.
    
    Returns:
        str: Stemmed phrase.
    """
    stemmer = PorterStemmer()
    tokens = phrase.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

# Deduplication function
def dedup(input_list: list) -> list:
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def dedup_stem(input_list: list) -> list:
    return dedup([stem_phrase(item) for item in input_list])

# Function to write results to a JSON file (asynchronous)
def print_to_json(data_name, score):
    """
    Write evaluation results to a JSON file.
    
    Parameters:
      data_name (str): Dataset name.
      score (list): List of evaluation results.
    """
    average_score = sum(score) / len(score) if score else 0
    result = {
        "dataset": data_name,
        "average_score": average_score,
    }

    
    with open(f"results/{data_name}.json", "w") as outfile:
        json.dump(result, outfile)

# Asynchronous main function
async def main():
    dataset = ['duc2001', 'inspec', 'krapivin', 'nus', 'semeval2010', 'sameval2017']
    
    for data_name in dataset:
        data, labels = await get_dataset_data(data_name)
        scores = []
        cnt = 0
        print(data_name, ": ", len(data))
        for id in data:
            keyphrases = await extract_keyphrases(data[id])
            labels[id] = dedup_stem(labels[id])
            keyphrases = dedup_stem(keyphrases)
            
            print("**** ", id, " : --> ", cnt, " / ", len(data))
            cnt += 1
            score = calculate_f1(keyphrases, labels[id])
            scores.append(score)
        print_to_json(data_name, scores)

if __name__ == "__main__":
    asyncio.run(main())
