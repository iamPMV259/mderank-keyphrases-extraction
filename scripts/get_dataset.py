#coding=utf-8
import re
import codecs
import json
import os

from torch.utils.data import Dataset
from transformers import T5Tokenizer

import nltk
from nltk.corpus import stopwords

from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm

MAX_LEN = None
enable_filter = None
temp_en = None
temp_de = None

StanfordCoreNLP_path = '../../stanford-corenlp-full-2018-02-27'

stopword_dict = set(stopwords.words('english'))
en_model = StanfordCoreNLP(StanfordCoreNLP_path, quiet=True)
tokenizer = None



