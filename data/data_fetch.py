import pandas as pd 
import os 
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.io_utils import load_corpus_from_csv

texts, doc_names = load_corpus_from_csv(path="data/english_fact_checks.csv")
print(texts[4])