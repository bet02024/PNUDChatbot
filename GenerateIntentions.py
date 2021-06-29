import time
import csv
import sys
import json
import spacy
from spacy.lang.en import English # updated
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
import re
from nltk.tokenize import word_tokenize
nlp = spacy.load('en_core_web_sm')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import yaml

def generate_intentions_file(file_name, file_name_yml, file_name_out):
    dict_file = []
    fieldnames = ['Website', 'Description', 'ScrapedText', 'Keywords', 'UserIntentions']
    with open(file_name_out, 'w', newline='') as file_out:
        writer = csv.DictWriter(file_out, fieldnames=fieldnames)
        writer.writeheader()
        with open(file_name, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                url = row['Website']
                keywords = row['Keywords'].lower().replace("\n", "").replace("  ", " ").replace(", ", ",")
                description = row['Description'].lower()
                scrapedtext = row['ScrapedText'].lower()
                intention = "get_" + description.replace("\n", "").replace("  ", " ").replace(" ", "_")
                row['UserIntentions'] = intention
                keyword_list = keywords.split(",")
                int_item = {}
                int_item[intention] = keyword_list
                dict_file.append(int_item)
                writer.writerow({'Website': url, 'Description': description, 'ScrapedText': scrapedtext, 'Keywords':keywords, 'UserIntentions':intention})

        with open(file_name_yml, 'w') as file:
            documents = yaml.safe_dump(dict_file, file, explicit_start=True)

file_name = "./dataset_final.csv"
file_name_yml = "./intentions.yml"
file_name_out = "./dataset_intentions.csv"

generate_intentions_file(file_name, file_name_yml, file_name_out)
