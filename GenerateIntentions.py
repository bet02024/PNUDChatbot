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


# Adds custom stopword into spaCy default stopword list
nlp.Defaults.stop_words |= {"o", "arcgis", "browser", "support", "date", "sort", "count", "vpn", "archive", "twitter", "coral", "gable", "coralgablea", "instagram", "linkedin", "tv", "large", "small", "non", "map", "key", "invalid", "response", "error"}


print(nlp.Defaults)

# Calculates the frequency of words in a document
def remove_stopwords(sentence):
    words = nlp(sentence)
    processed_sentence = ' '.join([token.text for token in words if token.is_stop != True ])
    words = word_tokenize(processed_sentence)
    words = [w for w in words if not w in stop_words]
    processed_sentence = ' '.join(words)
    return processed_sentence

def remove_punctuation_special_chars(sentence):
    sentence = nlp(sentence)
    processed_sentence = ' '.join([token.text for token in sentence
    if token.is_punct != True and
        token.is_quote != True and
        token.is_bracket != True and
        token.is_currency != True and
        token.is_digit != True])
    return processed_sentence# Lemmatization process with spaCy

def lemmatize_text(sentence):
    sentence = nlp(sentence)
    processed_sentence = ' '.join([word.lemma_ for word in sentence])
    return processed_sentence

def remove_numbers(text):
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]
    return ' '.join(words)

def remove_special_chars(text):
    bad_chars = ["%", "#", '"', "*"]
    for i in bad_chars:
        text = text.replace(i, '')
    return text

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
