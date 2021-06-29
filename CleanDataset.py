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

# Adds custom stopword into spaCy default stopword list
nlp.Defaults.stop_words |= {"o", "arcgis", "browser", "support", "date", "sort", "count", "vpn", "archive", "twitter", "coral", "gable", "coralgablea", "instagram", "linkedin", "tv", "large", "small", "non", "map", "key", "invalid", "response", "error"}


print(nlp.Defaults)

# Calculates the frequency of words in a document
def word_frequency(text):# all tokens that arent stop words or punctuations
    my_doc = nlp(text)
    words = [token.text for token in my_doc if token.is_stop != True and token.is_punct != True]# noun tokens that arent stop words or punctuations
    nouns = [token.text for token in my_doc if token.is_stop != True and token.is_punct != True and token.pos_ == "NOUN"]# verb tokens that arent stop words or punctuations
    verbs = [token.text for token in my_doc if token.is_stop != True and token.is_punct != True and token.pos_ == "VERB"]# five most common words
    word_freq = Counter(words)
    common_words = word_freq.most_common(5)
    print("---------------------------------------")
    print("5 MOST COMMON TOKEN")
    print(common_words)
    print("---------------------------------------")
    print("---------------------------------------")# five most common nouns
    noun_freq = Counter(nouns)
    common_nouns = noun_freq.most_common(5)
    print("5 MOST COMMON NOUN")
    print(common_nouns)
    print("---------------------------------------")
    print("---------------------------------------")# five most common verbs
    verb_freq = Counter(verbs)
    common_verbs = verb_freq.most_common(5)
    print("5 MOST COMMON VERB")
    print(common_verbs)
    print("---------------------------------------")
    print("---------------------------------------")# Removes stopwords from a sentence using spaCy (token.is_stop)



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

def clearScrapedText(file_name, file_name_out):
    fieldnames = ['Website', 'Description', 'ScrapedText', 'Keywords', 'UserIntentions']
    with open(file_name_out, 'w', newline='') as file_out:
        writer = csv.DictWriter(file_out, fieldnames=fieldnames)
        writer.writeheader()
        with open(file_name, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                url = row['Website']
                description = row['Description'].lower()
                scrapedtext = row['ScrapedText'].lower()
                #keywords = row['Keywords']
                text = description + " " + scrapedtext
                userintentions = row['UserIntentions']
                text = remove_special_chars(text)
                text = remove_punctuation_special_chars(text)
                text = remove_numbers(text)
                doc = nlp(text)
                #print(text)
                #print(doc)
                text = remove_stopwords(text)
                #text = lemmatize_text(text)
                print("######", url, " --- ", description)
                word_frequency(text)
                writer.writerow({'Website': url, 'Description': description, 'ScrapedText': scrapedtext, 'Keywords':text, 'UserIntentions':userintentions})

file_name = "./dataset_out.csv"
file_name_out = "./dataset_clean.csv"

clearScrapedText(file_name, file_name_out)
