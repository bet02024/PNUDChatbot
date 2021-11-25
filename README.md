# ONU PNUD Chatbot Project!

Scope

Explore the feasibility to create a Chatbot to help womens against domestic violence


# 1) Create a Dataset manually of all the user intentions, using existing conversations 

https://docs.google.com/spreadsheets/d/1Xjrs7jnW5wtXnRtB8XQx51w7KMhq0Xlj4f5jDeutNRX


# 2).  Data exploration & test with different Models

   #Run Model_Selection.py
 
 $python3 Model_Selection.py


# 3).  Clean the dataset 

input   "./dataset_clean.csv"
output1 = "./dataset_by_keywords.csv"
output2 = "./dataset_clean.csv". (updated)


# 4).  Test the Chatbot UI using RASA SDK

Pre-requsitites
    #Install RASA


    #Run GenerateIntentions.py
    $python3 GenerateIntentions.py

Generate all the Rasa dataset

		NLU user intentions File. ./data/nlu.yml
		Rules    ./data/rules.yml
		Domain.  ./domain.yml

Run Rasa Chatbot

    rasa init
    rasa train
    rasa run
