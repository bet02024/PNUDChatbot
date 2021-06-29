# Coral Gables Chatbot Project!

Scope

Explore the feasibility to create a Chatbot using the html text in the links of
 **The Coral Gables Smart Hub Portal**
https://cg-hubdev-cggis.opendata.arcgis.com/

Steeps

# 1) Create a Dataset manually of all the resources in the The Coral Gables Smart Hub Portal


https://docs.google.com/spreadsheets/d/1Xjrs7jnW5wtXnRtB8XQx51w7KMhq0Xlj4f5jDeutNRY/edit#gid=0

# 2) Create a HTML Scrapper in Python using selenium chromedriver


Pre-requsitites
    #Install chromedriver

    #Run ScrapBot.py
    $python3 ScrapBot.py


input   "./dataset.csv"
output = "./dataset_out.csv"

# 3).  Clean the html text


    #Run CleanDataset.py
    $python3 CleanDataset.py

input   "./dataset_out.csv"
output = "./dataset_clean.csv"


# 4).  Data exploration & test with different Models

    #Run Model_Selection.py
    $python3 Model_Selection.py
'

input   "./dataset_clean.csv"
output1 = "./dataset_by_keywords.csv"
output2 = "./dataset_clean.csv". (updated)



# 5).  Clean the dataset manually

(TODO)
Explore the dataset to clean duplicated user intentions
Explore the url resources to delete similar content
Use data augmentation to generate more data on the intentions with less keywords.


# 6).  Test the Chatbot UI

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
