import urllib.request
from inscriptis import get_text
from selenium import webdriver
import time
import csv
import sys
#url = "https://coralgablesfl.opengov.com/transparency#/11028/accountType=revenues&amp;breakdown=types&amp;currentYearAmount=cumulative&amp;currentYearPeriod=years&amp;graph=pie&amp;legendSort=desc&amp;proration=true&amp;saved_view=35992&amp;selection=D6C6B5F2756C65B5C52E6C663B8D477F&amp;year=NaN&amp;selectedDataSetIndex=6&amp;fiscal_start=earliest&amp;fiscal_end=latest"
def scrap_text(url):
    try:
        browser = webdriver.Chrome('./chromedriver')
        html =  browser.get(url)
        time.sleep(20)
        htmlSource = browser.page_source
        text = get_text(htmlSource)
        print(text)
        text = ''.join(text.splitlines())
        browser.quit()
        return text
    except:
        print("##Unexpected error:", sys.exc_info()[0])
        print(url)
        return ""

def getScrapedText(file_name, file_name_out):
    fieldnames = ['Website', 'Description', 'ScrapedText', 'Keywords', 'UserIntentions']
    with open(file_name_out, 'w', newline='') as file_out:
        writer = csv.DictWriter(file_out, fieldnames=fieldnames)
        writer.writeheader()

        with open(file_name, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                url = row['Website']
                description = row['Description']
                keywords = row['Keywords']
                userintentions = row['UserIntentions']
                text = scrap_text(url)
                writer.writerow({'Website': url, 'Description': description, 'ScrapedText': text, 'Keywords':keywords, 'UserIntentions':userintentions})

file_name = "./dataset.csv"
file_name_out = "./dataset_out.csv"

getScrapedText(file_name, file_name_out)
