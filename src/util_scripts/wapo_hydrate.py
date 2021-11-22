from newspaper import Article
from bs4 import BeautifulSoup
import pandas as pd 
import glob
from selenium import webdriver

authors = glob.glob("../datasets/wapo/*")
authors.sort()
df_dict = {"author":[], "link":[], "article":[]}

for author_file in authors:
    author_name = " ".join(author_file.split('/')[-1][:-4].split('-'))
    print(author_name)
    links = open(author_file, 'r').readlines()
    for link in links:
        full = ""
        article = Article(link)
        article.download()
        soup = BeautifulSoup(article.html, features="lxml")
        paras = soup.find_all("p", class_="font--article-body font-copy gray-darkest ma-0 pb-md")
        for para in paras:
            if(para.text!= "Read more:"):
                full+=para.text+'\n'
            else:
                break

        df_dict["author"].append(author_name)
        df_dict["link"].append(link)
        df_dict["article"].append(full)
        
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv('wapo.csv')

df = pd.DataFrame.from_dict(df_dict)
df.to_csv('wapo.csv')


    

    
