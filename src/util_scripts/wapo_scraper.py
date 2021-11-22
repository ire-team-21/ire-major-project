import re, urllib
import numpy as np
from bs4 import BeautifulSoup

from selenium import webdriver

class WapoScraper:
    def __init__(self):
        self.browser = webdriver.Firefox(executable_path="/home/green/Downloads/geckodriver-v0.30.0-linux64/geckodriver")
        self.browser.implicitly_wait(5)
    
    def load_page(self, page):
        self.browser.get(page)
        while(True):
            try:
                x = self.browser.find_element_by_css_selector(".skin")
                x.click()
            except:
                break
    
    def get_links(self, page):
        self.load_page(page)
        html = self.browser.page_source
        soup = BeautifulSoup(html)
        links = soup.findAll("a")
        t = set([ link["href"] for link in links if link.has_attr('href')])
        l = []
        for s in t:
            a = s.split('/')
            if(len(a) > 3):
                if a[3]=='opinions':
                    l.append(s+'\n')
        return l

scraper = WapoScraper()
links = open('wapo_authors.txt', 'r').readlines()

for link in links:
    author = link.split('/')[-2]
    articles = scraper.get_links(link)
    save_file = open(f'../datasets/wapo/{author}.txt', 'w')
    save_file.writelines(articles)
    save_file.close()

import glob 
authors = glob.glob("../datasets/wapo/*")

for author in authors:
    f = open(author, 'r')
    links = f.readlines()
    f.close()
    links = [l for l in links if len(l.split('/')) > 5]
    f = open(author, 'w')
    f.writelines(links)
    f.close()
