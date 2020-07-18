#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 13:43:06 2019

@author: deepank
"""
from selenium.webdriver.common.by import By
from selenium import webdriver
from bs4 import BeautifulSoup

driver = webdriver.Chrome()
driver.implicitly_wait(10)
driver.get('https://www.timeanddate.com/weather/united-arab-emirates/dubai/historic?month=12&year=2015')
val = driver.find_element_by_css_selector("table#wt-his")
driver.find_element_by_link_text('3 Dec').click()
text=val.text
d=text.split()
temp=[]
wind=[]
hum=[]
j=0
for i  in d:
    print(i)
    if 'km/h' in i:
        print(d[j-1])
        wind.append(d[j-1])
    j=j+1
print(wind)
print(text)
print(max(wind))

'''
r = requests.get(URL) 
soup = BeautifulSoup(r.content, 'html5lib') 
table = soup.find('div', attrs = {'class':'row pdflexi'})
td1=table.find('div',attrs={'class':'tb-scroll'})
td2=td1.find('tbody')
td3=td2.findAll('tr')
hum=[]
temp=[]
wind=[]
for row in td3:
    a=row.findAll('td')
    temp.append(int(a[1].text.split('\xa0')[0]))
    wind.append(int(a[3].text.split(' ')[0]))
    hum.append(int(a[5].text.split('%')[0]))
print(max(temp),max(wind),max (hum))
'''

