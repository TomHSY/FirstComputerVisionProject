# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:25:30 2021

@author: tom-h
"""

### Importing modules ########################################################
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementClickInterceptedException, ElementNotVisibleException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import os
import urllib.request
import random as rd


### Defining functions #######################################################

def click_on_button(browser, xpath):
    try:
        WebDriverWait(browser, 2).until(EC.presence_of_element_located(
            (By.XPATH, xpath)))
        button = browser.find_element_by_xpath(xpath)
    
        button.click()
    except TimeoutException:
        print(
            "Loading took too much time! Element probably not presented, so we continue.")
    except:
        pass
    
def fill_entry(browser, xpath, content):
    field = browser.find_element_by_xpath(xpath)
    field.send_keys(content)
    field.send_keys(Keys.RETURN)

def scraping(search):
    
    options = webdriver.ChromeOptions()
    options.add_experimental_option('w3c', False)
    
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    
    # Getting the chromedriver from cache or download it from internet
    browser = webdriver.Chrome('C://Users//tom-h//.wdm//drivers//chromedriver//win32//93.0.4577.63//chromedriver.exe',
                               options=options)
    # browser.minimize_window()
    
    # getting home page
    browser.get(URL)
    
    click_on_button(browser, '//*[@id="L2AGLb"]/div')
    
    # filling email entry
    fill_entry(browser, 
               xpath = '//*[@id="sbtc"]/div/div[2]/input',
               content = search)
    
    click_on_button(browser, '//*[@id="sbtc"]/button/div/span/svg')
    
    
    soup = BeautifulSoup(browser.page_source, 'html.parser')
    
    images =[]
    
    for item in soup.find_all('img'):
        
        try:
            print(item['src'])
            images.append(item['src'])
        except:
            pass
    
    print(f'{len(images)} trouvées')

    browser.close()
    
    return images
    
def saving(search, headers, images): 
    directory = f'scrapped_images_{search}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i, image_url in enumerate(images):
        
        try:
            request_ = urllib.request.Request(image_url, None, headers)  # The assembled request
            response = urllib.request.urlopen(request_)  # store the response
    
        except Exception as e:
            print(e)
        
        tmp = os.getcwd().replace("\\","/")
        name = rd.uniform(i,i+1000)
        path = f'{tmp}/{directory}/{search}_{name}.png'
        
        try:
            with open(path, 'wb') as f:
                f.write(response.read())
        except:
            pass
    
def main(search, D, URL):
    
    headers = {"User-Agent" : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'+
              'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 '+
              'Safari/537.36'}
    
    for word in D[search]:
        
        images = scraping(word)
        saving(search, headers, images)
        
        
        
### Script execution #########################################################
D = {'fork':['fork','fourchette'],
     'knife':['kitchen knife','couteau'],
     'spoon':['cuillère','spoon','petite cuillère','cuillère à soupe']}  
   
URL = """https://images.google.com/"""
    
            
main('fork', D, URL)
main('knife', D, URL)
main('spoon', D, URL)