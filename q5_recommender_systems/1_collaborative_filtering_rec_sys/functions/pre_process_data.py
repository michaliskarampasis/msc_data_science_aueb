# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 22:37:02 2023

@author: mkarampasis
"""

import pandas as pd
from termcolor import colored

# function to import data
def import_data(path_books,path_ratings):
    
    print('Import book data')
    #load books dataset
    books = pd.read_csv(path_books)
    print(colored('Succesfull', 'green'), end = '\n\n')
    
    print('Import rating data')
    #load ratings dataset
    ratings = pd.read_csv(path_ratings)
    print(colored('Succesfull', 'green'), end = '\n\n')
    
    return books, ratings

# function to select user
def select_user():
    
    user_id = int(input('Please provide a user ID (eg. a number from 1 to 53424): '))
    
    print(colored('Succesfull', 'green'), end = '\n\n')
    
    return user_id
