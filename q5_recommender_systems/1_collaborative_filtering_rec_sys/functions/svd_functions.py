# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 23:44:42 2023

@author: mkarampasis
"""

import pandas as pd
import numpy as np
from termcolor import colored
from surprise import  SVD
from surprise.model_selection.validation import cross_validate
from sklearn.metrics import mean_squared_error

def svd_algorithm(data):
    
    # use the SVD algorithm.
    svd = SVD(n_factors=50)
    
    print('[1] Compute the RMSE of the SVD algorithm')
    # compute the RMSE of the SVD algorithm.
    cross_validate(svd, data, measures=['RMSE'],cv=5,verbose=True)
    print(colored('Succesfull', 'green'), end = '\n\n')
    
    print('[2] Create training set')
    # create a training set
    trainset = data.build_full_trainset()
    print(colored('Succesfull', 'green'), end = '\n\n')
    
    print('[3] Fit SVD')
    # fit the svd
    svd.fit(trainset)
    print(colored('Succesfull', 'green'), end = '\n\n')
    
    return svd




def recommend_surprise(uid:int,
              ratings:pd.core.frame.DataFrame,
              model,
              books:np.ndarray,
              rec_num:int
             ):
    
    #get all the ratings by this user
    my_ratings = ratings[ratings.user_id==uid]

    #zip the ratings into a dict
    already_rated = dict(zip(my_ratings.book_id,my_ratings.rating))

    pred_dict={}# store predicted ratings

    for index,row in books.iterrows(): # for every movie 

        pred_dict[row.book_id] = model.predict(uid=uid,iid= row.book_id).est# get the pred for this user
        
    # sort the movies by predicted ratings
    srt=sorted(pred_dict.items(),key=lambda x:x[1],reverse=True)
    
    rec_set=set()# set of movie ids to be recommended
    
    for mid,pred in srt: # for each movie id 
        if mid not in already_rated: # movie has not already been rated
            
            rec_set.add(mid) # add to the set
            
            if len(rec_set)==rec_num:break 
       
    # make a data frame with only the recommended movies 
    rec_df=pd.DataFrame(books[books.book_id.isin(rec_set)])
    
    #add the predicted rating as a new column
    rec_df['predicted_rating']=rec_df['book_id'].map(pred_dict)
    
    #sort the df by the new col
    rec_df=rec_df.sort_values(['predicted_rating'], ascending=False)
    
    print(f'[4] The recommendation for user {uid} is:')
    print(rec_df.loc[:,('book_id','title')].to_string(index=False))
    print(colored('Succesfull', 'green'), end = '\n\n')
    
    return rec_df




def validate(uid:int,
              ratings:pd.core.frame.DataFrame,
              model,
              books:np.ndarray
             ):
    
    #get all the ratings by this user
    my_ratings=ratings[ratings.user_id==uid]

    #zip the ratings into a dict
    already_rated=dict(zip(my_ratings.book_id,my_ratings.rating))

    pred_dict={}# store predicted ratings

    for index,row in books.iterrows(): # for every book 

        pred_dict[row.book_id] = model.predict(uid=uid,iid= row.book_id).est# get the pred for this user
        
    actual,pred=[],[]
    for bid in already_rated: # for each movie id 
        actual.append(already_rated[bid])
        pred.append(pred_dict[bid])
    
    rmse = mean_squared_error(actual,pred,squared=False)
    print('[5] Validation of above recommendations')
    print(f'RMSE:{round(rmse,2)}')
    print(colored('Succesfull', 'green'), end = '\n\n')
    
    return rmse

