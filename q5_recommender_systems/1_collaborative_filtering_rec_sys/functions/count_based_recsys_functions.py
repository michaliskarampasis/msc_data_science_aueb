# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 22:42:10 2023

@author: mkarampasis
"""

# import libraries
import pandas as pd
from collections import defaultdict
from datasketch import MinHash, MinHashLSH
from termcolor import colored


# function to descretize ratings
def discretize_rating(ratings:float):
    
        '''
        Converts a given float rating to a string value
    
        '''
        polarity='A' # average
    
        if ratings<3: polarity='N' # negative
        elif ratings>3:polarity='P' # positive
    
        return polarity
        
    
    
# function to map each user to discretized ratings
def map_users_to_discretized_ratings(ratings:pd.DataFrame):
    
    '''
    Loads all the ratings submitted by each user
    and returns a dict that maps each user to discretized ratings
    
    '''
    print('[1] Map each user to discretized ratings')  
    
    # get all distinct users
    distinct_ids = set(ratings['user_id'])
    
    # store ratings per user
    ratings_ = {} 
    
    # for each user
    for id_ in distinct_ids: 
        
        # get the info for every rating submitted for this user
        my_ratings = ratings[ratings['user_id']==id_][['book_id','rating']]
        
        # discretize the ratings and attach them to the user or to the movie
        ratings_[id_] = set(zip(my_ratings['book_id'], my_ratings.rating.apply(discretize_rating)))
    
    
    print(colored('Succesfull', 'green'), end = '\n\n')
    
    return ratings_



# function create MinHashLSH inxdexes
def create_user_based_LSH_index(ratings_:pd.DataFrame, # descretized ratings
                                jaccard_threshold:float=0.2, # lower sim bound for the index
                                index_weights=(0.2,0.8), # false pos and false neg weights
                                num_perm:int=1000, # number of random permutations
                                min_num:int=5): # entites with less than this many ratings are ignored):
    
    '''
    Creates a user-based LSH index 

    '''
    print('[2] Create user-based LSH indexes')  
    
    #create the index
    index = MinHashLSH(threshold = jaccard_threshold, weights=index_weights, num_perm=num_perm)
    
    #remember the hashes
    hashes = {}
    
    #counter to inform how many users have been indexed
    cnt = 0
    
    # total number of users to index
    N = len(ratings_)
    
    # for all entities 
    for user_id, my_ratings in ratings_.items():
        
        cnt+=1
        
        if cnt%10000==0:
            print(cnt,'out of',N,'users indexed.')
        
        if len(ratings_) < 5:
            continue # not enough ratings for this user
            
        # create a new hash for this user
        rating_hash = MinHash(num_perm = 1000)
        
        # for each rating associated with this user 
        for id_,pol in my_ratings: 
            
            # create a string
            s = str(id_)+'_'+pol
            
            # add the string to the hash for this user
            rating_hash.update(s.encode('utf8')) 
   
        # remember the hash for this user
        hashes[user_id] = rating_hash
        
        # index the user based on the hash    
        index.insert(user_id, rating_hash) 
        
        
    print('index created')
    print(colored('Succesfull', 'green'), end = '\n\n')
    
    return index, hashes



# function to compute jaccard similarities
def jaccard(s1:set,
            s2:set
           ):
    
    '''
    Computes the jaccard coeff between two sets s1 and s2
    '''
    return len(s1.intersection(s2))/len(s1.union(s2))



# function to find the neighbors for a given user
def get_neighbors(ratings_:pd.DataFrame, # descretized ratings
                  index:MinHashLSH, # index of the user based on its hash
                  hashes:MinHashLSH, # hashes regarding the user
                  user_id:int, # user whose neighbors are to be retrieve
                  jaccard_threshold:float=0.2 # lower sim bound for the index
                 ):
    
    '''
    Uses the index to find and return the neighbors with a certain sim threshold for a given user. 
    '''
    
    # hash based on the index for the selected user
    result = index.query(hashes[user_id])
    
    # stores neighbors
    neighbors=[] 
    
    # for each candidate neighbor 
    for neighbor_id in result: 
        
        # ignore the user itself 
        if neighbor_id == user_id:
            continue 
            
         # compute jaccard 
        jacc = jaccard(ratings_[neighbor_id],ratings_[user_id])
        
        # if the threshold is respected, add
        if jacc >= jaccard_threshold: 
            neighbors.append((neighbor_id,jacc))
        
    return neighbors



# function to return recommendations for a given user
def get_recommendation_LSH(books:pd.DataFrame, # books infos
                           ratings_:pd.DataFrame,  # descretized ratings
                           user_id:int, # given user
                           index,
                           hashes
                           ):
    
    '''
    Delivers user-based recommendations. 
    Given a specific user:
    - Go over all the books rated by all neighbors
    - Each book gets +2 if a neighbor liked it, -2 if a neighbor didn't like it, -1 if  neighbor was neutral
    - +2,-1,and -2 are scaled based on user sim
    - Sort the books by their scores in desc order
    - Go over the sorted book list. If the user has already rated the book, store its rating. Otherwise print.
    '''
    
    print(f'[3] The recommendation for user {user_id} is:') 
    
    # find user's neigbors
    neighbors = get_neighbors(ratings_, index, hashes,  user_id)
    
    # count the votes per book
    votes = defaultdict(int) 
    
    # for each neighbor 
    for neighbor,sim_val in neighbors: 
        
        # for each book rated by this neighbor
        for bid,pol in ratings_[neighbor]:
            
            # positive neighbor rating
            if pol=='P': 
                votes[bid]+=2*sim_val
            # negative     
            elif pol=='N': 
                votes[bid]-=2*sim_val
            # average     
            else: 
                votes[bid]-=1*sim_val

    
    # sort the books in desc order 
    srt = sorted(votes.items(),key=lambda x:x[1], reverse=True)
    
    print('\nI suggest the following books because they have received positive ratings from users who tend to like what you like:\n')
    
    # find previous ratings
    previous_ratings = {x:y for x,y in ratings_[user_id]}
    
    # count number of recommendations made
    cnt=0
    
    # define dict, to store already rated books
    already_rated = {}
    
    # define list to 
    final_recommendation = []
    
    for book, score in srt: # for each book 
        try:
            title = books.loc[book]['title'] # get the title
        except KeyError:
            title='placeholder'
            
        rat = previous_ratings.get(book,None)
        
        if rat: # book already rated 
            
            #store rating
            already_rated[book] = rat
        
            continue
            
        # one more recommendation    
        cnt+=1 
        
        # add the recommendation to the list
        final_recommendation.append((book,title,score))
        
        if cnt == 10: break
            
    
    # add recommendation list to a df
    df_recommendations = pd.DataFrame(final_recommendation,columns=['book_id','title','similarity_value'])
    
    print(df_recommendations[['book_id','title']].to_string(index=False))
    
    print(colored('Succesfull', 'green'), end = '\n\n')
    
    return df_recommendations, already_rated



# function to evaluate the recsys
def validation_of_user_based_recsys(ratings:pd.DataFrame,  # descretized ratings
                                    already_rated:dict, # already rated books
                                    user_id:int #given user
                                    ):
    
    '''
    Computes precision and recall of the recommendation in order to evaluate the recsys
    
    '''
    
    print('[4] Validation of above recommendations')
    
    # get all the books rated by this user
    user_books = ratings[ratings.user_id == user_id][['book_id','rating']]
    
    #convert them to a dict
    user_books = dict(zip(user_books.book_id,user_books.rating.apply(discretize_rating)))
    
    # number of positive rated books in recommended books
    num_positives = list(already_rated.values()).count('P')
    
    # total number of rated books
    total_books = len(already_rated)
    
    if total_books == 0:
        precision = 0
    else:
        # compute precision (%)
        precision = int(round(num_positives / total_books,2)*100)
    
    print(f'Precision: {precision} %')
    
    # total number of positive rated books in recommended books
    total_num_positives = list(user_books.values()).count('P')
    
    if total_num_positives == 0:
        recall = 0
    else:
        # compute recall
        recall = int(round(num_positives / total_num_positives,2)*100)
    
    print(f'Recall: {recall} %')
    print(colored('Succesfull', 'green'), end = '\n\n')
          
    return precision, recall