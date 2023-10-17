# import libraries
import numpy as np
from collections import defaultdict
from random import random,sample
from dataclasses import dataclass
import csv 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# function to load pre-processed data regarding books
def load_books(input_file:str):
    
    # create class for books
    @dataclass
    class book: 
        authors:str
        original_publication_year:int
        original_title:str
        average_rating:float
        ratings_count_new:int
        per_positive_ratings:float
        per_negative_ratings:float
        per_average_ratings:float
        total_number_of_tags:int
        
    # title to index
    title_index={} 
    
    books = []

    with open(input_file,encoding='utf-8') as f:
        
        # define list to store tags for each book
        tags = []
        
        # for each book
        for row in csv.DictReader(f): 

            #make a new book object
            new_book = book(row['authors'],
                         int(row['original_publication_year']) if row['original_publication_year'].isnumeric() else None,
                         row['original_title'],
                         float(row['average_rating']),
                         int(row['ratings_count_new']),
                         float(row['per_positive_ratings']),
                         float(row['per_negative_ratings']),
                         float(row['per_average_ratings']),
                         int(row['total_number_of_tags'])
                    )
            
            #store the overview separately
            tags.append(row['tag_name'])
            
            #update the index             
            title_index[row['original_title']]=len(books)             
            books.append(new_book) 
    
    # Split the tags into individual keywords
    vectorizer = CountVectorizer(token_pattern=None, tokenizer=lambda x: x.split(', '))
    tags_matrix = vectorizer.fit_transform(tags)
    
    # Calculate the cosine similarities similarity matrix
    tags_similarity_matrix = cosine_similarity(tags_matrix)
    
    return books, tags_similarity_matrix, title_index


# function to get book-to-book similarity
def get_book_similarity(title1:str, 
                        title2:str, 
                        books:list,
                        weights:dict, 
                        tags_similarity_matrix:np.ndarray,
                        title_index:dict
                       ):
    
    # get the index of each book
    book_id_1 = title_index[title1]
    book_id_2 = title_index[title2]

    # get all the info regarding each book
    book1 = books[book_id_1]
    book2 = books[book_id_2]
    
    # define scores dict to store the score for each factor from the weights dict
    scores = dict() 
    
    #####################################
    #           authors                 #
    #####################################
    
    # define two sets for jaccard
    book1_authors = set(book1.authors)
    book2_authors = set(book2.authors)
    
    # compute jaccard sim regarding authors
    scores['authors'] = len(book1_authors.intersection(book2_authors))/len(book1_authors.union(book2_authors))
    
    #####################################
    #         publication year          #
    #####################################
    
    # publication year diff
    try:
        scores['publication_year'] = abs(book1.original_publication_year - book2.original_publication_year)/100
        
        # if score is more than one, then twto books are too far and by default we assign 0 value
        if scores['publication_year'] > 1: scores['publication_year'] = 0
            
    except:
        scores['publication_year'] = 0    
        
    #####################################
    #          average rating           #
    #####################################
    
    # normalized candidate rating
    scores['rating'] = round(book1.average_rating/10,2)
    
    #####################################
    #       % positive ratings          #
    #####################################
    
    # compute how close are the positive ratings of both books
    # the smaller the distance, the better the result, so we abstract it from 1 to give power
    scores['positive_ratings'] = 1 - (abs(book1.per_positive_ratings - book2.per_positive_ratings)/100)

    #####################################
    #              tags                 #
    #####################################
    
    # cosine sim for tags
    scores['tags'] = tags_similarity_matrix[book_id_1,book_id_2]
    
    #####################################
    #       final similarity dict       #
    #####################################
    
    # create the sim dict 
    factors = {x:round(scores[x]*weights[x],2) for x in scores}
    
    #sort factors by sim
    sorted_factors=[factor for factor in sorted(factors.items(), key=lambda x:x[1],reverse=True) if factor[1]>0]
    
    # compute overall score 
    overall_score = round(np.sum(list(factors.values())),2)
    
    return overall_score, sorted_factors


# function to get recommendations
def recommend_books(input_title:str, 
                    number_to_recommend:int,
                    books:dict,
                    weights:dict,
                    tags_similarity_matrix:np.ndarray,
                    title_index:dict
                    ):
    
    # initialize dict for recommendations 
    results = {} 
    
    # for each candidate
    for candidate in books: 
        
        # avoid to suggest the input title 
        if candidate.original_title != input_title:
        
            # get the similarity and the explanation
            my_sim, my_exp = get_book_similarity(candidate.original_title,input_title,\
                                                 books, weights,tags_similarity_matrix,title_index)
    
            # remember
            results[candidate.original_title] = (my_sim,my_exp)
            
        else: next

    # store recommendation and keep as much books as the number_to_recommend value
    recommendations = sorted(results.items(),key=lambda x:x[1][0],reverse=True)[:number_to_recommend]
    
    return recommendations


# function to generate fake users
def generate_users(books:list, # list of book objects 
             tags_similarity_matrix:np.ndarray, # book-to-book sim matrix
             title_index, # index that maps each title to a position in the books list
             factors:list, # factors to consider
             user_num:int=10,  # number of users to generaate
             seed_book_num:int=5, # number of random seed books to choose
             std_multiplier:float=1.5
            ):
    
    @dataclass
    class User:
        seed_books:list # latent books that the user likes
        likes:list # known books that the user has liked 
        dislikes:list # known books that the user has disliked
        weights:dict # user preferences 
        like_threshold:float # similarity threshold for liking a book
        
    # list fake users 
    users = []
    
    # for each fake user to create
    for i in range(user_num): 
        
        # user preferences for each factor
        weights = {} 
        
        # for each factor
        for factor in factors:
            
            # sample a random preference value (weight)
            weights[factor]=round(random(),2) 
            
        # sample 5 seed books    
        seed_books = sample(books,5) 
        
        '''
        Compute the "like" threshold for this user
        If a book has an above-threshold similarity with (at least) 
        one of the seed books, then we assume that the user will like it.
        The threshold is defined to be equal to the average sim of all books
        with the seed books, +1 stdev
        '''
        
        # define empty dict list to store similarity scores for each user
        sim_scores = [] 
        
        # for each seed book of this user
        for seed_book in seed_books: 
            
           # for each other book
            for candidate in books: 
                
                #compute seed-candidate sim
                book_simialrity, _ = get_book_similarity(
                           candidate.original_title, 
                           seed_book.original_title, 
                           books, 
                           weights,
                           tags_similarity_matrix,
                           title_index)
                
                # store similarity score
                sim_scores.append(book_simialrity) 
        
        # compute the "like" threshold for this user (mean + <std_multiplier> standard deviation)
        like_threshold = np.mean(sim_scores)+ std_multiplier*np.std(sim_scores) # threshold is mean + 1 std dev
     
        # remember the user 
        users.append(User(seed_books, 
                          [], # liked books 
                          [], # disliked books
                          weights, # preferences
                          round(like_threshold,2)) # like threshold
                    )
        
    print(f'{user_num} fake users have been created succesfully', end = '\n\n') 
    
    return users


# function to make book recommendations with exploit logic
def exploit_simulate(users:list, # fake users 
                     books:list, # books list 
                     title_index:dict, # maps titles to indices in the movies list 
                     tags_similarity_matrix:np.ndarray, # book to book sim matrix
                     recnum_per_user:int=50, # number of recommendations to make 
                     neighbor_num:int=10,# number of neighbors to consider when looking for candidates
                     liked_sample_size:int=10, # number of liked books to consider when looking for candidates
                     factors=['authors','publication_year','rating','positive_ratings','tags'], # factors to consider
                     ):
    
    # initialize dict to store recommended books for each user
    user_recommended_books = {}
    
    # for each fake user 
    for n_user, user in enumerate(users): 
        
        # intialize likes and dislikes for this user
        user.likes = []
        user.dislikes = []
        
        # estimated preference weights for this user
        estimated_weights = {factor:1 for factor in factors}
    
        # remembers previous recommendations
        recommended = set()
        
        # initialize dict to store recommended books for this user
        current_rec_books = {}
        
        # for each recommendation to be made
        for i in range(recnum_per_user): 
            
            # book to be recommended
            rec_book = None 
            
            # if the user has no known likes yet
            if len(user.likes)==0: 
                
                # sample a random book that has not been recommended before
                while rec_book == None or rec_book.original_title in recommended:
                    rec_book = sample(books,1)[0]
            else: 
                # remembers candidate book and maps them to a score
                # the score is equal to the sum of the similarity values with the 
                # user's seed books
                candidates = defaultdict(float)
                
                # get a sample of this user's likes (For speed)
                sampled_likes = sample(user.likes, min(len(user.likes),liked_sample_size))
                
                # for each book that has been liked by this user
                for liked_book in sampled_likes:
                
                    # find the top K most similar books to this liked book
                    neighbors = recommend_books(liked_book.original_title,
                                                neighbor_num,
                                                books,
                                                estimated_weights,
                                                tags_similarity_matrix,
                                                title_index)
                
                    # for each recommended book
                    for neighbor,metrics in neighbors:
                        
                        # if the book hasn't already been recommened, make it a candidate
                        if neighbor not in recommended: 
                            
                            # update the similarity score for this candidate
                            candidates[neighbor] += metrics[0] 
                            
                # if no candidates found
                if len(candidates)==0: 
                    
                    # pick a random book that has not been recommended already
                    while rec_book == None or rec_book.original_title in recommended:
                        rec_book = sample(books,1)[0]
                        
                # candidates found        
                else: 
                    
                    # pick the top-scoring candidate based on similarity score
                    rec_book = books[title_index[sorted(candidates.items(),key=lambda x:x[1],reverse=True)[0][0]]]
                    
            # add the recommended book to the set of recommended books
            recommended.add(rec_book.original_title)
            
            # becomes true if the recommended book is similar enough to one of the seed books
            found_seed=False 
            
            # for each seed book
            for seed_book in user.seed_books: 
                
                # compute the sim between the seed book and the recommended book
                similarity, _ = get_book_similarity(rec_book.original_title,
                                                    seed_book.original_title,
                                                    books,     
                                                    user.weights,
                                                    tags_similarity_matrix,
                                                    title_index)
                        
                # if the sim is over the like threshold for this user
                if similarity > user.like_threshold:
                    
                    # mark the recommended book as liked and exit the loop
                    found_seed=True
                    break
                    
            # # if the book is similar enough to at least one seed movie
            if found_seed: 
                
                # add the book to the user's liked books
                user.likes.append(rec_book)
                
                # add the book to recommended books for this user
                current_rec_books[rec_book.original_title] =   'Y'      
                
            else: 
                # add the book to the user's disliked books
                user.dislikes.append(rec_book)
                
                # add the book to recommended books for this user
                current_rec_books[rec_book.original_title] =   'N'
                    
        print(f'For user {n_user+1} the Total Likes out of {recnum_per_user} are : {len(user.likes)}')  
        
        # add recommended books for this user to list of recommended books
        user_recommended_books[f'user_{n_user+1}'] = current_rec_books
        
    return user_recommended_books


# function to evaluate recsys by computing avg precision
def compute_average_precision(user_recommended_books:dict):
    
    # get the number of liked books
    num_of_liked_books = [pol for pol in list(user_recommended_books.values())].count('Y')
    
    # check if there are any relevant items, else avg_precision = 0
    if num_of_liked_books == 0:
        return 0
    
    # initialize list to store precisions
    precisions = []
    
    # liked book counter
    liked_book_counter = 0
    
    # for each recommended book
    for i, (book,polarity) in enumerate(user_recommended_books.items()):
        
        # check if polarity is 'Y'
        if polarity == 'Y':
            
            # increase counter
            liked_book_counter += 1
            
            # compute current item precision
            precision = liked_book_counter / (i + 1)
            
            # append to precisions
            precisions.append(precision)
            
    # compute average precision
    avg_precision = round(sum(precisions) / num_of_liked_books,2)
    
    return avg_precision