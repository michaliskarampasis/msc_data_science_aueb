# import libraries
import pandas as pd

# function to import data
def import_data(path_books:str,
                path_ratings:str,
                path_book_tags:str,
                path_tags:str
               ):
    
    '''
    Function to import data.
    
    Takes as input the path where the CSVs are located.
    
    Returns 4 dataframes.
    
    '''
    
    #load books dataset
    books = pd.read_csv(path_books)
    
    #load ratings dataset
    ratings = pd.read_csv(path_ratings)
    
    #load book tags dataset
    book_tags = pd.read_csv(path_book_tags)
    
    #load tags dataset
    tags = pd.read_csv(path_tags)
   
    return books, ratings, book_tags, tags


# function to preprocess data
def preprocess_data(ratings:pd.DataFrame,
                    books:pd.DataFrame,
                    book_tags:pd.DataFrame,
                    tags:pd.DataFrame
                   ):
    
    '''
    Function to pre process data.
    
    Takes as input the 4 dataframes.
    
    Return 1 finalized dataframe, with all desired info merged, 
    the cosine similarities regarding tags and a dict with title indexes.
    
    '''
    
    ###############################
    #       ratings & books       #
    ###############################
    
    # drop duplicates from ratings table
    ratings.drop_duplicates(["user_id","book_id"], inplace = True)
    
    # drop duplicates from books table
    books.drop_duplicates(["original_title"], inplace = True) 
    
    # drop unwanted columns
    books = books.drop(['best_book_id', 'work_id', 'books_count', 'isbn', 'isbn13', 'language_code', 'work_ratings_count', 'work_text_reviews_count', 'image_url', 'small_image_url'], axis=1, errors='ignore')
    
    # drop duplicates from books table
    books.drop_duplicates('title', inplace=True)
    
    # find rows with NaN values and drop them from both tables
    books_with_nan = books.isnull().any(axis=1)
    
    for index, row in books[books_with_nan].iterrows():
        ratings = ratings[ratings.book_id != row.book_id]
    
    # drop NAs based on title
    books.dropna(subset=['original_title'], inplace=True)
    
    # compute percentages of positive ratings, negative ratings, average ratings
    books['ratings_count_new'] = books.ratings_1 + books.ratings_2 + books.ratings_3\
                                         + books.ratings_4 + books.ratings_5
    
    books['per_positive_ratings'] = round((books.ratings_4 + books.ratings_5)/ books.ratings_count_new,3)*100
    books['per_negative_ratings'] = round((books.ratings_1 + books.ratings_2)/ books.ratings_count_new,3)*100
    books['per_average_ratings'] = round((books.ratings_3)/ books.ratings_count_new,3)*100
    
    ###############################
    #      book_tags & tags       #
    ###############################
    
    tags_merged = pd.merge(book_tags, 
                tags, 
                how='left',
                left_on='tag_id',
                right_on='tag_id')

    # group the data by name and aggregate the tagValue values
    grouped_tags = tags_merged.groupby('goodreads_book_id').agg({'count':'sum','tag_name': lambda x: ', '.join(x)})

    # reset the index to make name a column again
    grouped_tags = grouped_tags.reset_index()


    #rename column
    grouped_tags = grouped_tags.rename(columns={'goodreads_book_id':'id','count':'total_number_of_tags'})
    
    # merge books and tags
    books_tags_merged = pd.merge(books, 
                grouped_tags, 
                how='inner',
                left_on='id',
                right_on='id')
    
    ###############################
    #    final format of data     #
    ###############################
    
    # drop unwanted columns
    df = books_tags_merged.drop(['book_id','title','ratings_count','ratings_1','ratings_2','ratings_3','ratings_4','ratings_5'], \
                   axis=1, errors='ignore')
    
    # set title as index
    df = df.set_index('id')

    return df
    


