# import libraries
import numpy as np

# function to investigate how much missing data there is in each column of the dataset
def return_missing_data(df):
    
    # get the total number of rows
    total_rows = df.shape[0]
    
    # create a dictionary where keys are column names, and values are the count of missing values in each column
    missing_data = dict(zip(df.columns,df.isnull().sum()))
    
    # ordered_missing_data = dict(sorted(missing_data.items(), key= lambda x: x[1], reverse=True))
    
    missing_values = 0
    # loop through the dict
    for key, value in missing_data.items():
        if value > 0:
            
            # count missing values in total
            missing_values += value
            # print information about missing data in each column
            print('Missing Data in column {:80} : {:8} ({:2.2f}%)'.format(key, value, value/total_rows*100))
            
    return missing_values

# function to create one column regarding race
def return_race(row):
    if row['Hispanic'] == 1:
        return 'Hispanic'
    elif row['Other'] == 1:
        return 'Other Race'
    elif row['African American/Black'] == 1: 
        return 'Non-Hispanic Black'
    elif row['White'] == 1:
        return 'White'
    else:
        return np.nan