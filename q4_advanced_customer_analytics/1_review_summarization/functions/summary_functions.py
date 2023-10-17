# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:12:26 2022

@author: mkarampasis
"""

#import libraries
#import libraries
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os.path
from collections import defaultdict
from typing import Callable
#! pip install imageio
import imageio
from PIL import Image
#!pip install wordcloud
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

#! pip install spacy-langdetect
import spacy
#! spacy download en_core_web_lg
from spacy.language import Language
from spacy_langdetect import LanguageDetector

from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.corpus import opinion_lexicon
#nltk.download('opinion_lexicon')

from scipy.stats import ttest_1samp


import networkx as nx
from itertools import combinations

import calendar
import sys
from termcolor import colored

from fpdf import FPDF

import warnings
warnings.filterwarnings("ignore")

#step 1

#function for the convertion of greek month names to integer
def convert_monthname(monthname):

    #dict of greek months
    table = {'Ιαν': 1, 'Φεβ': 2, 'Μαρ': 3, 'Απρ': 4, 'Μαΐ': 5, 'Ιουν': 6,
         'Ιουλ': 7, 'Αυγ': 8, 'Σεπ': 9, 'Οκτ': 10, 'Νοε': 11, 'Δεκ': 12,}

    return table.get(monthname, monthname)


#function for the correction of date, convert greek text date to date
def date_correction(reviews_eng):
    
        #from string date, split day, month, year to three new columns
        reviews_eng['day'] = reviews_eng.review_date.str[:2]
        reviews_eng['month'] = reviews_eng.review_date.str[2:-5]
        reviews_eng['year'] = reviews_eng.review_date.str[-4:]

        #get rid of whitespaces
        reviews_eng['day'] = reviews_eng.day.str.split(' ').str.join('')
        reviews_eng['month'] = reviews_eng.month.str.split(' ').str.join('')
        reviews_eng['year'] = reviews_eng.year.str.split(' ').str.join('')

        #apply the correction of greek month names to integer
        reviews_eng['month'] = reviews_eng['month'].apply(convert_monthname)

        #concatenate day, month, year to create the correct date
        reviews_eng['corrected_review_date'] = reviews_eng.day.astype(str) + '-' + \
                                                reviews_eng.month.astype(str) + '-' + \
                                                 reviews_eng.year.astype(str)

        #for each date, convert to timestamp
        reviews_eng['corrected_review_date'] = reviews_eng['corrected_review_date'].apply( \
                                                 lambda x: pd.to_datetime(str(x), format='%d-%m-%Y'))


        #keep only the columns regarding rating, content, corrected_review_date
        reviews_eng = reviews_eng.loc[:,('rating','content','corrected_review_date')]

        return reviews_eng


#step 2
#define 
nlp = spacy.load("en_core_web_lg")

#function for calculating the frequencies of all the words in all the reviews
def find_freq(reviews_eng):
        
        #load stopwords
        stopLex=set(stopwords.words('english')) 
        
        #define a dict for counting frequencies
        freq=defaultdict(int)
        
        #loop through each review
        for review in reviews_eng.content:
            
            #split reviews into sentences
            sentences = sent_tokenize(review)
            
            #loop through each sentence of the given review
            for sent in sentences:
                
                #remove emojies from text
                sent = sent.encode('ascii', 'ignore').decode('ascii')
                
                #remove non-word (special) characters such as punctuation, numbers etc
                #sent = re.sub(r'\W', ' ', str(sent))
                
                #split sentence into words
                #sent_tok = word_tokenize(sent)
                
                sent_tok = nlp(sent)
                
                for word in sent_tok:
        
                    #get the lemma of the term
                    lemma = word.lemma_.lower()
                            
                    #ignore stopwords, short words, non-nouns
                    if  (lemma not in stopLex) and (len(lemma)>=3) and (word.pos_=='NOUN'):
                        freq[lemma]+=1 # update freq
            
        return freq
    
#step 3

#function for getting aspects. Uses network analysis to group semantically similar aspects
def get_aspects_from_undirected_graphs(freq:dict, #a dict for counting frequencies of each word from all the reviews
                    component_function:Callable, #function used to find components in the term graph
                    lower_sim_bound: float=0.5, #minimum thresholds of semantic similarity between two terms
                    aspect_freq_bound=25):
        
        #create a new undirected graph
        G = nx.Graph()
        
        #map each aspect to its the spacy-tagged version
        tagged = {term:nlp(term) for term in freq} 
        
        #add a node for every aspect
        for term in tagged: 
            G.add_node(term)
        
        #all possible pairs of nodes  (aspects)
        pairs=combinations(freq.keys(),2)
        
        for x,y in pairs: 
            
            #semantically similar aspects
            if tagged[x].similarity(tagged[y]) > float(0.5): 
                
                #add an edge between them
                G.add_edge(x,y)
        
        #fild all connected components in this graph
        components = list(nx.find_cliques(G))
        
        component_info={}
        
        #for each component 
        for C in components:  
         
            term_freq_within_C={term:freq[term] for term in C}
            representative=sorted(term_freq_within_C.items(),key=lambda x:x[1],reverse=True)[0][0]
                
            total_freq=np.sum([term_freq_within_C[term] for term in term_freq_within_C])
                
            if total_freq>=25:
                
                component_info[representative]={'members':C,'freq':total_freq}
        
        return component_info


#step 4

#function for opinion mining
def get_opinions(reviews:list, # list of product reviews
                 aspects:dict # dictionary that maps each aspect to each component 
                ):
    
        #map each term to each representatives from the various aspects that it belongs to
        term_to_rep = defaultdict(set)
        for rep in aspects:
            for term in aspects[rep]['members']:
                term_to_rep[term].add(rep)
        
        #lexicons of positive and negative terms
        negLex = set(opinion_lexicon.negative())
        posLex = set(opinion_lexicon.positive())  
        
        my_opinions = defaultdict(list) # map each aspect rep to a list of sentiment scores
        
        #loop through reviews
        for review in reviews:
            
            sentences = sent_tokenize(review)
            
            #loop through each sentence of the current review
            for sent in sentences: 
                
                #remove emojies from text
                sent = sent.encode('ascii', 'ignore').decode('ascii')
                sent_tok = nlp(sent)
                
                #stores the aspects present in this sentence
                aspect_reps_in_sentence = set()
                #sentiment score for this sentence
                senti_score = 0
                
                #loop through each word of current sentence
                for word in sent_tok:
                    
                    if word.text.lower() in negLex: senti_score -= 1 #-1 for negative words
                    elif word.text.lower() in posLex: senti_score += 1 #+1 for positive words
                    
                    lemma = word.lemma_.lower() # get the lemma of the token
                    if lemma in term_to_rep: #if it is one of the known aspects 
                        for rep in  term_to_rep[lemma]: # for each rep of this term
                            aspect_reps_in_sentence.add(rep) # mark the aspect
               
                #smoothing
                if senti_score>0:
                    senti_score=1
                else: 
                    senti_score=-1
                    
                 # update the senti_scores for the aspects included in this sentence
                for rep in aspect_reps_in_sentence:
                    my_opinions[rep].append(senti_score)
                
                    
                    
        return my_opinions,aspect_reps_in_sentence


#step 5
#classification of postive and negative aspects
def classification_of_aspects(aspects,opinions):
        
        positive_aspects = list() #list of positive reviews
                
        negative_aspects = list() #list of negative reviews
        
        #loop through the aspects
        for aspect in aspects:
                
                #get the senti scores of each aspect
                senti_scores = opinions[0].get(aspect,[]) 
                    
                ''''
                perform one-sample t-test for the null hypothesis 
                that the expected value (mean) of all the opinions 
                of each aspect is equal to 0 which is a neutral opinion.
                '''
                
                #one sample t-test
                stat ,pval = ttest_1samp(senti_scores, 0.0, axis=0, nan_policy='propagate', alternative='two-sided') 
                
                # reject the null hypothesis, the opinions are significantly different from 0 which is a neutral opinion
                if pval < 0.05: 
                     
                    if np.mean(senti_scores) > 0:
                            positive_aspects.append(aspect) 
                    
                    elif np.mean(senti_scores) < 0:
                            negative_aspects.append(aspect) 
                            
                        
        return positive_aspects, negative_aspects   

#step 6

#function for classification
def classify(x):
        if x < 3.5:
            return "neg"
        else:
            return "pos"


#step 7

#function for yearly grouping
def yearly_group(reviews_eng):

    #average rating per year and number of reviews per year
    reviews_yearly = pd.DataFrame(reviews_eng.groupby(['review_year','classif']).agg({'rating':'mean','content':'count'})).reset_index()
    
    #rounding of ratings
    reviews_yearly['rating'] = round(reviews_yearly.rating,2)

    #rename columns
    reviews_yearly.rename(columns={'rating':'avg_rating','content': 'number_of_reviews'}, inplace=True)

    return reviews_yearly

#function to set the reviews df in a format in order to create graphs/plots
#Also, we create a yearly df
def manipulate_df(reviews_eng):
        
        #add year in dataframe
        reviews_eng['review_year'] = reviews_eng['corrected_review_date'].dt.year
        
        #add month in dataframe
        reviews_eng['review_month'] = reviews_eng['corrected_review_date'].dt.month
        
        #add month names
        reviews_eng['review_month_names'] = reviews_eng['review_month'].apply(lambda x: calendar.month_abbr[x])
    
        #convert rating to string
        reviews_eng['rating_char'] = reviews_eng.rating.astype('str')
        
        #yearly df
        reviews_eng_yearly = yearly_group(reviews_eng)
        
        return reviews_eng, reviews_eng_yearly
    
#step 8
#step 8.1
def wordcloud(freq:dict, #dict for counting frequencies of each word from all the reviews
                      ):

        nike_pic = imageio.imread('./images_for_summary/jordan.jpg')
                           
        cloud = WordCloud(max_font_size=1000, colormap='rocket_r' ,background_color="White",\
                          width=1, height=1, margin=2, random_state = 21,\
                          collocations=False, mask=nike_pic,\
                          font_step=2).generate_from_frequencies(freq)
        
        fig = plt.figure(figsize=(12,6)  , dpi=100)
        plt.imshow(cloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.title('Most Popular Words in Reviews'.format(fig,dpi=100))
        plt.savefig('./images_for_summary/wordcloud_j.png', bbox_inches='tight')
        #plt.show()
        plt.close()
        
        return
    
#step 8.2
def basic_graph(reviews_eng_yearly,):
        
        #split postive and negative yearly values
        positives = reviews_eng_yearly[reviews_eng_yearly.classif == 'pos']
        negatives = reviews_eng_yearly[reviews_eng_yearly.classif == 'neg']
        
        #define the labels
        labels=["neg reviews","pos reviews"]
        
        #create new dataframe column with the labels detailed for labeling the graph
        reviews_eng_yearly["classif_detailed"] = np.where(reviews_eng_yearly["classif"]=='pos','pos review','neg review')
        
        #merge positives and negatives into one df
        df_merge = positives.merge(negatives, on ='review_year',how ='outer')
        #rename columns
        df_merge.rename(columns={'avg_rating_x': 'avg_rating_pos', 'number_of_reviews_x': 'number_of_reviews_pos',\
                           'avg_rating_y':'avg_rating_neg','number_of_reviews_y':'number_of_reviews_neg'}, inplace=True)
        
        #drop unwanted columns
        df_merge.drop(['classif_x','classif_y'], axis=1, inplace=True)
        #fill NAs with zeros
        df_merge = df_merge.fillna(0)
        
        ###plot
        ax1 = sns.set_style("white")
        fig,ax1 = plt.subplots(figsize=(12,6)  , dpi=100)
        
        b = sns.barplot(data = reviews_eng_yearly, x='review_year', y='number_of_reviews', hue = 'classif_detailed', \
                    alpha=0.5,palette=['r', 'g'])
        b.set(xlabel=None)
        b.set(ylabel=None)
        #ax1.set_ylabel('number of reviews')
        #ax1.set_xlabel('year')
        ax2 = ax1.twinx()
        
        l1 = sns.lineplot(data=df_merge.avg_rating_neg, color='r', marker='o', sort = False, label="neg rating",ax=ax2)
        l2 = sns.lineplot(data=df_merge.avg_rating_pos,color='g',marker='o', sort = False,label="pos rating",ax=ax2)
        l1.set(ylabel=None)
        l2.set(ylabel=None)
        ax1.get_legend().remove()
        ax2.get_legend().remove()
        lgd = fig.legend(loc='center left', bbox_to_anchor=(0.95, 0.5), ncol=1)
        #ax2.set_ylabel('average rating per class')
        
        # title
        title = plt.suptitle('Number of Reviews and Average Rating per Year'.format(fig,dpi=100), #fontsize = 16, 
                    horizontalalignment='center')
        #set the title
        subtitle = plt.title('Postives vs Negatives'.format(fig,dpi=100), #fontsize = 8, 
                    horizontalalignment='center')
        
        fig.savefig('./images_for_summary/bars_with_lines.png', bbox_extra_artists=(lgd,title,subtitle), bbox_inches='tight')
        plt.close()
        
        return


#step 9

#function to create pdf report
def create_and_export_pdf(reviews_eng,positive_aspects,negative_aspects):
    
        #creating a new image file with black color with A4 size dimensions using PIL
        img = Image.new('RGB', (50,100), "#101010" )
        img.save('./images_for_summary/black_colored.png')
        
        #create pdf page
        pdf = FPDF()
        pdf.add_page(orientation ='L')
        pdf.set_font('Arial','B',16)
        
        #adding images to pdf page
        pdf.image('./images_for_summary/black_colored.png', x = 0, y = 0, w = 70, h = 210, type = '', link = '')
        pdf.image('./images_for_summary/nike_logo.jpg', x = 8, y = 3, w = 55, h = 30, type = '', link = '')
        pdf.image('./images_for_summary/screenshot.png', w=60, h=40, x =220, y = 0)
        
        #Defines the color used for all drawing operations (lines, rectangles and cell borders)
        pdf.set_draw_color(r=200, g = 200, b=200)
        
        # Adds a line beginning at point (10,30) and ending at point (110,30)
        pdf.line(0, 35, 300, 35)
        
        #header
        pdf.set_text_color(190,190,190)#white
        pdf.set_font('Arial','B',24)
        pdf.cell(ln=0, h=18.0, align='C', w=230, txt='Reviews Summary', border=50)
        pdf.ln(50)
        
        ###summary totals
        
        #number of reviews
        pdf.set_text_color(190,190,190)#white
        pdf.set_font('Arial','B',32)
        pdf.cell(ln=0, h=0.0, align='C', w=44, txt=rf'{len(reviews_eng)}', border=50)
        
        #title of number of reviews
        pdf.set_font('Arial','B',20)
        pdf.ln(10)
        pdf.cell(ln=0, h=10.0, align='left', w=0, txt='Total Reviews', border=0)
        
        
        if reviews_eng.rating.mean() > 3.5:
            
            #average rating
            pdf.ln(40)
            pdf.set_text_color(0,150,0)#green
            pdf.set_font('Arial','B',32)
            pdf.cell(ln=0, h=0.0, align='C', w=44, txt=rf'{round(reviews_eng.rating.mean(),1)}', border=50)
            
        else:
            #average rating
            pdf.ln(40)
            pdf.set_text_color(150,0,0)#red
            pdf.set_font('Arial','B',32)
            pdf.cell(ln=0, h=0.0, align='C', w=44, txt=rf'{round(reviews_eng.rating.mean(),1)}', border=50)
            
        #title of average rating
        pdf.set_text_color(190,190,190)#white
        pdf.set_font('Arial','B',20)
        pdf.ln(10)
        pdf.cell(ln=0, h=10.0, align='left', w=0, txt='Average Rating', border=0)
        
        if (len(reviews_eng[reviews_eng.classif == "pos"])/len(reviews_eng))*100 >= 50:
            
            #percentage of positivity 
            pdf.ln(40)
            pdf.set_text_color(0,150,0)#green
            pdf.set_font('Arial','B',32)
            pdf.cell(ln=0, h=0.0, align='C', w=44, txt=rf'{round((len(reviews_eng[reviews_eng.classif == "pos"])/len(reviews_eng))*100,1)}', border=50)
            
        else:
            #percentage of positivity 
            pdf.ln(40)
            pdf.set_text_color(150,0,0)#red
            pdf.set_font('Arial','B',32)
            pdf.cell(ln=0, h=0.0, align='C', w=44, txt=rf'{round((len(reviews_eng[reviews_eng.classif == "pos"])/len(reviews_eng))*100,1)}', border=50)
            
        #title of percentage of positive reviews
        pdf.set_text_color(190,190,190)#white
        pdf.set_font('Arial','B',20)
        pdf.ln(10)
        pdf.cell(ln=0, h=10.0, align='left', w=0, txt='  Positivity %', border=0)
        
        #add wordcloud
        pdf.image('./images_for_summary/wordcloud_j.png', x = 210, y = 38, w = 80, h = 90, type = 'PNG', link = '')
        
        #add barchart positives~negatives
        pdf.ln(10)
        pdf.image('./images_for_summary/bars_with_lines.png', x = 90, y = 110, w = 180, h = 90, type = 'PNG', link = '')
        
        
        pdf.set_xy(80,55)
        #text 
        
        dummy_text = rf'The total number of reviews shows {len(positive_aspects)} positive words and {len(negative_aspects)} negative.'
        if len(positive_aspects) != 0:
            dummy_text += rf" The positive words are related to: {', '.join(map(str,positive_aspects))}."
            
        if len(negative_aspects) != 0:
            dummy_text += rf" The negative words are related to: {', '.join(map(str,negative_aspects))}."
            
        pdf.set_text_color(50,50,50)#black
        pdf.set_font('Arial','',11)
        pdf.multi_cell(120,7, txt=dummy_text,align= "J")
                
        #output
        pdf.output('./Nike report.pdf','F')
        
        
        #in case you run it with linux
        if sys.platform.startswith("linux"):
            os.system("xdg-open ./Nike report.pdf")
        else:
            os.system("./Nike report.pdf")
            
        return