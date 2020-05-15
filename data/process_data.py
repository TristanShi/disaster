# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Tristan SHI
@Email: shizhendong@unionpayintl.com

@File: process_data.py
@Time: 2020/4/22 14:56
"""

import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np


def load_data(messages_filepath: str, categories_filepath: str)->pd.DataFrame:
    """
    load the dataset 
    :param messages_filepath:  The file path of the messages csv
    :param categories_filepath:  The file path of the categories cv
    :return df: Merged messages and categories df, merged on ID  
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='outer', on=['id'])

    return df


def clean_data(df:pd.DataFrame)->pd.DataFrame:
    """Clean the data
    :param df: combined messages and categories dataframe
    :return df: cleaned dataframe    
    """
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(np.array([val.split(';') for i, val in df['categories'].iteritems()]),
                              columns=['categories_%s' % str(i) for i in range(36)])

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2]).tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(x[-1:]))

    # drop the original categories column from `df`
    df.drop(columns=['categories'], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    
    # "related" column has 0, 1, 2 values
    # change 2 to 1, so that the value is only 1 or 0
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)

   
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df:pd.DataFrame, database_filename:str):
    """
    insert clean data into database
    :param df: clean dataset 
    :param database_filename
    :return 
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('message', engine, index=False)
    return True


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
