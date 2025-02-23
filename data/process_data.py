"""
    1)Path to the CSV file containing messages (e.g. disaster_messages.csv)
    2) Path to the CSV file containing categories (e.g. disaster_categories.csv)
    3) Path to SQLite destination database (e.g. disaster_response_db.db)
"""

#Import all libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
	Arguments:

	messages_filepath (str): Path to the CSV file containing messages.
	categories_filepath (str): Path to the CSV file containing categories.

    """
    
    # load the message file
    messages = pd.read_csv(messages_filepath)

    # load the categories file
    categories = pd.read_csv(categories_filepath)
    
    #Merge both datasets
    df = messages.merge(categories, on='id')

    return df 


def clean_data(df):
    # Split categories column into individual category columns
    categories = df["categories"].str.split(';', expand=True)

    # Select the first row of the categories
    row = categories[0:1]
   
    # Remove the last 2 characters to get category names
    category_colnames = row.apply(lambda x: x.str[:-2]).values.tolist()

    # rename the columns of categories
    categories.columns = category_colnames
 
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from df

    df.drop(['categories'], axis=1, inplace = True)

    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Save the cleaned dataframe to an SQLite database.

    Argument:
    database_filename (str)-> Path to the SQLite database.
    """ 
    # Create engine to connect to SQLite database
    engine = create_engine('sqlite:///'+ database_filename)

    # Save DataFrame to database table
    table_name = database_filename.replace(".db","") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')

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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'disaster_response_db.db')


if __name__ == '__main__':
    main()