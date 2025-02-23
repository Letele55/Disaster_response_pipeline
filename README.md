# Disaster_response_pipeline

## Table of contents
- [Background](#Background)
- [Installation](#Installation)
- [Instructions](#Instructions)
- [Licensing,Authors,Acknowledgements](#Licensing,Authors,Acknowledgements)

## Background
The dataset includes actual messages sent during disaster situations. I used an ETL (Extract, Transform, Load) pipeline to load and clean the dataset before developing a machine learning pipeline to classify these events. This classification enables the appropriate disaster relief agencies to be contacted for assistance.

## Installation
Run with Python 3 with libraries of numpy, pandas, sqlalchemy, re, NLTK, pickle, Sklearn, plotly and flask libraries

## Instructions

### ETL Pipeline
File `data/process_data.py` contains the data cleaning pipeline that:

- Loads the `messages` and `categories` dataset
- Merges the above datasets
- Cleans the data
- Stores it in a SQLite database

### ML Pipeline
File `models/train_classifier.py` contains the machine learning pipeline that:

- Loads data from the SQLite database
- Splits the data into training and testing sets
- Builds a text processing and machine learning pipeline for categorization.


1. Run the following commands in the project's root directory to set up your database and model.

 - To run ETL pipeline that cleans data and stores in database  `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db`
 - To run ML pipeline that trains classifier and saves  `python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl`
 
2. Run the following command in the app's directory to run your web app. `python app/run.py`

3. Go to http://0.0.0.0:3000/

## Licensing,Authors,Acknowledgements
I would like to thank Udacity for granting me the opportunity to learn data science further through this project
