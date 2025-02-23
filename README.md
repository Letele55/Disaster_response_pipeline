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
1. Run the following commands in the project's root directory to set up your database and model.

 To run ETL pipeline that cleans data and stores in database ''' bash python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'''
 To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
2. Run the following command in the app's directory to run your web app. python run.py

3. Go to http://0.0.0.0:3000/

## Licensing,Authors,Acknowledgements
I would like to thank Udacity for granting me the opportunity to learn data science further through this project
