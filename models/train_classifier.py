import sys
import nltk
import numpy as np
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.metrics import classification_report,accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import pickle

def fetch_data(db_filepath):
    """Retrieve data from the specified database file."""
    db_url = 'sqlite:///' + db_filepath
    engine = create_engine(db_url)
    df = pd.read_sql_table('disaster_response_db_table', con=engine) 
    print(df.head())
    X = df['message']
    y = df.iloc[:, 4:]
    categories = y.columns
    return X, y, categories

def tokenize(text):
    """Clean and tokenize the input text."""
    url_pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    for url in urls:
        text = text.replace(url, "urlplaceholder")
    
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
    return clean_tokens

def build_model():
    """Build and return a GridSearchCV model with a pipeline."""
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'classifier__estimator__max_depth': [10, 50, None],
        'classifier__estimator__min_samples_leaf': [2, 5, 10]
    }

    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''  
    This function aims to retrieve model metrics by collecting all metrics and saving them into a DataFrame, which we will store.
    '''
    # Predict
    predictions = model.predict(X_test)
    
    # Initiate metrics lists
    labels = category_names
    precision_list = []
    recall_list = []
    f1_score_list = []
    accuracy_list = []

    # Loop over labels and compute metrics
    for idx, label in enumerate(labels):
        y_true = Y_test.iloc[:, idx]  # Extract true labels for the current label
        y_pred = predictions[:, idx]  # Extract predicted labels for the current label
        
        # Compute precision, recall, F1-score, and accuracy for the current label
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        accuracy = accuracy_score(y_true, y_pred)
        
        # Handle zero division manually
        if precision == 0 and recall == 0:
            f1_score = 0
        
        # Append metrics to initiated lists
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)
        accuracy_list.append(accuracy)

    # Create a DataFrame with metrics for each label
    metrics_df = pd.DataFrame({
        'Label': labels,
        'Precision': precision_list,
        'Recall': recall_list,
        'F1-score': f1_score_list,
        'Accuracy': accuracy_list
    })

    # Transform the Label column
    metrics_df.set_index('Label', inplace=True)
    # Transpose so table is wider
    transposed_df = metrics_df.transpose()
    return transposed_df

def store_model(model, filepath):
    """Save the model to a file."""
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = fetch_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        store_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/disaster_response_db.db classifier.pkl')


if __name__ == '__main__':
    main()