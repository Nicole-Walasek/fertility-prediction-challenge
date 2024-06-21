"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""

import pandas as pd
import joblib
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,HistGradientBoostingClassifier, VotingClassifier

from sklearn.utils import resample
import random
import submission
from sklearn.metrics import precision_recall_fscore_support
import variable_selection

def train_save_model(df, target):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    
    ## This script contains a bare minimum working example
    random.seed(1) # not useful here because logistic regression deterministic
    data = submission.clean_df(df)
    data=data.drop('nomem_encr', axis=1)
    
    # remove missing values
    y_missing = target['new_child'].isna()
    # Drop samples in both `data` and `outcome`:
    data = data.drop(data[y_missing].index, axis='rows')
    target= target.drop(target[y_missing].index, axis='rows')
    target = target['new_child']

    
    
    # my model pipeline
    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")

    numerical_columns = numerical_columns_selector(data)
    categorical_columns = variable_selection.select_categorical_var()
    numerical_columns = [col for col in numerical_columns if col not in categorical_columns]


    numerical_preprocessor = StandardScaler()
    preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    ('standard_scaler', numerical_preprocessor, numerical_columns)])
    

    data_oversampled, target_oversampled = resample(data[target == 1],
                                        target[target== 1],
                                        replace=True,
                                        n_samples=target[target == 0].shape[0],
                                        random_state=123)
    #
    # Append the oversampled minority class to training data and related labels
    #
    data_balanced = pd.concat((data[target== 0], data_oversampled))
    target_balanced = pd.concat((target[target == 0], target_oversampled))
    

    #model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=500))

    #make an ensemble of classifiers
    clf1 = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=500))
    clf2 = make_pipeline(preprocessor, HistGradientBoostingClassifier(max_iter=500))
    clf3 = make_pipeline(preprocessor, LogisticRegression(max_iter=500))
    clf4 = make_pipeline(preprocessor, AdaBoostClassifier(n_estimators=500))

    model = VotingClassifier(estimators=[
        ('rf', clf1), ('histgrad', clf2), ('lr', clf3), ('ab', clf4)], voting='hard')

    model.fit(data_balanced, target_balanced)
    
    target_pred = model.predict(data)
    p, r, f, _ = precision_recall_fscore_support(target, target_pred, average='binary')
    print(f'Precision: {p}, recall: {r}, F1-score: {f} on train data')

    # Save the model
    joblib.dump(model, "model.joblib")
    


data = pd.read_csv('C:/Users/nicwa/data_prefer_full/training_data/PreFer_train_data.csv')
target = pd.read_csv('C:/Users/nicwa/data_prefer_full/training_data/PreFer_train_outcome.csv', encoding='cp1252')
train_save_model(data, target)