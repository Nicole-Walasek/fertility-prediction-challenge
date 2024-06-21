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
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import random
import submission
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import string

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

    lowercase_alphabet = string.ascii_lowercase
    num_list = np.concatenate((np.arange(7, 14), np.arange(15, 21)))
    healthRange = ['ch0%s%s004' % (num, letter) if num < 10 else 'ch%s%s004' % (num, letter) for num, letter in
                   zip(num_list, lowercase_alphabet[0:21 - 8])]  # ch07a004 ch20m004
    # this one is categorical:
    religiousRange = ['cr0%s%s012' % (num, letter) if num < 10 else 'cr%s%s012' % (num, letter) for num, letter in
                      zip(np.arange(8, 19), lowercase_alphabet[0:19 - 8])]  # cr08a012 - cr18k012
    childrenUnderEightRange = ['cw0%s%s439' % (num, letter) if num < 10 else 'cw%s%s439' % (num, letter) for num, letter
                               in zip(np.arange(8, 21), lowercase_alphabet[0:21 - 8])]  # cw08a439- cw20m439
    childrenNumRange = ['cf0%s%s036' % (num, letter) if num < 10 else 'cf%s%s036' % (num, letter) for num, letter in
                        zip(np.arange(8, 15), lowercase_alphabet[0:15 - 8])]  # cf08a036 - cf14g036
    # this one is categorical:
    fertilityIntRange = ['cf0%s%s128' % (num, letter) if num < 10 else 'cf%s%s128' % (num, letter) for num, letter in
                         zip(np.arange(8, 21), lowercase_alphabet[0:21 - 8])]  # cf08a128 - cf20m128
    fertilityNumRange = ['cf0%s%s129' % (num, letter) if num < 10 else 'cf%s%s129' % (num, letter) for num, letter in
                         zip(np.arange(8, 21), lowercase_alphabet[0:21 - 8])]  # cf08a129 - cf20m129
    num_list = np.concatenate((np.arange(8, 14), np.arange(15, 21)))
    mortalityBeliefRange = ['ch0%s%s006' % (num, letter) if num < 10 else 'ch%s%s006' % (num, letter) for num, letter in
                            zip(num_list, lowercase_alphabet[1:21 - 8])]  # ch08b006 - ch20m006

    numerical_columns = numerical_columns_selector(data)
    categorical_columns = ['woonvorm_2020', 'oplmet_2020', 'cf20m128', 'cf20m011', 'cs20m330',
                           'burgstat_2020'] + religiousRange + fertilityIntRange
    # domesticStatusRange+civilStatusRange+educationRange+woonvormRange

    numerical_columns = [col for col in numerical_columns if col not in categorical_columns]

    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
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
    

    model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=500))
    model.fit(data_balanced, target_balanced)
    
    target_pred = model.predict(data)
    p, r, f, _ = precision_recall_fscore_support(target, target_pred, average='binary')
    print(f'Precision: {p}, recall: {r}, F1-score: {f} on train data')

    # Save the model
    joblib.dump(model, "model.joblib")
    


data = pd.read_csv('C:/Users/nicwa/data_prefer_full/training_data/PreFer_train_data.csv')
target = pd.read_csv('C:/Users/nicwa/data_prefer_full/training_data/PreFer_train_outcome.csv', encoding='cp1252')
train_save_model(data, target)