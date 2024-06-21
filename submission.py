"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

clean_df should be used to clean (preprocess) the data.

run.py can be used to test your submission.
"""

# List your libraries and modules here. Don't forget to update environment.yml!
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.compose import make_column_selector as selector
import numpy as np
import string


def clean_df(df, background_df=None):
    """
    Preprocess the input dataframe to feed the model.
    # If no cleaning is done (e.g. if all the cleaning is done in a pipeline) leave only the "return df" command

    Parameters:
    df (pd.DataFrame): The input dataframe containing the raw data (e.g., from PreFer_train_data.csv or PreFer_fake_data.csv).
    background (pd.DataFrame): Optional input dataframe containing background data (e.g., from PreFer_train_background_data.csv or PreFer_fake_background_data.csv).

    Returns:
    pd.DataFrame: The cleaned dataframe with only the necessary columns and processed variables.
    """

    ## This script contains a bare minimum working example


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

    variable_selection_big =['nomem_encr']+ healthRange + religiousRange + childrenNumRange + childrenUnderEightRange + fertilityIntRange + fertilityNumRange + mortalityBeliefRange
    # +nettoIncomeRange
    # domesticStatusRange+ civilStatusRange+educationRange+stedRange+woonvormRange

    variable_selection_subset = ['cf20m011', 'cf20m003', 'cf20m004', 'cf20m007', 'birthyear_bg', 'cs20m330',
                                 'burgstat_2020', 'oplmet_2020', 'sted_2020', 'woonvorm_2020', 'nettohh_f_2020']

    variable_selection = variable_selection_big + variable_selection_subset
    variable_selection_drop = healthRange + childrenNumRange + childrenUnderEightRange + fertilityNumRange + mortalityBeliefRange

    df['meanHealth'] = df[healthRange].mean(axis=1)
    df['sdHealth'] = df[healthRange].std(axis=1)
    df['meanChildrenUnderEight'] = df[childrenUnderEightRange].mean(axis=1)
    df['sdChildrenUnderEight'] = df[childrenUnderEightRange].std(axis=1)
    df['meanChildrenNum'] = df[childrenNumRange].mean(axis=1)
    df['sdChildrenNum'] = df[childrenNumRange].std(axis=1)
    df['meanChildrenNum'] = df[fertilityNumRange].mean(axis=1)
    df['sdChildrenNum'] = df[fertilityNumRange].std(axis=1)
    df['meanMortalityBelief'] = df[mortalityBeliefRange].mean(axis=1)
    df['sdMortalityBelief'] = df[mortalityBeliefRange].std(axis=1)

    # drop individual years
    df.drop(columns=variable_selection_drop)





    df = df[variable_selection]
    filterNA = df.isna()


    # replace with value
    df[filterNA] = -99

    return df


def predict_outcomes(df, background_df=None, model_path="model.joblib"):
    """Generate predictions using the saved model and the input dataframe.

    The predict_outcomes function accepts a Pandas DataFrame as an argument
    and returns a new DataFrame with two columns: nomem_encr and
    prediction. The nomem_encr column in the new DataFrame replicates the
    corresponding column from the input DataFrame. The prediction
    column contains predictions for each corresponding nomem_encr. Each
    prediction is represented as a binary value: '0' indicates that the
    individual did not have a child during 2021-2023, while '1' implies that
    they did.

    Parameters:
    df (pd.DataFrame): The input dataframe for which predictions are to be made.
    background_df (pd.DataFrame): The background dataframe for which predictions are to be made.
    model_path (str): The path to the saved model file (which is the output of training.py).

    Returns:
    pd.DataFrame: A dataframe containing the identifiers and their corresponding predictions.
    """

    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # Load the model
    model = joblib.load(model_path)

    # Preprocess the fake / holdout data
    df = clean_df(df, background_df)

    # Exclude the variable nomem_encr if this variable is NOT in your model
    vars_without_id = df.columns[df.columns != 'nomem_encr']

    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    predictions = model.predict(df[vars_without_id])
    
    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": df["nomem_encr"], "prediction": predictions}
    )

    # Return only dataset with predictions and identifier
    return df_predict
