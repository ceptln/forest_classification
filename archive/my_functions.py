# Import useful packages
import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
import time

from my_dictionaries import *

# Import ML packages 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def split_trainset(df_train, test_size=0.2, random_state=10):

    '''
    Expands row training dataset offering some testing capabilities
    '''

    # Separate features and target
    X = df_train.copy()
    X.drop(columns='Cover_Type', inplace=True)
    y = df_train['Cover_Type']
    
    # Perform a train_test split (if asked)
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    
    # Render X and y (if no testset)
    elif test_size == 0:
        return X, y
    
    else:
        return 'Enter positive test_size'


def add_zone_info(df, climate=True, geographic=True, family=True, rock=True, stony=True, drop_initial_rows=True):

    """ 
    Add climate zone and/or geographical zone to df
    """

    df_temp = df.copy()
    
    if drop_initial_rows:
        soil_columns = [col for col in list(df.columns) if 'Soil_Type' in col]
        df_new = df.drop(columns=soil_columns)
    
    else: 
        df_new = df.copy()

    for col in elu_dict.keys():
        df_temp[col] = df_temp[col] * elu_dict[col]
        df_temp['ELU'] = df_temp[list(elu_dict.keys())].sum(axis=1)
    
    if climate:
        df_temp['ClimZone'] = df_temp['ELU'].apply(lambda x : int(str(x)[0:1]))
        df_new = df_new.join(pd.get_dummies(df_temp['ClimZone'], prefix='ClimZone'))

    if geographic:
        df_temp['GeoZone'] = df_temp['ELU'].apply(lambda x : int(str(x)[1:2]))
        df_new = df_new.join(pd.get_dummies(df_temp['GeoZone'], prefix='GeoZone'))

    if family or rock or stony:
        df_temp['ZoneDesc'] = df_temp['ELU'].apply(lambda x: desc_dict[x])

    if family:
        for fam in family_dict.keys():
            df_new[fam] = df_temp['ZoneDesc'].apply(lambda x: np.where(family_dict[fam] in x, 1, 0))
    
    if rock:
        for rock in rock_dict.keys():
            df_new[rock] = df_temp['ZoneDesc'].apply(lambda x: np.where(rock_dict[rock] in x, 1, 0))

    if stony:
        for stone in stony_dict.keys():
            df_new[stone] = df_temp['ZoneDesc'].apply(lambda x: np.where(stony_dict[stone] in x, 1, 0))

    return df_new



def climate_and_geo(df, climate=True, geographic=True):

    """ 
    Obsolete/Outdated
    Add climate zone and/or geographical zone to df
    """
    
    df_new = df.copy()

    correspondance_dict =  {'Soil_Type1': 2702, 'Soil_Type2': 2703, 'Soil_Type3': 2704, 'Soil_Type4': 2705, 'Soil_Type5': 2706, 'Soil_Type6': 2717, 'Soil_Type7': 3501, 'Soil_Type8': 3502, 'Soil_Type9': 4201, 'Soil_Type10': 4703, 'Soil_Type11': 4704, 'Soil_Type12': 4744, 'Soil_Type13': 4758, 'Soil_Type14': 5101, 'Soil_Type15': 5151, 'Soil_Type16': 6101, 'Soil_Type17': 6102, 'Soil_Type18': 6731, 'Soil_Type19': 7101, 'Soil_Type20': 7102, 'Soil_Type21': 7103, 'Soil_Type22': 7201, 'Soil_Type23': 7202, 'Soil_Type24': 7700, 'Soil_Type25': 7701, 'Soil_Type26': 7702, 'Soil_Type27': 7709, 'Soil_Type28': 7710, 'Soil_Type29': 7745, 'Soil_Type30': 7746, 'Soil_Type31': 7755, 'Soil_Type32': 7756, 'Soil_Type33': 7757, 'Soil_Type34': 7790, 'Soil_Type35': 8703, 'Soil_Type36': 8707, 'Soil_Type37': 8708, 'Soil_Type38': 8771, 'Soil_Type39': 8772, 'Soil_Type40': 8776}

    for col in correspondance_dict.keys():
        df_new[col] = df_new[col] * correspondance_dict[col]

    if climate and geographic:
        df_new['ELU'] = df_new[list(correspondance_dict.keys())].sum(axis=1)
        df_new['ClimZone'] = df_new['ELU'].apply(lambda x : int(str(x)[0:1]))
        df_new['GeoZone'] = df_new['ELU'].apply(lambda x : int(str(x)[1:2]))
        df_new.drop(columns=['ELU'], inplace=True)
        return df_new
    
    elif climate:
        df_new['ELU'] = df_new[list(correspondance_dict.keys())].sum(axis=1)
        df_new['ClimZone'] = df_new['ELU'].apply(lambda x : int(str(x)[0:1]))
        df_new.drop(columns=['ELU'], inplace=True)
        return df_new

    else:
        df_new['ELU'] = df_new[list(correspondance_dict.keys())].sum(axis=1)
        df_new['GeoZone'] = df_new['ELU'].apply(lambda x : int(str(x)[1:2]))
        df_new.drop(columns=['ELU'], inplace=True)
        return df_new


def local_metrics(
    df_train, model, test_size=0.2, 
    display_accuracy=True, display_matrix=True):
    
    '''
    Gives accuracy and confusion matrix of predictions after a train_test_split on training dataset
    '''

    if test_size<0.01:
        return 'test_size must be >0.01'
    
    # Perform a train_test split
    X_train, X_test, y_train, y_test = split_trainset(df_train=df_train, test_size=test_size, random_state=10)
   
    # Fit and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Compute accuracy and confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    matrix = pd.DataFrame(np.array(confusion_matrix(y_test, y_pred)), columns=range(1, 8), index=range(1, 8))
    
    if display_accuracy and display_matrix:
        return accuracy, matrix
    elif matrix:
        return matrix
    else:
        return accuracy


def test_predict(df_train, df_test, model, export_file=True, display_local_metrics=True):
    
    # Store unfitted model for local metrics computation
    unfitted_model = model

    # Separate features and target 
    print('spliting data...')
    X, y = split_trainset(df_train, test_size=0)
    
    # Fit and predict
    print('training model...')
    start = time.time()
    model.fit(X, y)
    print('-- training took', round(time.time() - start, 2), 'sec')
    
    print('predicting test_set...')
    print('can take several minutes...')
    start = time.time()
    pred = model.predict(df_test)
    print('predicting took', round(time.time() - start, 2), 'sec')

    # Store the predictions in a dataset
    print('creating dataframe...')
    df = pd.DataFrame(pred)
    df.reset_index(inplace=True)
    df.rename({'index':'Id', 0:'Cover_type'}, axis='columns', inplace=True)
    df['Id'] = df['Id'].apply(lambda x : x + 1)

    # Export the predictions by writing a csv file in the 'answers' folder (if asked)
    if export_file:
        print('exporting file...')
        os.chdir('/Users/camilleepitalon/Documents/DSB/11_machine_learning_2/Project/')
        try:
            os.chdir('answers')
        except: 
            os.mkdir('answers')
            os.chdir('answers')
        attempt_num = str(len(os.listdir()))
        df.to_csv('full_submission'+attempt_num+'.csv', index=False)
        os.chdir('..')
        print('file created!')

    # Display accuracy and confusion matrix (if asked)
    if display_local_metrics:
        print('computing local metrics...')
        accuracy, matrix = local_metrics(df_train=df_train, model=unfitted_model, test_size=0.2, display_accuracy=True, display_matrix=True)
        print(accuracy)
        print(matrix)
        print('done!')
        return pred, accuracy, matrix

    return pred