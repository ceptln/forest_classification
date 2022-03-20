# Import useful packages
import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
import time
from datetime import datetime
from csv import writer

# Import ML packages 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


""" 
#################################################
#           Explonatory Data Analysis           #
#################################################
"""

def plot_features_labelbreakdown(df, columns_of_interest=[]):

    """ 
    Plot numerical features with label-breakdown
    """
    if columns_of_interest == []:
        return print('Please specify the columns_of_interest argument')
    
    _ = plt.figure(figsize=(20, 20))
    i = 0
    for col in columns_of_interest:
        i += 1
        plt.subplot(5, 5, i)
        sns.distplot(df[df.Cover_Type == 1][col], hist=False, label='1')
        sns.distplot(df[df.Cover_Type == 2][col], hist=False, label='2')
        sns.distplot(df[df.Cover_Type == 3][col], hist=False, label='3')
        sns.distplot(df[df.Cover_Type == 4][col], hist=False, label='4')
        sns.distplot(df[df.Cover_Type == 5][col], hist=False, label='5')
        sns.distplot(df[df.Cover_Type == 6][col], hist=False, label='6')
        sns.distplot(df[df.Cover_Type == 7][col], hist=False, label='7');


def plot_features_hist(df, columns_of_interest=[]):

    """ 
    Plot numerical features histograms
    """

    if columns_of_interest == []:
        return print('Please specify the columns_of_interest argument')

    _ = plt.figure(figsize=(20, 20))
    i = 0
    for col in columns_of_interest:
        i += 1
        plt.subplot(5, 5, i)
        plt.hist(df[col], bins=30, alpha=0.3, label=columns_of_interest)
        plt.xlabel(col);


def pivot_binary_features(df, new_columns='Cover_Type', new_index='Wilder_Type', feature_key='Wilderness'):
    
    """ 
    Pivot a given binary feature
    """

    piv_columns = [col for col in df.columns if feature_key in col]
    df_temp = pd.DataFrame([(df[piv_columns] * np.arange(1, len(piv_columns)+1)).sum(axis=1),
                          df[new_columns]]).T
    df_temp.columns = [new_index, new_columns]
    df_temp["piv"] = 1
    piv = df_temp.pivot_table(values="piv", index=new_index, columns=new_columns, aggfunc="sum").fillna(0)
    piv['total'] = piv.apply(lambda x: sum(x), axis=1)
    return piv


def repartition_binary_features(df_train, df_test, feature_key='Wilderness'):

    """ 
    Show repartition binary features between training and testing datasets
    """

    feature_columns = [col for col in df_train.columns if feature_key in col]
    ftrain = list(df_train[feature_columns].sum(axis=0))
    ftest = list(df_test[feature_columns].sum(axis=0))
    repartition_table = pd.DataFrame([ftrain, ftest, np.round(np.array(ftrain) / sum(ftrain), 4), np.round(np.array(ftest) / sum(ftest), 4)])
    repartition_table.columns = [feature_key + str(i) for i in range(len(feature_columns))]
    repartition_table.index = ['train', 'test', 'train_ratio', 'test_ratio']
    return repartition_table


def one_hot_encode(df, column_of_interest, new_columns_prefix):
    """ 
    Perform OneHotEncoding on a given column (from 1 column to x)
    """
    df_prep = df.copy()
    df_prep = df_prep.join(pd.get_dummies(df_prep[column_of_interest],  prefix=new_columns_prefix))
    df_prep.drop(columns=column_of_interest, inplace=True)
    return df_prep

def reverse_one_hot_encode(df, columns_of_interest, new_column_name):
    """ 
    Perform a OneHotEncoding reverse (from x columns to a single one)
    """
    df_prep = df.copy()
    df_prep['prep'] = (df_prep[columns_of_interest] == 1).idxmax(1)
    df_prep[new_column_name] = df_prep['prep'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
    target = df_prep['Cover_Type']
    df_prep.drop(columns=['prep', 'Cover_Type']+columns_of_interest, inplace=True)
    df_prep['Cover_Type'] = target
    return df_prep


def outlier_function(df, col_name):

    ''' 
    Detect potential outliers
    '''

    first_quartile = np.percentile(np.array(df[col_name].tolist()), 25)
    third_quartile = np.percentile(np.array(df[col_name].tolist()), 75)
    IQR = third_quartile - first_quartile
                      
    upper_limit = third_quartile+(3*IQR)
    lower_limit = first_quartile-(3*IQR)
    outlier_count = 0
                      
    for value in df[col_name].tolist():
        if (value < lower_limit) | (value > upper_limit):
            outlier_count +=1
    return lower_limit, upper_limit, outlier_count


def relabel(df, first_group, second_group, third_group=None, drop_cover_type=True):
    
    """ 
    Create higher-level groups of cover types
    """
    
    df_relabeled = df.copy()
    if third_group == None:
        df_relabeled['Cover_Group'] = np.where(df['Cover_Type'].isin(first_group), 1, 0)
    else:
        df_relabeled['Cover_Group'] = np.where(df['Cover_Type'].isin(first_group), 2, np.where(df['Cover_Type'].isin(second_group), 1, 0))

    if drop_cover_type:
        df_relabeled.drop(columns=['Cover_Type'], inplace=True)

    return df_relabeled


def split_in_groups(df, first_group, second_group, third_group=None, target='Cover_Type'):
    
    df_prep = df.copy()
    if third_group == None:
        df_g1 = df_prep[np.where(df[target].isin(first_group), True, False)]
        df_g2 = df_prep[np.where(df[target].isin(second_group), True, False)]
        return df_g1, df_g2
    else:
        df_g1 = df_prep[np.where(df[target].isin(first_group), True, False)]
        df_g2 = df_prep[np.where(df[target].isin(second_group), True, False)]
        df_g3 = df_prep[np.where(df[target].isin(third_group), True, False)]
        return df_g1, df_g2, df_g3


""" 
#################################################
# Preprocessing and modelling class and methods #
#################################################
"""

class ClassifTools:
    def __init__(self, 
        df_train, df_test, model, 
        add_eng_features,
        add_climate, add_geographic, 
        add_family, add_rocky, add_stony, 
        keep_initial_rows, columns_to_drop,
        random_state):
        self.df_train =df_train
        self.df_test =df_test
        self.model=model
        self.add_eng_features=add_eng_features
        self.add_climate=add_climate
        self.add_geographic=add_geographic
        self.add_family=add_family
        self.add_rocky=add_rocky
        self.add_stony=add_stony
        self.keep_initial_rows=keep_initial_rows
        self.columns_to_drop=columns_to_drop
        self.random_state=random_state
    

    def split_trainset(self, target='Cover_Type', split_test_size=0):

        """
        Expand row training dataset offering some testing capabilities
        """

        df_train = self.df_train
        # Separate features and target
        X = df_train.copy()
        X.drop(columns=target, inplace=True)
        y = df_train[target]
        
        # Perform a train_test split (if asked)
        if split_test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=self.random_state)
            return X_train, X_test, y_train, y_test
        
        # Render X and y (if no testset)
        elif split_test_size == 0:
            return X, y
        
        else:
            return 'Enter positive test_size'


    def enrich_data(self, df):

        """ 
        Add climate zone and/or geographical zone to a given df (df_train/df_test)
        """
        
        df_temp = df.copy()
        #target = df_temp['Cover_Type']
        #df_new = df_temp.drop(columns=['Cover_Type'])
        df_new = df.copy()
        
        if self.keep_initial_rows:
            pass
        
        else: 
            soil_columns = [col for col in list(df.columns) if 'Soil_Type' in col]
            df_new.drop(columns=soil_columns, inplace=True)
        
        if self.add_eng_features:
            df_new['Euclidian_Distance_To_Hydrology'] = (df_new['Horizontal_Distance_To_Hydrology']**2 + df_new['Vertical_Distance_To_Hydrology'] ** 2) ** 0.5
            df_new['Mean_Hillshade'] = (df_new['Hillshade_9am'] + df_new['Hillshade_Noon'] + df_new['Hillshade_9am']) / 3
            df_new['Mean_HDistances'] = (df_new['Horizontal_Distance_To_Hydrology'] + df_new['Horizontal_Distance_To_Roadways'] + df_new['Horizontal_Distance_To_Fire_Points']) / 3
            df_new['Mean_Elevation_Vertical_Distance_Hydrology'] = (df_new['Elevation'] + df_new['Vertical_Distance_To_Hydrology']) / 2
            df_new['Mean_Distance_Hydrology_Firepoints'] = (df_new['Horizontal_Distance_To_Hydrology'] + df_new['Horizontal_Distance_To_Fire_Points']) / 2
            df_new['Mean_Distance_Hydrology_Roadways'] = (df_new['Horizontal_Distance_To_Hydrology'] + df_new['Horizontal_Distance_To_Roadways']) / 2
            df_new['Mean_Distance_Firepoints_Roadways'] = (df_new['Horizontal_Distance_To_Fire_Points'] + df_new['Horizontal_Distance_To_Roadways']) / 2

        for col in elu_dict.keys():
            df_temp[col] = df_temp[col] * elu_dict[col]
            df_temp['ELU'] = df_temp[list(elu_dict.keys())].sum(axis=1)
        
        if self.add_climate:
            df_temp['ClimZone'] = df_temp['ELU'].apply(lambda x : int(str(x)[0:1]))
            df_new = df_new.join(pd.get_dummies(df_temp['ClimZone'], prefix='ClimZone'))

        if self.add_geographic:
            df_temp['GeoZone'] = df_temp['ELU'].apply(lambda x : int(str(x)[1:2]))
            df_new = df_new.join(pd.get_dummies(df_temp['GeoZone'], prefix='GeoZone'))

        if self.add_family or self.add_rocky or self.add_stony:
            df_temp['ZoneDesc'] = df_temp['ELU'].apply(lambda x: desc_dict[x])

        if self.add_family:
            for fam in family_dict.keys():
                df_new[fam] = df_temp['ZoneDesc'].apply(lambda x: np.where(family_dict[fam] in x, 1, 0))
        
        if self.add_rocky:
            for rock in rock_dict.keys():
                df_new[rock] = df_temp['ZoneDesc'].apply(lambda x: np.where(rock_dict[rock] in x, 1, 0))

        if self.add_stony:
            for stone in stony_dict.keys():
                df_new[stone] = df_temp['ZoneDesc'].apply(lambda x: np.where(stony_dict[stone] in x, 1, 0))

        if self.columns_to_drop == None:
            columns_to_drop = []
        else:
            columns_to_drop = self.columns_to_drop

        to_drop = [col for col in df_new.columns if col in columns_to_drop]
        df_new.drop(columns=to_drop, inplace=True)
        #df_new['Cover_Type'] = target

        return df_new


    def local_metrics(self, unfitted_model=None, test_size=0.2, compute_accuracy=True, compute_matrix=True, target='Cover_Type'):

        """
        Computes accuracy and confusion matrix of predictions after a train_test_split on training dataset 
        """
        

        if test_size<0.01:
            return 'test_size must be >0.01'
        
        # Perform a train_test split
        X_train, X_test, y_train, y_test = self.split_trainset(target=target, split_test_size=0.2)
        X_train = self.enrich_data(df=X_train)
        X_test = self.enrich_data(df=X_test)
        print([col for col in X_train.columns if col not in X_test.columns])
    
        # Fit and predict
        if unfitted_model==None:
            model = self.model
        else:
            model = unfitted_model

        model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        #test_details = (X_test, y_test, y_pred)

        # Compute accuracy and confusion matrix
        accuracy = accuracy_score(y_test, y_pred)
        matrix = pd.DataFrame(np.array(confusion_matrix(y_test, y_pred)), columns=['true'+str(i) for i in list(y_test.unique())], index=['pred'+str(i) for i in list(y_test.unique())])
        
        if compute_accuracy and compute_matrix:
            return accuracy, matrix #, test_details
        elif matrix:
            return matrix
        else:
            return accuracy

    def test_predict(self, export_file=True, compute_local_metrics=True, target='Cover_Type'):

        """ 
        Fits the model, predicts the output and optionally export the results/computes metrics
        """
        pred_start = time.time()
        # Store unfitted model for local metrics computation
        unfitted_model = self.model
        model = self.model

        # Separate features and target 
        print('enriching the data...')
        start = time.time()
        X, y = self.split_trainset(target)
        X_train = self.enrich_data(df=X)
        print('number of features:', X_train.shape[1])
        X_test = self.enrich_data(df=self.df_test)
        print('   -- took', round(time.time() - start, 2), 'sec')
        
        # Fit the model
        print('training model...')
        start = time.time()
        model.fit(X_train, y)
        print('   -- took', round(time.time() - start, 2), 'sec')
        
        # Test the model
        print('predicting test_set...')
        start = time.time()
        pred = model.predict(X_test)
        print('   -- took', round(time.time() - start, 2), 'sec')

        # Store the predictions in a dataset
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

        # Compute accuracy and confusion matrix (if asked)
        if compute_local_metrics:
            print('computing local metrics...')
            start = time.time()
            accuracy, matrix = self.local_metrics(unfitted_model=unfitted_model, compute_accuracy=True, compute_matrix=True, target=target)
            print('   -- took', round(time.time() - start, 2), 'sec')
            print('> SUCCESS')
            pred_time = round(time.time() - pred_start, 2)
            return pred, accuracy, matrix, pred_time

        print('> SUCCESS')
        return pred


""" 
Output formatting functions
"""

def get_class_vars(cl):
    var_class = vars(cl)
    var_cl = var_class.copy()
    del var_cl['df_train']
    del var_cl['df_test']
    return var_cl

def print_results(acc, mat, pred_time, cl):
    print('-------------------------')
    print('- Parameters: ', get_class_vars(cl))
    print('- Execution time:', pred_time)
    print('- Accuracy:', round(acc, 4))
    print('- Confusion matrix:')
    return mat

def export_metrics(acc, mat, pred_time, cl):
    now = datetime.now()
    dt = now.strftime("%d/%m/%Y %H:%M:%S")
    export_list = ['---', dt, str(get_class_vars(cl)), str(pred_time), str(acc), [np.array(mat)]]
    with open('./answers/metrics_history.txt', 'a+', newline='') as txt:  
        # Pass the txt file object to the writer() function
        writer_object = writer(txt)
        # Pass the data in the list as an argument into the writerow() function
        for el in export_list:
            writer_object.writerow([el])
        # Close the file object
        txt.close()
    with open('./answers/metrics_history.csv', 'a+', newline='') as csv:  
        # Pass the CSV  file object to the writer() function
        writer_object = writer(csv)
        # Pass the data in the list as an argument into the writerow() function
        writer_object.writerow(export_list[1:5])
        # Close the file object
        csv.close()
    

""" 
Dictionaries of interest used in the above class methods
"""

elu_dict = {
    'Soil_Type1': 2702,
    'Soil_Type2': 2703,
    'Soil_Type3': 2704,
    'Soil_Type4': 2705,
    'Soil_Type5': 2706,
    'Soil_Type6': 2717,
    'Soil_Type7': 3501,
    'Soil_Type8': 3502,
    'Soil_Type9': 4201,
    'Soil_Type10': 4703,
    'Soil_Type11': 4704,
    'Soil_Type12': 4744,
    'Soil_Type13': 4758,
    'Soil_Type14': 5101,
    'Soil_Type15': 5151,
    'Soil_Type16': 6101,
    'Soil_Type17': 6102,
    'Soil_Type18': 6731,
    'Soil_Type19': 7101,
    'Soil_Type20': 7102,
    'Soil_Type21': 7103,
    'Soil_Type22': 7201,
    'Soil_Type23': 7202,
    'Soil_Type24': 7700,
    'Soil_Type25': 7701,
    'Soil_Type26': 7702,
    'Soil_Type27': 7709,
    'Soil_Type28': 7710,
    'Soil_Type29': 7745,
    'Soil_Type30': 7746,
    'Soil_Type31': 7755,
    'Soil_Type32': 7756,
    'Soil_Type33': 7757,
    'Soil_Type34': 7790,
    'Soil_Type35': 8703,
    'Soil_Type36': 8707,
    'Soil_Type37': 8708,
    'Soil_Type38': 8771,
    'Soil_Type39': 8772,
    'Soil_Type40': 8776
}

desc_dict = {
    2702: 'Cathedral family - Rock outcrop complex, extremely stony',
    2703: 'Vanet - Ratake families complex, very stony',
    2704: 'Haploborolis - Rock outcrop complex, rubbly',
    2705: 'Ratake family - Rock outcrop complex, rubbly',
    2706: 'Vanet family - Rock outcrop complex complex, rubbly',
    2717: 'Vanet - Wetmore families - Rock outcrop complex, stony',
    3501: 'Gothic family',
    3502: 'Supervisor - Limber families complex',
    4201: 'Troutville family, very stony',
    4703: 'Bullwark - Catamount families - Rock outcrop complex, rubbly',
    4704: 'Bullwark - Catamount families - Rock land complex, rubbly',
    4744: 'Legault family - Rock land complex, stony',
    4758: 'Catamount family - Rock land - Bullwark family complex, rubbly',
    5101: 'Pachic Argiborolis - Aquolis complex',
    5151: 'unspecified in the USFS Soil and ELU Survey',
    6101: 'Cryaquolis - Cryoborolis complex',
    6102: 'Gateview family - Cryaquolis complex',
    6731: 'Rogert family, very stony',
    7101: 'Typic Cryaquolis - Borohemists complex',
    7102: 'Typic Cryaquepts - Typic Cryaquolls complex',
    7103: 'Typic Cryaquolls - Leighcan family, till substratum complex',
    7201: 'Leighcan family, till substratum, extremely bouldery',
    7202: 'Leighcan family, till substratum - Typic Cryaquolls complex',
    7700: 'Leighcan family, extremely stony',
    7701: 'Leighcan family, warm, extremely stony',
    7702: 'Granile - Catamount families complex, very stony',
    7709: 'Leighcan family, warm - Rock outcrop complex, extremely stony',
    7710: 'Leighcan family - Rock outcrop complex, extremely stony',
    7745: 'Como - Legault families complex, extremely stony',
    7746: 'Como family - Rock land - Legault family complex, extremely stony',
    7755: 'Leighcan - Catamount families complex, extremely stony',
    7756: 'Catamount family - Rock outcrop - Leighcan family complex, extremely stony',
    7757: 'Leighcan - Catamount families - Rock outcrop complex, extremely stony',
    7790: 'Cryorthents - Rock land complex, extremely stony',
    8703: 'Cryumbrepts - Rock outcrop - Cryaquepts complex',
    8707: 'Bross family - Rock land - Cryumbrepts complex, extremely stony',
    8708: 'Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony',
    8771: 'Leighcan - Moran families - Cryaquolls complex, extremely stony',
    8772: 'Moran family - Cryorthents - Leighcan family complex, extremely stony',
    8776: 'Moran family - Cryorthents - Rock land complex, extremely stony'
}

family_dict = {
    'F_Cathedral' : 'Cathedral',
    'F_Ratake' : 'Ratake',
    'F_Vanet' : 'Vanet',
    'F_Gothic' : 'Gothic',
    'F_Troutville' : 'Troutville',
    'F_Legault' : 'Legault',
    'F_Catamount' : 'Catamount',
    'F_Bullwark' : 'Bullwark',
    'F_Gateview' : 'Gateview',
    'F_Rogert' : 'Rogert',
    'F_Leighcan' : 'Leighcan',
    'F_Como' : 'Como',
    'F_Bross' : 'Bross',
    'F_Moran' : 'Moran'
}

rock_dict = {
    'R_Rock_outcrop' : 'Rock outcrop',
    'R_Ratake_families' : 'Ratake families',
    'R_Limber_families' : 'Limber families',
    'R_Rock_land' : 'Rock land',
    'R_Aquolis' : 'Aquolis',
    'R_Cryoborolis' : 'Cryoborolis',
    'R_Cryaquolis' : 'Cryaquolis',
    'R_Borohemists' : 'Borohemists',
    'R_till_substratum' : 'till substratum',
    'R_Cryaquepts' : 'Cryaquepts',
    'R_Cryumbrepts' : 'Cryumbrepts',
    'R_Cryorthents' : 'Cryorthents',
    'R_Cryaquolls' : 'Cryaquolls',
    'Rock' : 'Rock'
}

stony_dict = { 
    'S_rubbly' : 'rubbly',
    'S_stony' : ', stony',
    'S_very stony' : 'very stony',
    'S_extremely stony' : 'extremely stony'
}
