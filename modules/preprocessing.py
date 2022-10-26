import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import numpy as np


def missing_values_columns(df):

    '''
    this function check for the presence of missing values in each column
    :param df: dataframe
    :return: dictionary with missing values for each column
    '''

    cols=df.columns

    nan_dict={key: None for key in cols}

    for col in cols:

        try:

            nan_dict[col]=df[col].isna().sum()

        except KeyError:

            continue

    return nan_dict


def encode_categorical_features(df,categorical:list):

    '''
    this function performs one-hot encoding of categorical features
    :param df: dataframe
    :param categorical: list of categorical features
    :return: dataframe with categorical columns encoded using one-hot vectors
    '''


    for column in categorical:


        try:

            encoder=OneHotEncoder()
            encoded_arr=encoder.fit_transform(np.array(df[[column]]).reshape(-1,1)).toarray()
            data_encoded=pd.DataFrame(encoded_arr)
            data_encoded=data_encoded.rename(lambda x:f'{column}_{str(x)}',axis=1)
            df=df.drop(columns=column)
            df=pd.concat([df,data_encoded],axis=1)



        except KeyError as e:

            print(e)
            continue



    return df


def standardize(df,categories:list):

    '''
    this function standardize numerical features
    :param df: dataframe
    :param categories: list of numerical columns
    :return: dataframe with standardize numerical columns
    '''


    for column in categories:


        try:

            scaler=StandardScaler()
            scaled_arr=scaler.fit_transform(np.array(df[[column]]).reshape(-1,1))
            df[column]=scaled_arr

        except KeyError as e:

            print(e)
            continue

    return df








