import pandas as pd
import os.path

import utils.config as config



def save_files(df_list):
    '''
    accepts dataframe list as input
    saves each dataframe in the tmp folder as csv
    the file name corresponds to the dataframe "name" attribute
    '''
    path = config.params["input_path"]
    [df.to_csv(path + df.name + '.csv' , sep=',', index=False) for df in df_list]


def load_files(names_list):
    '''
    accepts a list of names (str) as input
    load each csv file from the tmp folder with the input names
    returns a list of loaded dataframes
    '''
    df_list = []

    path = config.params["input_path"]
    [df_list.append(pd.read_csv(path + name + ".csv")) for name in names_list if
    os.path.isfile(path + name + '.csv')]

    return df_list


def data_spliter(X_train, X_test, y_train, y_test, train_labels, test_labels, index):
    """
    verileri kümeleme indekslerine göre ayrır.

    parametre:
    X_train, X_test, y_train, y_test: datalar
    train_labels: train datası için kümüleme sonuçları
    test_labels: test datası için kümeleme sonuçları
    index: train ve test labellerı içindeki hangi sınıfın filtreleceği

    return:
    X_train_loc,X_test_loc,y_train_loc,y_test_loc: datalar

    örneğin:

    input:
        X_train:      [1, 2, 3, 4, 5]
        y_train:      [6, 7, 8, 9, 10]
        train_labels: [0, 0, 0, 1, 1]

        X_test:       [11 , 12, 13]
        y_test:       [14 , 15, 16]
        test_labels:  [0  , 1 , 0]

        index = 1

    return:
         X_train_loc: [4, 5]
         y_train_loc: [0, 10]

         X_test_loc:  [12]
         y_test_loc:  [15]
    """
    train_index = np.where(train_labels == index)
    test_index = np.where(test_labels == index)
    X_train_loc = X_train[train_index]
    y_train_loc = y_train[train_index]
    X_test_loc = X_test[test_index]
    y_test_loc = y_test[test_index]
    return X_train_loc, X_test_loc, y_train_loc, y_test_loc
