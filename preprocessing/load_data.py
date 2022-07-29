import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from utils.file_util import scaler
from utils.file_util import save_files
import utils.config as config


main_input_path = config.params["input_path"]
file_name = config.params["input_file"]
file = main_input_path + file_name + '.csv'

def load_data(poly_features=True, scale=True):

    # read data from local path
    data = pd.read_csv(file)
    target = np.array(data["RMSD"])
    data = data.drop(columns="RMSD")

    # create polynomial features
    if poly_features:
        data = PolynomialFeatures(degree=2).fit_transform(data)

    # split data into train & test
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=52)

    if scale:
        X_train, X_test = scaler(X_train, X_test)

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    # set name for data read & write
    X_train.name = 'X_train'
    X_test.name = 'X_test'
    y_train.name = 'y_train'
    y_test.name = 'y_test'

    save_files([X_train,X_test,y_train,y_test])




