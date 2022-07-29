import pandas as pd
import numpy as np

from preprocessing.load_data import load_data
from modelling.experiment import cluster_predict
from modelling.comparison import find_difference

from utils.file_util import load_files
import utils.config as config


# read utils
parameters = {'alpha': config.params["param_alpha"]}
num_cluster = config.params["num_cluster"]
model = config.params["model"]


# load data
load_data()


# read prepared data
X_train, X_test, y_train, y_test = load_files(["X_train","X_test","y_train","y_test"])
print("X_train: ", X_train.shape, "/ny_train: ", y_train.shape)
print("X_test: ", X_test.shape, "/ny_test: ", y_test.shape)


# cluster predict
output_dict = cluster_predict(X_train, X_test, y_train, y_test,
                              parameters, model,
                              show_info=True, num_cluster=6)


# compare results
cluster_compare = pd.DataFrame()
for a in range (2,num_cluster-1):
    for b in range(1,a+1):
        empty_dict = {}
        one_cluster_error, current_cluster_error = find_difference(output_dict,"cluster_"+str(a),b,plot=False)
        empty_dict["cluster_num"] = a
        empty_dict["incluster_name"] = b
        empty_dict["shape"] = one_cluster_error.shape[0]
        empty_dict["mse_non_clustered"]  = np.average(one_cluster_error**2)
        empty_dict["mse_clustered"] =  np.average(current_cluster_error**2)
        empty_dict["std_clustered"] =  np.std(current_cluster_error)
        empty_dict["std_non_clustered"] =  np.std(one_cluster_error)
        cluster_compare = cluster_compare.append(empty_dict,ignore_index=True)

for _ in ["cluster_num","incluster_name","shape"]:
    cluster_compare[_] = cluster_compare[_].astype(int)

cluster_compare["mse_diff"] = cluster_compare["mse_non_clustered"] - cluster_compare["mse_clustered"]
cluster_compare["std_diff"] = cluster_compare["std_non_clustered"] - cluster_compare["std_clustered"]

print(cluster_compare)
