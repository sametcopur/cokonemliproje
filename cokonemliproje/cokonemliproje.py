from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Tuple
from cokonemliproje.modified_KMeans.modified_KMeans import modified_KMeans


class cokonemliproje:
    def __init__(self,model,
                selected_col = None,
                outlier_detection:bool = False,
                outlier_detection_cluster:bool = False) -> None:
        """_summary_

        Args:
            model (_type_): _description_
            selected_col (_type_, optional): _description_. Defaults to None.
        """ 
        if (outlier_detection == False) & (outlier_detection_cluster == True):
            raise ValueError("böyle bir şey olamaz.")
        self.model = model
        self.selected_col = selected_col
        self.outlier_detection = outlier_detection
        self.outlier_detection_cluster = outlier_detection_cluster
    
    def __repr__(self):
        return f"cokonemliproje({self.model.__class__.__name__})"

    def __k_meanscluster(self,X,y,n_clusters)-> Tuple[modified_KMeans,  np.array]:
        """_summary_

        Args:
            self (_type_): _description_

        Returns:
            _type_: _description_
        """        
        cluster = modified_KMeans(n_clusters=n_clusters,random_state=42,selected_col=self.selected_col)
        if callable(self.selected_col):
            if "y" in self.selected_col().fit.__code__.co_varnames:
                cluster.fit(X,y)
            else:
                cluster.fit(X)
            self.selected_col  = list(cluster.selected_col)
        else:
            cluster.fit(X) 
        train_labels = cluster.labels_
        return cluster, train_labels

    def __data_spliter(self,X:pd.DataFrame, y:pd.DataFrame, labels:np.array, index:int) -> Tuple[pd.DataFrame,pd.DataFrame]:
        """
        Args:
            X (pd.DataFrame): Feature input of dataframe.
            y (pd.DataFrame): Target input of dataframe.
            labels (np.array): Output of kmeans cluster.
            index (int): index in labels.

        Returns:
             Tuple[pd.DataFrame,pd.DataFrame] : X_loc and y_loc
        """        
        indexs = np.where(labels==index)
        X_loc = X.iloc[indexs]
        y_loc = y.iloc[indexs]
        
        return X_loc,y_loc

    def __scaler(self,X1,X2=None):
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X1 = scaler.fit_transform(X1)

        if X2 != None:
            X2 = scaler.transform(X2)
            return X1, X2
        else:
            return X1

    def __model_fit(self, X, y):
        grid = self.model
        grid.fit(X, y)

        best_estimator = grid.best_estimator_
        std = grid.cv_results_['std_test_score'][grid.best_index_]
        mean_error = grid.cv_results_['mean_test_score'][grid.best_index_]

        return best_estimator, std, mean_error

    def __model_predict(self,X,y,estimator):
            
        y_pred = estimator.predict(X)

        error_values = (y - y_pred).to_numpy()
        total_errors = (mean_squared_error(y, y_pred) * len(y))
        shape = y.shape[0]

        return total_errors, shape, error_values

    def __outlier_detection_fit(self,X,y):
        self.isolation  = IsolationForest(random_state=42).fit(X)
        
        self.fit_outlier_labels = self.isolation.predict(X)

        not_outlier_index = np.where(self.fit_outlier_labels==1) 
        outlier_index = np.where(self.fit_outlier_labels==-1)

        X_outlier = X.iloc[outlier_index]
        y_outlier = y.iloc[outlier_index]

        X_not_outlier = X.iloc[not_outlier_index]
        y_not_outlier = y.iloc[not_outlier_index]
        
        return X_outlier, y_outlier, X_not_outlier, y_not_outlier
    
    def __outlier_detection_predict(self,X,y):

        self.predict_outlier_labels = self.isolation.predict(X)

        not_outlier_index = np.where(self.predict_outlier_labels==1) 
        outlier_index = np.where(self.predict_outlier_labels==-1)

        X_outlier = X.iloc[outlier_index]
        y_outlier = y.iloc[outlier_index]

        X_not_outlier = X.iloc[not_outlier_index]
        y_not_outlier = y.iloc[not_outlier_index]
        
        return X_outlier, y_outlier, X_not_outlier, y_not_outlier

    def fit(self,X,y,n_clusters:int,scale:bool):

        self.clusters_ = {}
        self.fit_labels_ = {}
        self.best_estimator_ = {}
        self.cv_std_ = {}
        self.cv_mean_error_ = {}
        self.__n_clusters = n_clusters
        self.__scale = scale
        self.fit_shape_ = {}

        def __sub_fit(X,y,cluster_:int,scale:bool):
            cluster, labels = self.__k_meanscluster(X,y,n_clusters=np.abs(cluster_))
            self.clusters_[cluster_] = cluster
            self.fit_labels_[cluster_] = labels
            std_dict = {}
            best_estimator_dict = {}
            mean_error_dict = {}
            shape_list = {}
            for index in range(np.abs(cluster_)):
                X_loc,y_loc = self.__data_spliter(X, y, labels, index)
                if X_loc.shape[0] < self.model.cv:
                    break
                if self.__scale == True:
                    X_loc = self.__scaler(X_loc,X2=None)
                best_estimator, std, mean_error =  self.__model_fit(X_loc, y_loc)
                best_estimator_dict[index+1] = best_estimator
                std_dict[index+1] = std
                mean_error_dict[index+1] = mean_error
                shape_list[index+1] = X_loc.shape[0]
            self.best_estimator_[cluster_] = best_estimator_dict
            self.cv_std_[cluster_] = std_dict
            self.cv_mean_error_[cluster_] = mean_error_dict
            self.fit_shape_[cluster_] = shape_list
            if self.model.verbose > 0:
                print(f"Cluster {cluster_} completed.")

        if self.outlier_detection:
            X_outlier, y_outlier, X_not_outlier, y_not_outlier = self.__outlier_detection_fit(X,y)
            if self.outlier_detection_cluster:
                for cluster_ in range(1,n_clusters+1):
                    for key, X, y in [(1,X_not_outlier, y_not_outlier),(-1,X_outlier, y_outlier)]:
                        __sub_fit(X , y, cluster_ = cluster_ * key, scale=self.__scale)
            else:
                std_dict = {}
                best_estimator_dict = {}
                mean_error_dict = {}
                shape_list = {}
                if self.__scale == True:
                    X_outlier = self.__scaler(X_outlier,X2=None)
                best_estimator, std, mean_error =  self.__model_fit(X_outlier, y_outlier)
                best_estimator_dict[1] = best_estimator
                std_dict[1] = std
                mean_error_dict[1] = mean_error
                shape_list[1] = X_outlier.shape[0]
                self.best_estimator_[-1] = best_estimator_dict
                self.cv_std_[-1] = std_dict
                self.cv_mean_error_[-1] = mean_error_dict
                self.fit_shape_[-1] = shape_list
                for cluster_ in range(1,n_clusters+1):
                    __sub_fit(X_not_outlier, y_not_outlier, cluster_, self.__scale)
        else:
            for cluster_ in range(1,n_clusters+1):
                __sub_fit(X, y, cluster_, self.__scale)

        return self
    
    def predict(self,X,y):

        self.predict_labels_ = {}
        self.predict_total_errors = {}
        self.predict_shape_ = {}
        self.predict_error_values = {}

        def __sub_predict(X,y,cluster_):
            clusterer = self.clusters_[cluster_]
            labels = clusterer.predict(X)
            self.predict_labels_[cluster_] = labels

            total_errors_dict = {}
            shape_dict = {}
            error_values_dict = {}

            for index in range(np.abs(cluster_)):
                estimator = self.best_estimator_[cluster_][index+1]
                X_loc,y_loc = self.__data_spliter(X, y, labels, index)
                if X_loc.shape[0] == 0:
                    break
                if self.__scale == True:
                    X_loc = self.__scaler(X_loc,X2=None)
                total_errors, shape, error_values = self.__model_predict(X_loc,y_loc,estimator)
                
                total_errors_dict[index+1] = total_errors
                shape_dict[index+1] = shape
                error_values_dict[index+1] = error_values

            self.predict_total_errors[cluster_] = total_errors_dict
            self.predict_shape_[cluster_] = shape_dict
            self.predict_error_values[cluster_] = error_values_dict

        if self.outlier_detection:
            X_outlier, y_outlier, X_not_outlier, y_not_outlier = self.__outlier_detection_predict(X,y)
            if self.outlier_detection_cluster:
                for cluster_ in range(1,self.__n_clusters+1):
                    for key, X, y in [(1,X_not_outlier, y_not_outlier),(-1,X_outlier, y_outlier)]:
                        __sub_predict(X , y, cluster_ = cluster_ * key)
            else:
                total_errors_dict = {}
                shape_dict = {}
                error_values_dict = {}
                estimator = self.best_estimator_[-1][1]
                if self.__scale == True:
                    X_outlier = self.__scaler(X_outlier,X2=None)
                total_errors, shape, error_values = self.__model_predict(X_outlier,y_outlier,estimator)
                total_errors_dict[1] = total_errors
                shape_dict[1] = shape
                error_values_dict[1] = error_values
                self.predict_total_errors[-1] = total_errors_dict
                self.predict_shape_[-1] = shape_dict
                self.predict_error_values[-1] = error_values_dict
                for cluster_ in range(1,self.__n_clusters+1):
                    __sub_predict(X_not_outlier, y_not_outlier, cluster_)
        
        else:
            for cluster_ in range(1,self.__n_clusters+1):
                __sub_predict(X, y, cluster_)

        return self

    def predict_output(self,deep=True):

        def __calc(error_values):
            std = error_values.std()
            mae = np.mean(np.abs(error_values))
            mse = np.mean(np.square(error_values))
            shape = error_values.shape[0]
            return std, mae, mse, shape

        output = pd.DataFrame()

        if deep == True:
            for cluster in self.predict_error_values.keys():
                #if cluster in  self.predict_error_values.keys():
                    #for incluster in range(1,np.abs(cluster)+1):
                for incluster in self.predict_error_values[cluster].keys():
                        error_values = self.predict_error_values[cluster][incluster]
                        std, mae, mse, shape = __calc(error_values)
                        row = {"cluster":cluster,
                                "incluster":incluster,
                                "shape":shape,
                                "std":std,
                                "mae":mae,
                                "mse":mse}
                        row = pd.DataFrame(row,index=[0])
                        output = pd.concat([output, row], ignore_index=True)
            output[["incluster","shape"]] = output[["incluster","shape"]] .astype(int)
        else:
            for cluster in self.predict_error_values.keys():
                #if cluster in self.predict_error_values.keys():
                error_values_stacked = np.array([])
                for incluster in self.predict_error_values[cluster].keys():
                    error_values = self.predict_error_values[cluster][incluster]
                    error_values_stacked = np.concatenate([error_values_stacked,error_values])
                std, mae, mse, shape = __calc(error_values_stacked)
                row = {"cluster":cluster,
                        "std":std,
                        "mae":mae,
                        "mse":mse}
                row = pd.DataFrame(row,index=[0])
                output = pd.concat([output, row], ignore_index=True)

        output.cluster = output.cluster.astype(int)

        return output
    
    def fit_output(self):
        output = pd.DataFrame()
        for cluster in self.best_estimator_.keys():
            for incluster in self.best_estimator_[cluster].keys():
                std = self.cv_std_[cluster][incluster]
                mean = self.cv_mean_error_[cluster][incluster]
                shape = self.fit_shape_[cluster][incluster]
                row = {"cluster":cluster,
                        "incluster":incluster,
                        "shape":shape,
                        "cv_std":std,
                        "cv_mean_error":mean}
                row = pd.DataFrame(row,index=[0])
                output = pd.concat([output, row], ignore_index=True)
        output[["cluster","incluster","shape"]] = output[["cluster","incluster","shape"]].astype(int)
        return output
    
    def cluster_comparer(self):

        def __find_difference(cluster, index, control_item):
            error_values = self.predict_error_values
            one_cluster_error = error_values[control_item][1]
            current_cluster_error = error_values[cluster][index]
            cluster_labels = self.predict_labels_[cluster]
            cluster_index = np.where(cluster_labels == index - 1)
            one_cluster_error_loc = one_cluster_error[cluster_index]
            return one_cluster_error_loc, current_cluster_error

        cluster_compare = pd.DataFrame()

        for cluster_ in range(-self.__n_clusters,self.__n_clusters+1):
            if (cluster_ in self.best_estimator_.keys()) & (cluster_ != 1) & (cluster_!=-1) :
                    for index in self.predict_error_values[cluster_].keys():
                        empty_dict = {}
                        if cluster_ < -1:
                            control_item = -1
                        else:
                            control_item = 1
                        one_cluster_error, current_cluster_error = __find_difference(cluster_,index,control_item)
                        empty_dict["cluster_num"] = cluster_
                        empty_dict["incluster_name"] = index
                        empty_dict["shape"] = self.predict_shape_[cluster_][index]
                        empty_dict["mse_non_clustered"]  = np.average(one_cluster_error**2)
                        empty_dict["mse_clustered"] =  np.average(current_cluster_error**2)
                        empty_dict["std_clustered"] =  np.std(current_cluster_error)
                        empty_dict["std_non_clustered"] =  np.std(one_cluster_error)
                        empty_dict = pd.DataFrame(empty_dict,index=[0])
                        cluster_compare = pd.concat([cluster_compare,empty_dict],ignore_index=True)

        for _ in ["cluster_num","incluster_name","shape"]:
            cluster_compare[_] = cluster_compare[_].astype(int)

        cluster_compare["mse_diff"] = cluster_compare["mse_non_clustered"] - cluster_compare["mse_clustered"]
        cluster_compare["std_diff"] = cluster_compare["std_non_clustered"] - cluster_compare["std_clustered"]
        
        return cluster_compare
    