from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

from utils.file_util import data_spliter
from preprocessing.clustering import k_meanscluster


def model_predict(X_train, X_test, y_train, y_test, parameters, model):
    """
    tahminleme yapmayı sağlar. gridsearch içinde kullanılacak model ve parametre input olarak verilir.

    parametre:
    X_train, X_test, y_train, y_test: datalar
    model: hangi algoritmanın kullanılacağını seçilir örn. Lasso
    parameters: gridsearchteki kullanılacak algortimanın parametreleri

    return:
    error: test üzerindeki toplam hata oranı (ortalaması değl)
    mean_error: crossvalidasyonda en iyi modelin hata ortalaması
    std: crossvalidasyonda en iyi modelin stdsi
    len(y_test): clusterdaki toplam veri sayısı
    error_values: hata miktarları (array tipinde)

    """
    grid = GridSearchCV(model, parameters, scoring="neg_mean_squared_error", cv=5, verbose=1, n_jobs=15)
    grid.fit(X_train, y_train)
    std = grid.cv_results_['std_test_score'][grid.best_index_]
    mean_error = grid.cv_results_['mean_test_score'][grid.best_index_]
    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    error_values = (y_test - y_pred)
    error = mean_squared_error(y_test, y_pred) * len(y_test)
    return error, mean_error, std, len(y_test), error_values


def cluster_predict(X_train, X_test, y_train, y_test, parameters, model, show_info = True, num_cluster=6):
    output_dict = {}
    for cluster in range(1,num_cluster):
        train_labels,test_labels = k_meanscluster(X_train,X_test,n_clusters=cluster)
        total_error = 0
        index_output = {}
        for index in range(0,cluster):
            X_train_loc,X_test_loc,y_train_loc,y_test_loc = data_spliter(X_train, X_test, y_train, y_test,train_labels,test_labels,index=index)
            #X_train_loc, X_test_loc = scaler(X_train_loc,X_test_loc)
            error, mean_error, std, sample_size , error_values = model_predict(X_train_loc,X_test_loc,y_train_loc,y_test_loc,parameters,model)
            total_error += error
            index_output["cluster_"+str(index+1)] = {"mean_error":mean_error,"std":std,"sample_size":sample_size,"error_values":error_values}
        output_dict["cluster_"+str(cluster)] = {"mse":total_error / len(y_test), "test_labels":test_labels,
                                         "inside_cluster":index_output}
        if show_info == True:
            print("cluster"+str(cluster))
            print("mse", total_error / len(y_test))
            print("*****")

        return output_dict