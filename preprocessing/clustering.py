from sklearn.cluster import KMeans


def k_meanscluster(X_train, X_test, n_clusters):
    """
    veriyi kmeans kullanarak scale eder.

    parametre
    n_cluser : kmeans'in kaç cluster olacağını belirler:

    return:
    train_labels: train datası için cluster indexlerini verir ör: [0,1,1,1,1]
    test_labels: test datası için cluster indexlerini verir ör: [0,0,1]
    """
    cluster = KMeans(n_clusters=n_clusters, random_state=42).fit(X_train)
    train_labels = cluster.labels_
    test_labels = cluster.predict(X_test)
    return train_labels, test_labels