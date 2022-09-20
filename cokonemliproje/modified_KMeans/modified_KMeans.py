from sklearn.cluster import KMeans
class modified_KMeans(KMeans):
    def __init__(self, n_clusters=8, selected_col=None, init="k-means++", n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm="lloyd"):
        self.selected_col = selected_col
        super().__init__(n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state, copy_x=copy_x, algorithm=algorithm)
        
    def fit(self, X, y=None, sample_weight=None):
        if not ((type(self.selected_col) == list) | (callable(self.selected_col)) | (self.selected_col==None)):
            raise TypeError("liste ya da fonksiyon girmeniz gerekmekte. Yoksa None bırakın.")

        if callable(self.selected_col):
            selector = self.selected_col()
            if "y" in selector.fit.__code__.co_varnames:
                if y is None:
                    raise ValueError("y değeri girilmeli")
                selector.fit_transform(X,y)
            else:
                selector.fit_transform(X)
            self.selected_col = selector.get_feature_names_out()
        
        if self.selected_col is not None:
            X = X.loc[:,self.selected_col]
        return super().fit(X, y, sample_weight)

    def predict(self, X, sample_weight=None):
        if self.selected_col is not None:
            X = X.loc[:,self.selected_col]
            
        return super().predict(X, sample_weight)
