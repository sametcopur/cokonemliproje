from sklearn.linear_model import Ridge, LogisticRegression, Lasso
import numpy as np


params = {
    'input_file': 'CASP',
    'input_path': 'C:/Users/02485398/Documents/cokonemliproje/data/',
    'param_alpha': np.logspace(-4,1,50),
    'model': Lasso(tol=0.001)
}