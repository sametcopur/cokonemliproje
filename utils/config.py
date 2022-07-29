from sklearn.linear_model import Ridge, LogisticRegression, Lasso
import numpy as np


params = {
    'input_file': 'CASP',
    'input_path': 'C:/Users/02485398/Documents/cokonemliproje/data/',
    'n_job': 1,
    'num_cluster': 6,
    'param_alpha': [0, 0.05, 0.5, 1, 2],
    'model': Lasso()
}