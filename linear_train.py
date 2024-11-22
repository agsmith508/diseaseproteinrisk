import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split

# dataset - olink
x = pd.read_csv(protein_datafile, sep='\t', index_col=0)
input_size = x.shape[1]

# disease predictions
y = pd.read_csv(prev_disease_datafile, sep='\t', index_col=0)
    
# split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y,  test_size=0.2, shuffle=True, stratify=y, random_state=0)

# gridsearch param space
param_grid = {
			'elasticnet__alpha'     : [0.01, 0.1, 1, 10],
			'elasticnet__l1_ratio'  :  np.arange(0.40, 1.00, 0.10),
		}

# make elasticnet pipeline + fit model
cv = StratifiedShuffleSplit(n_splits=5, random_state=0)
gcv = GridSearchCV(
	make_pipeline(StandardScaler(), ElasticNet(max_iter=100)),
	param_grid=param_grid,
	scoring='roc_auc',
	cv=cv,
	n_jobs=1,
).fit(x_train, y_train)

# create best model
reg = make_pipeline(StandardScaler(), ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=100))
reg.set_params(**gcv.best_params_)
reg.fit(x_train, y_train)

# model performance
score_train = reg.score(x_train, y_train)
score_test = reg.score(x_test, y_test)

print('Finished %i -- Train: %.3f Test: %.3f' % (condition, score_train, score_test))