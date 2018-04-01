''' Load packages/data '''
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
warnings.filterwarnings('ignore')

class StackingEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator
		
    def fit(self, X, y=None, **params):
        self.estimator.fit(X, y, **params)
        return self
		
    def transform(self, X):
        X_transformed = np.copy(X)
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))
        return X_transformed

train = pd.read_csv('train.csv')
y_train = train['y'].values
y_mean = np.mean(y_train)
train.drop('y', axis=1, inplace=True)

test = pd.read_csv('test.csv')
id_test = test['ID'].values

''' Label Encoding '''
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

''' Seperate data for non-XGBoost models '''
train_nodeco = train.copy() 
test_nodeco = test.copy()

''' Decomposition '''
pca = PCA(12, random_state=0)
pca_train = pca.fit_transform(train, y_train)
pca_test = pca.transform(test)

ica = FastICA(12, random_state=0)
ica_train = ica.fit_transform(train, y_train)
ica_test = ica.transform(test)

tsvd = TruncatedSVD(12, random_state=0)
tsvd_train = tsvd.fit_transform(train, y_train)
tsvd_test = tsvd.transform(test)

grp = GaussianRandomProjection(12, eps=0.1, random_state=0)
grp_train = grp.fit_transform(train, y_train)
grp_test = grp.transform(test)

srp = SparseRandomProjection(12, dense_output=True, random_state=0)
srp_train = srp.fit_transform(train, y_train)
srp_test = srp.transform(test)

for i in range(12):
    train['pca_' + str(i)] = pca_train.T[i]
    test['pca_' + str(i)] = pca_test.T[i]

    train['ica_' + str(i)] = ica_train.T[i]
    test['ica_' + str(i)] = ica_test.T[i]

    train['tsvd_' + str(i)] = tsvd_train.T[i]
    test['tsvd_' + str(i)] = tsvd_test.T[i]

    train['grp_' + str(i)] = grp_train.T[i]
    test['grp_' + str(i)] = grp_test.T[i]

    train['srp_' + str(i)] = srp_train.T[i]
    test['srp_' + str(i)] = srp_test.T[i]

''' XGBoost '''
xgb_params = {
'n_trees': 520, 
'eta': 0.0045,
'max_depth': 4,
'subsample': 0.98,
'objective': 'reg:linear',
'eval_metric': 'rmse',
'base_score': y_mean,
'silent': 1,
}

dtrain = xgb.DMatrix(train, y_train)
dtest = xgb.DMatrix(test)

model = xgb.train(xgb_params, dtrain, num_boost_round=1250) # 700 for best CV
xgb_preds = model.predict(dtest)

''' Stacked Pipeline '''
stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV()),
    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=5, max_features=0.5, 
	    min_samples_leaf=18, min_samples_split=14, subsample=0.7, alpha=0.9, random_state=0)),
	LassoLarsCV()
	)

stacked_pipeline.fit(train_nodeco, y_train)
stacked_preds = stacked_pipeline.predict(test_nodeco)

''' Combine preds and write output '''
final_preds = (xgb_preds * 0.75) + (stacked_preds * 0.25)

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = final_preds
sub.to_csv('.//subs//subXXX.csv', index=False)

# 0.57001