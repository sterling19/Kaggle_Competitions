import numpy as np
from math import sqrt
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge

import feat_eng as fe
x_train, x_test, y_train, ntrain, ntest, test = fe.engineer_features()

# Credit: kaggle.com/eliotbarr
class SklearnWrapper(object):
    def __init__(self, clf, params=None):
        self.clf=clf(**params)
        
    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
        
    def predict(self, x):
        return self.clf.predict(x)

# Credit: kaggle.com/eliotbarr       
class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))

# Credit: kaggle.com/eliotbarr
def get_oof(clf): # Out-of-fold
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((5, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)        
        
xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.2,
    'silent': 1,
    'subsample': 0.2,
    'reg_alpha' : 0.9,
    'reg_lambda' : 0.6,
    'learning_rate': 0.01,
    'objective': 'reg:linear',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 1.5,
    'eval_metric': 'rmse',
    'nrounds': 30000
}

rd_params={'alpha': 4,
           'random_state': 0}

ls_params={'alpha': 0.00098,
           'random_state': 0,
           'max_iter' : 50000}
           
kr_params={'alpha' : 0.3, 
           'kernel' : 'polynomial', 
           'degree' : 2, 
           'coef0' : 1.85}

kf = KFold(ntrain, n_folds=5, shuffle=True, random_state=0)
           
xg = XgbWrapper(seed=0, params=xgb_params)
rd = SklearnWrapper(clf=Ridge, params=rd_params)
ls = SklearnWrapper(clf=Lasso, params=ls_params)
kr = SklearnWrapper(clf=KernelRidge, params=kr_params)

# TRAIN & PREDICT
def build_ensemble():
    xg_oof_train, xg_oof_test = get_oof(xg)
    rd_oof_train, rd_oof_test = get_oof(rd)
    ls_oof_train, ls_oof_test = get_oof(ls)
    kr_oof_train, kr_oof_test = get_oof(kr)

    """ 01/13/17 -- 12:03p -- Final Submission """
    print("XG-CV: {}".format(sqrt(mean_squared_error(y_train, xg_oof_train)))) # 0.122758
    print("RD-CV: {}".format(sqrt(mean_squared_error(y_train, rd_oof_train)))) # 0.136409
    print("LS-CV: {}".format(sqrt(mean_squared_error(y_train, ls_oof_train)))) # 0.125914
    print("KR-CV: {}".format(sqrt(mean_squared_error(y_train, kr_oof_train)))) # 0.179759

    x_train = np.concatenate((xg_oof_train, ls_oof_train, rd_oof_train), axis=1)
    x_test = np.concatenate((xg_oof_test, ls_oof_test, rd_oof_test), axis=1)

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test)

    xgb_params = {
        'seed': 0,
        'colsample_bytree': 0.8,
        'silent': 1,
        'subsample': 0.6,
        'learning_rate': 0.01,
        'objective': 'reg:linear',
        'max_depth': 1,
        'num_parallel_tree': 1,
        'min_child_weight': 1,
        'eval_metric': 'rmse',
    }

    res = xgb.cv(xgb_params, dtrain, num_boost_round=1000, nfold=5, seed=0, stratified=False,
             early_stopping_rounds=25, verbose_eval=10, show_stdv=True)

    best_nrounds = res.shape[0] - 1
    cv_mean = res.iloc[-1, 0]
    cv_std = res.iloc[-1, 1]
    print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std)) # 0.120805

    gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

    return(gbdt, dtest, test)
