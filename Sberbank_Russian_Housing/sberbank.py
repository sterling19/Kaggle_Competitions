''' LOADING/MERGING '''
import warnings
import numpy as np
import pandas as pd
import xgboost as xg
import lightgbm as lgbm
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
# from sklearn.linear_model import LinearRegression, SGDRegressor, BayesianRidge
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

train_df = pd.read_csv("train.csv")
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])

train_df = train_df[train_df.price_doc/train_df.full_sq <= 600000]
train_df = train_df[train_df.price_doc/train_df.full_sq >= 10000]

test_df = pd.read_csv("test.csv")
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

macro_df = pd.read_csv("macro.csv")
macro_df['timestamp'] = pd.to_datetime(macro_df['timestamp'])

all_train = train_df.merge(macro_df, how='inner', on='timestamp')
all_test = test_df.merge(macro_df, how='inner', on='timestamp')

y_train = np.log1p(train_df.price_doc.values)
test_ids = test_df.id.values

''' SUBSETTING FEATURES '''
feats_to_use = [x for x in test_df.columns[1:]]

for col in feats_to_use:
	if col[:2] == 'ID':
		feats_to_use.remove(col)	
		
x_train = all_train[feats_to_use]
x_test = all_test[feats_to_use]
	
''' OUTLIERS AND MISSING DATA '''
# Credit: kaggle.com/keremt
x_train.ix[x_train[x_train.life_sq > x_train.full_sq].index, "life_sq"] = np.nan
x_test.ix[[601,1896,2791], "life_sq"] = x_test.ix[[601,1896,2791], "full_sq"]
x_test.ix[x_test[x_test.life_sq > x_test.full_sq].index, "life_sq"] = np.nan
x_train.ix[x_train[x_train.life_sq < 5].index, "life_sq"] = np.nan
x_test.ix[x_test[x_test.life_sq < 5].index, "life_sq"] = np.nan
x_train.ix[x_train[x_train.full_sq < 5].index, "full_sq"] = np.nan
x_test.ix[x_test[x_test.full_sq < 5].index, "full_sq"] = np.nan
x_train.ix[13117, "build_year"] = x_train.ix[13117, "kitch_sq"]
x_train.ix[x_train[x_train.kitch_sq >= x_train.life_sq].index, "kitch_sq"] = np.nan
x_test.ix[x_test[x_test.kitch_sq >= x_test.life_sq].index, "kitch_sq"] = np.nan
x_train.ix[x_train[(x_train.kitch_sq == 0).values + (x_train.kitch_sq == 1).values].index, "kitch_sq"] = np.nan
x_test.ix[x_test[(x_test.kitch_sq == 0).values + (x_test.kitch_sq == 1).values].index, "kitch_sq"] = np.nan
x_train.ix[x_train[(x_train.full_sq > 210) & (x_train.life_sq / x_train.full_sq < 0.3)].index, "full_sq"] = np.nan
x_test.ix[x_test[(x_test.full_sq > 150) & (x_test.life_sq / x_test.full_sq < 0.3)].index, "full_sq"] = np.nan
x_train.ix[x_train[x_train.life_sq > 300].index, ["life_sq", "full_sq"]] = np.nan
x_test.ix[x_test[x_test.life_sq > 200].index, ["life_sq", "full_sq"]] = np.nan
x_train.product_type.value_counts(normalize= True)
x_test.product_type.value_counts(normalize= True)
x_train.ix[x_train[x_train.build_year < 1500].index, "build_year"] = np.nan
x_test.ix[x_test[x_test.build_year < 1500].index, "build_year"] = np.nan
x_train.ix[x_train[x_train.num_room == 0].index, "num_room"] = np.nan
x_test.ix[x_test[x_test.num_room == 0].index, "num_room"] = np.nan
x_train.ix[[10076, 11621, 17764, 19390, 24007, 26713, 29172], "num_room"] = np.nan
x_test.ix[[3174, 7313], "num_room"] = np.nan
x_train.ix[x_train[(x_train.floor == 0).values * (x_train.max_floor == 0).values].index, ["max_floor", "floor"]] = np.nan
x_train.ix[x_train[x_train.floor == 0].index, "floor"] = np.nan
x_train.ix[x_train[x_train.max_floor == 0].index, "max_floor"] = np.nan
x_test.ix[x_test[x_test.max_floor == 0].index, "max_floor"] = np.nan
x_train.ix[x_train[x_train.floor > x_train.max_floor].index, "max_floor"] = np.nan
x_test.ix[x_test[x_test.floor > x_test.max_floor].index, "max_floor"] = np.nan
x_train.ix[23584, "floor"] = np.nan
x_train.ix[33, "state"] = np.nan
x_train.loc[x_train.full_sq == 0, 'full_sq'] = 50

''' FEATURE ENGINEERING '''
x_train['life_prop'] = x_train['life_sq']/x_train['full_sq']
x_test['life_prop'] = x_test['life_sq']/x_test['full_sq']

x_train['room_prop'] = x_train['num_room']/x_train['full_sq']
x_test['room_prop'] = x_test['num_room']/x_test['full_sq']

x_train['pop_area'] = x_train['raion_popul']/x_train['area_m']
x_test['pop_area'] = x_test['raion_popul']/x_test['area_m']

x_train['kitch_life'] = x_train['kitch_sq']/x_train['life_sq']
x_test['kitch_life'] = x_test['kitch_sq']/x_test['life_sq']

month_year = (x_train.timestamp.dt.month + x_train.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
x_train['month_year_cnt'] = month_year.map(month_year_cnt_map)

month_year = (x_test.timestamp.dt.month + x_test.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
x_test['month_year_cnt'] = month_year.map(month_year_cnt_map)

week_year = (x_train.timestamp.dt.weekofyear + x_train.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
x_train['week_year_cnt'] = week_year.map(week_year_cnt_map)

week_year = (x_test.timestamp.dt.weekofyear + x_test.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
x_test['week_year_cnt'] = week_year.map(week_year_cnt_map)

x_train['month'] = x_train.timestamp.dt.month
x_train['dow'] = x_train.timestamp.dt.dayofweek

x_test['month'] = x_test.timestamp.dt.month
x_test['dow'] = x_test.timestamp.dt.dayofweek

x_train.drop('timestamp', axis=1, inplace=True)
x_test.drop('timestamp', axis=1, inplace=True)

''' DUMMIES AND LABEL ENCODING '''		
categorical = []
for col in x_train.columns:
	if x_train[col].dtype == 'object':
		categorical.append(col)

categorical.remove('sub_area')
categorical.remove('ecology')

for col in categorical:
	lbl = LabelEncoder()
	lbl.fit(list(x_train[col].values) + list(x_test[col].values))
	x_train[col] = lbl.transform(list(x_train[col].values))
	x_test[col] = lbl.transform(list(x_test[col].values))

x_train = pd.get_dummies(x_train, drop_first=False)
x_train.drop('sub_area_Poselenie Klenovskoe', axis=1, inplace=True)
x_test = pd.get_dummies(x_test, drop_first=False)

''' LGBM '''
lgbm_params = {'learning_rate' : 0.05,
'num_leaves' : 32,
'max_depth' : 5,
'objective' : 'regression',
'metric' : {'l2'},
'min_split_gain' : 0,
'min_child_weight' : 5,
'min_child_samples' : 10,
'subsample' : 1,
'subsample_freq' : 1,
'colsample_bytree' : 1,
'reg_alpha' : 5,
'reg_lambda' : 5,
'verbose' : 0,
'callbacks' : 'record_evaluation()',
'seed' : 0}

x_tr = lgbm.Dataset(x_train, y_train)
cv_log = lgbm.cv(lgbm_params, x_tr, num_boost_round=1000, nfold=5, verbose_eval=50, early_stopping_rounds=50)

best_iter = len(cv_log['l2-mean'])
print(cv_log['l2-mean'][best_iter-1])

model = lgbm.train(lgbm_params, x_tr, num_boost_round=best_iter)
preds0 = model.predict(x_test)
lgbm_preds = np.expm1(preds0)

''' EXTRA TREES '''
x_train.replace(np.nan, 0, inplace=True)
x_test.replace(np.nan, 0, inplace=True)

x_train.replace(np.inf, 0, inplace=True)
x_test.replace(np.inf, 0, inplace=True)

x_train.replace(-np.inf, 0, inplace=True)
x_test.replace(-np.inf, 0, inplace=True)
	
et = ExtraTreesRegressor(n_estimators=50, max_depth=None, max_features=None, max_leaf_nodes=None, 
	min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, min_impurity_split=1e-07, random_state=1)
et.fit(x_train, y_train)
preds1 = et.predict(x_test)
et_preds = np.expm1(preds1)

''' OTHER MODELS '''
# Credit: kaggle.com/aharless
xgb_df = pd.read_csv('xgbSub.csv')
xgb_preds = xgb_df['price_doc'].values

# Credit: kaggle.com/scirpus
gp_df = pd.read_csv('gpSub.csv')
gp_preds = gp_df['price_doc'].values

''' COMBINE AND WRITE OUTFILE '''
final_preds = (et_preds * 0.07) + (gp_preds * 0.07) + (xgb_preds * 0.85) + (lgbm_preds * 0.01)

outfile = pd.DataFrame()
outfile['id'] = test_ids
outfile['price_doc'] = final_preds

outfile.to_csv(".//subs//subXXX.csv", index=False)

# Public: 0.31020
# Private: 0.31571


