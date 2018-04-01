''' Load packages and data '''
import gc
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

train = pd.read_csv("train_1.csv")
test = pd.read_csv("key_1.csv")

''' Reshape data '''
train_flattened = pd.melt(train[list(train.columns[-49:])+['Page']], id_vars='Page', var_name='Date', value_name='Visits')
train_flattened['Date'] = train_flattened['Date'].astype('datetime64[ns]')
train_flattened['Weekend'] = ((train_flattened.Date.dt.dayofweek) // 5 == 1).astype(float)

del train
gc.collect()

''' Feature engineering '''
test['Date'] = test.Page.apply(lambda a: a[-10:])
test['Page'] = test.Page.apply(lambda a: a[:-11])
test['Date'] = test['Date'].astype('datetime64[ns]')
test['Weekend'] = ((test.Date.dt.dayofweek) // 5 == 1).astype(float)

''' Group by medians and merge '''
meds = train_flattened.groupby(['Page','Weekend']).median().reset_index()
test = test.merge(meds, how='left')

''' Fill NA's and write outfile '''
test.loc[test.Visits.isnull(), 'Visits'] = 1000
test[['Id','Visits']].to_csv(".//subs//sub_23.csv", index=False)
