''' Load packages and data '''
import os
import gc
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

train = pd.read_csv("train_1.csv")
test = pd.read_csv("key_1.csv")

''' Setup training data '''
train_flattened = pd.melt(train[list(train.columns[-49:])+['Page']], id_vars='Page', var_name='date', value_name='Visits')
train_flattened['date'] = train_flattened['date'].astype('datetime64[ns]')

del train
gc.collect()

''' Setup test data and label encoding '''
test['Date'] = test.Page.apply(lambda a: a[-10:])
test['Page'] = test.Page.apply(lambda a: a[:-11])
test['Date'] = test['Date'].astype('datetime64[ns]')
test = test.sort_values(['Page', 'Date'])

le = LabelEncoder()
train_flattened['Page'] = le.fit_transform(train_flattened.Page.values)
test['Page'] = le.transform(test.Page.values)

''' Group by medians and apply random multiplier '''
meds = train_flattened.groupby(['Page']).median().reset_index() # 1230 Nulls
meds.loc[meds.Visits.isnull(), 'Visits'] = 1000
st_devs = train_flattened.groupby(['Page']).std().reset_index()
st_devs.loc[st_devs.Visits.isnull(), 'Visits'] = 10

med_vals = np.array([[val] * 60 for val in meds.Visits.values]).flatten()
std_vals = np.array([[val] * 60 for val in st_devs.Visits.values]).flatten()
multipliers = np.array([list(range(60))] * len(meds)).flatten()

new_vals = [round(med_val + (std_val * multiplier * 0.0005), 2) for med_val, std_val, multiplier in zip(med_vals, std_vals, multipliers)]

del train_flattened, meds, st_devs
gc.collect()

''' Merge and write outfile '''
test['Visits'] = new_vals
test[['Id','Visits']].to_csv(".//subs//sub_22.csv", index=False)