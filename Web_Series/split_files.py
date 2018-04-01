import gc
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

train = pd.read_csv("train_1.csv")
test = pd.read_csv("key_1.csv")

test['date'] = test.Page.apply(lambda a: a[-10:])
test['date'] = test['date'].astype('datetime64[ns]')
test['Page'] = test.Page.apply(lambda a: a[:-11])

langs = ['en', 'es', 'de', 'fr', 'ja', 'ru', 'zh']

train['lang'] = train.Page.apply(lambda x: x.split(".wikipedia.org")[0][-2:])
test['lang'] = test.Page.apply(lambda x: x.split(".wikipedia.org")[0][-2:])
train['lang'] = train.lang.apply(lambda x: 'na' if x not in langs else x)
test['lang'] = test.lang.apply(lambda x: 'na' if x not in langs else x)

le = LabelEncoder()
train['Page'] = le.fit_transform(train.Page.values)
test['Page'] = le.transform(test.Page.values)

en_train = train.ix[train.lang=='en', :]
en_test = test.ix[test.lang=='en', :]
en_train.to_csv("en_train.csv", index=False)
en_test.to_csv("en_test.csv", index=False)
del en_train, en_test
gc.collect()

es_train = train.ix[train.lang=='es', :]
es_test = test.ix[test.lang=='es', :]
es_train.to_csv("es_train.csv", index=False)
es_test.to_csv("es_test.csv", index=False)
del es_train, es_test
gc.collect()

de_train = train.ix[train.lang=='de', :]
de_test = test.ix[test.lang=='de', :]
de_train.to_csv("de_train.csv", index=False)
de_test.to_csv("de_test.csv", index=False)
del de_train, de_test
gc.collect()

fr_train = train.ix[train.lang=='fr', :]
fr_test = test.ix[test.lang=='fr', :]
fr_train.to_csv("fr_train.csv", index=False)
fr_test.to_csv("fr_test.csv", index=False)
del fr_train, fr_test
gc.collect()

ja_train = train.ix[train.lang=='ja', :]
ja_test = test.ix[test.lang=='ja', :]
ja_train.to_csv("ja_train.csv", index=False)
ja_test.to_csv("ja_test.csv", index=False)
del ja_train, ja_test
gc.collect()

na_train = train.ix[train.lang=='na', :]
na_test = test.ix[test.lang=='na', :]
na_train.to_csv("na_train.csv", index=False)
na_test.to_csv("na_test.csv", index=False)
del na_train, na_test
gc.collect()

ru_train = train.ix[train.lang=='ru', :]
ru_test = test.ix[test.lang=='ru', :]
ru_train.to_csv("ru_train.csv", index=False)
ru_test.to_csv("ru_test.csv", index=False)
del ru_train, ru_test
gc.collect()

zh_train = train.ix[train.lang=='zh', :]
zh_test = test.ix[test.lang=='zh', :]
zh_train.to_csv("zh_train.csv", index=False)
zh_test.to_csv("zh_test.csv", index=False)
del zh_train, zh_test
gc.collect()