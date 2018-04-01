''' Load packages and data '''
import os
import gc
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

def create_preds(language):

    TRAIN_FILE = language + "_train.csv"
    TEST_FILE = language + "_test.csv"
    PREDS_FILE = language + "_preds.csv"

    TRAIN_PATH = os.path.join(os.getcwd() + "\\train_test\\" + TRAIN_FILE)
    TEST_PATH = os.path.join(os.getcwd() + "\\train_test\\" + TEST_FILE)
    PREDS_PATH = os.path.join(os.getcwd() + "\\preds\\" + PREDS_FILE)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train_flattened = pd.melt(train[list(train.columns[-61:-1])+['Page']], id_vars='Page', var_name='date', value_name='Visits')
    grouped = train_flattened.groupby(['Page'])['Visits']
    rolling_meds = pd.rolling_median(grouped, window=7)

    test['Visits'] = rolling_meds.values
    test.loc[test.Visits.isnull(), 'Visits'] = 0.0
	
    test[['Id','Visits']].to_csv(PREDS_PATH, index=False)

    del train, test, train_flattened, grouped, rolling_meds
    gc.collect()

def create_submission():
    en = pd.read_csv(".//preds//en_preds.csv")
    es = pd.read_csv(".//preds//es_preds.csv")
    de = pd.read_csv(".//preds//de_preds.csv")
    fr = pd.read_csv(".//preds//fr_preds.csv")
    ja = pd.read_csv(".//preds//ja_preds.csv")
    na = pd.read_csv(".//preds//na_preds.csv")
    ru = pd.read_csv(".//preds//ru_preds.csv")
    zh = pd.read_csv(".//preds//zh_preds.csv")
	
	sub_df = pd.concat((en, es, de, fr, ja, na, ru, zh), axis=0)
	sub_df.to_csv(".//subs//sub_17.csv", index=False)
	
if __name__ == '__main__':

    for lang in ['en', 'es', 'de', 'fr', 'ja', 'na', 'ru', 'zh']:
        create_preds(lang)

    create_submission()