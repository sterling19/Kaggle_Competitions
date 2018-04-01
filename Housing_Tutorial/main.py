import pandas as pd
import numpy as np

import models as mdls

def run_ensemble():
    gbdt, dtest, test = mdls.build_ensemble()
    
    y_pred = gbdt.predict(dtest)
    y_pred = np.exp(y_pred)

    pred_df = pd.DataFrame(y_pred, index=test["Id"], columns=["SalePrice"])
    pred_df.to_csv('subXX.csv', header=True, index_label='Id')

if __name__ == '__main__':
    run_ensemble()
