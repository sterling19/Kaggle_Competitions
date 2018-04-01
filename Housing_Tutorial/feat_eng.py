import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

def engineer_features():
    train = pd.read_csv('.\\data\\train.csv')
    test = pd.read_csv('.\\data\\test.csv')
    y_train = np.log(train['SalePrice']+1)

    test.loc[666, "GarageQual"] = "TA"
    test.loc[666, "GarageCond"] = "TA"
    test.loc[666, "GarageFinish"] = "Unf"
    test.loc[666, "GarageYrBlt"] = "1980"

    ntrain = train.shape[0]
    ntest = test.shape[0]
    all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], 
                     test.loc[:,'MSSubClass':'SaleCondition']), ignore_index=True)

    # NULL REPLACEMENT
    all_data.loc[all_data.Alley.isnull(), 'Alley'] = 'NoAlley'
    all_data.loc[all_data.MasVnrType.isnull(), 'MasVnrType'] = 'None'
    all_data.loc[all_data.MasVnrType == 'None', 'MasVnrArea'] = 0
    all_data.loc[all_data.BsmtQual.isnull(), 'BsmtQual'] = 'NoBsmt'
    all_data.loc[all_data.BsmtCond.isnull(), 'BsmtCond'] = 'NoBsmt'
    all_data.loc[all_data.BsmtExposure.isnull(), 'BsmtExposure'] = 'NoBsmt'
    all_data.loc[all_data.BsmtFinType1.isnull(), 'BsmtFinType1'] = 'NoBsmt'
    all_data.loc[all_data.BsmtFinType2.isnull(), 'BsmtFinType2'] = 'NoBsmt'
    all_data.loc[all_data.BsmtFinType1=='NoBsmt', 'BsmtFinSF1'] = 0
    all_data.loc[all_data.BsmtFinType2=='NoBsmt', 'BsmtFinSF2'] = 0
    all_data.loc[all_data.BsmtFinSF1.isnull(), 'BsmtFinSF1'] = 0
    all_data.loc[all_data.BsmtQual=='NoBsmt', 'BsmtUnfSF'] = 0
    all_data.loc[all_data.BsmtUnfSF.isnull(), 'BsmtUnfSF'] = 0
    all_data.loc[all_data.BsmtQual=='NoBsmt', 'TotalBsmtSF'] = 0
    all_data.loc[all_data.FireplaceQu.isnull(), 'FireplaceQu'] = 'NoFireplace'
    all_data.loc[all_data.GarageType.isnull(), 'GarageType'] = 'NoGarage'
    all_data.loc[all_data.GarageFinish.isnull(), 'GarageFinish'] = 'NoGarage'
    all_data.loc[all_data.GarageQual.isnull(), 'GarageQual'] = 'NoGarage'
    all_data.loc[all_data.GarageCond.isnull(), 'GarageCond'] = 'NoGarage'
    all_data.loc[all_data.BsmtFullBath.isnull(), 'BsmtFullBath'] = 0
    all_data.loc[all_data.BsmtHalfBath.isnull(), 'BsmtHalfBath'] = 0
    all_data.loc[all_data.KitchenQual.isnull(), 'KitchenQual'] = 'TA'
    all_data.loc[all_data.MSZoning.isnull(), 'MSZoning'] = 'RL'
    all_data.loc[all_data.Utilities.isnull(), 'Utilities'] = 'AllPub'
    all_data.loc[all_data.Exterior1st.isnull(), 'Exterior1st'] = 'VinylSd'
    all_data.loc[all_data.Exterior2nd.isnull(), 'Exterior2nd'] = 'VinylSd'
    all_data.loc[all_data.Functional.isnull(), 'Functional'] = 'Typ'
    all_data.loc[all_data.SaleCondition.isnull(), 'SaleCondition'] = 'Normal'
    all_data.ix[2489, 'SaleType'] = 'Normal'
    all_data.loc[all_data.SaleCondition.isnull(), 'SaleType'] = 'WD'
    all_data.loc[all_data['PoolQC'].isnull(), 'PoolQC'] = 'NoPool'
    all_data.loc[all_data['Fence'].isnull(), 'Fence'] = 'NoFence'
    all_data.loc[all_data['MiscFeature'].isnull(), 'MiscFeature'] = 'None'
    all_data.loc[all_data['Electrical'].isnull(), 'Electrical'] = 'SBrkr'
    all_data.loc[all_data['GarageYrBlt'].isnull(), 'GarageYrBlt'] = 'NoGarage'            
    all_data.loc[all_data['GarageArea'].isnull(), 'GarageArea'] = 0
    all_data.loc[all_data['GarageCars'].isnull(), 'GarageCars'] = 0

    lot_frontage_by_neighborhood = all_data["LotFrontage"].groupby(all_data["Neighborhood"])
    for key, group in lot_frontage_by_neighborhood:
        idx = (all_data["Neighborhood"] == key) & (all_data["LotFrontage"].isnull())
        all_data.loc[idx, "LotFrontage"] = group.median()

    # LOG TRANSFORM SKEWED VARS
    numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
    numeric_feats = numeric_feats.drop('MoSold')
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    skewed_feats = skewed_feats.drop(['MSSubClass', 'BsmtHalfBath', 'KitchenAbvGr'])
    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    # STANDARD SCALING
    scaler = StandardScaler()
    scaler.fit(all_data[numeric_feats])
    scaled = scaler.transform(all_data[numeric_feats])
    for i, col in enumerate(numeric_feats):
        all_data[col] = scaled[:, i]

    # FEATURE ENGINEERING
    all_data["IsRegularLotShape"] = (all_data["LotShape"] == "Reg") * 1
    all_data["IsLandLevel"] = (all_data["LandContour"] == "Lvl") * 1
    all_data["IsLandSlopeGentle"] = (all_data["LandSlope"] == "Gtl") * 1
    all_data["IsElectricalSBrkr"] = (all_data["Electrical"] == "SBrkr") * 1
    all_data["IsGarageDetached"] = (all_data["GarageType"] == "Detchd") * 1
    all_data["IsPavedDrive"] = (all_data["PavedDrive"] == "Y") * 1
    all_data["HasShed"] = (all_data["MiscFeature"] == "Shed") * 1.  
    all_data["Remodeled"] = (all_data["YearRemodAdd"] != all_data["YearBuilt"]) * 1
    all_data["RecentRemodel"] = (all_data["YearRemodAdd"] == all_data["YrSold"]) * 1
    all_data["VeryNewHouse"] = (all_data["YearBuilt"] == all_data["YrSold"]) * 1
    all_data["Has2ndFloor"] = (all_data["2ndFlrSF"] == 0) * 1
    all_data["HasMasVnr"] = (all_data["MasVnrArea"] == 0) * 1
    all_data["HasWoodDeck"] = (all_data["WoodDeckSF"] == 0) * 1
    all_data["HasOpenPorch"] = (all_data["OpenPorchSF"] == 0) * 1
    all_data["HasEnclosedPorch"] = (all_data["EnclosedPorch"] == 0) * 1
    all_data["Has3SsnPorch"] = (all_data["3SsnPorch"] == 0) * 1
    all_data["HasScreenPorch"] = (all_data["ScreenPorch"] == 0) * 1
    
    all_data["HighSeason"] = all_data["MoSold"].replace( 
        {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})

    all_data["NewerDwelling"] = all_data["MSSubClass"].replace(
        {20: 1, 30: 0, 40: 0, 45: 0,50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,
         90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})   
    
    all_data.loc[all_data.Neighborhood == 'NridgHt', "Neighborhood_Good"] = 1
    all_data.loc[all_data.Neighborhood == 'Crawfor', "Neighborhood_Good"] = 1
    all_data.loc[all_data.Neighborhood == 'StoneBr', "Neighborhood_Good"] = 1
    all_data.loc[all_data.Neighborhood == 'Somerst', "Neighborhood_Good"] = 1
    all_data.loc[all_data.Neighborhood == 'NoRidge', "Neighborhood_Good"] = 1
    all_data["Neighborhood_Good"].fillna(0, inplace=True)

    all_data["SaleCondition_PriceDown"] = all_data.SaleCondition.replace(
        {'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})

    all_data["BoughtOffPlan"] = all_data.SaleCondition.replace(
        {"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1})
    
    all_data["BadHeating"] = all_data.HeatingQC.replace(
        {'Ex': 0, 'Gd': 0, 'TA': 0, 'Fa': 1, 'Po': 1})

    area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea' ]
    
    all_data["TotalArea"] = all_data[area_cols].sum(axis=1)

    all_data["TotalArea1st2nd"] = all_data["1stFlrSF"] + all_data["2ndFlrSF"]
    all_data["Age"] = 2010 - all_data["YearBuilt"]
    all_data["TimeSinceSold"] = 2010 - all_data["YrSold"]

    all_data["SeasonSold"] = all_data["MoSold"].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 
                                                  6:2, 7:2, 8:2, 9:3, 10:3, 11:3}).astype(int)
    
    all_data["YearsSinceRemodel"] = all_data["YrSold"] - all_data["YearRemodAdd"]
 
    all_data["SimplOverallQual"] = all_data.OverallQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})
    all_data["SimplOverallCond"] = all_data.OverallCond.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})
    all_data["SimplPoolQC"] = all_data.PoolQC.replace(
        {1 : 1, 2 : 1, 3 : 2, 4 : 2})
    all_data["SimplGarageCond"] = all_data.GarageCond.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    all_data["SimplGarageQual"] = all_data.GarageQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    all_data["SimplFireplaceQu"] = all_data.FireplaceQu.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    all_data["SimplFireplaceQu"] = all_data.FireplaceQu.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    all_data["SimplFunctional"] = all_data.Functional.replace(
        {1 : 1, 2 : 1, 3 : 2, 4 : 2, 5 : 3, 6 : 3, 7 : 3, 8 : 4})
    all_data["SimplKitchenQual"] = all_data.KitchenQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    all_data["SimplHeatingQC"] = all_data.HeatingQC.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    all_data["SimplBsmtFinType1"] = all_data.BsmtFinType1.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2})
    all_data["SimplBsmtFinType2"] = all_data.BsmtFinType2.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2})
    all_data["SimplBsmtCond"] = all_data.BsmtCond.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    all_data["SimplBsmtQual"] = all_data.BsmtQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    all_data["SimplExterCond"] = all_data.ExterCond.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    all_data["SimplExterQual"] = all_data.ExterQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

    neighborhood_map = {
        "MeadowV" : 0,  #  88000
        "IDOTRR" : 1,   # 103000
        "BrDale" : 1,   # 106000
        "OldTown" : 1,  # 119000
        "Edwards" : 1,  # 119500
        "BrkSide" : 1,  # 124300
        "Sawyer" : 1,   # 135000
        "Blueste" : 1,  # 137500
        "SWISU" : 2,    # 139500
        "NAmes" : 2,    # 140000
        "NPkVill" : 2,  # 146000
        "Mitchel" : 2,  # 153500
        "SawyerW" : 2,  # 179900
        "Gilbert" : 2,  # 181000
        "NWAmes" : 2,   # 182900
        "Blmngtn" : 2,  # 191000
        "CollgCr" : 2,  # 197200
        "ClearCr" : 3,  # 200250
        "Crawfor" : 3,  # 200624
        "Veenker" : 3,  # 218000
        "Somerst" : 3,  # 225500
        "Timber" : 3,   # 228475
        "StoneBr" : 4,  # 278000
        "NoRidge" : 4,  # 290000
        "NridgHt" : 4,  # 315000
    }

    all_data["NeighborhoodBin"] = all_data["Neighborhood"].map(neighborhood_map)
    
    # PREPARING FOR SKLEARN
    all_data = pd.get_dummies(all_data, sparse=False)
    drop_cols = ["Exterior1st_ImStucc", "Exterior1st_Stone", "Exterior2nd_Other", 
    "HouseStyle_2.5Fin", "RoofMatl_Membran", "RoofMatl_Metal", "RoofMatl_Roll",
    "Condition2_RRAe", "Condition2_RRAn", "Condition2_RRNn", "Heating_Floor", 
    "Heating_OthW", "Electrical_Mix", "MiscFeature_TenC", "GarageQual_Ex", 
    "PoolQC_Fa",  "Condition2_PosN", "MSZoning_C (all)"]
    all_data.drop(drop_cols, axis=1, inplace=True)

    x_train = np.array(all_data[:train.shape[0]])
    x_test = np.array(all_data[train.shape[0]:])
    
    return(x_train, x_test, y_train, ntrain, ntest, test)
