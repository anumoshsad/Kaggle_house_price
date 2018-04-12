
# coding: utf-8

# In[4]:

# load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.stats import rv_continuous
from sklearn import neural_network
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# load Data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# helper method rmse of y_pred, y
def rmse(y_pred, y):
    return np.sqrt(((y-y_pred)**2).mean())


# Removing the Outliers and joining train + test
idx = train[train["GrLivArea"] > 4500].index
train.drop(idx, inplace=True)

train_without_label = train.ix[:,train.columns!='SalePrice']

All_DF = pd.concat([train_without_label, test])
All_DF = All_DF.set_index("Id")

# encoding ordinal categorical values to integers
from sklearn.preprocessing import LabelEncoder
lenc = LabelEncoder()
def encode_label(df, encoded_df, column, fill_na=None):
    encoded_df[column] = df[column]
    if fill_na is not None:
        encoded_df[column].fillna(fill_na, inplace=True)
    lenc.fit(encoded_df[column].unique())
    encoded_df[column] = lenc.transform(encoded_df[column])
    return encoded_df

# this method is for using one hot encoding
def onehot_encode(df, ans_df, col_name, na_filling = None, delete = True):
    ans_df[col_name] = df[col_name]
    if na_filling is not None:
        ans_df[col_name].fillna(na_filling, inplace=True)

    dummies = pd.get_dummies(ans_df[col_name], prefix="_"+col_name)
    
    ans_df = pd.merge(ans_df,dummies, left_index=True, right_index=True, how='outer')
    if delete: ans_df = ans_df.drop([col_name], axis=1)
    return ans_df

# make new data frame with all the numerical features, converted numerical festures 
def transformation(df):
    ans_df = pd.DataFrame(index = df.index)
    
    ans_df["LotFrontage"] = df["LotFrontage"]
    # filling NAs with median of the nbd
    lot_frontage_by_nbd = df["LotFrontage"].groupby(df["Neighborhood"])
    for nbd, grp in lot_frontage_by_nbd:
        idx = (df["Neighborhood"] == nbd) & (df["LotFrontage"].isnull())
        ans_df.loc[idx, "LotFrontage"] = grp.median()
    #lotshape
    lotShape_map = {"IR3": 1, "IR2": 2, "IR3": 2, "Reg": 3}                          
    ans_df["LotShape"] = df.LotShape.replace(lotShape_map)
    #Alley
    alley_map = {'Grvl':3, 'Pave':2, None:1}
    ans_df["Alley"] = df.Alley.replace(alley_map)
    
    # bathrooms, kitchen related features
    ans_df["BsmtFullBath"] = df["BsmtFullBath"]
    ans_df["BsmtFullBath"].fillna(0, inplace=True)

    ans_df["BsmtHalfBath"] = df["BsmtHalfBath"]
    ans_df["BsmtHalfBath"].fillna(0, inplace=True)

    ans_df["FullBath"] = df["FullBath"] 
    ans_df["HalfBath"] = df["HalfBath"] 
    ans_df["BedroomAbvGr"] = df["BedroomAbvGr"] 
    ans_df["KitchenAbvGr"] = df["KitchenAbvGr"] 
    ans_df["TotRmsAbvGrd"] = df["TotRmsAbvGrd"] 
    ans_df["Fireplaces"] = df["Fireplaces"] 

    ans_df["Street"] = df["Street"]
    ans_df["Condition1"] = df["Condition1"] 
    ans_df["Condition2"] = df["Condition2"]
    ans_df["RoofMatl"] = df["RoofMatl"] 
    ans_df["Heating"] = df["Heating"]
    
    ans_df["GarageYrBlt"] = df["GarageYrBlt"]
    
    
    
    # Area related features
    ans_df["LotArea"] = df["LotArea"]
    
    ans_df["MasVnrArea"] = df["MasVnrArea"]
    ans_df["MasVnrArea"].fillna(0., inplace=True)
   
    ans_df["BsmtFinSF1"] = df["BsmtFinSF1"]
    ans_df["BsmtFinSF1"].fillna(0, inplace=True)

    ans_df["BsmtFinSF2"] = df["BsmtFinSF2"]
    ans_df["BsmtFinSF2"].fillna(0, inplace=True)

    ans_df["BsmtUnfSF"] = df["BsmtUnfSF"]
    ans_df["BsmtUnfSF"].fillna(0, inplace=True)

    ans_df["TotalBsmtSF"] = df["TotalBsmtSF"]
    ans_df["TotalBsmtSF"].fillna(0, inplace=True)

    ans_df["1stFlrSF"] = df["1stFlrSF"]
    ans_df["2ndFlrSF"] = df["2ndFlrSF"]
    ans_df["GrLivArea"] = df["GrLivArea"]
    
    ans_df["GarageArea"] = df["GarageArea"]
    ans_df["GarageArea"].fillna(0, inplace=True)

    ans_df["WoodDeckSF"] = df["WoodDeckSF"]
    ans_df["OpenPorchSF"] = df["OpenPorchSF"]
    ans_df["EnclosedPorch"] = df["EnclosedPorch"]
    ans_df["3SsnPorch"] = df["3SsnPorch"]
    ans_df["ScreenPorch"] = df["ScreenPorch"]
    
    ans_df["PoolArea"] = df["PoolArea"]
    ans_df["PoolArea"].fillna(0, inplace=True)
    
    # quality features
    ans_df["CentralAir"] = (df["CentralAir"] == "Y") * 1.0
   
    ans_df["OverallQual"] = df["OverallQual"]
    ans_df["OverallCond"] = df["OverallCond"]
    
    # Change Quality features to numerical values
    # higher number, better quality
    quality = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    ans_df["ExterQual"] = df["ExterQual"].map(quality).astype(int)
    ans_df["ExterCond"] = df["ExterCond"].map(quality).astype(int)
    ans_df["BsmtQual"] = df["BsmtQual"].map(quality).astype(int)
    ans_df["BsmtCond"] = df["BsmtCond"].map(quality).astype(int)
    ans_df["HeatingQC"] = df["HeatingQC"].map(quality).astype(int)
    ans_df["KitchenQual"] = df["KitchenQual"].map(quality).astype(int)
    ans_df["FireplaceQu"] = df["FireplaceQu"].map(quality).astype(int)
    ans_df["GarageQual"] = df["GarageQual"].map(quality).astype(int)
    ans_df["GarageCond"] = df["GarageCond"].map(quality).astype(int)
    ans_df["PoolQC"] = df["PoolQC"].map(quality).astype(int)
    
    exposure = {None: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}                          
    ans_df["BsmtExposure"] = df["BsmtExposure"].map(exposure).astype(int)

    BsmtFinType = {None: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
    ans_df["BsmtFinType1"] = df["BsmtFinType1"].map(BsmtFinType).astype(int)
    ans_df["BsmtFinType2"] = df["BsmtFinType2"].map(BsmtFinType).astype(int)

    functionality = {None: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}                           
    ans_df["Functional"] = df["Functional"].map(functionality).astype(int)

    finished = {None: 0, "Unf": 1, "RFn": 2, "Fin": 3}
    ans_df["GarageFinish"] = df["GarageFinish"].map(finished).astype(int)

    fence = {None: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}                          
    ans_df["Fence"] = df["Fence"].map(fence).astype(int)

    ans_df["YearBuilt"] = df["YearBuilt"]
    ans_df["YearRemodAdd"] = df["YearRemodAdd"]

    ans_df["GarageYrBlt"] = df["GarageYrBlt"]
    ans_df["GarageYrBlt"].fillna(0.0, inplace=True)

    ans_df["MoSold"] = df["MoSold"]
    ans_df["YrSold"] = df["YrSold"]
    
    ans_df["LowQualFinSF"] = df["LowQualFinSF"]
    ans_df["MiscVal"] = df["MiscVal"]
    
    #garage
    ans_df["GarageCars"] = df["GarageCars"]
    ans_df["GarageCars"].fillna(0, inplace=True)
    
    #land
    landContour_map = {'Lvl':3, 'Bnk':2, 'HLS':2, 'Low':1}                          
    ans_df["LandContour"] = df.LandContour.replace(landContour_map)

    #land Slope
    landSlope_map = {'Gtl':3, 'Mod':2, 'Sev':1}                          
    ans_df["LandSlope"] = df.LandSlope.replace(landSlope_map)
    
    #electrical
    electrical_map = {'SBrkr':3, 'FuseA':2,'FuseF':1,'FuseP':1,'Mix':0, None:3}
    ans_df["Electrical"] = df.Electrical.replace(electrical_map)
    
    #garage type
    garage_type_map = {'2Types':6 ,'Attchd':5,'Basment':4, 'BuiltIn':3, 'CarPort': 2, 'Detchd':1, None:0}
    ans_df["GarageType"] = df.GarageType.replace(garage_type_map)
    
    #PavedDrive type
    paved_drive_map = {'Y':2 ,'P':1,'N':0}
    ans_df["PavedDrive"] = df.PavedDrive.replace(paved_drive_map)
    
    #PavedDrive type
    paved_drive_map = {'Y':2 ,'P':1,'N':0}
    ans_df["PavedDrive"] = df.PavedDrive.replace(paved_drive_map)

    # misc features
    ans_df["MiscFeature"] = df["MiscFeature"]
    ans_df["MiscFeature"].fillna("NA", inplace=True)
    
    
    
    # now encode categorical features into numericals
    ans_df = encode_label(df, ans_df, "MSSubClass")
    ans_df = encode_label(df, ans_df, "MSZoning","RL")
    ans_df = encode_label(df, ans_df, "LotConfig")
    ans_df = encode_label(df, ans_df, "Neighborhood")
    ans_df = encode_label(df, ans_df, "Condition1")
    ans_df = encode_label(df, ans_df, "BldgType")
    ans_df = encode_label(df, ans_df, "HouseStyle")
    ans_df = encode_label(df, ans_df, "RoofStyle")
    ans_df = encode_label(df, ans_df, "Exterior1st", "Other")
    ans_df = encode_label(df, ans_df, "Exterior2nd", "Other")
    ans_df = encode_label(df, ans_df, "MasVnrType", "None")
    ans_df = encode_label(df, ans_df, "Foundation")
    ans_df = encode_label(df, ans_df, "SaleType", "Oth")
    ans_df = encode_label(df, ans_df, "SaleCondition")
    
    
    # Also add some more features
    # info about lot shape
    ans_df["IsRegularLotShape"] = (df["LotShape"] == "Reg") * 1

    # make just two groups of level land or not
    ans_df["IsLandLevel"] = (df["LandContour"] == "Lvl") * 1

    # make two groups of land slope
    ans_df["IsSlopeGentle"] = (df["LandSlope"] == "Gtl") * 1

    # standard circuit breakers or not
    ans_df["IsElectricalSBrkr"] = (df["Electrical"] == "SBrkr") * 1

    # attached garage or not
    ans_df["IsGarageDetached"] = (df["GarageType"] == "Detchd") * 1

    # paved drive or not
    ans_df["IsPavedDrive"] = (df["PavedDrive"] == "Y") * 1

    # interesting "misc. feature" is the presence of a shed.
    ans_df["HasShed"] = (df["MiscFeature"] == "Shed") * 1.  
    ans_df["HasTenC"] = (df["MiscFeature"] == "Tenc") * 1.  
    ans_df["HasElev"] = (df["MiscFeature"] == "Elev") * 1.  

    
    # If remodeling took place at some point.
    ans_df["Remodeled"] = (ans_df["YearRemodAdd"] != ans_df["YearBuilt"]) * 1
    
    # remodeling in the year the house was sold?
    ans_df["RecentRemodel"] = (ans_df["YearRemodAdd"] == ans_df["YrSold"]) * 1
    
    # house sold in the year it was built?
    ans_df["VeryNew"] = (ans_df["YearBuilt"] == ans_df["YrSold"]) * 1

    ans_df["Has2ndFloor"] = (ans_df["2ndFlrSF"] == 0) * 1
    ans_df["HasMasVnr"] = (ans_df["MasVnrArea"] == 0) * 1
    ans_df["HasWoodDeck"] = (ans_df["WoodDeckSF"] == 0) * 1
    ans_df["HasOpenPorch"] = (ans_df["OpenPorchSF"] == 0) * 1
    ans_df["HasEnclosedPorch"] = (ans_df["EnclosedPorch"] == 0) * 1
    ans_df["Has3SsnPorch"] = (ans_df["3SsnPorch"] == 0) * 1
    ans_df["HasScreenPorch"] = (ans_df["ScreenPorch"] == 0) * 1

    # Months with the largest number of deals may be significant.
    season = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}                          
    ans_df["HighSeason"] = df["MoSold"].replace( season)
    
    # make distinction newer dwelling or not                          
    newer = {20: 1, 30: 0, 40: 0, 45: 0,50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,
         90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0}
    ans_df["NewerDwelling"] = df["MSSubClass"].replace(newer)   
    
    ans_df.loc[df.Neighborhood == 'NridgHt', "Neighborhood_Good"] = 1
    ans_df.loc[df.Neighborhood == 'Crawfor', "Neighborhood_Good"] = 1
    ans_df.loc[df.Neighborhood == 'StoneBr', "Neighborhood_Good"] = 1
    ans_df.loc[df.Neighborhood == 'Somerst', "Neighborhood_Good"] = 1
    ans_df.loc[df.Neighborhood == 'NoRidge', "Neighborhood_Good"] = 1
    ans_df.loc[df.Neighborhood == 'Timber', "Neighborhood_Good"] = 1
    ans_df.loc[df.Neighborhood == 'Veenker', "Neighborhood_Good"] = 1
    ans_df["Neighborhood_Good"].fillna(0, inplace=True)

    ans_df["SaleCondition_PriceDown"] = df.SaleCondition.replace(
        {'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})

    # House completed before sale or not
    ans_df["BuyOffPlan"] = df.SaleCondition.replace(
        {"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1})
    
    ans_df["BadHeating"] = df.HeatingQC.replace(
        {'Ex': 0, 'Gd': 0, 'TA': 0, 'Fa': 1, 'Po': 1})

    ans_df["Age"] = 2010 - ans_df["YearBuilt"]
    ans_df["TimeSinceSold"] = 2010 - ans_df["YrSold"]

    ans_df["SeasonSold"] = ans_df["MoSold"].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 
                                                  6:2, 7:2, 8:2, 9:3, 10:3, 11:3}).astype(int)
    
    ans_df["YearsSinceRemodel"] = ans_df["YrSold"] - ans_df["YearRemodAdd"]
    
                              
    # now adding some simplified quality features
    ans_df["SimplHeatingQC"] = ans_df.HeatingQC.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    ans_df["SimplBsmtFinType1"] = ans_df.BsmtFinType1.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2})
    ans_df["SimplBsmtFinType2"] = ans_df.BsmtFinType2.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2})
    ans_df["SimplBsmtCond"] = ans_df.BsmtCond.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    ans_df["SimplBsmtQual"] = ans_df.BsmtQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    ans_df["SimplExterCond"] = ans_df.ExterCond.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    ans_df["SimplExterQual"] = ans_df.ExterQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    ans_df["SimplOverallQual"] = ans_df.OverallQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})
    ans_df["SimplOverallCond"] = ans_df.OverallCond.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})
    ans_df["SimplPoolQC"] = ans_df.PoolQC.replace(
        {1 : 1, 2 : 1, 3 : 2, 4 : 2})
    ans_df["SimplGarageCond"] = ans_df.GarageCond.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    ans_df["SimplGarageQual"] = ans_df.GarageQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    ans_df["SimplFireplaceQu"] = ans_df.FireplaceQu.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    ans_df["SimplFireplaceQu"] = ans_df.FireplaceQu.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    ans_df["SimplFunctional"] = ans_df.Functional.replace(
        {1 : 1, 2 : 1, 3 : 2, 4 : 2, 5 : 3, 6 : 3, 7 : 3, 8 : 4})
    ans_df["SimplKitchenQual"] = ans_df.KitchenQual.replace(
        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    
    # make neighborhood bin according to their mean/median price computed by: 
    # train_df["SalePrice"].groupby(train_df["Neighborhood"]).median().sort_values()
    neighborhood_map = { 
    					# median price			
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

    ans_df["NeighborhoodBin"] = df["Neighborhood"].map(neighborhood_map)
    return ans_df
    
df_transformed = transformation(All_DF)

####################################################
# log transformation and normalization
df = df_transformed
numeric_features = ['LotFrontage',
 'BsmtFullBath',
 'BsmtHalfBath',
 'FullBath',
 'HalfBath',
 'BedroomAbvGr',
 'KitchenAbvGr',
 'TotRmsAbvGrd',
 'Fireplaces',
 'LotArea',
 'MasVnrArea',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'BsmtUnfSF',
 'TotalBsmtSF',
 '1stFlrSF',
 '2ndFlrSF',
 'GrLivArea',
 'GarageArea',
 'WoodDeckSF',
 'OpenPorchSF',
 'EnclosedPorch',
 '3SsnPorch',
 'ScreenPorch',
 'PoolArea',
 'OverallQual',
 'OverallCond',
 'ExterCond',
 'BsmtQual',
 'BsmtCond',
 'HeatingQC',
 'KitchenQual',
 'FireplaceQu',
 'GarageQual',
 'GarageCond',
 'PoolQC',
 'BsmtExposure',
 'Functional',
 'GarageFinish',
 'Fence',
 'LowQualFinSF',
 'MiscVal',
 'GarageCars']

# taking log(1+feature) transformation if skewedness is large
skewness = df[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))
sk_idx = skewness[skewness > 0.75]
sk_idx = sk_idx.index

df[sk_idx] = np.log1p(df[sk_idx])

# standardizing each column   
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(df[numeric_features])

scaled_df = scale.transform(df[numeric_features])
for i, col in enumerate(numeric_features):
    df[col] = scaled_df[:, i]
    
 #############################################################
# now use one-hot-encoding on the categorical features
def onehot_encode_transformation(df):
    ans_df = df.copy()

    ans_df = onehot_encode(df, ans_df, "MSSubClass")
    ans_df = onehot_encode(df, ans_df, "MSZoning")
    ans_df = onehot_encode(df, ans_df, "LotConfig")
    ans_df = onehot_encode(df, ans_df, "Neighborhood")
    ans_df = onehot_encode(df, ans_df, "Condition1")
    ans_df = onehot_encode(df, ans_df, "BldgType")
    ans_df = onehot_encode(df, ans_df, "HouseStyle")
    ans_df = onehot_encode(df, ans_df, "RoofStyle")
    ans_df = onehot_encode(df, ans_df, "Exterior1st")
    ans_df = onehot_encode(df, ans_df, "Exterior2nd")
    ans_df = onehot_encode(df, ans_df, "Foundation")
    ans_df = onehot_encode(df, ans_df, "SaleType")
    ans_df = onehot_encode(df, ans_df, "SaleCondition")
    ans_df = onehot_encode(df, ans_df, "MasVnrType")

    ans_df = onehot_encode(df, ans_df, "LotShape")
    ans_df = onehot_encode(df, ans_df, "LandContour")
    ans_df = onehot_encode(df, ans_df, "LandSlope")
    ans_df = onehot_encode(df, ans_df, "Electrical")
    ans_df = onehot_encode(df, ans_df, "GarageType")
    ans_df = onehot_encode(df, ans_df, "PavedDrive")
    ans_df = onehot_encode(df, ans_df, "MiscFeature")
    
    # this are numerical variables but also add them as categorical
    ans_df = onehot_encode(df, ans_df, "ExterQual", "None")
    ans_df = onehot_encode(df, ans_df, "ExterCond", "None")
    ans_df = onehot_encode(df, ans_df, "BsmtQual", "None")
    ans_df = onehot_encode(df, ans_df, "BsmtCond", "None")
    ans_df = onehot_encode(df, ans_df, "HeatingQC", "None")
    ans_df = onehot_encode(df, ans_df, "KitchenQual", "TA")
    ans_df = onehot_encode(df, ans_df, "FireplaceQu", "None")
    ans_df = onehot_encode(df, ans_df, "GarageQual", "None")
    ans_df = onehot_encode(df, ans_df, "GarageCond", "None")
    ans_df = onehot_encode(df, ans_df, "PoolQC", "None")
    ans_df = onehot_encode(df, ans_df, "BsmtExposure", "None")
    ans_df = onehot_encode(df, ans_df, "BsmtFinType1", "None")
    ans_df = onehot_encode(df, ans_df, "BsmtFinType2", "None")
    ans_df = onehot_encode(df, ans_df, "Functional", "Typ")
    ans_df = onehot_encode(df, ans_df, "GarageFinish", "None")
    ans_df = onehot_encode(df, ans_df, "MoSold", None)
    # this features we can probably be ignored (but want to include anyway to see
    # if they make any positive difference).
    ans_df = onehot_encode(df, ans_df, "Street")
    ans_df = onehot_encode(df, ans_df, "Alley")
    ans_df = onehot_encode(df, ans_df, "Condition2")
    ans_df = onehot_encode(df, ans_df, "Condition1")
    ans_df = onehot_encode(df, ans_df, "RoofMatl")
    ans_df = onehot_encode(df, ans_df, "Heating")

    
    # Dividing the years between 1871 and 2010 in slices of 20 years.
    year_map = pd.concat(pd.Series("YearBin" + str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))

    yearbin_df = pd.DataFrame(index = df.index)
    yearbin_df["GarageYrBltBin"] = df.GarageYrBlt.map(year_map)
    yearbin_df["GarageYrBltBin"].fillna("NoGarage", inplace=True)

    yearbin_df["YearBuiltBin"] = df.YearBuilt.map(year_map)
    yearbin_df["YearRemodAddBin"] = df.YearRemodAdd.map(year_map)
    
    ans_df = onehot_encode(yearbin_df, ans_df, "GarageYrBltBin")
    ans_df = onehot_encode(yearbin_df, ans_df, "YearBuiltBin")
    ans_df = onehot_encode(yearbin_df, ans_df, "YearRemodAddBin")

    return ans_df

df_final = onehot_encode_transformation(df_transformed)


#Split data
train_x = df_final.iloc[:1458]
train_x = train_x.reset_index(drop=True)
test_x = df_final.iloc[1458:]
test_x = test_x.reset_index(drop=True)
train_y = pd.DataFrame(index = train.index)
train_y["Log1pSalePrice"] = np.log1p(train["SalePrice"])
train_y = train_y.reset_index(drop=True)


# In[5]:

#Helper functions


#Split data into n random sets
def makeNSets(train_x,train_y,n):
    originalTrain=train_x
    originalLabel=train_y
    foldsX=list()
    foldsY=list()
    for i in range(0, n):
        ind=np.random.rand(len(train_x)) < (1/((1.0*n)-i))
        newFold = pd.DataFrame(train_x[ind])
        newFold = newFold.reset_index(drop=True)
        train_x = pd.DataFrame(train_x[~ind])
        train_x = train_x.reset_index(drop=True)
        newFoldLabel = pd.DataFrame(train_y[ind])
        newFoldLabel = newFoldLabel.reset_index(drop=True)
        train_y = pd.DataFrame(train_y[~ind])
        train_y = train_y.reset_index(drop=True)
        foldsX.append(newFold)
        foldsY.append(newFoldLabel)
        
    combFoldsX=list()
    combFoldsY=list()
    for j in range(0, n):
        newFold=pd.DataFrame()
        newFoldLabel=pd.DataFrame()
        for k in range(0, n):
            if(j!=k):
                newFold=pd.concat([newFold,foldsX[k]],axis=0)
                newFoldLabel=pd.concat([newFoldLabel,foldsY[k]],axis=0)
        newFold=newFold.reset_index(drop=True)
        newFoldLabel=newFoldLabel.reset_index(drop=True)
        combFoldsX.append(newFold)
        combFoldsY.append(newFoldLabel)
        
    return foldsX, foldsY, combFoldsX, combFoldsY


# In[6]:

#####Stacking with XGB and Lasso

def simple_stacking(trainX,testX,trainY):
    #Define models
    lassoModel=linear_model.Lasso(alpha=0.0005)
    xgbModel = xgb.XGBRegressor(colsample_bytree=0.2,
                 gamma=0.01,
                 learning_rate=.005,
                 max_depth=6,
                 min_child_weight=1.5,
                 n_estimators=9000,                                                                  
                 reg_alpha=0.5,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)
    
    trainPredictions=pd.DataFrame()

    lassoMod=lassoModel.fit(trainX,trainY)
    lassoPre=pd.DataFrame(lassoMod.predict(trainX))
        
    xgbMod=xgbModel.fit(trainX, trainY)
    xgbPre=pd.DataFrame(xgbMod.predict(trainX))
    
    trainPredictions=pd.concat([lassoPre,xgbPre],axis=1)

    
    
    #Train second level model
    secLevelLassoModel=linear_model.Lasso(alpha=0.0005)
    secLevelLasso=secLevelLassoModel.fit(trainPredictions,trainY)
    
    #Compute second level test data
    testPredictions=pd.DataFrame()
    
    xgbModelT=xgbModel.fit(trainX,trainY)
    lassoModelT=lassoModel.fit(trainX,trainY)

    xgbTestPrediction=pd.DataFrame(xgbModelT.predict(testX))
    lassoTestPrediction=pd.DataFrame(lassoModelT.predict(testX))

    testPredictions=pd.concat([lassoTestPrediction,xgbTestPrediction],axis=1)
    
    #Compute second level test prediction
    testPrediction=pd.DataFrame(secLevelLasso.predict(testPredictions))
    testPrediction=testPrediction.rename(columns={0:'SalePrice'})
    
    return testPrediction


# In[7]:

#predict and write to csv
testPrediction=simple_stacking(train_x,test_x,train_y)
testPrediction=pd.DataFrame(np.expm1(testPrediction))
testIndex=test.pop('Id')
testPrediction=pd.concat([testIndex,testPrediction], axis=1)
testPrediction=testPrediction.rename(columns={0:'SalePrice'})
testPrediction.to_csv("SimpleStacking.csv", index=False)


# In[ ]:



