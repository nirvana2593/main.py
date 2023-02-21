import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, Imputer


# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")
X = data.iloc[:, 0:79]
Y = data.iloc[:, -1]

#Take care of missing values-Nan
null_columns = []
for ele in X.columns:
    if (X[ele].isna().any().any()):
       null_columns.append(ele)
       
for idx in range(0, X.shape[0]):
    for ele in null_columns:
        X.iloc[idx, X.columns.get_loc(ele)]=0 
#X[ele].fillna(0, inplace = True)

#Checks if there are still NaNs
count = 0
for idx in range(0, X.shape[0]):
    for idy in range(0, X.shape[1]):
        if (pd.isnull(X.iloc[idx, idy])) == True:
            count += 1
            break
#No more null columns

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
        
        
#Encoding categorical columns
X = MultiColumnLabelEncoder(columns = ['MSZoning','Street','LotShape','LandContour','Utilities',
'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType',
'HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
'ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure',
'BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical',
'KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual',
'GarageCond','PavedDrive','SaleType']).fit_transform(X)
#Checks if there are still strings
for ele in X.columns:
    if type(X.at[3, ele]) == str:
        print (ele)
        

"""One hot Encode"""
onehot_encoder = OneHotEncoder(sparse=False)
X = onehot_encoder.fit_transform(X)


"""Feature Scaling"""
sc = StandardScaler()
X = sc.fit_transform(X)
#regr = DecisionTreeRegressor(random_state=0)
#model = regr.fit()
        
"""Training and Testing"""
regr = DecisionTreeRegressor(random_state=0)
regr.fit(X, Y)