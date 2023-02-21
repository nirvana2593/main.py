import pandas as pd
import warnings
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, skew
from scipy import stats
import cmath

warnings.filterwarnings("ignore")

train = pd.read_csv("boston_house/train.csv")
train


color = sns.color_palette()
sns.set_style('darkgrid')

fig, ax = plt.subplots()
# 绘制散点图
ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
# plt.show()

# 删除异常值点
train_drop = train.drop(
    train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

# 重新绘制图
fig, ax = plt.subplots()
ax.scatter(train_drop['GrLivArea'], train_drop['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
# plt.show()

var = 'OverallQual'
data = pd.concat([train_drop['SalePrice'], train_drop[var]], axis=1)
# 画出箱线图
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)


k = 10
corrmat = train_drop.corr()  # 获得相关性矩阵
# 获得相关性最高的 K 个特征
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# 获得相关性最高的 K 个特征组成的子数据集
cm = np.corrcoef(train_drop[cols].values.T)
# 绘制热图
# plt.show()

train_drop1 = train_drop.drop("Id", axis=1)

# 绘制散点图
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea',
        'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# plt.show()


train_drop1['SalePrice'].describe()
"""
count      1458.000000
mean     180932.919067
std       79495.055285
min       34900.000000
25%      129925.000000
50%      163000.000000
75%      214000.000000
max      755000.000000
Name: SalePrice, dtype: float64
"""


sns.distplot(train_drop1['SalePrice'], fit=norm)

# 获得均值和方差
(mu, sigma) = norm.fit(train_drop1['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# 画出数据分布图
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
           loc='best')
plt.ylabel('Frequency')
# 设置标题
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(train_drop1['SalePrice'], plot=plt)
#plt.show()


# 平滑数据
train_drop1["SalePrice"] = np.log1p(train_drop1["SalePrice"])

# 重新画出数据分布图
sns.distplot(train_drop1['SalePrice'], fit=norm)

# 重新计算平滑后的均值和方差
(mu, sigma) = norm.fit(train_drop1['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
           loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# 画出 Q-Q 图

fig = plt.figure()
res = stats.probplot(train_drop1['SalePrice'], plot=plt)
#plt.show()


# 查看数据集中的缺失值
train_drop1.isnull().sum().sort_values(ascending=False)[:20]  # 取前 20 个数据
"""
PoolQC          1452
MiscFeature     1404
Alley           1367
Fence           1177
FireplaceQu      690
LotFrontage      259
GarageYrBlt       81
GarageCond        81
GarageType        81
GarageFinish      81
GarageQual        81
BsmtExposure      38
BsmtFinType2      38
BsmtCond          37
BsmtQual          37
BsmtFinType1      37
MasVnrArea         8
MasVnrType         8
Electrical         1
MSSubClass         0
dtype: int64
"""

# 计算缺失率
train_na = (train_drop1.isnull().sum() / len(train)) * 100
train_na = train_na.drop(
    train_na[train_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio': train_na})
missing_data.head(20)
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation=90)
sns.barplot(x=train_na.index, y=train_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

feature = ['PoolQC', 'MiscFeature', 'Alley', 'Fence',
           'FireplaceQu', 'GarageType', 'GarageFinish',
           'GarageQual', 'GarageCond', 'BsmtQual',
           'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
           'BsmtFinType2', 'MasVnrType', 'MSSubClass']
for col in feature:
    train_drop1[col] = train_drop1[col].fillna('None')

train_drop1["LotFrontage"] = train_drop1.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

feature = []
train_drop1['MSZoning'] = train_drop1['MSZoning'].fillna(
    train_drop1['MSZoning'].mode()[0])

train_drop2 = train_drop1.drop(['Utilities'], axis=1)

data_y = train_drop2['SalePrice']
data_X = train_drop2.drop(['SalePrice'], axis=1)

data_X_oh = pd.get_dummies(data_X)
print(data_X_oh.shape)

data_y_v = data_y.values  # 转换为 NumPy 数组
data_X_v = data_X_oh.values
length = int(len(data_y) * 0.7)

# 划分数据集
train_y = data_y_v[:length]
train_X = data_X_v[:length]
test_y = data_y_v[length:]
test_X = data_X_v[length:]

model = Lasso()
model.fit(train_X, train_y)

# 使用训练好的模型进行预测。并使用均方差来衡量预测结果的好坏。

y_pred = model.predict(test_X)
a = mean_squared_error(test_y, y_pred)
print('rmse:', cmath.sqrt(a))

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit


xgb_model = XGBRegressor(nthread=7)
cv_split = ShuffleSplit(n_splits=6, train_size=0.7, test_size=0.2)
grid_params = dict(
    max_depth = [4, 5, 6, 7],
    learning_rate = np.linspace(0.03, 0.3, 10),
    n_estimators = [100, 200]
)
grid_model = GridSearchCV(xgb_model, grid_params, cv=cv_split, scoring='neg_mean_squared_error')
grid_model.fit(train_X, train_y)

GridSearchCV(cv=ShuffleSplit(n_splits=6, random_state=None, test_size=0.2, train_size=0.7),
             estimator=XGBRegressor(base_score=None, booster=None,
                                    colsample_bylevel=None,
                                    colsample_bynode=None,
                                    colsample_bytree=None,
                                    enable_categorical=False, gamma=None,
                                    gpu_id=None, importance_type=None,
                                    interaction_constraints=None,
                                    learning_rate=None, max_delta_step=None,
                                    n_estimators=100, n_jobs=None, nthread=7,
                                    num_parallel_tree=None, predictor=None,
                                    random_state=None, reg_alpha=None,
                                    reg_lambda=None, scale_pos_weight=None,
                                    subsample=None, tree_method=None,
                                    validate_parameters=None, verbosity=None),
             param_grid={'learning_rate': array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3 ]),
                         'max_depth': [4, 5, 6, 7],
                         'n_estimators': [100, 200]},
             scoring='neg_mean_squared_error')
print(grid_model.best_params_)
print('rmse:', (-grid_model.best_score_) ** 0.5)
"""
{'learning_rate': 0.12000000000000001, 'max_depth': 4, 'n_estimators': 200}
rmse: 0.014665279597622335
"""