import numpy as np
import pandas as pd

train_x = pd.read_csv("train_x.csv")
test_x = pd.read_csv("test_x.csv")
train_y = pd.read_csv("log1p_train_y.csv")

# helper method rmse of y_pred, y
def rmse(y_pred, y):
    return np.sqrt(((y-y_pred)**2).mean())

###############################
#####   LASSO  ################
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': [.0005]} # found by gridsearch

model = Lasso()
regr_lasso = GridSearchCV(model, param_grid=param_grid, cv = 10, scoring='neg_mean_squared_error')
regr_lasso.fit(train_x, train_y)

# print("Best parameters set found on development set:")
# print()
# print(regr_lasso.best_params_)
# print()
# print("Grid scores on development set:")
# print()
# means = regr_lasso.cv_results_['mean_test_score']
# stds = regr_lasso.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, regr_lasso.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))
    
# print()
print('lasso_Rmse: ', rmse(regr_lasso.predict(train_x), (train_y["Log1pSalePrice"])))

lasso_pred_train = np.expm1(regr_lasso.predict(train_x))
lasso_pred = np.expm1(regr_lasso.predict(test_x))
pred_df = pd.DataFrame(lasso_pred, index=range(1461,2920), columns=["SalePrice"])
pred_df.to_csv('lasso_pred.csv', header=True, index_label='Id')


###############################
#####   XGBoost  ################

import xgboost as xgb

# manually tuned parameter
param_grid = {   'colsample_bytree': [0.2],
                 'gamma': [0.01],
                 'learning_rate': [.005],
                 'max_depth': [6],
                 'min_child_weight': [1.5],
                 'n_estimators': [9000],                                                                  
                 'reg_alpha': [0.5],
                 'reg_lambda': [0.6],
                 'subsample': [0.2],
                 'seed': [42],
                 'silent': [1]
             }
model = xgb.XGBRegressor()
regr = GridSearchCV(model, param_grid=param_grid, cv = 3, verbose = 1)
regr.fit(train_x, train_y)

# print("Best parameters set found on development set:")
# print()
# print(regr.best_params_)
# print()
# print("Grid scores on development set:")
# print()
# means = regr.cv_results_['mean_test_score']
# stds = regr.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, regr.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))

print('xgb_Rmse: ', rmse((regr.predict(train_x)), (train_y["Log1pSalePrice"])))

xgb_pred_train = np.expm1(regr.predict(train_x))
xgb_pred = np.expm1(regr.predict(test_x))
pred_df = pd.DataFrame(xgb_pred, index=range(1461,2920), columns=["SalePrice"])
pred_df.to_csv('xgb_pred.csv', header=True, index_label='Id')


# Weighted average of models
y_pred = .45*xgb_pred+.55*lasso_pred
pred_df = pd.DataFrame(y_pred+1, index=range(1461,2920), columns=["SalePrice"])
pred_df.to_csv('0.45*xgb_pred+0.55*lasso.csv', header=True, index_label='Id')

