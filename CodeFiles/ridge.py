import numpy as np
import pandas as pd

train_x = pd.read_csv("train_x.csv")
test_x = pd.read_csv("test_x.csv")
train_y = pd.read_csv("log1p_train_y.csv")

# helper method rmse of y_pred, y
def rmse(y_pred, y):
    return np.sqrt(((y-y_pred)**2).mean())


################################
######### Ridge Regression ########

from sklearn.linear_model import RidgeCV

regr = RidgeCV()

regr.fit(train_x, train_y)

print('ridge_train_Rmse: ', rmse(regr.predict(train_x).reshape(1458,), (train_y["Log1pSalePrice"])))


ridge_pred_train = np.expm1(regr.predict(train_x))
ridge_pred = np.expm1(regr.predict(test_x))
pred_df = pd.DataFrame(ridge_pred, index=range(1461,2920), columns=["SalePrice"])
pred_df.to_csv('ridge_pred.csv', header=True, index_label='Id')