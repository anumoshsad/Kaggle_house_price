import numpy as np
import pandas as pd

train_x = pd.read_csv("train_x.csv")
test_x = pd.read_csv("test_x.csv")
train_y = pd.read_csv("log1p_train_y.csv")

# helper method rmse of y_pred, y
def rmse(y_pred, y):
    return np.sqrt(((y-y_pred)**2).mean())


################################
######### Random Forest ########

from sklearn.ensemble import RandomForestRegressor

rng = np.random.RandomState(0)
regr = RandomForestRegressor(random_state=0, n_estimators=50)
regr.fit(train_x, train_y)

print('rf_train_Rmse: ', rmse((regr.predict(train_x)), (train_y["Log1pSalePrice"])))


rf_pred_train = np.expm1(regr.predict(train_x))
rf_pred = np.expm1(regr.predict(test_x))
pred_df = pd.DataFrame(rf_pred, index=range(1461,2920), columns=["SalePrice"])
pred_df.to_csv('rf_pred.csv', header=True, index_label='Id')