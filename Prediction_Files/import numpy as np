import numpy as np
import pandas as pd

ridge = pd.read_csv("ridge_pred.csv")["SalePrice"]
lasso = pd.read_csv("lasso_pred.csv")["SalePrice"]
xgb = pd.read_csv("xgb_pred.csv")["SalePrice"]
poly = pd.read_csv("PolynomialFeaturePredictions.csv")["SalePrice"]
stack = pd.read_csv("SimpleStacking.csv")["SalePrice"]


y_pred = .2*ridge + .4*lasso + .3*xgb + .05*poly + .05*stack
pred_df = pd.DataFrame(np.array(y_pred+1), index=range(1461,2920), columns=["SalePrice"])
pred_df.to_csv('final_try.csv', header=True, index_label='Id')

y_pred = .2*ridge + .4*lasso + .3*xgb + .05*poly + .05*stack
pred_df = pd.DataFrame(np.array(y_pred+1), index=range(1461,2920), columns=["SalePrice"])
pred_df.to_csv('final_try.csv', header=True, index_label='Id')

##############################
# this is our best combination 
y_pred = .25*ridge + .35*lasso + .35*xgb + .03*poly + .02*stack
pred_df = pd.DataFrame(np.array(y_pred+1), index=range(1461,2920), columns=["SalePrice"])
pred_df.to_csv('final_try2.csv', header=True, index_label='Id')