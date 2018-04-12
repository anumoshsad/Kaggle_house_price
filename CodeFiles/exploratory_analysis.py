# load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# load Data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# outlier detection
fig = plt.figure()
x_var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[x_var]], axis=1)
data.plot.scatter(x=x_var, y='SalePrice', ylim=(0,800000));
plt.savefig('outliers.png')


# Histogram of saleprice
fig = plt.figure()
x = train['SalePrice']
plt.hist(x, bins= 30)
plt.ylabel('Count', font)
plt.xlabel('Saleprice Bins')
plt.savefig('saleprice_hist.png')

# histogram of log(1+saleprice)
fig = plt.figure()
x = np.log1p(train['SalePrice'])
plt.hist(x, bins= 30)
plt.ylabel('Count', fontsize = 18 )
plt.xlabel('Saleprice Bins', fontsize = 18 )
plt.savefig('log1p_saleprice_hist.png')


#box plot overallqual/saleprice
fig = plt.figure()
var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xlabel('Sale Price', fontsize=18)
plt.ylabel('Overall Quality', fontsize=18)
plt.savefig('overall.png')



