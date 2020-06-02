"""Modelling expected 4-year AV with a linear model"""

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)

LOCAL_PATH = "C:/Users/jopentak/Documents/Data/draft_data/"
#%%Read in Data

draft_df = pd.read_csv(LOCAL_PATH + "draftAV2000_2015.csv")
draft_2012 = pd.read_csv(LOCAL_PATH + "2012_draft.csv")
#Something went wrong with 2012 pick values
draft_df.loc[draft_df.Year == 2012, "Pick"] = draft_df.loc[draft_df.Year == 2012].index - 3059
#%%Groupbys on pick or round number

pick_av = draft_df.groupby(["Pick"]).mean()["FourYearAV"]
pick_av = pd.DataFrame(pick_av)
#%%Split data into Train/Test

#Cuts the last round to make predicting easier
train_data = draft_df[draft_df.Pick <= 224]
X = train_data.Pick
y = train_data.FourYearAV
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X.values.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2,
                                                    random_state=42)
#%%Train a linear model (with polynomial degree 3)

model_k = linear_model.LinearRegression(fit_intercept=False)
model_k.fit(X_train, y_train)
#R^2 coefficient
test_score = model_k.score(X_test, y_test)
print("R-squared of {:.3f}".format(test_score))
#%%Make model predictions

full_draft = np.arange(1, 257)
full_poly = poly.fit_transform(full_draft.reshape(-1, 1))
av_predictions = model_k.predict(full_poly)
#%%Plot Model Predictions

plt.scatter(X_test[:, 1], y_test, s=10)
plt.plot(av_predictions, color="black")
plt.ylim(0, 60)
plt.xlim(0, 230)
plt.xlabel("Draft Position")
plt.ylabel("4-year AV")
plt.title("Career Value by Draft Position")
plt.show()
#%%Save model

pickle.dump(model_k, open(LOCAL_PATH + "linearmodel_2000_2015.sav", "wb"))
