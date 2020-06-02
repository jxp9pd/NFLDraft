"""Modelling expected 4-year AV"""

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

X = draft_df.Pick
y = draft_df.FourYearAV
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X.values.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2,
                                                    random_state=42)
#%%Train a linear model (with polynomial degree 3)
model_k = linear_model.LinearRegression(fit_intercept=True)
model_k.fit(X_train, y_train)
model_k.coef_ #Gets final coefficients
#R^2 coefficient
test_score = model_k.score(X_test, y_test)
print("R-squared of {:.3f}".format(test_score))