"""Modelling expected 4-year AV"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)

LOCAL_PATH = "C:/Users/jopentak/Documents/Data/draft_data/"
#%%Read in Data

draft_df = pd.read_csv(LOCAL_PATH + "draftAV2000_2015.csv")
#%%Groupbys on pick or round number

pick_av = draft_df.groupby(["Pick"]).mean()["FourYearAV"]
pick_av = pd.DataFrame(pick_av)
#%%Split data into Train/Test

X = draft_df.Pick
y = draft_df.FourYearAV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=42)

#%%

model_k = linear_model.LinearRegression(fit_intercept=True)
model_k.fit(X, y)
model_k.coef_ #Gets final coefficients
model_k.score(X, y)