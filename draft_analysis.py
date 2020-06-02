"""Use the expected baseline model to analyze different drafts"""

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
#%%Read in Data and model

draft_2016 = pd.read_csv(LOCAL_PATH + "draftAV2016.csv")
draft_2016 = draft_2016.loc[draft_2016.index < 224]
draft_2016["Pick"] = np.arange(1, 225)

draft_df = pd.read_csv(LOCAL_PATH + "draftAV2000_2015.csv")
#Something went wrong with 2012 pick values
draft_df.loc[draft_df.Year == 2012, "Pick"] = draft_df.loc[draft_df.Year == 2012].index - 3059
draft_df = draft_df.loc[draft_df.Pick <= 224]

#It's a 3rd degree polynomial model
poly_model = pickle.load(open(LOCAL_PATH + "linearmodel_2000_2015.sav", 'rb'))
#%%Prep Data

def get_baselines(picks, model):
    """Expects a draft positions vector. Outputs a vector of the expected AV"""
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(picks.reshape(-1, 1))
    return model.predict(X_poly)
#%%Find Biggest "winners" and "losers"

predicted = get_baselines(np.arange(1, 225), poly_model)
draft_2016["ExpectedValue"] = predicted
draft_2016["PickDelta"] = draft_2016["FourYearAV"] - draft_2016["ExpectedValue"]
draft_df["ExpectedValue"] = get_baselines(draft_df.Pick.values, poly_model)
draft_df["PickDelta"] = draft_df["FourYearAV"] - draft_df["ExpectedValue"]

draft_df.PickDelta.sum()/draft_df.ExpectedValue.sum()
draft_df.PickDelta.sum()
draft_df.ExpectedValue.sum()

best_finds = draft_2016.sort_values(by="PickDelta", ascending=False)[
    ["Pos", "PlayerName", "Pick", "FourYearAV", "ExpectedValue", "PickDelta"]]
best_finds.head(10)
#%%Plot Baseline Career Value vs. Actual

plt.plot(predicted, label="Expected Value")
plt.plot(seven_rounds.FourYearAV, label="Actual Draft Value")
plt.legend()
plt.title("Comparing Actual Draft Value with Expected Baseline")
plt.show()
#%%Grigson Review

grigson_df = draft_df[(draft_df.Tm == "IND") & (draft_df.Year>=2012)]
# grigson_df["ExpectedValue"] = get_baselines(grigson_df.Pick.values, poly_model)
# grigson_df["PickDelta"] = grigson_df["FourYearAV"] - grigson_df["ExpectedValue"]
grigson_df = pd.concat([grigson_df, draft_2016[draft_2016.Tm == "IND"]])
sorted_picks = grigson_df.sort_values(by="PickDelta", ascending=False)[
    ["Pos", "PlayerName", "Pick", "FourYearAV", "ExpectedValue", "PickDelta"]]
added_value = grigson_df.PickDelta.sum()

print("Grigson's Five Best Picks:")
print(sorted_picks.head(5))
print("Grigson's Five Worst Picks:")
print(sorted_picks.tail(5))
print("Grigson overall delivered {:.2f} in additional value, averaging {:.2f} added value per draft".format(added_value, added_value/5))
#%%Ozzie ozzie Review

ozzie_df = draft_df[draft_df.Tm == "BAL"]
ozzie_df = pd.concat([ozzie_df, draft_2016[draft_2016.Tm == "BAL"]])
sorted_picks = ozzie_df.sort_values(by="PickDelta", ascending=False)[
    ["Pos", "PlayerName", "Pick", "FourYearAV", "ExpectedValue", "PickDelta"]]
added_value = ozzie_df.PickDelta.sum()

print("Ozzie Newsome's Five Best Picks:")
print(sorted_picks.head(5))
print("Ozzie Newsome's Five Worst Picks:")
print(sorted_picks.tail(5))
print("Ozzie Newsome drafted {:.2f} in additional value, averaging {:.2f} per draft"
      .format(added_value, added_value/ozzie_df.Year.nunique()))
