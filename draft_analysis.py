"""Use the expected baseline model to analyze different drafts"""

import pickle
import pdb

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

#%%GM Evaluator

def gm_evaluate(team, start_year, end_year, name, include_2016=True):
    """Prints out a summary of a GM's draft performance"""
    gm_df = draft_df[(draft_df.Tm == team) & (draft_df.Year>=start_year) &
                     (draft_df.Year<=end_year)]
    if include_2016:
        gm_df = pd.concat([gm_df, draft_2016[draft_2016.Tm == team]])

    sorted_picks = gm_df.sort_values(by="PickDelta", ascending=False)[
        ["Pos", "PlayerName", "Pick", "FourYearAV", "ExpectedValue", "PickDelta"]]
    # pdb.set_trace()
    added_value = gm_df.PickDelta.sum()
    
    print(name + "'s Five Best Picks:")
    print(sorted_picks.head(5))
    print()
    print(name + "'s Five Worst Picks:")
    print(sorted_picks.tail(5))
    
    print(name + " overall drafted {:.2f} in marginal value.".format(added_value))
    print("He added on average {:.2f} per draft.".format(added_value/gm_df.Year.nunique()))
    return gm_df
    
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
plt.plot(draft_2016.FourYearAV, label="Actual Draft Value")
plt.legend()
plt.title("Comparing Actual Draft Value with Expected Baseline")
plt.show()
#%%Grigson Review
grigson_df = gm_evaluate("IND", 2012, 2016, "Grigson", True)
thompson_df = gm_evaluate("GNB", 2005, 2016, "Ted Thompson", True)
bruce_df = gm_evaluate("WAS", 2009, 2016, "Bruce Allen", True)
roseman_df = gm_evaluate("PHI", 2010, 2014, "Howie Roseman", True)

#%%Team Evals

cleveland_df = gm_evaluate("CLE", 2000, 2016, "Cleveland Browns", True)
buffalo_df = gm_evaluate("BUF", 2000, 2016, "Buffalo Bills", True)
kc_df = gm_evaluate("KAN", 2000, 2016, "Kansas City Chiefs", True)
ind_df = gm_evaluate("IND", 2000, 2016, "Colts", True)
pit_df = gm_evaluate("DET", 2000, 2016, "Pittsburgh Steelers", True)
#%%Team Groupbys

draft_df.groupby("Pos")["PickDelta"].mean().sort_values(ascending=False)



