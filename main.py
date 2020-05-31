"""
Created on Sat May 30 23:54:40 2020

@author: jopentak
"""
import pdb

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)

LOCAL_PATH = "C:/Users/jopentak/Documents/Data/draft_data/"
#%%Reading in Draft Data
START_YEAR = 2010

draft_years = [str(year) + "_draft.csv" for year in np.arange(START_YEAR, 2016)]
draft_data = [pd.read_csv(LOCAL_PATH + filename, skiprows=1)
              for filename in draft_years]
for df in draft_data:
    df["Year"] = START_YEAR
    START_YEAR += 1
START_YEAR -= len(draft_data)

draft_df = pd.concat(draft_data)
#%%
draft_2010 = pd.read_csv(LOCAL_PATH + "2010_draft.csv", skiprows=1)
draft_2010.columns

#%%
