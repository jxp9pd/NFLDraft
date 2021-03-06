"""
Historic Draft Data Collection and Processing
The output of this is a concatenated group of all the 
"""
import pdb
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from bs4 import BeautifulSoup

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)

LOCAL_PATH = "C:/Users/jopentak/Documents/Data/draft_data/"
#%%Reading in Draft Data

START_YEAR = 2000
draft_years = [str(year) + "_draft.csv" for year in np.arange(START_YEAR, 2017)]
draft_data = [pd.read_csv(LOCAL_PATH + filename, skiprows=1)
              for filename in draft_years]
for df in draft_data:
    df["Year"] = START_YEAR
    START_YEAR += 1
START_YEAR -= len(draft_data)
draft_df = pd.concat(draft_data)
#%%Processing Draft Data

name_list = draft_df.Player.str.split('\\')
draft_df['PlayerName'] = name_list.apply(lambda x: x[0])
draft_df['PlayerId'] = name_list.apply(lambda x: x[1] if len(x) > 1 else None)
# draft_df.drop("Player", inplace=True, axis=1)

draft_df.G.fillna(0, inplace=True)
draft_df.CarAV.fillna(0, inplace=True)
#%%Pull a players AV stat
# Example URL https://www.pro-football-reference.com/players/S/SuhxNd99.htm
# Player: Ndamkong Suh
def player_url(player_id):
    """Produces the URL for a player's PFF page"""
    url = "https://www.pro-football-reference.com/players/"
    url += player_id[0] + "/"
    url += player_id + ".htm"
    return url

def player_av(player_id, years=4):
    """Supplies the Approximate Value for a player over # of years"""
    # pdb.set_trace()
    if player_id == None:
        return None
    url = player_url(player_id)
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    
    table = soup.select_one("table")
    if table == None:
        return [0]
    table = table.select("tbody")[0].find_all("tr")[:years]
    #Pulls out the AV stat, then converts those to strs
    av = [tr.find("td", {'data-stat':"av"}).text if tr.find("td",
        {'data-stat':"av"}) != None else '0' for tr in table]
    av_int = [int(av_val) if len(av_val) > 0 else 0 for av_val in av]
    return av_int

TEST_ID = 'SuhxNd99'
player_vals = player_av(TEST_ID)
player_vals
#%%Pull 4-yr AV for all players

#This step takes some time. Has to make ~1500 HTTP Requests
draft_df["AVList"] = draft_df.apply(lambda x: player_av(x["PlayerId"])
                                    if x["G"] > 0 else [0], axis=1)

#%%Process 4-yr AV list into summation

# draft_df["FourYearAV"] = np.array(draft_df["AVList"].sum(skipna=False))
draft_df["FourYearAV"] = draft_df["AVList"].apply(lambda x: np.sum(x))
draft_2016 = draft_df[draft_df.Year == 2016]
draft_df = draft_df[draft_df.Year < 2016]
draft_df.to_csv(LOCAL_PATH + "draftAV2000_2015.csv", index=None)
draft_2016.to_csv(LOCAL_PATH + "draftAV2016.csv", index=None)
