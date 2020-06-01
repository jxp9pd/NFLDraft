"""
Comparing the value NFL drafts bring year to year
"""
import pdb
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from bs4 import BeautifulSoup

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
#%%Reading in Jimmy J Trade Value Chart
jimmy_j = pd.read_csv(LOCAL_PATH + "jimmy_j_chart.csv")
#%%Scaler Function
def standard_scaler(signal):
    """Converts everything into z-scores (sklearn standard scaler)"""
    scaler = StandardScaler()
    signal_vals = signal.values.reshape(len(signal), 1)
    return scaler.fit_transform(signal_vals)
#%%Bin Function

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
    table = soup.select_one("table").select("tbody")[0].find_all("tr")[:years]
    #Pulls out the AV stat, then converts those to strs
    av = [tr.find("td", {'data-stat':"av"}).text if tr.find("td",
        {'data-stat':"av"}) != None else '0' for tr in table]
    av_int = [int(av_val) if len(av_val) > 0 else 0 for av_val in av]
    return av_int

TEST_ID = 'SuhxNd99'
player_vals = player_av(TEST_ID)
player_vals
#%%Process 4-yr AV for all players
#This step takes some time. Has to make ~1500 HTTP Requests
# sample_df = draft_df.sample(frac=0.1, replace=False)
# sample_df.PlayerId = sample_df.PlayerId.astype("string")
# sample_df = sample_df[sample_df["G"] > 0]
draft_df["AVList"] = draft_df.apply(lambda x: player_av(x["PlayerId"])
                                    if x["G"] > 0 else [0], axis=1)
#%%Save Draft Dataframe to CSV
# draft_df["FourYearAV"] = np.array(draft_df["AVList"].sum(skipna=False))
draft_df["FourYearAV"] = draft_df["AVList"].apply(lambda x: np.sum(x))

#%%Groupbys on pick or round number
pick_av = draft_df.groupby(["Pick"]).mean()["FourYearAV"]
rnd_av = draft_df.groupby(["Rnd"]).mean()["FourYearAV"]
#%%Binning value by Draft Position
#end point needs to be divisible by bin-size
ENDPOINT = 256
BIN_SIZE = 16
two_rounds = pick_av[pick_av.index<=ENDPOINT]
two_rounds_bin = two_rounds.values.reshape(-1, BIN_SIZE)
two_rounds_bin = np.mean(two_rounds_bin, axis=1)
bins_n = np.arange(1, ENDPOINT, BIN_SIZE) + (BIN_SIZE)/2 - 1
#%%Binning Values for Jimmy Johnson Trade Value
ENDPOINT = 224
chart = jimmy_j[["Pick", "Value"]][:-1]
jimmy_bin = chart.Value.values.reshape(-1, BIN_SIZE)
jimmy_bin = np.mean(jimmy_bin, axis=1)
bins_j = np.arange(1, ENDPOINT, BIN_SIZE) + (BIN_SIZE)/2 - 1
#%%Normalizing values for direct comparison



#%%Bar Chart of Pick Value
plt.bar(height=two_rounds_bin, x=bins_n, edgecolor='black', width=BIN_SIZE)
plt.xlabel("Pick Position")
plt.ylabel("Mean Approximate Value")
plt.xticks(bins_n)
plt.title("Approximate Value by Draft Position")
plt.show()
#%%Jimmy J Trade Value Chart
jimmy_j["Value"].plot()
plt.title("Jimmy Johnson Trade Value Chart")
plt.xlabel("Draft Position")
plt.show()

plt.bar(height=jimmy_bin, x=bins_j, edgecolor="black", width=BIN_SIZE)
plt.title("Jimmy Johnson Trade Value Bar Chart")
plt.xlabel("Pick Position")
plt.ylabel("Trade Value")
plt.show()
#%%


#%%
pick_av.plot()
plt.title("Approximate Value by Draft Pick Position")
plt.show()
#%%
rnd_av.plot()
plt.title("Approximate Value by Draft Round")
plt.xlabel("Draft Round")
plt.ylabel("Average Approximate Value")
plt.show()
