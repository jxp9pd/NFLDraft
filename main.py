"""
Comparing the value NFL drafts bring year to year
"""
# import pdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)

LOCAL_PATH = "C:/Users/jopentak/Documents/Data/draft_data/"
#%%Reading in AV drafts and Trade Value Chart

jimmy_j = pd.read_csv(LOCAL_PATH + "jimmy_j_chart.csv")
draft_df = pd.read_csv(LOCAL_PATH + "draftAV2000_2015.csv")
#Something went wrong with 2012 pick values
draft_df.loc[draft_df.Year == 2012, "Pick"] = draft_df.loc[draft_df.Year == 2012].index - 3059
#%%Scaler Function

def standard_scaler(signal):
    """Converts everything into z-scores (sklearn standard scaler)"""
    scaler = MinMaxScaler()
    signal_vals = signal.values.reshape(len(signal), 1)
    return scaler.fit_transform(signal_vals)    
#%%Bin Function

def bin_values(player_vals, end, bin_size):
    """
    Returns values binned into the given size along with bin tick marks.
    Paramaters
    Values: Dataframe with a Draft position column and a Value column
    end: Last pick to consider
    bin_size: How many rows to group together

    Output
    binned_vals: Actual values binned
    bins_n: The index values they were binned between.
    """
    # pdb.set_trace()
    player_vals = player_vals[player_vals.index <= end]
    binned_vals = player_vals.values.reshape(-1, bin_size)
    binned_vals = np.mean(binned_vals, axis=1)
    bins_n = np.arange(1, end, bin_size) + (bin_size)/2 - 1
    return binned_vals, bins_n
#%%Grouped bar chart labels

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
#%%Groupbys on pick or round number

pick_av = draft_df.groupby(["Pick"]).mean()["FourYearAV"]
pick_av = pd.DataFrame(pick_av)
rnd_av = draft_df.groupby(["Rnd"]).mean()["FourYearAV"]
#%%Binning value by Draft Position

#end point needs to be divisible by bin-size
ENDPOINT = 224
BIN_SIZE = 16
draft_vals, draft_bins = bin_values(pick_av.FourYearAV, ENDPOINT, BIN_SIZE)

# ENDPOINT = 224
chart = jimmy_j[["Pick", "Value"]][:-1]
chart.set_index("Pick", inplace=True) 
jimmy_bin, bin_j = bin_values(chart, ENDPOINT, BIN_SIZE)
#%%Normalizing AV by Draft Position

pick_av["ScaledVals"] = standard_scaler(pick_av.FourYearAV)
chart["ScaledVals"] = standard_scaler(chart.Value)

draft_scaled, draft_bins = bin_values(pick_av.ScaledVals, ENDPOINT, BIN_SIZE)
jimmy_scaled, bins_j = bin_values(chart.ScaledVals, ENDPOINT, BIN_SIZE)

pick_av["PctAV"] = pick_av.FourYearAV/pick_av.FourYearAV.sum()
chart["PctValue"] = chart.Value/chart.Value.sum()

draft_pct, draft_bins = bin_values(pick_av.PctAV, ENDPOINT, BIN_SIZE)
jimmy_pct, bins_j = bin_values(chart.PctValue, ENDPOINT, BIN_SIZE)
#%%Grouped Bar Chart 

# x = np.arange(len(draft_bins))  # the label locations
WIDTH = BIN_SIZE/3  # the width of the bars

fig, ax = plt.subplots()
rects1 = plt.bar(draft_bins - WIDTH/2, draft_pct*100, BIN_SIZE/3, label='Expected Value')
rects2 = plt.bar(draft_bins + WIDTH/2, jimmy_pct*100, BIN_SIZE/3, label='Jimmy\'s Trade Value')

plt.title("Draft Position by Expected Career Value and Pick Trade Value")
plt.xlabel("Draft Position")
plt.ylabel("Avg. Percent of Total Value in Draft Class") 
plt.legend()
# autolabel(rects1)
# autolabel(rects2)
fig.tight_layout()
plt.show()
#%%Bar Chart of Pick Value

plt.bar(height=draft_vals, x=draft_bins, edgecolor='black', width=BIN_SIZE)
plt.xlabel("Pick Position")
plt.ylabel("Mean Approximate Value")
plt.xticks(draft_bins)
plt.title("Approximate Value by Draft Position")
plt.show()

#%%Jimmy J Trade Value Chart

chart["Value"].plot()
plt.title("Jimmy Johnson Trade Value Chart")
plt.xlabel("Draft Position")
plt.show()

plt.bar(height=jimmy_bin, x=bins_j, edgecolor="black", width=BIN_SIZE)
plt.xticks(bins_j)
plt.title("Jimmy Johnson Trade Value")
plt.xlabel("Pick Position")
plt.ylabel("Trade Value")
plt.show()
#%%Standard Scaled Charts

plt.bar(height=jimmy_scaled, x=bins_j, edgecolor="black", width=BIN_SIZE)
plt.xticks(bins_j)
plt.title("Jimmy Johnson Trade Value")
plt.xlabel("Pick Position")
plt.ylabel("Trade Value")
plt.show()

plt.bar(height=draft_scaled, x=draft_bins, edgecolor="black", width=BIN_SIZE)
plt.xticks(draft_bins)
plt.title("Jimmy Johnson Trade Value")
plt.xlabel("Pick Position")
plt.ylabel("Trade Value")
plt.show()
