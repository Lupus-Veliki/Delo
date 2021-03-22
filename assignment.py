# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 07:20:29 2021

@author: Tomaz Volf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/Tomaz Volf/Desktop/ctr-prediction-train-1.txt", sep=",",
                   header="infer")
print(data)

series = data.groupby("hour", as_index=False).agg({"hour": "count", "click": "sum"})
series["ctr"] = series["click"]/series["hour"]
series["MA_ctr"], series["STD_ctr"] = (series["ctr"].rolling(window=7, min_periods=1).mean(),
                                       series["ctr"].rolling(window=7, min_periods=1).std())

plt.title("CTR Time Series")
plt.ylabel("CTR Value")
plt.xlabel("Time")
plt.plot(series[["ctr"]])
plt.show()

pd.set_option("display.max_rows", None)
series["outlier"] = np.sign(1.5*series["STD_ctr"]-abs(series["ctr"]-series["MA_ctr"]))
# negative value indicates an outlier.

print(series)
weirdos = series[series["outlier"]==-1]
print(weirdos)

plt.title("CTR Time Series with Highlighted Outliers")
plt.ylabel("CTR Value")
plt.xlabel("Time")
plt.plot(weirdos["ctr"], linestyle="none", marker="X", color="red", markersize=6)
plt.plot(series[["ctr"]])
plt.show()