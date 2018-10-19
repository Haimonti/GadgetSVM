import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("../data/gisette/gisette_gadget_means.csv",
			header=0)

df2 = pd.read_csv("../data/gisette/gisette_pegasos_means.csv",
			header=0)

skip= 20
fig, ax = plt.subplots(1,2)
ax[0].plot(df["train_time"][::skip], df["obj_value"][::skip], 'b')
ax[0].set_title("Objective Value vs Train Time")
ax[0].set_xlabel("Train time")
ax[0].plot(df2["train_time"][::skip], df2["obj_value"][::skip], 'r')
ax[0].set_title("Zero One Error vs Train Time")
ax[0].set_xlabel("Train time")
ax[0].set_yscale('log')
plt.show()