import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


#Load Data
df = pd.read_pickle("../../data/interim/02_data_outliers_removed_chavunets.pkl")

predictor_columns = list(df.columns[:6])

#Plot Settings
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] =(20,5)
plt.rcParams["figure.dpi"] = 100 
plt.rcParams["lines.linewidth"] = 2


# Dealing with missing values (imputation)
for col in predictor_columns:
    df[col] = df[col].interpolate()

# Calculating set duration
for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    end = df[df["set"] == s].index[-1]
    
    duration = end - start
    df.loc[df["set"] == s, "duration"] = duration.seconds

duration_df = df.groupby(["category"])["duration"].mean()

    

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------


# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------