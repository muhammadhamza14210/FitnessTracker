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

# Butterworth lowpass filter
df_lowpass = df.copy()
LowPass = LowPassFilter()

fs = 1000/5
cutoffs = 1

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoffs, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]


# Principal component analysis PCA
df_pca = df.copy()
pca = PrincipalComponentAnalysis()

pc_values = pca.determine_pc_explained_variance(df_pca,predictor_columns)

plt.figure(figsize=(10,10))
plt.plot(range(1,len(predictor_columns)+1),pc_values)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance")
plt.show()   

df_pca = pca.apply_pca(df_pca,predictor_columns,3)

# Sum of squares attributes
df_squared = df.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)


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