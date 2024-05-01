import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


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

plt.figure(figsize=(10,5))
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

# Temporal abstraction
df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]
ws = int(1000/200)

for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")
    
#Make it unique to each exercise as currently it overlaps from previous data
df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

# Frequency features
df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

fs = int(1000/200)
ws = int(2800/200)

df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier Transformation to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset,predictor_columns,ws,fs)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# Dealing with overlapping windows
df_freq = df_freq.dropna()
df_freq = df_freq.iloc[::2]

# Clustering
df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2,10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters = k, n_init = 20, random_state = 0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10,10))
plt.plot(k_values,inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()


kmeans = KMeans(n_clusters = 5, n_init = 20, random_state = 0)
subset = kmeans.fit_predict(df_cluster[cluster_columns])
df_cluster["cluster"] = subset

#Plot Clusters
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=f"Cluster {c}")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

#Plot accelerometer graph to compare
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster ["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

#Plot gyrometer graph to compare
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster ["label"] == l]
    ax.scatter(subset["gyr_x"], subset["gyr_y"], subset["gyr_z"], label=l)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# Export dataset
df_cluster.to_pickle("../../data/interim/03_data_features.pkl")