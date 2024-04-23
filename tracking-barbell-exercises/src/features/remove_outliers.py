import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn

#Load Data
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

outlier_columns = list(df.columns[:6])

#Plotting outliers
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] =(20,5)
plt.rcParams["figure.dpi"] = 100 

#Accelerometer parameters
df[outlier_columns[:3] + ["label"]].boxplot(by="label",figsize = (20,10), layout = (1,3))

#Gyroscope parameters
df[outlier_columns[3:] + ["label"]].boxplot(by="label",figsize = (20,10), layout = (1,3))

def plot_binary_outliers(dataset, columns, outlier_columns, reset_index):
    """Plot outliers in case of binary outlier score. Here, the col specifies the real data column 
    and outlier_column with a binary value (outlier or not).

    Args:
        dataset (pd.Dataframe): The dataset
        columns (string): Column that you want to plot
        outlier_columns (string):  Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """
    
    dataset = dataset.dropna(axis = 0, subset = [columns,outlier_columns])
    dataset[outlier_columns] = dataset[outlier_columns].astype("bool")
    
    if reset_index:
        dataset = dataset.reset_index()
        
    fig,ax = plt.subplots()
    
    plt.xlabel("samples")
    plt.ylabel("values")
    
    #Plot non outliers in default colour
    ax.plot(
        dataset.index[~dataset[outlier_columns]],
        dataset[columns][~dataset[outlier_columns]],
        "+"
    )
    
    #Plot outliers in red
    ax.plot(
        dataset.index[dataset[outlier_columns]],
        dataset[columns][dataset[outlier_columns]],
        "r+"
    )
    
    plt.legend(
        ["outlier" + columns, "non-outlier" + columns],
        loc = "upper center",
        ncol = 2,
        fancybox = True,
        shadow = True
    )

#Insert IQR funciton
def mark_outliers_iqr(dataset, column):
    """Function to mark values as outliers using the IQR method

    Args:
        dataset (pd.Dataframe): The dataset
        column (string): The column you want to apply outlier detection to 
        
    Returns:
        dataset (pd.Dataframe): The dataset with the extra boolean column
        indicating whether value is an outlier or not
    """
    
    dataset = dataset.copy()
    
    Q1 = dataset[column].quantile(0.25)
    Q3 = dataset[column].quantile(0.75)
    Iqr = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * Iqr
    upper_bound = Q3 + 1.5 * Iqr
    
    dataset[column+"_outlier"] = (dataset[column] < lower_bound) | (dataset[col] > upper_bound)
    
    return dataset


#Plot single column
col = "acc_x"
dataset = mark_outliers_iqr(df,col)
plot_binary_outliers(dataset, col, col+"_outlier", reset_index = True)

#Plot all columns
for col in outlier_columns:
    dataset = mark_outliers_iqr(df,col)
    plot_binary_outliers(dataset, col, col+"_outlier", reset_index = True)
    

#Check for normal distribution
df[outlier_columns[:3] + ["label"]].plot.hist(by="label",figsize = (20,10), layout = (3,3))
df[outlier_columns[3:] + ["label"]].plot.hist(by="label",figsize = (20,10), layout = (3,3))

#Chavuenet's criterion
def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high.iloc[i]) - scipy.special.erf(low.iloc[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset

#Plot all columns
for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df,col)
    plot_binary_outliers(dataset, col, col+"_outlier", reset_index = True)
    

#Local Outlier Factor    
def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.
    
    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores

#Plot all columns
dataset, outliers, X_scores = mark_outliers_lof(df,outlier_columns)
for col in outlier_columns:
    plot_binary_outliers(dataset, col, "outlier_lof", reset_index = True)

exercise = list(df["label"].unique())

#Plot all columns for Iqr by label
for i in exercise:
    for col in outlier_columns:
        dataset = mark_outliers_iqr(df[df["label"] == i],col)
        plot_binary_outliers(dataset, col, col+"_outlier", reset_index = True)

#Plot all columns for Chauvenet by label
for i in exercise:
    for col in outlier_columns:
        dataset = mark_outliers_chauvenet(df[df["label"] == i],col)
        plot_binary_outliers(dataset, col, col+"_outlier", reset_index = True)

#Plot all columns for LOF by label
for i in exercise:
    dataset, outliers, X_scores = mark_outliers_lof(df[df["label"] == i],outlier_columns)
    for col in outlier_columns:
        plot_binary_outliers(dataset, col, "outlier_lof", reset_index = True)


outliers_removed_df = df.copy()
for col in outlier_columns:
    for label in df["label"].unique():
        dataset = mark_outliers_chauvenet(df[df["label"] == label],col)
        
        # Replace outliers with NaN
        dataset.loc[dataset[col+"_outlier"], col] = np.nan
        
        #Update the column in original frame
        outliers_removed_df.loc[outliers_removed_df["label"] == label, col] = dataset[col]
        
        n_outliers = len(dataset) - len(dataset[col].dropna())
        print(f"Removed {n_outliers} outliers from {col} for {label}")
    
#Export dataset
outliers_removed_df.to_pickle("../../data/interim/02_data_outliers_removed_chavunets.pkl")