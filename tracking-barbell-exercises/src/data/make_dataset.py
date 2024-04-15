import pandas as pd
from glob import glob

def read_from_files(files):
    data_path = "../../data/raw/MetaMotion/"
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        df = pd.read_csv(f)

        df["participant"] = f.split("-")[0].replace(data_path, "")
        df["label"] = f.split("-")[1]
        df["category"] = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        if "Accelerometer" in f:
            df['set'] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
        if "Gyroscope" in f:
            df['set'] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]
    
    return acc_df, gyr_df

files = glob("../../data/raw/MetaMotion/*csv")
acc_df, gyr_df = read_from_files(files)

data_merged = pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1)
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

sampling = {
    "acc_x":"mean",
    "acc_y":"mean",
    "acc_z":"mean",
    "gyr_x":"mean",
    "gyr_y":"mean",
    "gyr_z":"mean",
    "participant":"last",
    "label":"last",
    "category":"last",
    "set":"last",
}

data_merged[:1000].resample(rule="200ms").apply(sampling)

#split by days
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

#resampled data
resampled_data = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])
resampled_data["set"] = resampled_data["set"].astype("int")

#export database
resampled_data.to_pickle("../../data/interim/01_data_processed.pkl")