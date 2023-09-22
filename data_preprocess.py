import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)  # No limit for column on display
pd.set_option('display.max_rows', 10)        # Max rows on display = 10


def europe_data():
    df = pd.read_csv("data/eu_net/ec_data_set.csv")
    df2 = pd.read_csv("data/eu_net/isp.csv")

    df["region"] = "eu"
    df = df.rename(columns={df.columns[0]: "Time", df.columns[1]: "usage"})
    df["time_idx"] = df.index
    df["usage"] /= 1000000
    df2["region"] = "uk"
    df2 = df2.rename(columns={df2.columns[0]: "Time", df2.columns[1]: "usage"})
    df2["time_idx"] = df2.index
    merge_df = pd.concat([df, df2], ignore_index=True)
    merge_df.to_csv("data/merged_data.csv")
    merge_df["Time"] = pd.to_datetime(merge_df["Time"], format='mixed')
    df_plot = pd.DataFrame({"eu": df["usage"], "uk": df2["usage"]})
    merge_df["month"] = merge_df["Time"].dt.month.astype(str)
    merge_df["day"] = merge_df["Time"].dt.day
    merge_df["hour"] = merge_df["Time"].dt.hour
    merge_df["minute"] = merge_df["Time"].dt.minute
    print(merge_df)
    merge_df.to_csv("data/merged_data.csv")

    # df_plot.plot()
    # plt.show()


def network_data_preprocess(data="data/network/network_data_5min.csv"):
    df = pd.read_csv(data)
    # df = pd.melt(df, id_vars=["time", "customer", "line"], var_name="direction", value_name="usage")
    if "time_idx" not in df.columns:
        df["time_idx"] = df.groupby("line").cumcount() + 1
    df["time"] = pd.to_datetime(df["time"], format='mixed')
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["hour"] = df["time"].dt.hour
    df["minute"] = df["time"].dt.minute
    df["line"] = df["line"].astype('category')
    data_name = data.split('.')[0]
    df.to_csv(f"{data_name}_processed.csv")
    print(f"{data_name}_processed.csv")


if __name__ == "__main__":
    data = "data/network/network_data_15min.csv"

    data_processed = f"{(data.split('.')[0])}_processed.csv"
    network_data_preprocess(data)
    pre = pd.read_csv(data)
    print(pre.shape)
    post = pd.read_csv(data_processed)
    print(post.shape)
