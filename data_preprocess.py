import numpy as np
import pandas as pd
from typing import Union, List
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def network_data_preprocess(data: str):
    df = pd.read_csv(data)
    # df = pd.melt(df, id_vars=["time", "customer", "line"], var_name="direction", value_name="usage")
    # pytorch_forecasting/data/encoders.py/softplus_inv will make zero to NaN, caused by log(negative number). BUT WHY??????
    df["in_variance"] = df.groupby('line')['in'].transform('var')
    # df["in_mean"] = df.groupby('line')['in'].transform('mean')
    df["out_variance"] = df.groupby('line')['out'].transform('var')
    # df["out_mean"] = df.groupby('line')['out'].transform('mean')
    df["in_norm"] = df['in'] / np.sqrt(df["in_variance"])
    df["out_norm"] = df['out'] / np.sqrt(df["out_variance"])
    # df["in_log"] = np.where(df["in"] == 0, 0, np.log(df["in"]))
    # df["out_log"] = np.where(df["out"] == 0, 0, np.log(df["out"]))
    df["in_log"] = np.log1p(df["in"])
    df["out_log"] = np.log1p(df["out"])
    df["in_log_avg"] = df.groupby('line')['in_log'].transform('mean')
    df["out_log_avg"] = df.groupby('line')['out_log'].transform('mean')
    df["in_log_var"] = df.groupby('line')['in_log'].transform('var')
    df["out_log_var"] = df.groupby('line')['out_log'].transform('var')
    df["in_log_norm"] = df['in_log'] / np.sqrt(df["in_log_var"])
    df["out_log_norm"] = df['out_log'] / np.sqrt(df["out_log_var"])
    # df["in_log_norm_mean"] = df.groupby('line')['in_log_norm'].transform('mean')
    # df["out_log_norm_mean"] = df.groupby('line')['out_log_norm'].transform('mean')
    if "time_idx" not in df.columns:
        df["time_idx"] = df.groupby("line").cumcount() + 1
    df["time"] = pd.to_datetime(df["time"], format='%Y-%m-%d %H:%M:%S')
    df["month"] = df["time"].dt.month
    # df["weekday"] = df["time"].dt.day_name()
    weekday_to_num = {
        "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7
    }
    df["weekday"] = df["time"].dt.day_name().map(weekday_to_num)
    df["day"] = df["time"].dt.day
    df["hour"] = df["time"].dt.hour
    df["minute"] = df["time"].dt.minute
    df["line"] = df["line"].astype('category')
    df = add_date_cyclical_features(df, ["month", "weekday", "day", "hour", "minute"])
    data_name = data.split('.')[0]
    df.to_csv(f"{data_name}_processed.csv")
    print(f"{data_name}_processed.csv")


def add_date_cyclical_features(df: pd.DataFrame, features: Union[str, List[str]]) -> pd.DataFrame:
    cycles = {
        "month": 12,
        "weekday": 7,
        "day": [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        "hour": 24,
        "minute": 60
    }
    for feat in features:
        assert feat in cycles.keys(), f"feature must be one of {(x for x in cycles.keys())}"
        if feat == "day":
            df[f"{feat}_cos"] = np.cos(np.pi * df[feat] /
                                       df["month"].apply(lambda x: cycles[feat][x-1]))  # select max day
        else:
            df[f"{feat}_cos"] = np.cos(np.pi * df[feat] / cycles[feat])
        df[f"{feat}_cos"] = np.where(np.isclose(df[f"{feat}_cos"], 0, atol=1e-10), 0, df[f"{feat}_cos"])    # Make zero
    return df


def show_data_plot(raw_data="network_data.csv", target=None, groupby=None, timecol=None):
    target = [] if target is None else target
    groupby = [] if groupby is None else groupby
    data = pd.read_csv(raw_data)
    start_time = data.min()[timecol]
    end_time = data.max()[timecol]
    date_range = pd.date_range(start=start_time, end=end_time, freq="W")
    print(date_range)
    data[timecol] = pd.to_datetime(data[timecol], format='%Y-%m-%d %H:%M:%S')
    groups = data[groupby].unique()
    fig, ax = plt.subplots(len(target), 1, figsize=(10, 6), sharex=True)

    for group in groups:
        group_data = data[data[groupby] == group]
        if len(target) > 1:
            for i, t in enumerate(target):
                ax[i].plot(group_data[timecol].to_numpy(), group_data[t].to_numpy(), label=group)
                # ax[i].set_yscale('log')
                ax[i].set_title(t)
                ax[i].legend()
        else:
            ax.plot(group_data[timecol].to_numpy(), group_data[target[0]].to_numpy(), label=group)
            # ax.set_yscale('log')
            ax.set_title(target[0])
            ax.legend()

    plt.xticks(date_range, rotation=30)
    plt.xlabel('Time')

    plt.tight_layout()
    plt.show()


def show_origdata_hist(raw_data="network_data.csv", target=None, groupby=None):
    target = [] if target is None else target
    groupby = [] if groupby is None else groupby
    data = pd.read_csv(raw_data)
    bin_num = data.shape[0] // 500
    print("bin number", bin_num)

    groups = data[groupby].unique()
    fig, ax = plt.subplots(len(groups), len(target), figsize=(10, 6))
    ax = ax.flatten()
    for i, group in enumerate(groups):
        j = i * len(target)
        for k, t in enumerate(target):
            group_data = data[data[groupby] == group]
            ax[j+k].hist(group_data[t].to_numpy(), bins=bin_num, label=group)
            ax[j+k].set_yscale('log')
            # ax[j].set_xscale('log')
            ax[j+k].set_title(f'{group} {t}')

    # plt.xticks(date_range, rotation=30)
    # plt.xlabel('Time')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = "data/network/network_data_5min.csv"
    data_processed = f"{(data.split('.')[0])}_processed.csv"
    network_data_preprocess(data)
    pre = pd.read_csv(data)
    print(pre.shape)
    post = pd.read_csv(data_processed)
    print(post.shape)

    show_data_plot("data/network/network_data_5min_processed.csv", ["in_log", "in_log_norm", "out_log", "out_log_norm"], groupby="line", timecol='time')
    # show_data_plot("data/eu_net/merged_data.csv", ["usage"], groupby="region", timecol='Time')

    # show_origdata_hist(raw_data="data/network/network_data_15min_processed.csv", target=["in_log", "in_log_norm", "out_log", "out_log_norm"], groupby="line")
    # show_origdata_hist(raw_data="data/eu_net/merged_data.csv", target=["usage"], groupby="region")


    # pd.options.display.float_format = '{:.10f}'.format
    # print('log in mean\n', post.groupby('line')['log_in'].mean())
    # print('log in std\n', np.sqrt(post.groupby('line')['log_in'].var()))
    # print('log in standard\n', post.groupby('line')['log_in'].mean() / np.sqrt(post.groupby('line')['log_in'].var()))
    # print('in std\n', np.sqrt(post.groupby('line')['in'].var()))
    # print('in mean\n', post.groupby('line')['in'].mean())
    # print('in standard\n', post.groupby('line')['in'].mean() / np.sqrt(post.groupby('line')['in'].var()))


