import matplotlib.pyplot as plt
import requests

requests.packages.urllib3.disable_warnings()

import json
from datetime import datetime, timedelta
import pandas as pd
import os
import openpyxl

token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2OTUyNTAzOTU0NDIsInBheWxvYWQiOiIxNDYifQ.WE5CzBWCHJGJeOt9LjIr_V1Uk0E7egqZJp4-3_WUaU0"
# url_login = "https://sdwan-uiv1.dyxnet.com:10082/api/auth/login"
# curl -k -i -X POST "https://sdwan-uiv1.dyxnet.com:10082/api/auth/login" -H  "Content-Type: application/json" -d "{\"account\":\"jeffz\", \"password\": \"jeffz#202309\"}"


def try_data():
    headers = {"Content-Type": "application/json",
               "Authorization": token}
    url_get = (f"https://sdwan-uiv1.dyxnet.com:10082/api/cpe/65/peers/networkUsage/wanType/VRF_PEER/" +
                         f"wan/0?from=2023-09-01%2000:00:00&to=2023-09-04%2023:59:59")
    login_response = requests.get(url_get, headers=headers, verify=False)
    data = login_response.json()
    data = data["data"]["peerNetworkUsageList"][0]["data"]
    if data == []:
        print("None")
    else:
        for item in data:
            # print(json.dumps(item))
            print(item)


def get_data(file_name="network_data.csv"):
    format = file_name.split('.')[-1]
    start_date = datetime(2023, 9, 18)
    end_date = datetime(2023, 9, 18)
    headers = {"Content-Type": "application/json",
               "Authorization": token}
    customers = [65, 31, 33, 36]
    customers = [33, 36]
    df_combined = pd.DataFrame(columns=['time', 'in', 'out', 'customer', 'line'])
    for customer in customers:
        current_date = start_date
        while current_date <= end_date:
            month = current_date.strftime('%m')
            day = current_date.strftime('%d')
            url_get_1 = (f"https://sdwan-uiv1.dyxnet.com:10082/api/cpe/{customer}/peers/networkUsage/wanType/VRF_PEER/" +
                         f"wan/0?from=2023-{month}-{day}%2000:00:00&to=2023-{month}-{day}%2011:59:59")
            url_get_2 = (f"https://sdwan-uiv1.dyxnet.com:10082/api/cpe/{customer}/peers/networkUsage/wanType/VRF_PEER/" +
                         f"wan/0?from=2023-{month}-{day}%2012:00:00&to=2023-{month}-{day}%2023:59:59")
            for url in [url_get_1, url_get_2]:
                login_response = requests.get(url, headers=headers, verify=False)
                data = login_response.json()
                data = data["data"]["peerNetworkUsageList"][0]["data"]
                if data == []:
                    print(f"xxxx  customer<{customer}>, {month}-{day}: None")
                    continue
                else:
                    df_new = pd.DataFrame(data)
                    # print(df_new)
                    df_new["customer"] = "A" if customer in [65, 31] else "B"
                    df_new["line"] = customer
                    df_combined = pd.concat([df_combined, df_new])
                    print(f"customer<{customer}>, {month}-{day}: SUCCESS")
            current_date += timedelta(days=1)
        if os.path.exists(file_name):
            df_existing = pd.read_csv(file_name)
            df_output = pd.concat([df_existing, df_combined], ignore_index=True)
        else:
            df_output = df_combined
        df_output.to_csv(file_name, index=False)


def get_data_5min(file_name="network_data.csv"):
    format = file_name.split('.')[-1]
    start_date = datetime(2023, 6, 1)
    end_date = datetime(2023, 9, 18)
    headers = {"Content-Type": "application/json",
               "Authorization": token}
    customers = [65, 31, 33, 36]
    # customers = [36]
    # df_combined = pd.DataFrame(columns=['time', 'in', 'out', 'customer', 'line'])
    for customer in customers:
        current_start_date = start_date
        while current_start_date <= end_date:
            current_end_date = current_start_date + timedelta(days=2)
            start_month = current_start_date.strftime('%m')
            start_day = current_start_date.strftime('%d')
            end_month = current_end_date.strftime('%m')
            end_day = current_end_date.strftime('%d')
            url_get = (f"https://sdwan-uiv1.dyxnet.com:10082/api/cpe/{customer}/peers/networkUsage/wanType/VRF_PEER/" +
                        f"wan/0?from=2023-{start_month}-{start_day}%2000:00:00&to=2023-{end_month}-{end_day}%2023:59:59")
            login_response = requests.get(url_get, headers=headers, verify=False)
            data = login_response.json()
            data = data["data"]["peerNetworkUsageList"][0]["data"]

            current_start_date += timedelta(days=3)       # Next time window, 3 days later

            if data == []:
                print(f"xxxx  customer<{customer}>, {start_month}-{start_day} ~ {end_month}-{end_day}: None")
                continue
            else:
                df_new = pd.DataFrame(data)
                # print(df_new)
                df_new["customer"] = "A" if customer in [65, 31] else "B"
                df_new["line"] = customer
                # df_combined = pd.concat([df_combined, df_new])
                print(f"customer<{customer}>, {start_month}-{start_day} ~ {end_month}-{end_day}: SUCCESS")

            if os.path.exists(file_name):
                df_existing = pd.read_csv(file_name)
                df_output = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_output = df_new
            df_output.to_csv(file_name, index=False)


def show_data_plot(raw_data="network_data.csv"):
    data = pd.read_csv(raw_data)
    start_time = data.min()['time']
    end_time = data.max()['time']
    date_range = pd.date_range(start=start_time, end=end_time, freq="W")
    print(date_range)
    data['time'] = pd.to_datetime(data['time'], format='mixed')
    # lines = data['line'].unique()
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for group in [65, 31, 33, 36]:
        group_data = data[data['line'] == group]
        ax[0].plot(group_data["time"].to_numpy(), group_data["in"].to_numpy(), label=group)
        ax[1].plot(group_data["time"].to_numpy(), group_data["out"].to_numpy(), label=group)

    # LOG scale
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[0].set_title('IN')
    ax[1].set_title('OUT')
    ax[0].legend()
    ax[1].legend()
    plt.xticks(date_range, rotation=45)
    plt.xlabel('Time')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # get_data()
    # get_data_5min("network_data_5min.csv")
    # try_data()
    show_data_plot("network_data_5min.csv")

