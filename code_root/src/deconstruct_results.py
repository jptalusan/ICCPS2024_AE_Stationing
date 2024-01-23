import pandas as pd
from os import listdir
from os.path import isfile, join
import re
from tqdm import tqdm

numeric_bus_cols = [
    "bus_id",
    "dwell_time",
    "agg_delay",
    "service_time",
    "total_served",
    "service_kms",
    "status",
    "total_deadsecs",
]

numeric_stop_cols = ["total_passenger_ons", "total_passenger_offs", "total_passenger_walk_away"]


def logs_to_df(mypath):
    log_file_arr = {}

    bus_metrics = []
    stop_metrics = []

    start_bus_metrics = False
    start_stop_metrics = False

    log_path = f"{mypath}"
    with open(log_path) as file:
        for line in file:
            data = line.rstrip()

            if (
                "bus_id,dwell_time,agg_delay,service_time,total_served,service_kms,current_stop,status,type,total_deadsecs,starting_stop"
                in line
            ):
                bus_metrics.append(data)
                start_stop_metrics = False
                start_bus_metrics = True
                continue

            if "stop_id,total_passenger_ons,total_passenger_offs,total_passenger_walk_away" in line:
                stop_metrics.append(data)
                start_stop_metrics = True
                start_bus_metrics = False
                continue

            if "Count of all passengers" in line:
                break

            if "passenger waiting dict list" in line:
                break

            if start_bus_metrics:
                bus_metrics.append(data)

            if start_stop_metrics:
                stop_metrics.append(data)

        # break
        bus_df = pd.DataFrame([sub.split(",") for sub in bus_metrics])
        # if bus_df.empty:
        #     continue
        # display(bus_df.head(1))
        new_header = bus_df.iloc[0]  # grab the first row for the header
        bus_df = bus_df[1:]  # take the data less the header row
        bus_df.columns = new_header  # set the header row as the df header

        bus_df[numeric_bus_cols] = bus_df[numeric_bus_cols].apply(pd.to_numeric, errors="coerce")

        stop_df = pd.DataFrame([sub.split(",") for sub in stop_metrics])
        # display(stop_df.head(1))
        new_header = stop_df.iloc[0]  # grab the first row for the header
        stop_df = stop_df[1:]  # take the data less the header row
        stop_df.columns = new_header  # set the header row as the df header

        stop_df[numeric_stop_cols] = stop_df[numeric_stop_cols].apply(pd.to_numeric, errors="coerce")
        stop_df = stop_df[["stop_id", "total_passenger_ons", "total_passenger_offs", "total_passenger_walk_away"]]
    #     # stop_df
    #     label = "MCTS"
    #     if baselines:
    #         label = "Greedy"
    #     nr_label = "With Reallocation"
    #     if norealloc:
    #         nr_label = "No Reallocation"
    #     log_file_arr[(date, label, nr_label)] = [bus_df, stop_df]
    # return log_file_arr
    return stop_df, bus_df


# res_path = (
#     "/Users/josepaolo-t/Developer/git/mta_simulator_redo/code_root/experiments/TEST/results/20210211_test/results.csv"
# )
# sdf, bdf = logs_to_df(res_path)
# bdf.head()
