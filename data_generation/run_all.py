import os
import random
import pickle
import itertools
import smtplib
import warnings
import geopandas as gpd
import pandas as pd
import osmnx as ox
import pandas as pd
import datetime as dt

from os import getenv
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")

import sys
sys.path.insert(1, "..")
from code_root.src import utils

warnings.filterwarnings("ignore")
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from multiprocessing import Pool, cpu_count
from multiprocessing import Process, Queue, Manager

spark = (
    SparkSession.builder.config("spark.executor.cores", "8")
    .config("spark.executor.memory", "80g")
    .config("spark.sql.session.timeZone", "UTC")
    .config("spark.driver.memory", "40g")
    .master("local[26]")
    .appName("wego-daily")
    .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
    .config("spark.executor.extraJavaOptions", "-Duser.timezone=UTC")
    .config("spark.sql.datetime.java8API.enabled", "true")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.sql.autoBroadcastJoinThreshold", -1)
    .config("spark.driver.maxResultSize", 0)
    .config("spark.shuffle.spill", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

random.seed(100)
queue = Queue()
set_name = "_2022"


def get_traveltimes(tdf):
    tdf = tdf.sort_values("stop_sequence")
    # HACK: For null arrival times in the middle.
    tdf["stop_sequence"] = range(1, len(tdf) + 1)
    if len(tdf) <= 2:
        return pd.DataFrame()
    # HACK: This is for correcting the issue that the first stop's arrival_time starts much earlier than the scheduled time

    tdf = tdf.reset_index(drop=True)
    # tdf['scheduled_timestamp'] = (tdf['arrival_time'] - dt.datetime(1970,1,1)).dt.total_seconds()
    tdf["arrival_at_next_stop"] = tdf.arrival_time.shift(-1)
    tdf["time_to_next_stop"] = tdf["arrival_at_next_stop"] - tdf["departure_time"]
    tdf.at[0, "time_to_next_stop"] = (tdf.at[1, "arrival_time"] - tdf.at[0, "scheduled_time"]).total_seconds()
    # tdf = tdf.drop('scheduled_timestamp', axis=1)
    tdf = tdf.fillna(0)
    return tdf


def applyParallel(dfGrouped, func):
    with Pool(cpu_count() - 2) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list)


def mp_worker(L, queue):
    while queue.qsize() > 0:
        record = queue.get()

        current_node = record[0]
        next_node = record[1]

        if current_node == next_node:
            tt = 0
            dd = 0
        else:
            try:
                r = ox.shortest_path(G, current_node, next_node, weight="length")
                cols = ["osmid", "length", "travel_time"]
                attrs = ox.utils_graph.get_route_edge_attributes(G, r)
                tt = pd.DataFrame(attrs)[cols]["travel_time"].sum()
                dd = pd.DataFrame(attrs)[cols]["length"].sum()
            except:
                tt = -1
                dd = -1
        L.append((current_node, next_node, tt, dd))


def mp_handler():
    with Manager() as manager:
        L = manager.list()

        # Spawn two processes, assigning the method to be executed
        # and the input arguments (the queue)
        processes = [
            Process(
                target=mp_worker,
                args=(
                    L,
                    queue,
                ),
            )
            for _ in range(cpu_count() - 4)
        ]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        fp = os.path.join(f"results/pair_dd_tt{set_name}.pkl")
        with open(fp, "wb") as f:
            pickle.dump(list(L), f)

if __name__ == "__main__":
    start_date = dt.datetime.strptime("2022-01-01 06:28:00", "%Y-%m-%d %H:%M:%S")
    end_date = dt.datetime.strptime("2022-12-10 06:38:00", "%Y-%m-%d %H:%M:%S")

    # Read APC data.
    f = os.path.join("/home/jptalusan/mta_stationing_problem/data/processed/apc_weather_gtfs_20221216.parquet")
    apcdata = spark.read.load(f)
    get_columns = [
        "trip_id",
        "transit_date",
        "arrival_time",
        "departure_time",
        "block_abbr",
        "scheduled_time",
        "vehicle_id",
        "stop_sequence",
        "stop_id_original",
        "route_id",
        "route_direction_name",
    ]
    get_str = ", ".join([c for c in get_columns])
    apcdata.createOrReplaceTempView("apc")
    # # filter subset
    query = f"""
        SELECT {get_str}
        FROM apc
        WHERE (transit_date >= '{start_date.date()}') AND (transit_date <= '{end_date.date()}')
        """
    print(query)
    apcdata = spark.sql(query)
    apcdata = apcdata.withColumn(
        "route_id_direction", F.concat_ws("_", apcdata.route_id, apcdata.route_direction_name)
    )
    apcdata = apcdata.drop("route_id", "route_direction_name")
    baseline_data = apcdata.toPandas()
    baseline_data = baseline_data.dropna(subset=["arrival_time", "departure_time"])
    baseline_data["dow"] = baseline_data["scheduled_time"].dt.dayofweek
    baseline_data["IsWeekend"] = (baseline_data["scheduled_time"].dt.weekday >= 5).astype("int")
    baseline_data["time"] = baseline_data["scheduled_time"].dt.time

    # print(baseline_data.head())
    # Compute travel times
    out_arr = applyParallel(
        baseline_data.groupby(["block_abbr", "route_id_direction", "transit_date", "trip_id"]), get_traveltimes
    )
    tdf = out_arr.groupby(
        ["route_id_direction", "block_abbr", "stop_sequence", "stop_id_original", "time", "IsWeekend"]
    ).agg({"time_to_next_stop": list})
    fp = os.path.join(f"results/travel_time_by_scheduled_time{set_name}.pkl")
    tdf.to_pickle(fp)

    tdf = tdf.reset_index()
    tdf["sampled_travel_time"] = tdf.reset_index()["time_to_next_stop"].apply(lambda x: random.choice(x))
    tdf["sampled_travel_time"] = abs(tdf["sampled_travel_time"])
    fp = os.path.join(f"results/sampled_travel_times{set_name}.pkl")
    tdf.to_pickle(fp)

    tdf = tdf.drop("time_to_next_stop", axis=1)
    tdf["key_pair"] = list(
        zip(tdf.route_id_direction, tdf.block_abbr, tdf.stop_sequence, tdf.stop_id_original, tdf.time, tdf.IsWeekend)
    )
    tdf = tdf.set_index("key_pair")
    tdf = tdf.drop(
        ["route_id_direction", "block_abbr", "stop_sequence", "stop_id_original", "time", "IsWeekend"], axis=1
    ).to_dict("index")
    with open(f"results/sampled_travel_times_dict{set_name}.pkl", "wb") as handle:
        pickle.dump(tdf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    f = os.path.join("/home/jptalusan/mta_stationing_problem/data/processed/apc_weather_gtfs_20221216.parquet")
    apcdata = spark.read.load(f)
    apcdata.columns
    get_columns = ["stop_sequence", "stop_id_original", "stop_name", "map_latitude", "map_longitude"]
    get_str = ", ".join([c for c in get_columns])
    apcdata.createOrReplaceTempView("apc")

    # # filter subset
    query = f"""
        SELECT {get_str}
        FROM apc
        WHERE (transit_date >= '{start_date.date()}') AND (transit_date <= '{end_date.date()}')
    """
    # LIMIT 1000
    apcdata = spark.sql(query)
    apcdata = apcdata.drop_duplicates(["stop_id_original"])
    apcdf = apcdata.toPandas()

    fp = os.path.join("data", "shapefiles", "tncounty")
    gdf_county = gpd.read_file(fp)
    gdf_dav = gdf_county[gdf_county["NAME"] == "Davidson"]
    gdf_dav = gdf_dav.to_crs("EPSG:4326")

    G = ox.graph_from_polygon(gdf_dav.geometry.iloc[0], network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    apcdf["nearest_node"] = ox.nearest_nodes(G, apcdf["map_longitude"], apcdf["map_latitude"])
    apcdf["nearest_edge"] = ox.nearest_edges(G, apcdf["map_longitude"], apcdf["map_latitude"])

    fp = os.path.join("results", f"stops_node_matching{set_name}.pkl")
    apcdf.to_pickle(fp)
    fp = os.path.join("data", "davidson_graph.graphml")
    ox.save_graphml(G, fp)

    fp = os.path.join("results", f"stops_node_matching{set_name}.pkl")
    stop_nodes = pd.read_pickle(fp)
    fp = os.path.join("data", "davidson_graph.graphml")
    G = ox.load_graphml(fp)

    all_stops_list = stop_nodes.stop_id_original.tolist()
    all_nodes_list = stop_nodes.nearest_node.tolist()
    all_stops_combos = list(itertools.product(all_stops_list, repeat=2))
    all_nodes_combos = list(itertools.product(all_nodes_list, repeat=2))

    for pair in all_nodes_combos:
        queue.put(pair)

    mp_handler()

    fp = os.path.join(f"results/pair_dd_tt{set_name}.pkl")
    with open(fp, "rb") as f:
        pair_dd_tt = pickle.load(f)
    # [print(all_stops_combos[i], r) for i, r in enumerate(res)]

    # Setting it up with stop names

    stops_node_matching = pd.read_pickle("results/stops_node_matching_2022.pkl")

    df = pd.DataFrame.from_records(pair_dd_tt, columns=["current_node", "next_node", "travel_time_s", "distance_m"])

    df = pd.merge(
        left=df,
        right=stops_node_matching[["nearest_node", "stop_id_original"]],
        left_on="current_node",
        right_on="nearest_node",
        how="inner",
    )
    df = df.rename({"stop_id_original": "current_stop"}, axis=1).drop("nearest_node", axis=1)
    df = pd.merge(
        left=df,
        right=stops_node_matching[["nearest_node", "stop_id_original"]],
        left_on="next_node",
        right_on="nearest_node",
        how="inner",
    )
    df = df.rename({"stop_id_original": "next_stop"}, axis=1).drop("nearest_node", axis=1)

    df.to_pickle(f"results/pair_tt_dd_stops{set_name}.pkl")

    # Creating a node dictionary
    tdf = df.drop_duplicates(subset=["current_node", "next_node"])
    tdf["key_pair"] = list(zip(tdf.current_node, tdf.next_node))
    tdf = tdf.set_index("key_pair")
    stops_tt_dd_dict = tdf.drop(["current_node", "next_node", "current_stop", "next_stop"], axis=1).to_dict("index")
    with open(f"results/stops_tt_dd_node_dict{set_name}.pkl", "wb") as handle:
        pickle.dump(stops_tt_dd_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Creating a dictionary
    df["key_pair"] = list(zip(df.current_stop, df.next_stop))
    df = df.set_index("key_pair")
    df = df.drop_duplicates(subset=["current_node", "next_node", "current_stop", "next_stop"])
    stops_tt_dd_dict = df.drop(["current_node", "next_node", "current_stop", "next_stop"], axis=1).to_dict("index")
    with open(f"results/stops_tt_dd_dict{set_name}.pkl", "wb") as handle:
        pickle.dump(stops_tt_dd_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fp = os.path.join("results", f"stops_node_matching{set_name}.pkl")
    stop_nodes = pd.read_pickle(fp)
    stop_nodes = stop_nodes.set_index("stop_id_original")
    stop_nodes_dict = stop_nodes.drop(
        ["stop_sequence", "stop_name", "map_latitude", "map_longitude", "nearest_edge"], axis=1
    ).to_dict("index")
    with open(f"results/stops_node_matching_dict{set_name}.pkl", "wb") as handle:
        pickle.dump(stop_nodes_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    utils.send_email("GENERATION", "Done")
