import logging
import os

logFormatter = logging.Formatter("[%(asctime)s][%(levelname)s][%(lineno)d] %(message)s", "%m-%d %H:%M:%S")
logger = logging.getLogger("generate_day_trips")
logger.setLevel(logging.DEBUG)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(logFormatter)
streamHandler.setLevel(logging.DEBUG)
logger.addHandler(streamHandler)

from tensorflow.keras import backend as K

K.clear_session()
import os
from os import getenv
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.insert(1, "..")
from code_root.src import utils

import datetime as dt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import numpy as np
import random
import json
import shutil
import argparse
import joblib
import matplotlib as mpl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
import smtplib
import time

from multiprocessing import Process, Queue, cpu_count, Manager, Pool
from re import search

mpl.rcParams["figure.facecolor"] = "white"

import warnings

import pandas as pd

pd.set_option("display.max_columns", None)
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
tf.get_logger().setLevel("INFO")

spark = (
    SparkSession.builder.config("spark.executor.cores", "8")
    .config("spark.executor.memory", "80g")
    .config("spark.sql.session.timeZone", "UTC")
    .config("spark.driver.memory", "40g")
    .master("local[*]")
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

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

CURR_DIR = "/home/jpt/gits/mta_simulator_redo/data_generation"
rng = np.random.default_rng(12345)


def pick_random_until_zero(row):
    st = row["scheduled_time"]
    val = row["ons"]
    interval = pd.date_range(pd.Timestamp(st) - pd.Timedelta(minutes=10), pd.Timestamp(st), freq="min")
    arrival_arr = []
    val_arr = []
    if val == 0:
        return list(zip([0], [st]))
    while val > 0:
        try:
            pick = random.randint(1, val)
            val -= pick
            val_arr.append(pick)
        except ValueError as e:
            logger.debug(val, pick, e)

    try:
        arrtimes = rng.choice(interval, size=len(val_arr), replace=False)
        arrival_arr.append(list(zip(val_arr, arrtimes)))
        arrival_arr = np.reshape(arrival_arr, (-1, 2))
    except ValueError as e:
        logger.debug(len(interval), len(val_arr))
        arrival_arr.append(list(zip(val_arr, interval)))
        arrival_arr = np.reshape(arrival_arr, (-1, 2))

    return arrival_arr


def process_to_parquet(path):
    logger.info(f"Processing: {path}")
    df = pd.read_parquet(path).reset_index()
    # df = pd.DataFrame(df).reset_index()
    df = df[
        [
            "route_id_dir",
            "block_id",
            "stop_sequence",
            "stop_id",
            "scheduled_time",
            "trip_id",
            "sampled_loads",
            "ons",
            "offs",
        ]
    ]

    new_passenger_arrival_arr = []

    for k, v in df.iterrows():
        route_id_dir = v["route_id_dir"]
        block_id = v["block_id"]
        stop_sequence = v["stop_sequence"]
        stop_id = v["stop_id"]
        trip_id = v["trip_id"]
        scheduled_time = v["scheduled_time"]
        sampled_loads = v["sampled_loads"]
        offs = v["offs"]
        # scheduled_time = v['scheduled_time']
        arrival_arr = pick_random_until_zero(v)

        if len(arrival_arr) > 0:
            idx = random.choice(list(range(len(arrival_arr))))
            for i, (ons, arrival_time) in enumerate(arrival_arr):
                if i != idx:
                    offs = 0
                _input = [
                    route_id_dir,
                    block_id,
                    trip_id,
                    stop_sequence,
                    stop_id,
                    scheduled_time,
                    sampled_loads,
                    ons,
                    arrival_time,
                    offs,
                ]
                new_passenger_arrival_arr.append(_input)
        else:
            _input = [
                route_id_dir,
                block_id,
                trip_id,
                stop_sequence,
                stop_id,
                scheduled_time,
                sampled_loads,
                0,
                arrival_time,
                offs,
            ]
            new_passenger_arrival_arr.append(_input)

    tdf = pd.DataFrame(
        new_passenger_arrival_arr,
        columns=[
            "route_id_dir",
            "block_id",
            "trip_id",
            "stop_sequence",
            "stop_id",
            "scheduled_time",
            "sampled_loads",
            "ons",
            "arrival_time",
            "offs",
        ],
    )
    tdf.to_parquet(f'{path.split(".")[0]}.parquet')
    logger.info(f"Done: {path.split('.')[0]}.parquet")


def get_apc_data_for_daterange(start_date, end_date):
    logger.info(f"Start get_apc_data_for_date range: {start_date} to {end_date}")
    filepath = f"{CURR_DIR}/data/apc_weather_gtfs_20221216.parquet"
    apcdata = spark.read.load(filepath)
    apcdata.createOrReplaceTempView("apc")

    get_columns = [
        "trip_id",
        "transit_date",
        "arrival_time",
        "scheduled_time",
        "block_abbr",
        "stop_sequence",
        "stop_id_original",
        "vehicle_id",
        "vehicle_capacity",
        "load",
        "darksky_temperature",
        "darksky_humidity",
        "darksky_precipitation_probability",
        "route_direction_name",
        "route_id",
        "overload_id",
        "dayofweek",
        "year",
        "month",
        "hour",
        "zero_load_at_trip_end",
        "sched_hdwy",
    ]
    get_str = ", ".join([c for c in get_columns])
    query = f"""
    SELECT {get_str}
    FROM apc
    WHERE (transit_date >= '{start_date}' and transit_date <= '{end_date}')
    ORDER BY arrival_time
    """
    apcdata = spark.sql(query)
    apcdata = apcdata.withColumn("route_id_dir", F.concat_ws("_", apcdata.route_id, apcdata.route_direction_name))
    apcdata = apcdata.withColumn("day", F.dayofmonth(apcdata.arrival_time))
    apcdata = apcdata.drop("route_direction_name")
    apcdata = apcdata.withColumn("load", F.when(apcdata.load < 0, 0).otherwise(apcdata.load))
    apcdata = apcdata.na.fill(value=0, subset=["zero_load_at_trip_end"])

    exclude_routes = [88, 89, 94, 95]
    apcdata = apcdata.filter(~F.col("route_id").isin(exclude_routes))
    apcdata = apcdata.filter(F.col("overload_id") == 0)
    apcdata = apcdata.dropna(subset=["arrival_time"])

    exclude_stops = ["PEARL", "JOHASHEN", "ROS10AEN", "PLSNTVW", "CLK E11"]
    apcdata = apcdata.filter(~F.col("stop_id_original").isin(exclude_stops))
    return apcdata


def get_apc_data_for_date(filter_date):
    logger.info(f"Start get_apc_data_for_date: {filter_date}")
    filepath = f"{CURR_DIR}/data/apc_weather_gtfs_20221216.parquet"
    apcdata = spark.read.load(filepath)
    apcdata.createOrReplaceTempView("apc")

    get_columns = [
        "trip_id",
        "transit_date",
        "arrival_time",
        "scheduled_time",
        "block_abbr",
        "stop_sequence",
        "stop_id_original",
        "vehicle_id",
        "vehicle_capacity",
        "load",
        "darksky_temperature",
        "darksky_humidity",
        "darksky_precipitation_probability",
        "route_direction_name",
        "route_id",
        "overload_id",
        "dayofweek",
        "year",
        "month",
        "hour",
        "zero_load_at_trip_end",
        "sched_hdwy",
    ]
    get_str = ", ".join([c for c in get_columns])
    query = f"""
        SELECT {get_str}
        FROM apc
        WHERE (transit_date == '{filter_date}')
        ORDER BY arrival_time
    """
    apcdata = spark.sql(query)
    apcdata = apcdata.withColumn("route_id_dir", F.concat_ws("_", apcdata.route_id, apcdata.route_direction_name))
    apcdata = apcdata.withColumn("day", F.dayofmonth(apcdata.arrival_time))
    apcdata = apcdata.drop("route_direction_name")
    apcdata = apcdata.withColumn("load", F.when(apcdata.load < 0, 0).otherwise(apcdata.load))
    apcdata = apcdata.na.fill(value=0, subset=["zero_load_at_trip_end"])

    exclude_routes = [88, 89, 94, 95]
    apcdata = apcdata.filter(~F.col("route_id").isin(exclude_routes))
    apcdata = apcdata.filter(F.col("overload_id") == 0)
    apcdata = apcdata.dropna(subset=["arrival_time"])

    exclude_stops = ["PEARL", "JOHASHEN", "ROS10AEN", "PLSNTVW", "CLK E11"]
    apcdata = apcdata.filter(~F.col("stop_id_original").isin(exclude_stops))

    return apcdata


def prepare_input_data(input_df, ohe_encoder, label_encoders, num_scaler, columns, keep_columns=[], target="y_class"):
    num_columns = ["darksky_temperature", "darksky_humidity", "darksky_precipitation_probability", "sched_hdwy"]
    cat_columns = ["month", "hour", "day", "stop_sequence", "stop_id_original", "year", "time_window"]
    ohe_columns = ["dayofweek", "route_id_dir", "is_holiday", "is_school_break", "zero_load_at_trip_end"]

    # OHE
    input_df[ohe_encoder.get_feature_names_out()] = ohe_encoder.transform(input_df[ohe_columns]).toarray()

    # Label encoder
    for cat in cat_columns:
        encoder = label_encoders[cat]
        input_df[cat] = encoder.transform(input_df[cat])

    # Num scaler
    input_df[num_columns] = num_scaler.transform(input_df[num_columns])
    input_df[target] = input_df[target].astype("int")

    if keep_columns:
        columns = keep_columns + columns
    # Rearrange columns
    input_df = input_df[columns]

    return input_df


def assign_data_to_bins(df, TARGET="load"):
    bins = pd.IntervalIndex.from_tuples([(-1, 6.0), (6.0, 12.0), (12.0, 55.0), (55.0, 75.0), (75.0, 100.0)])
    mycut = pd.cut(df[TARGET].tolist(), bins=bins)
    df["y_class"] = mycut.codes
    return df


def add_features(df, TIMEWINDOW=15):
    df = df[df.arrival_time.notna()]
    df = df.fillna(method="bfill")

    df["day"] = df["arrival_time"].dt.day
    df = df.sort_values(by=["block_abbr", "arrival_time"]).reset_index(drop=True)

    # Adding extra features
    # Holidays
    fp = os.path.join("data", "US Holiday Dates (2004-2021).csv")
    holidays_df = pd.read_csv(fp)
    holidays_df["Date"] = pd.to_datetime(holidays_df["Date"])
    holidays_df["is_holiday"] = True
    df = df.merge(holidays_df[["Date", "is_holiday"]], left_on="transit_date", right_on="Date", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(False)
    df = df.drop(columns=["Date"])

    # School breaks
    fp = os.path.join("data", "School Breaks (2019-2022).pkl")
    school_break_df = pd.read_pickle(fp)
    school_break_df["is_school_break"] = True
    df = df.merge(school_break_df[["Date", "is_school_break"]], left_on="transit_date", right_on="Date", how="left")
    df["is_school_break"] = df["is_school_break"].fillna(False)
    df = df.drop(columns=["Date"])

    df["minute"] = df["arrival_time"].dt.minute
    df["minuteByWindow"] = df["minute"] // TIMEWINDOW
    df["temp"] = df["minuteByWindow"] + (df["hour"] * 60 / TIMEWINDOW)
    df["time_window"] = np.floor(df["temp"]).astype("int")
    df = df.drop(columns=["minute", "minuteByWindow", "temp"])

    # HACK
    # df = df[df['hour'] != 3]
    # df = df[df['stop_sequence'] != 0]

    df = df.sort_values(by=["block_abbr", "arrival_time"]).reset_index(drop=True)

    df = assign_data_to_bins(df, TARGET="load")
    return df


def setup_simple_lstm_generator(num_features, num_classes, learning_rate=1e-4):
    # define model
    model = tf.keras.Sequential()
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    # compile model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["sparse_categorical_accuracy"],
    )

    input_shape = (None, None, num_features)
    model.build(input_shape)
    return model


def generate_simple_lstm_predictions(input_df, model, past, future):
    past_df = input_df[0:past]
    future_df = input_df[past:]
    predictions = []
    pred_probs = []
    if future == None:
        future = len(future_df)
    for f in range(future):
        pred = model.predict(past_df.to_numpy().reshape(1, *past_df.shape), verbose=0)
        pred_probs.append(pred)
        y_pred = np.argmax(pred)
        predictions.append(y_pred)

        # Add information from future
        last_row = future_df.iloc[[0]]

        # Commenting this out means it will use the y_true for predicting next stops (which will lead to higher accuracy without cheating)
        # last_row['y_class'] = y_pred
        past_df = pd.concat([past_df[1:], last_row])

        # Move future to remove used row
        future_df = future_df[1:]
    return predictions, pred_probs


def compute_ons_offs(s):
    curr_load = s["sampled_loads"]
    next_load = s["next_load"]
    if next_load > curr_load:
        ons = next_load - curr_load
        offs = 0
    elif next_load < curr_load:
        ons = 0
        offs = curr_load - next_load
    else:
        ons = 0
        offs = 0

    return ons, offs


# Load model
latest = tf.train.latest_checkpoint(f"{CURR_DIR}/models/no_speed")
columns = joblib.load(f"{CURR_DIR}/models/LL_X_columns.joblib")
label_encoders = joblib.load(f"{CURR_DIR}/models/LL_Label_encoders.joblib")
ohe_encoder = joblib.load(f"{CURR_DIR}/models/LL_OHE_encoder.joblib")
num_scaler = joblib.load(f"{CURR_DIR}/models/LL_Num_scaler.joblib")

ohe_columns = ["dayofweek", "route_id_dir", "is_holiday", "is_school_break", "zero_load_at_trip_end"]
percentiles = [(0, 6.0), (6.0, 12.0), (12.0, 55.0), (55.0, 75.0), (75.0, 100.0)]
NUM_CLASSES = 5
FUTURE = None
PAST = 5
NUM_TRIPS = None


def save_plans(trip_res_df, config, DATE):
    overall_vehicle_plan = {}
    DEFAULT_CAPACITY = config["capacities_of_regular_buses"]
    for vehicle_id, vehicle_df in trip_res_df.groupby("vehicle_id"):
        vehicle_df = vehicle_df.dropna(subset=["arrival_time"]).sort_values(["scheduled_time"])
        if config["limit_regular_bus_capacity"]:
            vehicle_capacity = DEFAULT_CAPACITY
        else:
            vehicle_capacity = vehicle_df.iloc[0].vehicle_capacity
            if np.isnan(vehicle_capacity):
                vehicle_capacity = DEFAULT_CAPACITY

        # Assume buses travel from the garage to the first stop in their trip at the start of the day
        starting_depot = "MCC5_1"
        service_type = "regular"
        blocks = [block for block in vehicle_df.block_abbr.unique().tolist()]
        trips = []
        got_first_trip_stop = False
        for block in blocks:
            block_df = vehicle_df.query("block_abbr == @block")
            for trip in block_df.trip_id.unique().tolist():
                trip_df = trip_res_df[trip_res_df["trip_id"] == str(trip)]
                stop_id_original = trip_df.stop_id_original.tolist()
                if len(stop_id_original) < 3:
                    continue
                trips.append((str(block), str(trip)))
                if not got_first_trip_stop:
                    if len(stop_id_original) < 3:
                        continue
                    starting_depot = stop_id_original[0]
                    got_first_trip_stop = True

        overall_vehicle_plan[vehicle_id] = {
            "vehicle_capacity": vehicle_capacity,
            "trips": trips,
            "starting_depot": starting_depot,
            "service_type": service_type,
        }

    OVERLOAD_BUSES = 5
    for vehicle_id in range(41, 41 + OVERLOAD_BUSES):
        overall_vehicle_plan[str(vehicle_id)] = {
            "vehicle_capacity": 55.0,
            "trips": [],
            "starting_depot": "MTA",
            "service_type": "overload",
        }

    if config["limit_regular_bus_capacity"]:
        with open(f'{CURR_DIR}/results/vehicle_plan_{DATE.replace("-", "")}_{vehicle_capacity}CAP.json', "w") as fp:
            json.dump(overall_vehicle_plan, fp, sort_keys=True, indent=2)
            logger.info(
                f'Saved vehicle json to {CURR_DIR}/results/vehicle_plan_{DATE.replace("-", "")}_{vehicle_capacity}CAP.json'
            )
    else:
        with open(f'{CURR_DIR}/results/vehicle_plan_{DATE.replace("-", "")}.json', "w") as fp:
            json.dump(overall_vehicle_plan, fp, sort_keys=True, indent=2)
            logger.info(f'Saved vehicle json to {CURR_DIR}/results/vehicle_plan_{DATE.replace("-", "")}.json')

    overall_block_plan = {}
    for block_abbr, block_df in trip_res_df.groupby("block_abbr"):
        block_df = block_df.dropna(subset=["arrival_time"]).sort_values(["scheduled_time"])
        trip_ids = block_df.trip_id.unique().tolist()
        start_time = (
            block_df[block_df["trip_id"] == trip_ids[0]].iloc[0]["scheduled_time"].strftime("%Y-%m-%d %H:%M:%S")
        )
        end_time = (
            block_df[block_df["trip_id"] == trip_ids[-1]].iloc[-1]["scheduled_time"].strftime("%Y-%m-%d %H:%M:%S")
        )
        overall_block_plan[block_abbr] = {"trip_ids": trip_ids, "start_time": start_time, "end_time": end_time}

    overall_trip_plan = {}
    for trip_id, trip_df in trip_res_df.groupby("trip_id"):
        trip_df = trip_df.dropna(subset=["arrival_time"]).sort_values(["scheduled_time"])
        route_id_dir = trip_df.iloc[0].route_id_dir
        block_abbr = int(trip_df.iloc[0].block_abbr)
        route_id = int(route_id_dir.split("_")[0])
        route_direction = route_id_dir.split("_")[1]
        zero_load_at_trip_end = trip_df.iloc[-1].zero_load_at_trip_end.tolist()
        scheduled_time = trip_df.scheduled_time.dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        stop_sequence = trip_df.stop_sequence.tolist()
        stop_sequence = list(range(0, len(stop_sequence)))
        stop_id_original = trip_df.stop_id_original.tolist()
        start_trip_time = scheduled_time[0]

        if len(stop_id_original) < 3:
            continue

        overall_trip_plan[trip_id] = {
            "route_id": route_id,
            "block_abbr": block_abbr,
            "route_direction": route_direction,
            "scheduled_time": scheduled_time,
            "stop_sequence": stop_sequence,
            "stop_id_original": stop_id_original,
            "zero_load_at_trip_end": zero_load_at_trip_end,
            "last_stop_sequence": stop_sequence[-1],
            "last_stop_id": stop_id_original[-1],
            "start_trip_time": start_trip_time,
        }

    with open(f'results/trip_plan_{DATE.replace("-", "")}.json', "w") as fp:
        json.dump(overall_trip_plan, fp, sort_keys=True, indent=2)
        logger.info(f'Saved trip json to results/trip_plan_{DATE.replace("-", "")}.json')


def generate_traffic_data_for_date(df, DATE, config, CHAINS):
    logger.info(f"DF has shape: {df.shape}")
    if CHAINS <= 0:
        CHAINS = 1
    logger.info(f"Start generate_traffic_data_for_date: {DATE}")
    # date_to_predict = dt.datetime.strptime(DATE, '%Y-%m-%d')
    df = df.query("overload_id == 0")
    df = df.dropna(subset=["arrival_time"])

    # HACK
    df = df.query("route_id != 94")
    df = df.query("route_id != 95")
    df = df.query("route_id != 89")
    df = df.query("route_id != 88")
    df = df[~df["stop_id_original"].isin(["PEARL", "JOHASHEN", "ROS10AEN", "PLSNTVW", "CLK E11"])]

    df = add_features(df)
    raw_df = deepcopy(df)

    input_df = prepare_input_data(df, ohe_encoder, label_encoders, num_scaler, columns, target="y_class")
    drop_cols = [ohe_column for ohe_column in ohe_columns if ohe_column in input_df.columns]
    input_df = input_df.drop(columns=drop_cols)

    if NUM_TRIPS == None:
        rand_trips = df.trip_id.unique().tolist()
    else:
        rand_trips = random.sample(df.trip_id.unique().tolist(), NUM_TRIPS)
    model = setup_simple_lstm_generator(input_df.shape[1], NUM_CLASSES)
    model.load_weights(latest)

    trip_res = []
    for trip_id in tqdm(rand_trips):
        _df = df.query("trip_id == @trip_id")
        try:
            _input_df = input_df.loc[_df.index]
            _, y_pred_probs = generate_simple_lstm_predictions(_input_df, model, PAST, FUTURE)

            # Introducing stochasticity
            y_pred = [np.random.choice(len(ypp.flatten()), size=1, p=ypp.flatten())[0] for ypp in y_pred_probs]
            loads = [random.randint(percentiles[yp][0], percentiles[yp][1]) for yp in y_pred]

            _raw_df = raw_df.loc[_df.index]
            y_true = _raw_df[0:PAST]["load"].tolist()
            a = y_true + loads
            _raw_df["sampled_loads"] = a

            y_true_classes = _raw_df[0:PAST]["y_class"].tolist()
            _raw_df["y_pred_classes"] = y_true_classes + y_pred
            _raw_df["y_pred_probs"] = [[-1] * NUM_CLASSES] * len(y_true_classes) + [ypp[0] for ypp in y_pred_probs]

            trip_res.append(_raw_df)
        except:
            logger.error(f"FAILED:{trip_id}")
            continue

    trip_res = pd.concat(trip_res)
    _columns = [
        "trip_id",
        "transit_date",
        "arrival_time",
        "scheduled_time",
        "block_abbr",
        "stop_sequence",
        "stop_id_original",
        "route_id_dir",
        "zero_load_at_trip_end",
        "y_pred_classes",
        "y_pred_probs",
        "sampled_loads",
        "vehicle_id",
        "vehicle_capacity",
    ]
    trip_res_df = trip_res[_columns]

    ##### FOR LIMITING TIME

    if config["limit_service_hours"]:
        start_time = config["limit_service_hours_start_time"]
        end_time = config["limit_service_hours_end_time"]
        start_datetime = dt.datetime.strptime(f"{DATE} {start_time}", "%Y-%m-%d %H:%M:%S")
        end_datetime = dt.datetime.strptime(f"{DATE} {end_time}", "%Y-%m-%d %H:%M:%S")

        arr = []
        for trip_id, trip_df in trip_res_df.groupby("trip_id"):
            if (trip_df.scheduled_time.min() >= start_datetime) and (trip_df.scheduled_time.max() <= end_datetime):
                if len(trip_df.index) > 1:
                    arr.append(trip_df)

        trip_res_df = pd.concat(arr)

    save_plans(trip_res_df=trip_res_df, config=config, DATE=DATE)

    for chain in tqdm(range(CHAINS)):
        loads = [random.randint(percentiles[yp][0], percentiles[yp][1]) for yp in trip_res_df.y_pred_classes]
        trip_res_df["sampled_loads"] = loads

        sampled_ons_offs = []
        for trip_id, trip_id_df in trip_res_df.groupby(["transit_date", "trip_id"]):
            tdf = trip_id_df.sort_values("stop_sequence").reset_index(drop=True)
            tdf["stop_sequence"] = list(range(1, len(tdf) + 1))
            tdf["ons"] = 0
            tdf["offs"] = 0
            tdf["next_load"] = tdf["sampled_loads"].shift(-1)

            # Intermediate stops
            tdf[["ons", "offs"]] = tdf.apply(compute_ons_offs, axis=1, result_type="expand")

            # first and last stops
            tdf.at[0, "ons"] = tdf.iloc[0]["next_load"]
            tdf.at[len(tdf) - 1, "offs"] = tdf.iloc[-1]["sampled_loads"]
            sampled_ons_offs.append(tdf)

        df = pd.concat(sampled_ons_offs)
        out_columns = [
            "route_id_dir",
            "block_id",
            "trip_id",
            "stop_sequence",
            "stop_id",
            "scheduled_time",
            "sampled_loads",
            "ons",
            "arrival_time",
            "offs",
        ]
        df = df.rename({"block_abbr": "block_id", "stop_id_original": "stop_id"}, axis=1)

        if chain == 0:
            # FIX arrival times equal to scheduled time when chains are not used.
            df["arrival_time"] = df["scheduled_time"]
            df[out_columns].to_parquet(f'results/sampled_ons_offs_dict_{DATE.replace("-", "")}.parquet')
            pass
        else:
            Path(f'results/chains/{DATE.replace("-","")}').mkdir(parents=True, exist_ok=True)
            df[out_columns].to_parquet(
                f'results/chains/{DATE.replace("-","")}/ons_offs_dict_chain_{DATE.replace("-","")}_{chain - 1}.parquet'
            )

        extra_label = reorganize_files(date=config["date"])
        if config.get("arrival_noise", False):
            if extra_label:
                dir = f"{CURR_DIR}/results/test_data/{config['date'].replace('-','')}_{extra_label}"
            else:
                dir = f"{CURR_DIR}/results/test_data/{config['date'].replace('-','')}"
            logger.debug(f"Start to add noise for: {dir}")
            add_noise_to_arrivals(dir=dir)

    return True


def generate_noisy_data_for_date(trip_res_df, DATE, config, CHAINS):
    save_plans(trip_res_df=trip_res_df, config=config, DATE=DATE)
    noise_levels = config["noise_pct"]

    logger.info(f"Number of noise levels: {len(noise_levels)}.")
    load_noise_range = 10
    for noise_level in noise_levels:
        logger.info(f"Processing noise level {noise_level:.2f}.")
        noise_percentage = noise_level / 100
        # Add noise to the numeric column

        logger.info(f"DF has shape: {trip_res_df.shape}")
        if CHAINS <= 0:
            CHAINS = 1
        logger.info(f"Start generate_traffic_data_for_date: {DATE}")
        # date_to_predict = dt.datetime.strptime(DATE, '%Y-%m-%d')

        for chain in tqdm(range(CHAINS)):
            logger.info(f"Starting chain: {chain}")

            sampled_ons_offs = []
            for trip_id, trip_id_df in trip_res_df.groupby(["transit_date", "trip_id"]):
                tdf = trip_id_df.sort_values("stop_sequence").reset_index(drop=True)

                # Add noise to the numeric column and ensure it's non-negative
                noise = rng.uniform(-noise_percentage, noise_percentage, len(tdf))
                noisy_values = tdf["load"] + noise * tdf["load"]
                noisy_values = np.floor(noisy_values)
                tdf["sampled_loads"] = np.clip(noisy_values, 0, None)

                tdf["stop_sequence"] = list(range(1, len(tdf) + 1))
                tdf["ons"] = 0
                tdf["offs"] = 0
                tdf["next_load"] = tdf["sampled_loads"].shift(-1)

                # Intermediate stops
                tdf[["ons", "offs"]] = tdf.apply(compute_ons_offs, axis=1, result_type="expand")

                # first and last stops
                tdf.at[0, "ons"] = tdf.iloc[0]["next_load"]
                tdf.at[len(tdf) - 1, "offs"] = tdf.iloc[-1]["sampled_loads"]
                sampled_ons_offs.append(tdf)

            df = pd.concat(sampled_ons_offs)
            out_columns = [
                "route_id_dir",
                "block_id",
                "trip_id",
                "stop_sequence",
                "stop_id",
                "scheduled_time",
                "sampled_loads",
                "ons",
                "arrival_time",
                "offs",
            ]
            df = df.rename({"block_abbr": "block_id", "stop_id_original": "stop_id"}, axis=1)
            if chain == 0:
                # FIX arrival times equal to scheduled time when chains are not used.
                df["arrival_time"] = df["scheduled_time"]
                df[out_columns].to_parquet(f'{CURR_DIR}/results/sampled_ons_offs_dict_{DATE.replace("-", "")}.parquet')
                logger.info(
                    f"Saving df to {CURR_DIR}/results/sampled_ons_offs_dict_noise_{noise_level}_{DATE.replace('-', '')}.parquet"
                )
            else:
                Path(f'{CURR_DIR}/results/chains/{DATE.replace("-","")}').mkdir(parents=True, exist_ok=True)
                df[out_columns].to_parquet(
                    f'{CURR_DIR}/results/chains/{DATE.replace("-","")}/ons_offs_dict_chain_{DATE.replace("-","")}_{chain - 1}.parquet'
                )
                logger.info(
                    f"Saving df to results/chains/{DATE.replace('-','')}/ons_offs_dict_noise_{noise_level}_chain_{DATE.replace('-','')}_{chain - 1}.parquet"
                )

        extra_label = reorganize_files(date=config["date"], extra_label=f"noise_{noise_level}")
        if config.get("arrival_noise", False):
            if extra_label:
                dir = f"{CURR_DIR}/results/test_data/{config['date'].replace('-','')}_{extra_label}"
            else:
                dir = f"{CURR_DIR}/results/test_data/{config['date'].replace('-','')}"
            logger.debug(f"Start to add noise for: {dir}")
            add_noise_to_arrivals(dir=dir)
    return True


def reorganize_files(date, extra_label=None):
    # Reorganize files after generation
    logger.info("Reorganizing files.")
    chains_dir = f"{CURR_DIR}/results/chains"
    Path(chains_dir).mkdir(parents=True, exist_ok=True)

    results_dir = f"{CURR_DIR}/results/test_data"
    date_str = date.replace("-", "")
    if extra_label:
        date_dir = f"{results_dir}/{date_str}_{extra_label}"
    else:
        date_dir = f"{results_dir}/{date_str}"
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)

    if os.path.isfile(f"{CURR_DIR}/results/sampled_ons_offs_dict_{date_str}.parquet"):
        os.rename(
            f"{CURR_DIR}/results/sampled_ons_offs_dict_{date_str}.parquet",
            f"{date_dir}/sampled_ons_offs_dict_{date_str}.parquet",
        )
    if os.path.isfile(f"results/trip_plan_{date_str}.json"):
        # os.rename(f"results/trip_plan_{date_str}.json", f"{date_dir}/trip_plan_{date_str}.json")
        shutil.copy(f"results/trip_plan_{date_str}.json", f"{date_dir}/trip_plan_{date_str}.json")

    if config["limit_regular_bus_capacity"]:
        shutil.copy(
            f"{CURR_DIR}/results/vehicle_plan_{date_str}_{config['capacities_of_regular_buses']}CAP.json",
            f"{date_dir}/vehicle_plan_{date_str}_{config['capacities_of_regular_buses']}CAP.json",
        )
    else:
        shutil.copy(f"{CURR_DIR}/results/vehicle_plan_{date_str}.json", f"{date_dir}/vehicle_plan_{date_str}.json")

    # if os.path.exists(f"{chains_dir}/{date_str}"):
    #     os.rename(f"{chains_dir}/{date_str}", f"{date_dir}/chains")

    source_dir = f"{chains_dir}/{date_str}"
    target_dir = f"{date_dir}/chains/"
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    if os.path.isdir(source_dir):
        file_names = os.listdir(source_dir)

        for file_name in file_names:
            shutil.move(os.path.join(source_dir, file_name), target_dir)

    return extra_label


def add_noise_to_arrivals(dir=None):
    paths = []
    for root, dirs, files in os.walk(dir):
        if "chains" in dirs:
            chain_dir = f"{root}/{dirs[0]}"
            for croot, cdirs, cfiles in os.walk(chain_dir):
                for file in cfiles:
                    if search(r"ons_offs_dict_chain.*\d{8}_\d+.parquet", file):
                        logger.debug(f"Starting: {file}")
                        _pkl = os.path.join(croot, file)
                        paths.append(_pkl)

    with Pool(processes=cpu_count() - 2) as pool:
        pool.map(process_to_parquet, paths)

        pool.close()
        pool.join()

    logger.info("Finished randomizing arrival times.")


def namespace_to_dict(namespace):
    return {k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v for k, v in vars(namespace).items()}


# Setting noise_pct to 0 and chains to 0 will be equivalent to real_world.
# noise_pct is for the load not arrival.
if __name__ == "__main__":
    print(os.getcwd())
    # Create the ArgumentParser object
    parser = argparse.ArgumentParser(description="Stationing and dispatch executor.")
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_level", type=str, default="DEBUG")
    parser.add_argument("-c", "--config", type=str, default="configs/test")
    args = parser.parse_args()
    args = namespace_to_dict(args)

    config_path = f'{args["config"]}.json'
    with open(config_path) as f:
        c = json.load(f)

        # config = {
        #     "number_of_overload_buses": 5,
        #     "capacities_of_overload_buses": 55,
        #     "limit_regular_bus_capacity": False,
        #     "capacities_of_regular_buses": 40,
        #     "is_date_range": False,
        #     "date": d,
        #     "start_date": "2022-10-01",
        #     "end_date": "2022-10-31",
        #     "frequency_h": 24,
        #     "chains": 50,
        #     "limit_service_hours": False,
        #     "limit_service_hours_start_time": "03:00:00",
        #     "limit_service_hours_end_time": "06:00:00",
        #     "send_mail": True,
        #     "use_generative_models": False,
        #     "noise_pct": [1, 5, 10],
        #     "arrival_noise": True,
        # }

    _start_date = c.get("start_date", "2022-10-01")
    _end_date = c.get("end_date", "2022-10-01")
    date_range = pd.date_range(_start_date, _end_date, freq="1D")
    date_range = [d.strftime("%Y-%m-%d") for d in date_range]

    # Date_range needs to be a list with elements in the format above STR "%Y-%m-%d"
    for d in date_range:
        start_time = time.time()
        logger.info(f"Generating data for {d}.")
        config = deepcopy(c)
        config["date"] = d
        # chains should be 1 .. N

        logger.info(f"Config: {config}")
        CHAINS = config["chains"]
        extra_label = None

        if config["is_date_range"]:
            dates = pd.date_range(config["start_date"], config["end_date"], freq=f'{config["frequency_h"]}h')
            dates = [dr.strftime("%Y-%m-%d") for dr in dates]
            apcdata = get_apc_data_for_daterange(config["start_date"], config["end_date"])
            df = apcdata.toPandas()
            for date in tqdm(dates):
                a = df.query("transit_date == @date")
                if config.get("use_generative_models", False):
                    extra_label = generate_traffic_data_for_date(a, date, config, CHAINS)
                else:
                    pass
        else:
            apcdata = get_apc_data_for_date(config["date"])
            df = apcdata.toPandas()
            if config.get("use_generative_models", False):
                extra_label = generate_traffic_data_for_date(df, config["date"], config, CHAINS)
            else:
                res = generate_noisy_data_for_date(df, config["date"], config, CHAINS)

        elapsed = time.time() - start_time
        if config["send_mail"]:
            try:
                utils.send_email(f"GENERATE_DAY_TRIPS: {d}", f"Done in: {elapsed} seconds.")
            except:
                pass

        logger.info(f"Done generating for {d}")

    logger.info("Done generating trips.")
