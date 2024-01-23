# nohup python3 -u process.py > process.out &
# process.out -> 784123 final count lines
import pandas as pd
import geopandas as gpd
import os
import gtfs_kit as gk
import osmnx as ox
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from shapely.geometry import Polygon, LineString
import warnings
warnings.filterwarnings('ignore')
import gtfs_kit as gk
import glob
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
from pyspark import SparkContext,SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F
from pyspark import SparkConf
import pandas as pd
import pickle
from tqdm import tqdm
spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '40g')\
        .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '40g').master("local[26]")\
        .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .config("spark.sql.autoBroadcastJoinThreshold", -1)\
        .config("spark.driver.maxResultSize", 0)\
        .config("spark.shuffle.spill", "true")\
        .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")\
        .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")\
        .config("spark.ui.showConsoleProgress", "false")\
        .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
apc_data_path = 'data/apc_weather_gtfs_20221216.parquet'
apc_data = spark.read.load(apc_data_path)
apc_data.createOrReplaceTempView("apc")

get_columns = ['transit_date', 'route_id', 'trip_id', 'stop_id_original', 'stop_sequence', 'load', 'arrival_time', 'departure_time', 'scheduled_time']
get_str = ",".join([c for c in get_columns])

query = f"""
        SELECT {get_str}
        FROM apc
        WHERE transit_date < '2022-05-01'
        """
        
apc_data = spark.sql(query)
apc_data = apc_data.filter(F.col("load") >= 0)
apc_data = apc_data.orderBy(F.col("transit_date"))
df = apc_data.toPandas()
df_arr = []
i = 0
curr_month = set()
print("Starting...", flush=True)

for (transit_date, trip_id), trip_df in df.groupby(['transit_date', 'trip_id']):
    tdf = trip_df.sort_values(['stop_sequence'])
    tdf['next_stop_id'] = tdf['stop_id_original'].shift(-1)
    tdf['next_arrival'] = tdf['arrival_time'].shift(-1)
    tdf['tt_to_next_stop'] = tdf['next_arrival'] - tdf['departure_time']
    tdf['tt_to_next_stop'] = tdf['tt_to_next_stop'].dt.total_seconds()
    df_arr.append(tdf)
    print(transit_date, trip_id, flush=True)
    x = str(transit_date.year) + '_' + str(transit_date.month)
    if x not in curr_month:
        print(f"Starting on: {x}", flush=True)
        curr_month.add(x)
        
_df = pd.concat(df_arr)
_df = _df.drop('arrival_time', axis=1)
_df = _df.reset_index(drop=True)
_df.to_parquet('data/tt_to_next_stop_2022_04_31.parquet')

print("Done...", flush=True)