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
pd.set_option('display.max_columns', None)
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
# apc_data = spark.read.load('data/tt_to_next_stop_2022_04_31.parquet')
# apc_data.createOrReplaceTempView("apc")
from shapely import wkt
waze_df = pd.read_csv('data/nashville_2020_2022.csv')
waze_df['geometry'] = waze_df['geo'].apply(wkt.loads)
waze_df = gpd.GeoDataFrame(waze_df, crs='epsg:4326')
# waze_df['datetime'] = pd.to_datetime()
cols = ['date', 'hour', 'minute', 'second']
waze_df['datetime'] = waze_df[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
waze_df['datetime'] = pd.to_datetime(waze_df['datetime'], format="%Y-%m-%d %H %M %S")
waze_df = waze_df.query("type == 'ACCIDENT'")
waze_df.head(1)
listfiles = glob.glob('data/raw_gtfs/*.zip')
routes_arr = []
stops_arr = []
for lf in listfiles:
    feed = gk.read_feed(lf, dist_units='mi')
    routes = feed.geometrize_routes()
    stops = feed.geometrize_stops()
    routes_arr.append(routes)
    stops_arr.append(stops)
    
routes = pd.concat(routes_arr)
routes = routes.drop_duplicates()

stops = pd.concat(stops_arr)
stops = stops.drop_duplicates(subset='stop_id')
stops = stops.reset_index(drop=True)
stops
# maps
fp = os.path.join('data', 'shapefiles', "tncounty")
gdf_county = gpd.read_file(fp)
gdf_dav = gdf_county[gdf_county["NAME"] == "Davidson"]
# gdf_dav = gdf_dav.to_crs("EPSG:4326")
xmin, ymin, xmax, ymax = gdf_dav.total_bounds
gdf_dav.total_bounds
# gdf_dav.plot()
# GTFS

length = 5280 #feet
wide = 5280

cols = list(np.arange(xmin, xmax + wide, wide))
print(len(cols))
rows = list(np.arange(ymin, ymax + length, length))
print(len(rows))

polygons = []
for x in cols[:-1]:
    for y in rows[:-1]:
        polygons.append(Polygon([(x,y), (x+wide, y), (x+wide, y+length), (x, y+length)]))

grid = gpd.GeoDataFrame({'geometry':polygons})
grids = grid.set_crs("EPSG:2274")
dav_grids = gpd.overlay(gdf_dav, grids, how='intersection')

dav_grids['row_num'] = np.arange(len(dav_grids))
dav_grids2 = dav_grids.to_crs("EPSG:4326")
dav_grids2.plot()

fp = os.path.join('data', 'inrix_grouped.pkl')
inrix_grouped = pd.read_pickle(fp)
inrix_grouped = inrix_grouped.set_geometry('geometry')

gdf_dav = gdf_dav.to_crs("EPSG:4326")
inrix_grouped = inrix_grouped[inrix_grouped.within(gdf_dav.geometry.iloc[0])]
# Match incidents per grid
grid_incidents = {}
grid_stops = {}
for k, v in tqdm(dav_grids2.iterrows()):
    polygon = v['geometry']
    
    spatial_index = waze_df.sindex
    possible_matches_index = list(spatial_index.intersection(polygon.bounds))
    possible_matches = waze_df.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(polygon)]
    if not precise_matches.empty:
        grid_incidents[k] = precise_matches.index

    spatial_index = stops.sindex
    possible_matches_index = list(spatial_index.intersection(polygon.bounds))
    possible_matches = stops.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(polygon)]
    if not precise_matches.empty:
        grid_stops[k] = precise_matches.index
        
grids_with_incidents_and_roads = list(set(list(grid_incidents.keys())) & set(list(grid_stops.keys())))
grids_with_incidents_and_roads.sort()

import random
k = 63 # good example
# k = random.choice(list(grid_incidents.keys()))
incident_idx = grid_incidents[k]
poly = dav_grids2.iloc[k].geometry
ax = waze_df.loc[incident_idx].plot(markersize=20)
ax.plot(*poly.exterior.xy, color='red')
inrix_grouped[inrix_grouped.within(poly)].plot(ax=ax, color='k')

stops.loc[grid_stops[k]].plot(marker='^', ax=ax)
apc_data = spark.read.load('data/tt_to_next_stop_2022_04_31.parquet')
apc_data.createOrReplaceTempView("apc")

TOLERANCE = "10min"
dr = pd.date_range("2020-01-01 00:00:00", "2022-12-31 23:59:59", freq="10min")
dr = list(zip(dr, dr[1:]))

all_df = []
for k in grids_with_incidents_and_roads:
# for k in random.sample(grids_with_incidents_and_roads, 1):
    print(f"Grid:{k}", flush=True)
    _grid_incidents_idx = grid_incidents[k]
    _grid_stops_idx = grid_stops[k]
    
    _grid_stops = stops.loc[_grid_stops_idx]
    _grid_stops = _grid_stops.stop_id.tolist()
    _apc_data = apc_data.filter(F.col("stop_id_original").isin(_grid_stops))
    df = _apc_data.toPandas()
    
    _grid_incidents = waze_df.loc[_grid_incidents_idx]
    _grid_incidents = _grid_incidents.set_index('datetime').sort_index()
    _grid_incidents['incident'] = 1

    i = 0
    df_arr = []
    for _, v in df.groupby('stop_id_original'):
        v = v.dropna(subset='departure_time')
        v = v.set_index('departure_time')
        v = v.sort_index()
        v = pd.merge_asof(v, _grid_incidents[['street', 'date', 'hour', 'minute', 'second', 'subtype', 'confidence', 'reliability', 'magvar', 'geometry', 'incident']], 
                          left_index=True, right_index=True, tolerance=pd.Timedelta(TOLERANCE))
        v['incident'] = v['incident'].fillna(value=0, inplace=False)
        df_arr.append(v)

    if len(df_arr) > 0:
        merged_df = pd.concat(df_arr)
        merged_df.sort_index()
        merged_df = merged_df.query("tt_to_next_stop >= 0")
        merged_df = merged_df.dropna(subset='tt_to_next_stop')
        merged_df['stop_pairs'] = merged_df["stop_id_original"] + "_" + merged_df["next_stop_id"]
        
        df_arr = []
        for stop_pair, stop_pair_df in merged_df.groupby('stop_pairs'):
            a = stop_pair_df.resample("10min").agg({'tt_to_next_stop':'mean', 'incident':'max'}).dropna()
            a = a.loc[a['tt_to_next_stop'].shift() != a['tt_to_next_stop']]
            # a = stop_pair_df.resample("10min").agg({'tt_to_next_stop':'mean', 'incident':'max'}).bfill()
            a['stop_pair'] = stop_pair
            df_arr.append(a)
    
        if len(df_arr) > 0:
            merged_df = pd.concat(df_arr)
            merged_df['grid_id'] = k
            all_df.append(merged_df)
        
all_df = pd.concat(all_df)
print(all_df, flush=True)
all_df.to_parquet(path='data/processed_tt_incident_data_per_grid.parquet')