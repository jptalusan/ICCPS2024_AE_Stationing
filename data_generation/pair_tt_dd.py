"""
This code generate all stop pairs' travel time and distance.
It requires the following:
- stops_node_matching.pkkl
- davidson_graph.graphml

Generates:
- stops_tt_dd_dict.pkl
"""

import os
import osmnx as ox
import networkx as nx
import pandas as pd
import itertools
import pickle
from multiprocessing import Process, Queue, cpu_count, Manager
from time import sleep

queue = Queue()


def mp_worker(L, queue):
    while queue.qsize() > 0:
        record = queue.get()
        # current_stop = record[0][0]
        # next_stop = record[0][1]

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

        fp = os.path.join("/home/jptalusan/mta_simulator/code/data/pair_dd_tt.pkl")
        with open(fp, "wb") as f:
            pickle.dump(list(L), f)

        # print(L)


if __name__ == "__main__":
    fp = "/home/jptalusan/mta_simulator/code/data/stops_node_matching.pkl"
    stop_nodes = pd.read_pickle(fp)
    # stop_nodes = stop_nodes[0:3]
    fp = "/home/jptalusan/mta_simulator/code/data/davidson_graph.graphml"
    G = ox.load_graphml(fp)

    all_stops_list = stop_nodes.stop_id_original.tolist()
    all_nodes_list = stop_nodes.nearest_node.tolist()
    all_stops_combos = list(itertools.product(all_stops_list, repeat=2))
    all_nodes_combos = list(itertools.product(all_nodes_list, repeat=2))

    for pair in all_nodes_combos:
        queue.put(pair)

    mp_handler()

    fp = os.path.join("/home/jptalusan/mta_simulator/code/data/pair_dd_tt.pkl")
    with open(fp, "rb") as f:
        pair_dd_tt = pickle.load(f)
    # [print(all_stops_combos[i], r) for i, r in enumerate(res)]

    # Setting it up with stop names

    stops_node_matching = pd.read_pickle("results/stops_node_matching.pkl")

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

    df.to_pickle("results/pair_tt_dd_stops.pkl")

    # Creating a dictionary
    df["key_pair"] = list(zip(df.current_stop, df.next_stop))
    df = df.set_index("key_pair")
    df = df.drop_duplicates(subset=["current_node", "next_node", "current_stop", "next_stop"])
    stops_tt_dd_dict = df.drop(["current_node", "next_node", "current_stop", "next_stop"], axis=1).to_dict("index")
    with open("results/stops_tt_dd_dict.pkl", "wb") as handle:
        pickle.dump(stops_tt_dd_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('results/stops_tt_dd_dict.pkl', 'rb') as handle:
    #     stops_tt_dd_dict = pickle.load(handle)
    # print(stops_tt_dd_dict)
