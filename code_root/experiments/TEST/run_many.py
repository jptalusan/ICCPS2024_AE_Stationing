import sys

BASE_DIR = "../../../code_root"
sys.path.append(BASE_DIR)

# All dates and times should just be datetime!
from DecisionMaking.Coordinator.DecisionMaker import DecisionMaker
from DecisionMaking.Coordinator.NearestCoordinator import NearestCoordinator
from DecisionMaking.Dispatch.SendNearestDispatchPolicy import SendNearestDispatchPolicy
from DecisionMaking.Dispatch.HeuristicDispatch import HeuristicDispatch
from DecisionMaking.DecisionEnvironmentDynamics import DecisionEnvironmentDynamics
from DecisionMaking.CentralizedMCTS.Rollout import BareMinimumRollout
from Environment.DataStructures.Bus import Bus
from Environment.DataStructures.Event import Event
from Environment.DataStructures.State import State
from Environment.DataStructures.Stop import Stop
from Environment.EmpiricalTravelModelLookup import EmpiricalTravelModelLookup
from Environment.enums import BusStatus, BusType, EventType, MCTSType, LogType
from Environment.EnvironmentModelFast import EnvironmentModelFast
from Environment.Simulator import Simulator
from src.utils import *
from src.deconstruct_results import *
import copy
import time
import logging
import dateparser
import ast
import json
import pickle
import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
from copy import deepcopy

_dir = "."


def load_initial_state(starting_date, bus_plan, trip_plan, config, random_seed=100):
    # print("Loading initial states...")
    active_stops = []

    _starting_date_str = dt.datetime.strptime(starting_date, "%Y%m%d").strftime("%Y-%m-%d")
    Buses = {}
    for bus_id, bus_info in bus_plan.items():
        bus_type = bus_info["service_type"]
        if bus_type == "regular":
            bus_type = BusType.REGULAR
            bus_starting_depot = bus_info["starting_depot"]
        else:
            if bus_id not in config["overload_start_depots"]:
                # print(f"Bus {bus_id} not in config.")
                continue
            if config["overload_start_depots"].get(bus_id) == "":
                # print(f"Bus {bus_id} not in config.")
                continue
            bus_type = BusType.OVERLOAD
            if config.get("overload_start_depots", False):
                bus_starting_depot = config["overload_start_depots"].get(bus_id, bus_info["starting_depot"])
            else:
                bus_starting_depot = bus_info["starting_depot"]

        bus_status = BusStatus.IDLE
        bus_capacity = bus_info["vehicle_capacity"]
        bus_block_trips = np.asarray(bus_info["trips"])

        if bus_type != BusType.OVERLOAD and len(bus_block_trips) <= 0:
            continue
        bus_block_trips = [tuple(bus_block_trip) for bus_block_trip in bus_block_trips]
        for i, bus_block_trip in enumerate(bus_block_trips):
            block_id = bus_block_trip[0]
            trip_id = bus_block_trip[1]
            trip_info = trip_plan[trip_id]
            stop_id_original = trip_info["stop_id_original"]
            # Make all MCC a single stop
            stop_id_original = ["MCC" if "MCC" in stop_id[0:3] else stop_id for stop_id in stop_id_original]

            active_stops.extend(stop_id_original)
            if i == 0:
                st = trip_plan[trip_id]["scheduled_time"]
                st = [str_timestamp_to_datetime(st).time().strftime("%H:%M:%S") for st in st][0]
                # Add when the bus should reach next stop as state change
                t_state_change = str_timestamp_to_datetime(f"{_starting_date_str} {st}")

        if "MCC" in bus_starting_depot[0:3]:
            bus_starting_depot = "MCC"

        bus = Bus(bus_id, bus_type, bus_status, bus_capacity, bus_block_trips)
        bus.current_stop = bus_starting_depot
        bus.starting_stop = bus_starting_depot
        bus.current_load = 0
        bus.t_state_change = t_state_change
        Buses[bus_id] = bus

    Stops = {}
    for active_stop in active_stops:
        stop = Stop(stop_id=active_stop)
        Stops[active_stop] = stop

    # print(f"Added {len(Buses)} buses and {len(Stops)} stops.")
    return Buses, Stops, active_stops


def load_events(travel_model, starting_date, Buses, trip_plan, event_file="", random_seed=100):
    datetime_str = dt.datetime.strptime(starting_date, "%Y%m%d")
    # print("Adding events...")
    np.random.seed(random_seed)

    # Initial events
    # Includes: Trip starts, passenger sampling
    # all active stops that buses will pass
    events = []
    saved_events = f"scenarios/testset/{datetime_str}/{event_file}"

    _starting_date_str = dt.datetime.strptime(starting_date, "%Y%m%d").strftime("%Y-%m-%d")

    active_trips = []
    pbar = Buses.items()
    # if not os.path.exists(saved_events):
    if True:
        for bus_id, bus in pbar:
            if bus.type == BusType.OVERLOAD:
                continue
            blocks_trips = bus.bus_block_trips

            if len(blocks_trips) <= 0:
                continue
            # Start trip (assuming trips are in sequential order)
            block = blocks_trips[0][0]
            trip = blocks_trips[0][1]
            st = trip_plan[trip]["scheduled_time"]
            st = [str_timestamp_to_datetime(st).time().strftime("%H:%M:%S") for st in st][0]
            first_stop_scheduled_time = str_timestamp_to_datetime(f"{_starting_date_str} {st}")
            current_block_trip = bus.bus_block_trips.pop(0)
            current_stop_number = bus.current_stop_number
            current_depot = bus.current_stop

            active_trips.append(current_block_trip[1])

            bus.current_block_trip = current_block_trip
            bus.current_stop_number = current_stop_number

            travel_time, distance = travel_model.get_traveltime_distance_from_depot(
                current_block_trip, current_depot, bus.current_stop_number, first_stop_scheduled_time
            )

            time_to_state_change = first_stop_scheduled_time + dt.timedelta(seconds=travel_time)
            bus.t_state_change = time_to_state_change
            bus.distance_to_next_stop = distance
            bus.status = BusStatus.IN_TRANSIT

            route_id_direction = travel_model.get_route_id_direction(current_block_trip)

            event = Event(
                event_type=EventType.VEHICLE_ARRIVE_AT_STOP,
                time=time_to_state_change,
                type_specific_information={
                    "bus_id": bus_id,
                    "current_block_trip": current_block_trip,
                    "stop": bus.current_stop_number,
                    "stop_id": bus.current_stop,
                    "route_id_direction": route_id_direction,
                },
            )
            events.append(event)

        events.sort(key=lambda x: x.time, reverse=False)

        # with open(saved_events, "wb") as f:
        #     pickle.dump(events, f)
    else:
        print("loading events...")
        with open(saved_events, "rb") as f:
            events = pickle.load(f)

    return events, active_trips


# TODO: Double check if all vehicles are there.
def manually_insert_disruption(buses, bus_id, time):
    if bus_id not in list(buses.keys()):
        # raise "Bus does not exist."
        return None

    # If overload bus, don't add
    if buses[bus_id].type == BusType.OVERLOAD:
        return None
    event = Event(event_type=EventType.VEHICLE_BREAKDOWN, time=time, type_specific_information={"bus_id": bus_id})
    return event


# TODO: Fix config json to add CHAIN_DIR
# Only load passenger events which will be visited by buses (to keep it fair)
# passenger_waiting_dict_list should be a 2D list (one list per chain) then just use the chain required.
# first element should be the real world. But what a
def load_passengers_events(Stops, active_stops, config, active_trips=[]):
    chain = config.get("pool_thread_count", 0)
    starting_date_str = config["starting_date_str"]
    noise_label = str(config.get("noise_level", ""))
    print(f"Start passenger data load.")
    start = time.time()
    list_of_stops = []
    all_ons = []
    # Chain + 1 so that the first chain belongs to the "real-world" chain
    for c in range(chain + 1):
        _stops = copy.deepcopy(Stops)
        # HACK Switched df for testing
        if not c:
            REALWORLD_DIR = f'{BASE_DIR}/scenarios/{config["real_world_dir"]}/{starting_date_str}'
            df = pd.read_parquet(f"{REALWORLD_DIR}/sampled_ons_offs_dict_{starting_date_str}.parquet")
            print(f"Using initial chain from {REALWORLD_DIR}.")
        else:
            if noise_label:
                MCTSWORLD_DIR = (
                    f'{BASE_DIR}/scenarios/{config["mcts_world_dir"]}/{starting_date_str}_noise_{noise_label}'
                )
            else:
                MCTSWORLD_DIR = f'{BASE_DIR}/scenarios/{config["mcts_world_dir"]}/{starting_date_str}'
            df = pd.read_parquet(f"{MCTSWORLD_DIR}/chains/ons_offs_dict_chain_{starting_date_str}_{c - 1}.parquet")

            print(f"Using chains from: {MCTSWORLD_DIR} with chain: {c - 1}")
        df["stop_id"] = np.where(df["stop_id"].str.startswith("MCC"), "MCC", df["stop_id"])
        df = df[df["stop_id"].isin(active_stops)]
        df = df[df["trip_id"].isin(active_trips)]

        # df = df[(df["ons"] > 0) | (df["offs"] > 0)]
        df = df.drop(columns=["sampled_loads", "next_load"], errors="ignore")
        all_ons.append(df["ons"].sum())
        for stop_id, stop_df in df.groupby("stop_id"):
            if stop_id not in active_stops:
                continue
            arrival_input = stop_df.to_dict(orient="records")
            _stops[stop_id].passenger_waiting_dict_list = arrival_input
        list_of_stops.append(_stops)
    elapsed = time.time() - start
    print(f"Done loading in {elapsed:0f} seconds")
    return list_of_stops


def manually_insert_allocation_events(bus_arrival_events, starting_date, buses, trip_plan, intervals=15):
    earliest_datetime = starting_date + dt.timedelta(days=2)
    latest_datetime = starting_date
    for bus_id, bus_obj in buses.items():
        if bus_obj.type == BusType.OVERLOAD:
            continue

        bus_block_trips = bus_obj.bus_block_trips + [bus_obj.current_block_trip]
        for bbt in bus_block_trips:
            trip = bbt[1]
            plan = trip_plan[trip]
            schedule = plan["scheduled_time"]
            first_stop = str_timestamp_to_datetime(schedule[0])
            last_stop = str_timestamp_to_datetime(schedule[-1])
            if first_stop <= earliest_datetime:
                earliest_datetime = first_stop
            if last_stop >= latest_datetime:
                latest_datetime = last_stop

    # Use actual times and dont round down/up
    # earliest_datetime = earliest_datetime - dt.timedelta(minutes=30)
    if latest_datetime.hour < 23:
        # latest_datetime = latest_datetime.replace(hour=latest_datetime.hour + 1, minute=0, second=0, microsecond=0)
        latest_datetime = latest_datetime + dt.timedelta(minutes=15)
    else:
        latest_datetime = latest_datetime.replace(hour=latest_datetime.hour, minute=59, second=0, microsecond=0)

    datetime_range = pd.date_range(earliest_datetime, latest_datetime, freq=f"{intervals}min")
    # datetime_range = np.arange(earliest_datetime, latest_datetime, np.timedelta64(intervals, 'm'))
    for _dt in datetime_range:
        event = Event(
            event_type=EventType.DECISION_ALLOCATION_EVENT, time=_dt, type_specific_information={"bus_id": None}
        )
        bus_arrival_events.append(event)

    bus_arrival_events.sort(key=lambda x: x.time, reverse=False)
    return bus_arrival_events


def load_disruption_chains(config, buses):
    chains = config.get("pool_thread_count", 0)
    disruptions_df = pd.read_parquet(f"./prepped_disruptions.parquet")
    date_str = dateparser.parse(config["starting_date_str"], date_formats=["%Y%m%d"]).strftime("%Y-%m-%d")
    disruptions_df = disruptions_df.query("date == @date_str")
    disruption_event_chains = []
    for chain in range(chains + 1):
        disruption_chain = []
        chain_disruption_df = disruptions_df[disruptions_df["chain"] == chain]
        if not chain_disruption_df.empty:
            dc = dict(zip(chain_disruption_df.bus_id, chain_disruption_df.scheduled_time))
            for bus_id, breakdown_datetime_str in dc.items():
                disruption_event = manually_insert_disruption(
                    buses=buses,
                    bus_id=bus_id,
                    time=str_timestamp_to_datetime(breakdown_datetime_str),
                )
                if disruption_event is not None:
                    disruption_chain.append(disruption_event)
        if len(disruption_chain) <= 0:
            disruption_event_chains.append([])
        else:
            disruption_event_chains.append(disruption_chain)
    return disruption_event_chains


# Almost the same as the main function.
def run_simulation(config):
    LOOKUP_DIR = f"{BASE_DIR}/scenarios"
    logger = logging.getLogger("debuglogger")
    starting_date_str = config["starting_date_str"]
    REALWORLD_DIR = f'{LOOKUP_DIR}/{config["real_world_dir"]}/{starting_date_str}'

    vehicle_count = config["vehicle_count"]
    starting_date_str = config["starting_date_str"]
    trip_plan_path = f"{REALWORLD_DIR}/trip_plan_{starting_date_str}.json"

    with open(trip_plan_path) as f:
        trip_plan = json.load(f)

    log(logger, dt.datetime.now(), json.dumps(config), LogType.INFO)

    if vehicle_count != "":
        bus_plan_path = f"{REALWORLD_DIR}/vehicle_plan_{starting_date_str}_{vehicle_count}.json"
    else:
        bus_plan_path = f"{REALWORLD_DIR}/vehicle_plan_{starting_date_str}.json"

    with open(bus_plan_path) as f:
        bus_plan = json.load(f)

    travel_model = EmpiricalTravelModelLookup(LOOKUP_DIR, starting_date_str, config=config)

    start_time = time.time()
    elapsed_time = time.time() - start_time
    print(f"Lookup loading time: {elapsed_time:.2f} s")

    # TODO: Move to environment model once i know it works
    valid_actions = None

    starting_date = dt.datetime.strptime(starting_date_str, "%Y%m%d")

    Buses, Stops, active_stops = load_initial_state(starting_date_str, bus_plan, trip_plan, config, random_seed=100)
    # print(f"Count buses: {len(Buses)}")

    bus_arrival_events, active_trips = load_events(travel_model, starting_date_str, Buses, trip_plan)

    stop_chains = load_passengers_events(Stops, active_stops, config, active_trips=active_trips)

    # Load disruption chains
    disruption_chains = load_disruption_chains(config, buses=Buses)

    # Adding interval events
    if config["reallocation"]:
        before_count = len(bus_arrival_events)
        bus_arrival_events = manually_insert_allocation_events(
            bus_arrival_events, starting_date, Buses, trip_plan, intervals=60
        )
        after_count = len(bus_arrival_events)

        log(logger, dt.datetime.now(), f"Initial interval decision events: {after_count - before_count}", LogType.INFO)

    # Chain 0 is the "real world" chain
    bus_arrival_events.extend(disruption_chains[0])
    bus_arrival_events.sort(key=lambda x: x.time, reverse=False)

    # HACK because of my weird simulation event pop, duplicate the first event
    bus_arrival_events.insert(0, bus_arrival_events[0])

    starting_state = copy.deepcopy(
        State(stops=stop_chains[0], buses=Buses, bus_events=bus_arrival_events, time=bus_arrival_events[0].time)
    )

    starting_state.stop_chains = stop_chains
    starting_state.disruption_chains = disruption_chains

    mcts_discount_factor = config["mcts_discount_factor"]
    lookahead_horizon_delta_t = config["lookahead_horizon_delta_t"]
    rollout_horizon_delta_t = config["rollout_horizon_delta_t"]
    uct_tradeoff = config["uct_tradeoff"]
    pool_thread_count = config["pool_thread_count"]
    iter_limit = config["iter_limit"]
    allowed_computation_time = config["allowed_computation_time"]
    mcts_type = MCTSType.MODULAR_MCTS

    # Class setup
    dispatch_policy = SendNearestDispatchPolicy(travel_model)  # RandomDispatch(travel_model)
    rollout_policy = BareMinimumRollout(rollout_horizon_delta_t, dispatch_policy=dispatch_policy)

    heuristic_dispatch = HeuristicDispatch(travel_model)
    mdp_environment_model = DecisionEnvironmentDynamics(
        travel_model, dispatch_policy=heuristic_dispatch, config=config
    )

    sim_environment = EnvironmentModelFast(travel_model, config)
    if config["method"].upper() == "MCTS":
        decision_maker = DecisionMaker(
            environment_model=sim_environment,
            travel_model=travel_model,
            dispatch_policy=None,
            pool_thread_count=pool_thread_count,
            mcts_type=mcts_type,
            discount_factor=mcts_discount_factor,
            mdp_environment_model=mdp_environment_model,
            rollout_policy=rollout_policy,
            uct_tradeoff=uct_tradeoff,
            iter_limit=iter_limit,
            lookahead_horizon_delta_t=lookahead_horizon_delta_t,
            allowed_computation_time=allowed_computation_time,  # 5 seconds per thread
            starting_date=starting_date_str,
            oracle=config["oracle"],
            base_dir=f"{BASE_DIR}/scenarios",
            config=config,
        )
    elif config["method"].upper() == "BASELINE":
        decision_maker = NearestCoordinator(travel_model=travel_model, dispatch_policy=dispatch_policy, config=config)

    simulator = Simulator(
        starting_event_queue=copy.deepcopy(bus_arrival_events),
        starting_state=starting_state,
        environment_model=sim_environment,
        event_processing_callback=decision_maker.event_processing_callback_funct,
        valid_actions=valid_actions,
        config=config,
        travel_model=travel_model,
    )
    start_time = time.time()
    score = simulator.run_simulation()
    print(score)
    elapsed = time.time() - start_time

    csvlogger = logging.getLogger("csvlogger")

    csvlogger.info(f"Simulator run time: {elapsed:.2f} s")
    print(f"Simulator run time: {elapsed:.2f} s")
    return score


def read_last_n_lines(file_path, n):
    with open(file_path, "r") as file:
        lines = file.readlines()
        last_n_lines = lines[-n:]
    return last_n_lines


def setup_and_run(config):
    filename = config["mcts_log_name"]
    exp_log_path = f"{_dir}/logs/{filename}"
    exp_res_path = f"{_dir}/results/{filename}"
    Path(exp_log_path).mkdir(parents=True, exist_ok=True)
    Path(exp_res_path).mkdir(parents=True, exist_ok=True)

    logFormatter = logging.Formatter("[%(asctime)s][%(levelname)-.1s] %(message)s", "%m-%d %H:%M:%S")
    logger = logging.getLogger("debuglogger")
    logger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(f"{exp_log_path}/stream.log", mode="w")
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.DEBUG)

    csvlogger = logging.getLogger("csvlogger")
    csvlogger.setLevel(logging.DEBUG)
    res_file = f"{exp_res_path}/results.csv"
    csvFormatter = logging.Formatter("%(message)s")
    csvHandler = logging.FileHandler(f"{res_file}", mode="w")
    csvHandler.setFormatter(csvFormatter)
    csvHandler.setLevel(logging.DEBUG)
    csvlogger.addHandler(csvHandler)
    # csvlogger.addHandler(logging.NullHandler())

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(logFormatter)
    streamHandler.setLevel(logging.DEBUG)

    if config.get("save_debug_log", False):
        logger.addHandler(fileHandler)
    else:
        logger.addHandler(logging.NullHandler())

    if config.get("show_stream_log", False):
        logger.addHandler(streamHandler)

    logger.debug("Starting process.")

    # config["pool_thread_count"] = 0
    config["save_debug_log"] = True
    run_simulation(config)

    ### Sending stats over email.
    n_lines = 9
    stats = []
    last_n_lines = read_last_n_lines(res_file, n_lines)
    for line in last_n_lines:
        stats.append(line.strip())  # Print or process each line as needed
    stats = "\n".join(stats)

    if config.get("send_mail", False):
        emailer(config_dict=config, msg_content=stats)

    stops_df, bus_df = logs_to_df(res_file)
    stops_df.to_csv(f"{exp_res_path}/stops_results.csv")
    bus_df.to_csv(f"{exp_res_path}/buses_results.csv")

    logger.debug("Finished experiment")

    from importlib import reload

    logging.shutdown()
    reload(logging)
    pass


def generate_plots(config_arr):
    # Loop through all configs and generate union of all DF
    all_buses = []
    all_stops = []

    for config in config_arr:
        filename = config["mcts_log_name"]
        starting_date_str = config["starting_date_str"]
        exp_log_path = f"{_dir}/logs/{filename}"
        exp_res_path = f"{_dir}/results/{filename}"
        bus_results_path = f"{exp_res_path}/buses_results.csv"
        stops_results_path = f"{exp_res_path}/stops_results.csv"

        buses_df = pd.read_csv(bus_results_path, index_col=0)
        stops_df = pd.read_csv(stops_results_path, index_col=0)

        buses_df["filename"] = filename
        stops_df["filename"] = filename

        buses_df["date"] = starting_date_str
        stops_df["date"] = starting_date_str

        all_buses.append(buses_df)
        all_stops.append(stops_df)

    all_buses_df = pd.concat(all_buses)
    all_stops_df = pd.concat(all_stops)

    all_buses_df.to_csv(f"{_dir}/plots/all_buses.csv")
    all_stops_df.to_csv(f"{_dir}/plots/all_stops.csv")
    pass


if __name__ == "__main__":
    logger = logging.getLogger("runner_logger")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    # Params
    run_execute = True
    date_list = ["20201023", "20210211", "20200505"]
    method_list = ["baseline"]  # ["baseline", "mcts"]
    overload_start_depots_option = ["garage", "agency", "search"]  # ["garage", "agency", "search", "mcts"]
    real_world_dirs = ["TEST_WORLD"]  # ["TEST_WORLD", "REAL_WORLD"]
    mcts_world_dirs = ["TEST_WORLD"]  # ["TEST_WORLD", "REAL_WORLD"]
    noise_levels = [0]  # 0, 1, 5, 10
    parallel = False
    run_plotting = True
    send_email = True

    # Needed data
    # Best stationing results
    best_solutions_searched = f"{BASE_DIR}/../data_analysis/data/best_solutions.parquet"
    searched_df = pd.read_parquet(best_solutions_searched)
    searched_df["solution"] = searched_df["solution"].apply(ast.literal_eval)
    subsitute_buses_ids = ["41", "42", "43", "44", "45"]

    # Base config
    base_config = {
        "starting_date_str": "20231018",
        "real_world_dir": "TEST_WORLD",
        "mcts_world_dir": "TEST_WORLD",
        "noise_level": "",
        "iter_limit": 1,
        "pool_thread_count": 1,
        "mcts_discount_factor": 0.99997,
        "uct_tradeoff": 1000,
        "lookahead_horizon_delta_t": 900,
        "rollout_horizon_delta_t": 900,
        "allowed_computation_time": 15,
        "vehicle_count": "",
        "oracle": False,
        "method": "baseline",
        "save_metrics": False,
        "send_mail": send_email,
        "reallocation": True,
        "early_end": False,
        # "early_end": "2021-02-11 12:00:00",
        "scenario": "1B",
        "mcts_log_name": "test",
        "save_debug_log": True,
        "show_stream_log": False,
        "decision_interval": 5,
        "default_vehicle_capacity": 40,
        "overage_threshold": 0.05,
        "passenger_time_to_leave_min": 30,
        "overload_start_depots": {"41": "MTA", "42": "MTA", "43": "MTA", "44": "MTA", "45": "MTA"},
    }

    incomplete_data = []
    config_arr = []

    for date in date_list:
        for method in method_list:
            for depot in overload_start_depots_option:
                for real_world_dir in real_world_dirs:
                    for mcts_world_dir in mcts_world_dirs:
                        for noise in noise_levels:
                            # Only baseline can use these, mcts defaults to mcts depots
                            if method == "baseline":
                                if depot == "agency":
                                    depots = ["MCC", "MCC", "MCSRVRF", "FAILEWN", "MXITHOMP"]
                                elif depot == "garage":
                                    depots = ["MTA", "MTA", "MTA", "MTA", "MTA"]
                                elif depot == "hub":
                                    depots = ["MCC", "MCC", "MCC", "MCC", "MCC"]
                                elif depot == "mcts":
                                    continue
                                elif depot == "search":
                                    try:
                                        depots = searched_df.loc[searched_df["date"] == date].solution.iloc[0]
                                    except:
                                        # raise Exception(f"Date: {date} has no valid searched depots.")
                                        logger.error(f"Date: {date} has no valid searched depots.")
                                        incomplete_data.append(date)
                                        continue
                            elif method == "mcts":
                                if depot != "mcts":
                                    continue
                                else:
                                    depots = ["MTA", "MTA", "MTA", "MTA", "MTA"]
                            else:
                                raise Exception(f"Depot: {depot} is not valid a depot option.")

                            log_name = f"{date}_{method}_{depot}_{real_world_dir.split('_')[0]}_{mcts_world_dir.split('_')[0]}_{noise}"
                            if noise == 0:
                                noise = ""

                            _config = deepcopy(base_config)
                            _config_depots = dict(zip(subsitute_buses_ids, depots))
                            _config["overload_start_depots"] = _config_depots
                            _config["starting_date_str"] = date
                            _config["real_world_dir"] = real_world_dir
                            _config["mcts_world_dir"] = mcts_world_dir
                            _config["method"] = method
                            _config["noise_level"] = noise
                            _config["mcts_log_name"] = log_name

                            if method != "mcts":
                                _config["send_mail"] = False

                            config_arr.append(_config)

    logger.debug(len(config_arr))

    if run_execute:
        logger.debug(f"Starting execution")
        for config in config_arr:
            date = config["starting_date_str"]
            if date in incomplete_data:
                logger.debug(f"Date: {date} is not run since it is missing search data.")
                continue

            logger.debug(f"Starting: {config['mcts_log_name']}")
            setup_and_run(config)
            logger.debug(f"\tFinished: {config['mcts_log_name']}")
        logger.debug(f"Finished execution")
    else:
        logger.debug(f"Not running execution")

    if run_plotting:
        logger.debug(f"Starting plotting")
        exp_plot_path = f"{_dir}/plots"
        Path(exp_plot_path).mkdir(parents=True, exist_ok=True)

        generate_plots(config_arr)
        logger.debug(f"Finished plotting")
    else:
        logger.debug(f"Not running plotting")

    logger.debug(f"Finished script")
