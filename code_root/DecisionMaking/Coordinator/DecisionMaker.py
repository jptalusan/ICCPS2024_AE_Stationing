import copy
import time
import pandas as pd
import numpy as np
import datetime as dt
from multiprocessing import Pool
from Environment.enums import BusStatus, BusType, ActionType
from DecisionMaking.CentralizedMCTS.ModularMCTS import ModularMCTS
from src.utils import *
import math
import logging

"""
Combine the two files here:
/media/seconddrive/JP/gits/EMS_DSS/code_root/decision_making/LowLevel/CentralizedMCTS/LowLevelCentMCTSPolicy.py
/media/seconddrive/JP/gits/EMS_DSS/code_root/decision_making/coordinator/LowLevelCoordTest.py

# Here you define the solver which is the ModularMCTS

# The output here is the action (which is obtained by running MCTS solver)
"""


def run_low_level_mcts(arg_dict):
    """
    arg dict needs:
    current_state,
    event_queue,
    iter_limit,
    allowed_compu_time,
    exploration_constant,
    discount_factor,
    rollout_policy,
    mdp_environment,
    MCTS_type
    :param arg_dict:
    :return:
    """

    solver = ModularMCTS(
        mdp_environment_model=arg_dict["mdp_environment_model"],
        discount_factor=arg_dict["discount_factor"],
        iter_limit=arg_dict["iter_limit"],
        allowed_computation_time=arg_dict["allowed_computation_time"],
        rollout_policy=arg_dict["rollout_policy"],
        exploit_explore_tradoff_param=arg_dict["exploit_explore_tradoff_param"],
        action_type=arg_dict["action_type"],
    )

    res = solver.solve(arg_dict["current_state"], arg_dict["bus_arrival_events"])

    return {"region_id": arg_dict["tree_number"], "mcts_res": res}


class DecisionMaker:
    def __init__(
        self,
        environment_model,
        travel_model,
        dispatch_policy,
        pool_thread_count,
        mcts_type,
        discount_factor,
        mdp_environment_model,
        rollout_policy,
        uct_tradeoff,
        iter_limit,
        lookahead_horizon_delta_t,
        allowed_computation_time,
        starting_date,
        oracle,
        base_dir,
        config,
    ):
        self.environment_model = environment_model
        self.travel_model = travel_model
        self.dispatch_policy = dispatch_policy
        self.event_counter = 0
        self.pool_thread_count = pool_thread_count
        self.mcts_type = mcts_type

        self.discount_factor = discount_factor
        self.mdp_environment_model = mdp_environment_model
        self.rollout_policy = rollout_policy
        self.uct_tradeoff = uct_tradeoff
        self.iter_limit = iter_limit
        self.allowed_computation_time = allowed_computation_time
        self.lookahead_horizon_delta_t = lookahead_horizon_delta_t
        self.oracle = oracle
        self.base_dir = base_dir
        self.config = config

        self.starting_date = starting_date
        self.time_taken = {}

        self.action_type = None
        self.logger = logging.getLogger("debuglogger")

    # Call the MCTS in parallel here
    def get_trips_with_remaining_passengers(self, state, limit=None):
        VEHICLE_CAPACITY = int(self.config.get("default_vehicle_capacity", 40))
        OVERAGE_THRESHOLD = float(self.config.get("overage_threshold", 0.05))
        trips_with_remaining = []
        # for stop_id, stop_obj in state.stops.items():
        for p_set in state.people_left_behind:
            if p_set.get("left_behind", False):
                remaining_passengers = p_set["ons"]
                if remaining_passengers >= (VEHICLE_CAPACITY * OVERAGE_THRESHOLD):
                    block_id = p_set["block_id"]
                    trip_id = p_set["trip_id"]
                    arrival_time = p_set["arrival_time"]
                    if arrival_time + pd.Timedelta(30, unit="min") < state.time:
                        continue
                    block_trip = (block_id, str(trip_id))
                    if block_trip in state.served_trips:
                        continue
                    trips_with_remaining.append(
                        (
                            remaining_passengers,
                            block_trip,
                        )
                    )
        if limit:
            trips_with_remaining = sorted(trips_with_remaining, key=lambda x: x[0], reverse=True)
            trips_with_remaining = trips_with_remaining[0 : limit - 1]
        trips_with_remaining = [i for i in trips_with_remaining if i]
        return trips_with_remaining

    # Check if current stop and scheduled times are timepoints, only do decisions then.
    def event_processing_callback_funct(self, actions, state, action_type):
        # Only do something when buses are available?

        if len(self.get_trips_with_remaining_passengers(state)) <= 0 and action_type == ActionType.OVERLOAD_DISPATCH:
            # print(f"Event counter: {self.event_counter}")
            # print(f"Event: {state.bus_events[0]}")
            # print(f"Time: {state.time}")
            # print("no incidents detected")
            # print()
            return None

        if self.any_available_overload_buses(state):
            self.action_type = action_type
            self.event_counter += 1
            chosen_action = self.process_mcts(state)
            if chosen_action is None:
                return None
            return chosen_action
        else:
            # print(f"Event counter: {self.event_counter}")
            # print(f"Event: {state.bus_events[0]}")
            # print(f"Time: {state.time}")
            # print("no available buses")
            # print()
            return None

    def any_available_overload_buses(self, state):
        num_available_buses = len(
            [
                _
                for _ in state.buses.values()
                if ((_.status == BusStatus.IDLE) or (_.status == BusStatus.ALLOCATION)) and _.type == BusType.OVERLOAD
            ]
        )
        return num_available_buses > 0

    def process_mcts(self, state):
        ORACLE = self.oracle
        if ORACLE:
            CHAINS = 0
            event_queues = self.load_events(state)
            states = [state]
        else:
            CHAINS = self.pool_thread_count
            event_queues = self.get_event_chains(state, CHAINS)
            if CHAINS > 0:
                event_queues = event_queues * CHAINS
            # states = [state] * CHAINS
            states = []
            # TODO Can probably prune this based on state.time so it doesn't copy all
            # chain == 0 is real world chain so its not used here.
            for chain in range(1, CHAINS + 1):
                _state = copy.deepcopy(state)
                _state.stops = copy.deepcopy(state.stop_chains[chain - 1])
                states.append(_state)

        if len(event_queues[0]) <= 0:
            return None

        result = self.get_action(states, event_queues)
        return result

    def get_action(self, states, event_queues, passenger_arrival_distribution=None):
        # print(event_queues)
        final_action = {}
        # For display
        sorted_actions = []
        decision_start = time.time()

        if self.pool_thread_count == 0:
            res_dict = []
            inputs = self.get_mcts_inputs(
                states=states,
                bus_arrival_events=event_queues,
                discount_factor=self.discount_factor,
                mdp_environment_model=self.mdp_environment_model,
                rollout_policy=self.rollout_policy,
                uct_tradeoff=self.uct_tradeoff,
                iter_limit=self.iter_limit,
                allowed_computation_time=self.allowed_computation_time,
                mcts_type=self.mcts_type,
                action_type=self.action_type,
            )

            for input in inputs:
                result = run_low_level_mcts(input)
                res_dict.append(result)

            best_actions = dict()

            for i in range(len(res_dict)):
                results = [_["mcts_res"] for _ in res_dict if _["region_id"] == i]
                actions = [_["action"] for _ in results[0]["scored_actions"]]

                all_action_scores = []
                for action in actions:
                    action_scores = []
                    action_visits = []
                    for result in results:
                        action_score = next((_ for _ in result["scored_actions"] if _["action"] == action), None)
                        action_scores.append(action_score["score"])
                        action_visits.append(action_score["num_visits"])

                    all_action_scores.append({"action": action, "scores": action_scores, "num_visits": action_visits})

                avg_action_scores = list()
                for res in all_action_scores:
                    avg_action_scores.append(
                        {
                            "action": res["action"],
                            "avg_score": np.mean(res["scores"]),
                            "num_visits": np.mean(res["num_visits"]),
                        }
                    )

                # We want the actions which result in the least passengers left behind
                best_actions[i] = max(avg_action_scores, key=lambda _: _["avg_score"])

            best_score = -math.inf
            overall_best_action = None
            for _, actions in best_actions.items():
                if actions["avg_score"] >= best_score:
                    best_score = actions["avg_score"]
                    overall_best_action = actions["action"]
            final_action = overall_best_action

        else:
            start_pool_time = time.time()
            with Pool(processes=self.pool_thread_count) as pool:
                pool_creation_time = time.time() - start_pool_time

                inputs = self.get_mcts_inputs(
                    states=states,
                    bus_arrival_events=event_queues,
                    discount_factor=self.discount_factor,
                    mdp_environment_model=self.mdp_environment_model,
                    rollout_policy=self.rollout_policy,
                    uct_tradeoff=self.uct_tradeoff,
                    iter_limit=self.iter_limit,
                    allowed_computation_time=self.allowed_computation_time,
                    mcts_type=self.mcts_type,
                    action_type=self.action_type,
                )

                # run_start_ = time.time()
                res_dict = pool.map(run_low_level_mcts, inputs)

            best_actions = dict()

            all_actions = []

            for i in range(len(res_dict)):
                results = [_["mcts_res"] for _ in res_dict if _["region_id"] == i]
                actions = [_["action"] for _ in results[0]["scored_actions"]]

                for action in actions:
                    for result in results:
                        action_score = next((_ for _ in result["scored_actions"] if _["action"] == action), None)
                        if action not in [_a["action"] for _a in all_actions]:
                            all_actions.append(
                                {
                                    "action": action,
                                    "scores": [action_score["score"]],
                                    "visits": [action_score["num_visits"]],
                                }
                            )
                        else:
                            for _a in all_actions:
                                if _a["action"] == action:
                                    _a["scores"].append(action_score["score"])
                                    _a["visits"].append(action_score["num_visits"])

            avg_action_scores = list()
            for action in all_actions:
                avg_action_scores.append(
                    {
                        "action": action["action"],
                        "avg_score": np.mean(action["scores"]),
                        "sum_visits": np.sum(action["visits"]),
                    }
                )

            final_action = max(avg_action_scores, key=lambda _: _["avg_score"])["action"]

        self.time_taken["decision_maker"] = time.time() - decision_start

        # sorted_actions = res_dict[0]['mcts_res']['scored_actions']
        # sorted_actions.sort(key=lambda _: _['score'], reverse=True)
        avg_action_scores.sort(key=lambda _: _["avg_score"], reverse=True)
        time_taken = res_dict[0]["mcts_res"]["time_taken"]

        # print(f"Event counter: {self.event_counter}")
        # print(f"Event: {event_queues[0][0]}")
        # print(f"Time: {states[0].time}")
        # if self.pool_thread_count == 0:
        #     [print(f"{sa['action']}, {sa['avg_score']:.0f}, {sa['num_visits']}") for sa in avg_action_scores]
        # else:
        #     [print(f"{sa['action']}, {sa['avg_score']:.0f}, {sa['sum_visits']}") for sa in avg_action_scores]
        # print(f"time_taken:{time_taken}")
        # print(f"Decision maker time: {self.time_taken}")
        # # self.logger.debug(f"Decision maker time: {self.time_taken['decision_maker']}")
        # print(f"Final action: {final_action}")
        # print()

        return final_action

    def get_mcts_inputs(
        self,
        states,
        bus_arrival_events,
        discount_factor,
        mdp_environment_model,
        rollout_policy,
        uct_tradeoff,
        iter_limit,
        allowed_computation_time,
        mcts_type,
        action_type,
    ):
        inputs = []

        # Based on how many parallel mcts we want
        # QUESTION: Copy? deepcopy? plain?
        for i in range(len(states)):
            input_dict = {}
            input_dict["tree_number"] = i
            input_dict["MCTS_type"] = mcts_type
            input_dict["mdp_environment_model"] = mdp_environment_model
            input_dict["discount_factor"] = discount_factor
            input_dict["iter_limit"] = iter_limit
            input_dict["exploit_explore_tradoff_param"] = uct_tradeoff
            input_dict["allowed_computation_time"] = allowed_computation_time
            input_dict["rollout_policy"] = rollout_policy
            input_dict["bus_arrival_events"] = copy.deepcopy(bus_arrival_events[i])
            input_dict["current_state"] = copy.deepcopy(states[i])
            input_dict["action_type"] = action_type
            inputs.append(input_dict)

        return inputs

    def load_events(self, state):
        events = copy.copy(state.bus_events)

        if state.time.time() == dt.time(0, 0, 0):
            start_time = events[0].time
        else:
            start_time = state.time

        # Rollout lookahead_horizon
        if self.lookahead_horizon_delta_t:
            lookahead_horizon = start_time + dt.timedelta(seconds=self.lookahead_horizon_delta_t)
            _events = [event for event in events if start_time <= event.time <= lookahead_horizon]
        else:
            _events = [event for event in events if start_time <= event.time]

        if len(_events) <= 0:
            _events = [events[0]]

        return [_events]

    # Generate processed chains using generate_chains_pickles.ipynb
    def get_event_chains(self, state, chains):
        event_chains = []
        for chain in range(1, chains + 1):
            state_events = copy.copy(state.bus_events)
            state_events.extend(state.disruption_chains[chain])
            # Rollout lookahead_horizon
            if self.lookahead_horizon_delta_t:
                lookahead_horizon = state.time + dt.timedelta(seconds=self.lookahead_horizon_delta_t)
                _events = [event for event in state_events if state.time <= event.time <= lookahead_horizon]
            else:
                _events = [event for event in state_events if state.time <= event.time]

            _events.sort(key=lambda x: x.time, reverse=False)

            event_chains.append(_events)
        return event_chains
