from Environment.enums import BusStatus, BusType, ActionType, EventType
from src.utils import *
import itertools
import random
import copy


class GreedyCoordinator:
    def __init__(self, travel_model, dispatch_policy, config):
        self.travel_model = travel_model
        self.dispatch_policy = dispatch_policy
        self.trips_already_covered = []
        self.broken_buses_covered = []
        self.config = config

    def event_processing_callback_funct(self, actions, state, action_type):
        valid_actions = self.generate_possible_actions(state, action_type)
        action_to_take = self.select_overload_to_dispatch(state, valid_actions)
        if action_to_take["type"] == ActionType.OVERLOAD_DISPATCH:
            self.trips_already_covered.append(action_to_take["info"][4])
        # if action_to_take['type'] == ActionType.OVERLOAD_TO_BROKEN:
        #     self.broken_buses_covered.append(action_to_take['info'][4])

        return action_to_take

    def generate_possible_actions(self, state, action_type=ActionType.OVERLOAD_ALL):
        # Find idle overload buses
        idle_overload_buses = []
        for bus_id, bus_obj in state.buses.items():
            if (bus_obj.type == BusType.OVERLOAD) and (
                (bus_obj.status == BusStatus.IDLE) or (bus_obj.status == BusStatus.ALLOCATION)
            ):
                # Prevent overload from being used when IDLE but has TRIPS left...
                if len(bus_obj.bus_block_trips) <= 0:
                    idle_overload_buses.append(bus_id)

        valid_actions = []
        if len(idle_overload_buses) <= 0:
            valid_actions = [{"type": ActionType.NO_ACTION, "overload_bus": None, "info": "No available buses."}]
            return valid_actions

        if action_type == ActionType.OVERLOAD_ALLOCATE:
            _valid_actions = self.get_valid_allocations(state)
            valid_actions.extend(_valid_actions)
        elif action_type == ActionType.OVERLOAD_DISPATCH:
            stops_with_left_behind_passengers = []
            for stop_id, stop_obj in state.stops.items():
                passenger_waiting = stop_obj.passenger_waiting
                if not passenger_waiting:
                    continue

                for route_id_dir, route_pw in passenger_waiting.items():
                    if not route_pw:
                        continue

                    for arrival_time, pw in route_pw.items():
                        remaining_passengers = pw["remaining"]
                        block_trip = pw["block_trip"]

                        if remaining_passengers > 0:
                            if block_trip not in self.trips_already_covered:
                                current_stop_number = self.travel_model.get_stop_number_at_id(block_trip, stop_id)
                                stops_with_left_behind_passengers.append(
                                    (stop_id, current_stop_number, arrival_time, remaining_passengers, block_trip)
                                )

            _valid_actions = [[ActionType.OVERLOAD_DISPATCH], idle_overload_buses, stops_with_left_behind_passengers]
            _valid_actions = list(itertools.product(*_valid_actions))
            valid_actions.extend(_valid_actions)
            # elif action_type == ActionType.OVERLOAD_TO_BROKEN:
            broken_buses = []
            for bus_id, bus_obj in state.buses.items():
                if bus_obj.status == BusStatus.BROKEN:
                    if bus_obj.current_block_trip is not None:
                        broken_buses.append(bus_id)

            if len(broken_buses) > 0:
                _valid_actions = [[ActionType.OVERLOAD_TO_BROKEN], idle_overload_buses, broken_buses]
                _valid_actions = list(itertools.product(*_valid_actions))
                valid_actions.extend(_valid_actions)

        do_nothing_action = {"type": ActionType.NO_ACTION, "overload_bus": None, "info": "NO actions."}
        if len(valid_actions) > 0:
            valid_actions = [{"type": _va[0], "overload_bus": _va[1], "info": _va[2]} for _va in valid_actions]
        else:
            # No action
            valid_actions = [do_nothing_action]

        return valid_actions

    def get_valid_allocations(self, state):
        num_available_buses = len(
            [_ for _ in state.buses.values() if _.status == BusStatus.IDLE and _.type == BusType.OVERLOAD]
        )

        if num_available_buses <= 0:
            return []

        # MTA, MCC5_1, HICHICNN, WESWILEN (based on MTA)
        valid_stops = ["MTA", "MCC5_1", "NOLTAYSN", "DWMRT", "WHICHASF"]

        # Based on spatial clustering k = 10
        idle_overload_buses = []
        for bus_id, bus_obj in state.buses.items():
            if (bus_obj.type == BusType.OVERLOAD) and (
                (bus_obj.status == BusStatus.IDLE) or (bus_obj.status == BusStatus.ALLOCATION)
            ):
                if len(bus_obj.bus_block_trips) <= 0:
                    idle_overload_buses.append(bus_id)

        valid_actions = []
        _valid_actions = [[ActionType.OVERLOAD_ALLOCATE], idle_overload_buses, valid_stops]
        _valid_actions = list(itertools.product(*_valid_actions))

        # Remove allocations to the same stop the bus is currently in.
        _valid_actions = [va for va in _valid_actions if va[2] != state.buses[va[1]].current_stop]
        valid_actions.extend(_valid_actions)
        return valid_actions

    def select_overload_to_dispatch(self, state, actions):
        passenger_time_to_leave = self.config.get("passenger_time_to_leave_min", 30)
        random.seed(100)
        if len(actions) <= 0:
            return None

        actions_with_distance = []

        is_all_allocation = True
        for action in actions:
            if action["type"] != ActionType.OVERLOAD_ALLOCATE and action["type"] != ActionType.NO_ACTION:
                is_all_allocation = False

        if is_all_allocation:
            return random.choice(actions)

        for action in actions:
            action_type = action["type"]
            overload_bus = action["overload_bus"]
            info = action["info"]

            if (action_type == ActionType.NO_ACTION) and (len(actions) == 1):
                return actions[0]

            elif (action_type == ActionType.NO_ACTION) and (len(actions) > 1):
                continue

            current_stop = state.buses[overload_bus].current_stop
            next_stop = None

            if action_type == ActionType.OVERLOAD_TO_BROKEN:
                broken_bus = info
                next_stop = state.buses[info].current_stop
                remaining_passengers = 100
            elif action_type == ActionType.OVERLOAD_DISPATCH:
                next_stop = info[0]
                remaining_passengers = info[3]
                pass
            elif action_type == ActionType.OVERLOAD_ALLOCATE:
                next_stop = info
                pass
            else:
                raise "Action not supported"

            travel_time = self.travel_model.get_travel_time_from_stop_to_stop(current_stop, next_stop, state.time)
            if travel_time <= (passenger_time_to_leave * 60):
                distance = self.travel_model.get_distance_from_stop_to_stop(current_stop, next_stop, state.time)
                actions_with_distance.append((action, distance, remaining_passengers))

        if len(actions_with_distance) == 0:
            action_to_take = {"type": ActionType.NO_ACTION, "overload_bus": None, "info": None}
            return action_to_take

        # Rank greedily based on number of people left behind (get most people left)
        actions_with_distance = sorted(actions_with_distance, key=lambda x: x[2], reverse=True)

        # Rank greedily based on distance to stop (go to nearest dispatch point)
        # actions_with_distance = sorted(actions_with_distance, key=lambda x: x[1], reverse=False)
        # print(actions_with_distance)
        actions_with_distance = actions_with_distance[0][0]
        return actions_with_distance
