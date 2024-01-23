from Environment.enums import BusStatus, BusType, ActionType, EventType
from src.utils import *
import itertools
import pandas as pd
import copy


class NearestCoordinator:
    def __init__(self, travel_model, dispatch_policy, config):
        self.travel_model = travel_model
        self.dispatch_policy = dispatch_policy
        self.config = config

    def event_processing_callback_funct(self, actions, state, action_type):
        valid_actions = self.generate_possible_actions(state, action_type)
        action_to_take = self.select_overload_to_dispatch(state, valid_actions)

        return action_to_take

    def generate_possible_actions(self, state, action_type=ActionType.OVERLOAD_ALL):
        VEHICLE_CAPACITY = int(self.config.get("default_vehicle_capacity", 40))
        OVERAGE_THRESHOLD = float(self.config.get("overage_threshold", 0.05))
        PASSENGER_TIME_TO_LEAVE = self.config.get("passenger_time_to_leave_min", 30)
        
        # Find idle overload buses
        idle_overload_buses = []
        broken_buses = []
        for bus_id, bus_obj in state.buses.items():
            if bus_obj.type == BusType.REGULAR:
                continue
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
            # _valid_actions = self.get_valid_allocations(state)
            # valid_actions.extend(_valid_actions)
            pass

        # TODO Should we also consider all future stops in the trip?
        elif action_type == ActionType.OVERLOAD_DISPATCH:
            # TODO: Check if passengers left behind is >= 5% of the vehicle capacity
            pass
            stops_with_left_behind_passengers = []
            candidate_stops = []
            for p_set in state.people_left_behind:
                if p_set.get("left_behind", False):
                    remaining_passengers = p_set["ons"]
                    if remaining_passengers >= (VEHICLE_CAPACITY * OVERAGE_THRESHOLD):
                        arrival_time = p_set["arrival_time"]
                        current_stop_number = p_set["stop_sequence"]
                        block_id = p_set["block_id"]
                        trip_id = p_set["trip_id"]
                        stop_id = p_set["stop_id"]
                        block_trip = (block_id, str(trip_id))
                        if block_trip in state.served_trips:
                            continue
                        route_id_dir = p_set["route_id_dir"]
                        stops_with_left_behind_passengers.append(
                            (
                                stop_id,
                                current_stop_number,
                                arrival_time,
                                remaining_passengers,
                                block_trip,
                                route_id_dir,
                            )
                        )

            _valid_actions = [[ActionType.OVERLOAD_DISPATCH], idle_overload_buses, stops_with_left_behind_passengers]
            _valid_actions = list(itertools.product(*_valid_actions))
            valid_actions.extend(_valid_actions)

        # TODO: Check if they will reach there within the next bus based on headway.
        elif action_type == ActionType.OVERLOAD_TO_BROKEN:
            for bus_id, bus_obj in state.buses.items():
                if bus_obj.status == BusStatus.BROKEN:
                    if bus_obj.current_block_trip is not None:
                        if state.time <= (
                            bus_obj.time_at_last_stop + pd.Timedelta(PASSENGER_TIME_TO_LEAVE, unit="min")
                        ):
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

        # Constraint on broken bus (broken buses should be serviced immediately if possible)
        if len(broken_buses) > 0:
            # if event and (event.event_type == EventType.VEHICLE_BREAKDOWN):
            constrained_combo_actions = []
            for action in valid_actions:
                _action_type = action["type"]
                if _action_type == ActionType.OVERLOAD_TO_BROKEN:
                    if action not in constrained_combo_actions:
                        constrained_combo_actions.append(action)
            valid_actions = copy.copy(constrained_combo_actions)

        if len(valid_actions) <= 0:
            valid_actions = [do_nothing_action]

        return valid_actions

    def select_overload_to_dispatch(self, state, actions):
        actions_with_distance = self.dispatch_policy.select_overload_to_dispatch(state, actions)
        return actions_with_distance
