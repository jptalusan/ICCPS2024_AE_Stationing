from argparse import Action

import pandas as pd
from Environment.enums import BusStatus, BusType, ActionType, LogType
from Environment.EnvironmentModelFast import EnvironmentModelFast
from src.utils import *
from Environment.DataStructures.Event import Event
from Environment.enums import EventType
import itertools
import copy
import datetime as dt
import time
import numpy as np

# Should have get possible actions function


# TODO: Add a heuristic to keep same action count, but expand search space (most number of passengers left behind?)
class DecisionEnvironmentDynamics(EnvironmentModelFast):
    def __init__(self, travel_model, dispatch_policy, reward_policy=None, config=None):
        EnvironmentModelFast.__init__(self, travel_model=travel_model, config=config)
        self.dispatch_policy = dispatch_policy
        self.reward_policy = reward_policy
        self.travel_model = travel_model
        self.config = config

    def generate_possible_actions(self, state, event, action_type=ActionType.OVERLOAD_ALL):
        VEHICLE_CAPACITY = int(self.config.get("default_vehicle_capacity", 40))
        OVERAGE_THRESHOLD = float(self.config.get("overage_threshold", 0.05))
        # Find idle overload buses
        idle_overload_buses = []
        for bus_id, bus_obj in state.buses.items():
            if bus_obj.type == BusType.REGULAR:
                continue
            if (bus_obj.type == BusType.OVERLOAD) and (
                (bus_obj.status == BusStatus.IDLE) or (bus_obj.status == BusStatus.ALLOCATION)
            ):
                # Prevent overload from being used when IDLE but has TRIPS left...
                if len(bus_obj.bus_block_trips) <= 0:
                    idle_overload_buses.append(bus_id)

        if len(idle_overload_buses) <= 0:
            valid_actions = [{"type": ActionType.NO_ACTION, "overload_bus": None, "info": "No available buses."}]
            action_taken_tracker = [(_[0], False) for _ in enumerate(valid_actions)]
            return valid_actions, action_taken_tracker

        # Create matrix of overload buses, original bus id, block/trips, stop_id
        valid_actions = []

        if action_type == ActionType.OVERLOAD_ALLOCATE or action_type == ActionType.OVERLOAD_ALL:
            _valid_actions = self.get_valid_allocations(state)
            valid_actions.extend(_valid_actions)

        PASSENGER_TIME_TO_LEAVE = self.config.get("passenger_time_to_leave_min", 30)

        if action_type == ActionType.OVERLOAD_DISPATCH or action_type == ActionType.OVERLOAD_ALL:
            if (
                event.event_type == EventType.VEHICLE_ARRIVE_AT_STOP
                or event.event_type == EventType.DECISION_DISPATCH_EVENT
                or event.event_type == EventType.PASSENGER_LEFT_BEHIND
            ):
                bus_id = event.type_specific_information["bus_id"]
                # Get current trip
                if state.buses[bus_id].type != BusType.OVERLOAD and state.buses[bus_id].status == BusStatus.IN_TRANSIT:
                    stops_with_left_behind_passengers = []
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
                                # Only consider actions for trip of the current bus?
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

                    _valid_actions = [
                        [ActionType.OVERLOAD_DISPATCH],
                        idle_overload_buses,
                        stops_with_left_behind_passengers,
                    ]
                    _valid_actions = list(itertools.product(*_valid_actions))
                    valid_actions.extend(_valid_actions)

        broken_buses = []
        for bus_id, bus_obj in state.buses.items():
            if bus_obj.status == BusStatus.BROKEN:
                # Without checking if a broken bus has already been covered, we try to cover it again
                # Leading to null values
                if bus_obj.current_block_trip is not None:
                    if state.time <= (bus_obj.time_at_last_stop + pd.Timedelta(PASSENGER_TIME_TO_LEAVE, unit="min")):
                        broken_buses.append(bus_id)

        if len(broken_buses) > 0:
            _valid_actions = [[ActionType.OVERLOAD_TO_BROKEN], idle_overload_buses, broken_buses]
            _valid_actions = list(itertools.product(*_valid_actions))
            valid_actions.extend(_valid_actions)

        do_nothing_action = {"type": ActionType.NO_ACTION, "overload_bus": None, "info": "No actions"}

        if len(valid_actions) > 0:
            valid_actions = [{"type": _va[0], "overload_bus": _va[1], "info": _va[2]} for _va in valid_actions]
            # Always add do nothing as a possible option
            valid_actions.append(do_nothing_action)
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

        action_taken_tracker = [(_[0], False) for _ in enumerate(valid_actions)]
        return valid_actions, action_taken_tracker

    def get_valid_allocations(self, state):
        num_available_buses = len(
            [
                _
                for _ in state.buses.values()
                if (_.status == BusStatus.IDLE or _.status == BusStatus.ALLOCATION) and _.type == BusType.OVERLOAD
            ]
        )

        if num_available_buses <= 0:
            return []

        # Based on k-means clustering of disruption events and finding the center most stop from that cluster's envelope
        # TODO Expand this to 25
        valid_stops = ["MTA", "MCC", "NOLTAYSN", "DWMRT", "WHICHASF"]

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

    def take_action(self, state, action):
        action_type = action["type"]
        ofb_id = action["overload_bus"]
        # ('HILLOMNF', '7_TO DOWNTOWN', datetime.datetime(2021, 8, 23, 14, 13, 11), 8.0, ('5692', '246343'))

        new_events = []

        # Send to stop

        if ActionType.OVERLOAD_DISPATCH == action_type:
            #### NEW
            res = self.dispatch_policy.select_overload_to_dispatch(state, action)
            if res:
                stop_id = res["stop_id"]
                stop_no = res["stop_no"]
                current_block_trip = res["current_block_trip"]
                ofb_id = res["bus_id"]

                ofb_obj = state.buses[ofb_id]
                ofb_obj.bus_block_trips = [current_block_trip]
                ofb_obj.current_stop_number = stop_no
                ofb_obj.status = BusStatus.IN_TRANSIT

                ofb_obj.current_block_trip = ofb_obj.bus_block_trips.pop(0)

                scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(current_block_trip, stop_no)

                travel_time, distance = self.travel_model.get_traveltime_distance_from_depot(
                    current_block_trip, ofb_obj.current_stop, stop_no, state.time
                )
                route_id_direction = self.travel_model.get_route_id_direction(current_block_trip)
                ofb_obj.total_deadkms_moved += distance
                ofb_obj.total_deadsecs_moved += travel_time
                # log(self.logger, state.time, f"Bus {ofb_id} moves {distance:.2f} deadkms.", LogType.DEBUG)

                time_of_activation = state.time
                time_to_state_change = time_of_activation + dt.timedelta(seconds=travel_time)
                # Buses should start either at the scheduled time, or if they are late, should start as soon as possible.
                time_to_state_change = max(time_to_state_change, scheduled_arrival_time)
                ofb_obj.t_state_change = time_to_state_change
                ofb_obj.time_at_last_stop = time_of_activation

                event = Event(
                    event_type=EventType.VEHICLE_ARRIVE_AT_STOP,
                    time=time_to_state_change,
                    type_specific_information={
                        "bus_id": ofb_id,
                        "current_block_trip": current_block_trip,
                        "stop": state.buses[ofb_id].current_stop_number,
                        "stop_id": stop_id,
                        "route_id_direction": route_id_direction,
                        "action": ActionType.OVERLOAD_DISPATCH,
                    },
                )

                new_events.append(event)

                # log(
                #     self.logger,
                #     state.time,
                #     f"Dispatching overflow bus {ofb_id} from {ofb_obj.current_stop} @ stop {stop_id}",
                #     LogType.ERROR,
                # )

                state.served_trips.append(current_block_trip)
            ##### END NEW

        # Take over broken bus
        elif ActionType.OVERLOAD_TO_BROKEN == action_type:
            ofb_obj = state.buses[ofb_id]
            action_info = action["info"]
            broken_bus_id = action_info
            broken_bus_obj = state.buses[broken_bus_id]

            current_block_trip = broken_bus_obj.current_block_trip
            stop_no = broken_bus_obj.current_stop_number

            ofb_obj.bus_block_trips = copy.copy([broken_bus_obj.current_block_trip] + broken_bus_obj.bus_block_trips)
            ofb_obj.bus_block_trips = [x for x in ofb_obj.bus_block_trips if x is not None]

            ofb_obj.current_block_trip = None
            # In case bus has not yet started trip.
            if stop_no == 0:
                ofb_obj.current_stop_number = 0
            # Because at this point we already set the state to the next stop.
            else:
                ofb_obj.current_stop_number = stop_no - 1

            ofb_obj.status = BusStatus.IN_TRANSIT

            # Deactivate broken_bus_obj
            broken_bus_obj.current_block_trip = None
            broken_bus_obj.bus_block_trips = []
            broken_bus_obj.total_passengers_served -= broken_bus_obj.current_load
            broken_bus_obj.current_load = 0

            ofb_obj.current_block_trip = ofb_obj.bus_block_trips.pop(0)

            # TODO: It should just compute the time it takes from where it is right now, to go to the last passed stop of the broken bus.
            broken_bus_last_passed_stop = broken_bus_obj.current_stop
            current_overload_bus_stop = ofb_obj.current_stop

            travel_time, distance = self.travel_model.get_traveltime_distance_from_stops(
                current_overload_bus_stop, broken_bus_last_passed_stop, state.time
            )
            ofb_obj.total_deadkms_moved += distance
            ofb_obj.total_deadsecs_moved += travel_time
            route_id_direction = self.travel_model.get_route_id_direction(ofb_obj.current_block_trip)

            # In case it arrives earlier than scheduled time (it should wait?)
            scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(
                ofb_obj.current_block_trip, ofb_obj.current_stop_number
            )
            time_of_activation = state.time
            time_to_state_change = time_of_activation + dt.timedelta(seconds=travel_time)
            # HACK kind of.
            time_to_state_change = max(time_to_state_change, scheduled_arrival_time)

            event = Event(
                event_type=EventType.VEHICLE_ARRIVE_AT_STOP,
                time=time_to_state_change,
                type_specific_information={
                    "bus_id": ofb_id,
                    "current_block_trip": ofb_obj.current_block_trip,
                    "stop": ofb_obj.current_stop_number,
                    "stop_id": broken_bus_last_passed_stop,
                    "route_id_direction": route_id_direction,
                    "action": ActionType.OVERLOAD_TO_BROKEN,
                },
            )

            new_events.append(event)

            # log(
            #     self.logger,
            #     state.time,
            #     f"[SIM] Sending takeover overflow bus {ofb_id} from {ofb_obj.current_stop} to stop {broken_bus_obj.current_stop} {broken_bus_id}",
            #     LogType.ERROR,
            # )

        elif ActionType.OVERLOAD_ALLOCATE == action_type:
            ofb_obj = state.buses[ofb_id]
            current_stop = ofb_obj.current_stop
            action_info = action["info"]
            reallocation_stop = action_info

            travel_time = self.travel_model.get_travel_time_from_stop_to_stop(
                current_stop, reallocation_stop, state.time
            )
            distance_to_next_stop = self.travel_model.get_distance_from_stop_to_stop(
                current_stop, reallocation_stop, state.time
            )
            time_to_state_change = state.time + dt.timedelta(seconds=travel_time)
            ofb_obj.current_stop = reallocation_stop
            ofb_obj.t_state_change = time_to_state_change
            ofb_obj.distance_to_next_stop = distance_to_next_stop
            ofb_obj.time_at_last_stop = state.time
            ofb_obj.status = BusStatus.ALLOCATION

        elif ActionType.NO_ACTION == action_type:
            # Do nothing
            pass

        reward = self.compute_reward(state)
        return reward, new_events, state.time

    def compute_reward(self, state):
        total_walk_aways = 0
        total_remaining = 0
        total_passenger_ons = 0
        total_deadkms = 0
        total_passengers_served = 0
        total_aggregate_delay = 0

        # total_broken_buses = 0
        # for _, stop_obj in state.stops.items():
        #     total_walk_aways += stop_obj.total_passenger_walk_away
        #     total_passenger_ons += stop_obj.total_passenger_ons
        #     passenger_waiting = stop_obj.passenger_waiting
        #     if not passenger_waiting:
        #         continue

        #     for route_id_dir, route_pw in passenger_waiting.items():
        #         if not route_pw:
        #             continue

        #         for arrival_time, pw in route_pw.items():
        #             remaining_passengers = pw['remaining']
        #             total_remaining += remaining_passengers

        for _, bus_obj in state.buses.items():
            total_deadkms += bus_obj.total_deadkms_moved
            total_passengers_served += bus_obj.total_passengers_served
            total_aggregate_delay += bus_obj.delay_time

        return total_passengers_served
        # return (-1 * total_remaining)
        # return (-1 * total_remaining) + (-1 * total_deadkms)
