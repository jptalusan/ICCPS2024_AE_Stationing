from src.utils import *
import json
import logging
import pickle
import pandas as pd
import numpy as np
import datetime as dt

# from pandas.core.common import SettingWithCopyWarning
from Environment.enums import EventType, ActionType, BusType

# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
# warnings.simplefilter(action='ignore', category=FutureWarning)


# For now should contain all travel related stuff (ons, loads, travel times, distances)
class EmpiricalTravelModelLookup:
    def __init__(self, base_dir, date_str, config):
        noise_label = str(config.get("noise_level", ""))

        config_path = f'{base_dir}/{config["real_world_dir"]}/{date_str}/trip_plan_{date_str}.json'
        with open(config_path) as f:
            self.trip_plan = json.load(f)

        self.logger = logging.getLogger("debuglogger")

        with open(f"{base_dir}/common/sampled_travel_times_dict.pkl", "rb") as handle:
            self.sampled_travel_time = pickle.load(handle)

        with open(f"{base_dir}/common/stops_tt_dd_node_dict.pkl", "rb") as handle:
            self.stops_tt_dd_dict = pickle.load(handle)

        with open(f"{base_dir}/common/stops_node_matching_dict.pkl", "rb") as handle:
            self.stop_nodes_dict = pickle.load(handle)

        self.config = config

    # pandas dataframe: route_id_direction, block_abbr, stop_id_original, time, IsWeekend, sample_time_to_next_stop
    def get_travel_time(self, current_block_trip, current_stop_number, _datetime):
        current_trip = str(current_block_trip[1])
        IsWeekend = 0 if _datetime.weekday() < 5 else 1
        block_abbr = int(current_block_trip[0])

        trip_data = self.trip_plan[current_trip]
        route_id = trip_data["route_id"]
        route_direction = trip_data["route_direction"]
        route_id_dir = str(route_id) + "_" + route_direction
        stop_id_original = trip_data["stop_id_original"]
        stop_id_original = stop_id_original[current_stop_number]

        scheduled_time = self.trip_plan[current_trip]["scheduled_time"][current_stop_number].split(" ")[1]
        scheduled_time = dt.datetime.strptime(scheduled_time, "%H:%M:%S").time()
        # rid, bid, ssq, sid, tme, wkd
        key = (route_id_dir, block_abbr, current_stop_number + 1, stop_id_original, scheduled_time, IsWeekend)

        if key in self.sampled_travel_time:
            tt = self.sampled_travel_time[key]["sampled_travel_time"]
            return tt

        # log(self.logger, _datetime, f'Failed to get travel time for: {key}', LogType.ERROR)
        return self.get_travel_time_from_depot(current_block_trip, stop_id_original, current_stop_number, _datetime)

    def get_travel_time_from_depot(self, current_block_trip, current_stop, current_stop_number, _datetime):
        current_trip = str(current_block_trip[1])
        trip_data = self.trip_plan[current_trip]
        next_stop = trip_data["stop_id_original"][current_stop_number]

        tt, _ = self.get_traveltime_distance_from_stops(current_stop, next_stop, _datetime)
        return tt

    # tt is in seconds
    def get_travel_time_from_stop_to_stop(self, current_stop, next_stop, _datetime):
        tt, _ = self.get_traveltime_distance_from_stops(current_stop, next_stop, _datetime)
        return tt

    # dd is in meters
    def get_distance_from_stop_to_stop(self, current_stop, next_stop, _datetime):
        _, dd = self.get_traveltime_distance_from_stops(current_stop, next_stop, _datetime)
        return dd

    # Can probably move to a different file next time
    def get_next_stop(self, current_block_trip, current_stop_sequence):
        """
        trip_plan[trip_id][stop_id_original]

        """
        current_trip = str(current_block_trip[1])
        trip_data = self.trip_plan[current_trip]
        last_stop_id = trip_data["last_stop_id"]
        last_stop_sequence = trip_data["last_stop_sequence"]

        if current_stop_sequence == None:
            return None

        if current_stop_sequence == last_stop_sequence:
            return None

        else:
            return trip_data["stop_sequence"][current_stop_sequence + 1]

    def get_route_id_direction(self, current_block_trip):
        try:
            current_trip = str(current_block_trip[1])
            trip_data = self.trip_plan[current_trip]
            route_id = trip_data["route_id"]
            route_direction = trip_data["route_direction"]
            res = f"{route_id}_{route_direction}"
            return res
        except Exception as e:
            self.logger.error(f"get_route_id_direction error {e}, with block trip: {current_block_trip}")
            return None

    def get_stop_id_at_number(self, current_block_trip, current_stop_sequence):
        if current_stop_sequence == -1:
            return None
        current_trip = str(current_block_trip[1])
        trip_data = self.trip_plan[current_trip]
        return trip_data["stop_id_original"][current_stop_sequence]

    def get_stop_number_at_id(self, current_block_trip, current_stop_id):
        stop_idx = 0
        current_trip = str(current_block_trip[1])
        trip_data = self.trip_plan[current_trip]
        try:
            stop_idx = trip_data["stop_id_original"].index(current_stop_id)
        except:
            for i, s in enumerate(trip_data["stop_id_original"]):
                if current_stop_id in s[0:3]:
                    stop_idx = i
                    break
        return trip_data["stop_sequence"][stop_idx]

    def get_scheduled_arrival_time(self, current_block_trip, current_stop_sequence):
        current_trip = str(current_block_trip[1])
        trip_data = self.trip_plan[current_trip]
        arrival_time_str = trip_data["scheduled_time"][current_stop_sequence]
        scheduled_arrival_time = str_timestamp_to_datetime(arrival_time_str)
        return scheduled_arrival_time

    def get_route_id_dir_for_trip(self, current_block_trip):
        current_trip = str(current_block_trip[1])
        trip_data = self.trip_plan[current_trip]
        route_id = trip_data["route_id"]
        route_direction = trip_data["route_direction"]
        return str(route_id) + "_" + route_direction

    def get_last_stop_number_on_trip(self, current_block_trip):
        current_trip = str(current_block_trip[1])
        trip_data = self.trip_plan[current_trip]
        return trip_data["last_stop_sequence"]

    def get_traveltime_distance_from_depot(self, current_block_trip, current_stop, current_stop_number, _datetime):
        dd = -1
        tt = -1
        current_trip = str(current_block_trip[1])
        trip_data = self.trip_plan[current_trip]
        next_stop = trip_data["stop_id_original"][current_stop_number]
        tt, dd = self.get_traveltime_distance_from_stops(current_stop, next_stop, _datetime)
        return tt, dd

    def get_traveltime_distance_from_stops(self, current_stop, next_stop, _datetime):
        if current_stop == "MCC":
            current_stop = "MCC5_1"
        if next_stop == "MCC":
            next_stop = "MCC5_1"

        # HACK: Need to regenerate run_all.py for all years
        if self.stop_nodes_dict.get(current_stop, False) and self.stop_nodes_dict.get(next_stop, False):
            cn = self.stop_nodes_dict[current_stop]["nearest_node"]
            nn = self.stop_nodes_dict[next_stop]["nearest_node"]

            # if self.config.get("incident_factor", False):
            #     tt_factor = self.get_travel_time_incident_factor(current_stop, next_stop, _datetime)
            # else:
            tt_factor = 1.0

            if (cn, nn) in self.stops_tt_dd_dict:
                tt = self.stops_tt_dd_dict[(cn, nn)]["travel_time_s"]
                dd = self.stops_tt_dd_dict[(cn, nn)]["distance_m"]
            elif (nn, cn) in self.stops_tt_dd_dict:
                tt = self.stops_tt_dd_dict[(nn, cn)]["travel_time_s"]
                dd = self.stops_tt_dd_dict[(nn, cn)]["distance_m"]
            else:
                # print(f"Could not find tt/dd for {cn}/{current_stop} to {nn}/{next_stop}.")
                return 60 * tt_factor, 0.5
                # raise "Error"
            if tt < 0:
                # print(f"Could not find tt/dd for {cn}/{current_stop} to {nn}/{next_stop}.")
                return 60 * tt_factor, 0.5
                # raise "Error"
            return tt * tt_factor, dd / 1000
        else:
            tt_factor = 1.0
            return 60 * tt_factor, 0.5

    def is_bus_at_last_stop(self, current_stop, trip):
        trip_data = self.trip_plan[trip]
        last_stop_id = trip_data["last_stop_id"]
        if current_stop == last_stop_id or ((current_stop == "MCC") and (current_stop in last_stop_id[0:3])):
            return True
        return False

    def get_list_of_stops_for_trip(self, trip, current_stop_number):
        trip_data = self.trip_plan[trip]
        stop_id_original = trip_data["stop_id_original"]
        if current_stop_number == 0:
            current_stop_number = current_stop_number + 1
        stop_id_original = stop_id_original[0:current_stop_number]
        return stop_id_original

    def is_event_a_timepoint(self, curr_event, state):
        info = curr_event.type_specific_information
        if curr_event.event_type == EventType.VEHICLE_BREAKDOWN:
            return True

        if curr_event.event_type == EventType.VEHICLE_ARRIVE_AT_STOP:
            # bus_id = info['bus_id']
            current_block_trip = info["current_block_trip"]
            current_trip = str(current_block_trip[1])
            current_stop_number = info["stop"]
            # bus_object = state.buses[bus_id]
            current_stop_id = self.get_stop_id_at_number(current_block_trip, current_stop_number)
            scheduled_time = str_timestamp_to_datetime(
                self.trip_plan[current_trip]["scheduled_time"][current_stop_number]
            )

            key = (int(current_trip), current_stop_id, scheduled_time)
            if key in self.time_point_dict:
                timepoint = self.time_point_dict[key]["timepoint"]
            else:
                print(f"{key}: not in timepoint dictionary.")
                return True
            return timepoint == 1

        return False
