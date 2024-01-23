from Environment.enums import BusStatus, BusType, ActionType
import random
import datetime as dt
from src.utils import *


class HeuristicDispatch:
    def __init__(self, travel_model) -> None:
        self.travel_model = travel_model

    # Plan is to check which stop to actually send the bus to.
    def select_overload_to_dispatch(self, state, action, baseline=False):
        if action["type"] != ActionType.OVERLOAD_DISPATCH:
            return None

        action_info = action["info"]
        stop_id = action_info[0]
        current_stop_number = action_info[1]
        # remaining = action_info[3]
        current_block_trip = action_info[4]

        if baseline:
            return {
                "bus_id": action["overload_bus"],
                "current_block_trip": current_block_trip,
                "stop_no": current_stop_number,
                "stop_id": stop_id,
            }

        past_stops = self.travel_model.get_list_of_stops_for_trip(current_block_trip[1], current_stop_number + 1)
        max_remaining = -1
        dispatch_stop = stop_id
        max_remain_arrival_time = None

        for stop_id in past_stops:
            if "MCC" in stop_id[0:3]:
                stop_id = "MCC"
            stop_obj = state.stops[stop_id]
            passenger_waiting = stop_obj.passenger_waiting
            if not passenger_waiting:
                continue

            for route_id_dir, route_pw in passenger_waiting.items():
                if not route_pw:
                    continue

                for arrival_time, pw in route_pw.items():
                    remaining_passengers = pw["remaining"]
                    block_trip = pw["block_trip"]

                    if remaining_passengers > max_remaining:
                        max_remaining = remaining_passengers
                        dispatch_stop = stop_id
                        max_remain_arrival_time = arrival_time

        # for stop_id in past_stops:
        #     _stop_no = self.travel_model.get_stop_number_at_id(current_block_trip, stop_id)
        #     scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(current_block_trip, _stop_no)
        #     if scheduled_arrival_time <= max_remain_arrival_time - dt.timedelta(minutes=PASSENGER_TIME_TO_LEAVE/2):
        #         dispatch_stop = stop_id
        #         break

        current_stop_number = self.travel_model.get_stop_number_at_id(current_block_trip, dispatch_stop)
        # stop_id = self.travel_model.get_stop_id_at_number(current_block_trip, current_stop_number)

        # Nearest overflow
        actions_with_distance = []
        for bus_id in ["41", "42", "43", "44", "45"]:
            bus_obj = state.buses.get(bus_id, None)
            if bus_obj is None:
                continue
            # for bus_id, bus_obj in state.buses.items():
            if (bus_obj.type == BusType.OVERLOAD) and (
                (bus_obj.status == BusStatus.IDLE) or (bus_obj.status == BusStatus.ALLOCATION)
            ):
                # Prevent overload from being used when IDLE but has TRIPS left...
                if len(bus_obj.bus_block_trips) <= 0:
                    current_stop = bus_obj.current_stop
                    distance = self.travel_model.get_distance_from_stop_to_stop(
                        current_stop, dispatch_stop, state.time
                    )
                    actions_with_distance.append((bus_id, distance))

        actions_with_distance = sorted(actions_with_distance, key=lambda x: x[1], reverse=False)
        bus_id = actions_with_distance[0][0]
        # # For now, just send to first stop.
        return {
            "bus_id": bus_id,
            "current_block_trip": current_block_trip,
            "stop_no": current_stop_number,
            "stop_id": dispatch_stop,
        }
