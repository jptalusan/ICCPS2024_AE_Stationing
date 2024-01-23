from Environment.enums import BusStatus, EventType, ActionType, LogType, BusType
from Environment.DataStructures.Event import Event
from DecisionMaking.Dispatch.HeuristicDispatch import HeuristicDispatch
from src.utils import *
import datetime as dt
import copy
import pandas as pd
import logging


class EnvironmentModelFast:
    def __init__(self, travel_model, config):
        self.travel_model = travel_model
        self.csvlogger = logging.getLogger("csvlogger")
        self.logger = logging.getLogger("debuglogger")
        self.config = config
        self.dispatch_policy = HeuristicDispatch(travel_model)
        self.update_timer = 0

    def update(self, state, curr_event, should_log=False):
        """
        Updates the state to the given time. This is mostly updating the responders
        :param state:
        :param curr_event:
        :param passenger_alights:
        :return:
        """
        new_events = []
        new_time = curr_event.time

        # Special case for breakdowns (to save time)
        if curr_event.event_type == EventType.VEHICLE_BREAKDOWN:
            if new_time < state.time:
                return new_events
        try:
            assert new_time >= state.time
        except AssertionError:
            # print(curr_event)
            # print(new_time)
            # print(state.time)
            assert new_time >= state.time

        # print(curr_event)
        # log(self.logger, new_time, f"Event: {curr_event}", LogType.DEBUG)

        new_events = []
        if curr_event.event_type == EventType.PASSENGER_ARRIVE_STOP:
            pass

        # Buses should transfer their passengers as waiting to the nearest(?) stop as a new entry to passenger_waiting_dict_list
        elif curr_event.event_type == EventType.VEHICLE_BREAKDOWN:
            type_specific_information = curr_event.type_specific_information
            event_bus_id = type_specific_information["bus_id"]
            current_block_trip = state.buses[event_bus_id].current_block_trip
            if not should_log and current_block_trip is None:
                pass
            else:
                state.buses[event_bus_id].status = BusStatus.BROKEN
                current_stop = state.buses[event_bus_id].current_stop
                if should_log:
                    log(
                        self.logger,
                        new_time,
                        f"Bus {event_bus_id} on trip {current_block_trip[1]}, broke down near stop {current_stop}",
                        LogType.ERROR,
                    )
                self.handle_disruption_event(state, event_bus_id, current_stop, should_log)

        elif curr_event.event_type == EventType.VEHICLE_ARRIVE_AT_STOP:
            additional_info = curr_event.type_specific_information
            bus_id = additional_info["bus_id"]
            bus_state = state.buses[bus_id].status
            bus_type = state.buses[bus_id].type

            if BusStatus.IDLE == bus_state:
                # raise "Should not have an IDLE bus arriving at a stop."
                pass

            elif BusStatus.IN_TRANSIT == bus_state:
                time_of_arrival = state.buses[bus_id].t_state_change
                current_block_trip = state.buses[bus_id].current_block_trip
                bus_block_trips = state.buses[bus_id].bus_block_trips
                current_stop_number = state.buses[bus_id].current_stop_number

                if current_block_trip is None:
                    return new_events
                current_stop_id = self.travel_model.get_stop_id_at_number(current_block_trip, current_stop_number)

                if "MCC" in current_stop_id[0:3]:
                    current_stop_id = "MCC"

                last_stop_number = self.travel_model.get_last_stop_number_on_trip(current_block_trip)

                # Bus running time
                if state.buses[bus_id].time_at_last_stop:
                    state.buses[bus_id].total_service_time += (
                        new_time - state.buses[bus_id].time_at_last_stop
                    ).total_seconds()
                state.buses[bus_id].time_at_last_stop = new_time
                state.buses[bus_id].current_stop = current_stop_id

                if current_stop_number >= 0:
                    pickup_events = self.handle_bus_arrival(curr_event, state, should_log)
                    new_events.extend(pickup_events)

                # No next stop but maybe has next trips? (will check in idle_update)
                if current_stop_number == last_stop_number:
                    state.buses[bus_id].current_stop_number = 0
                    state.buses[bus_id].status = BusStatus.IDLE
                    state.buses[bus_id].t_state_change = new_time

                    if len(state.buses[bus_id].bus_block_trips) > 0:
                        # time_of_activation = new_time
                        time_of_activation = state.buses[bus_id].t_state_change

                        state.buses[bus_id].status = BusStatus.IN_TRANSIT
                        current_block_trip = state.buses[bus_id].bus_block_trips.pop(0)
                        state.buses[bus_id].current_block_trip = current_block_trip
                        current_depot = state.buses[bus_id].current_stop
                        current_stop_number = state.buses[bus_id].current_stop_number
                        route_id_direction = self.travel_model.get_route_id_direction(current_block_trip)

                        scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(
                            current_block_trip, current_stop_number
                        )

                        travel_time, distance = self.travel_model.get_traveltime_distance_from_depot(
                            current_block_trip, current_depot, current_stop_number, state.time
                        )
                        if BusType.OVERLOAD == bus_type:
                            state.buses[bus_id].total_deadkms_moved += distance
                            state.buses[bus_id].total_deadsecs_moved += travel_time
                            # log(
                            #     self.logger,
                            #     state.time,
                            #     f"Bus {bus_id} moves {distance:.2f} deadkms.",
                            #     LogType.DEBUG,
                            # )
                        else:
                            state.buses[bus_id].distance_to_next_stop = distance

                        time_to_state_change = time_of_activation + dt.timedelta(seconds=travel_time)
                        # Buses should start either at the scheduled time, or if they are late, should start as soon as possible.
                        time_to_state_change = max(time_to_state_change, scheduled_arrival_time)
                        state.buses[bus_id].t_state_change = time_to_state_change
                        state.buses[bus_id].time_at_last_stop = time_of_activation

                        arrival_event = Event(
                            event_type=EventType.VEHICLE_ARRIVE_AT_STOP,
                            time=pd.Timestamp(time_to_state_change),
                            type_specific_information={
                                "bus_id": bus_id,
                                "current_block_trip": current_block_trip,
                                "stop": state.buses[bus_id].current_stop_number,
                                "stop_id": state.buses[bus_id].current_stop,
                                "route_id_direction": route_id_direction,
                            },
                        )
                        new_events.append(arrival_event)

                # Going to next stop
                else:
                    state.buses[bus_id].current_stop_number = current_stop_number + 1
                    next_stop_id = self.travel_model.get_stop_id_at_number(
                        current_block_trip, state.buses[bus_id].current_stop_number
                    )

                    if "MCC" in next_stop_id[0:3]:
                        next_stop_id = "MCC"

                    route_id_direction = self.travel_model.get_route_id_direction(current_block_trip)

                    travel_time, distance = self.travel_model.get_traveltime_distance_from_depot(
                        current_block_trip, current_stop_id, state.buses[bus_id].current_stop_number, state.time
                    )
                    scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(
                        current_block_trip, state.buses[bus_id].current_stop_number
                    )
                    time_to_state_change = time_of_arrival + dt.timedelta(seconds=travel_time)

                    # Taking into account delay time
                    if scheduled_arrival_time < time_to_state_change:
                        delay_time = time_to_state_change - scheduled_arrival_time
                        state.buses[bus_id].delay_time += delay_time.total_seconds()

                    # TODO: Not the best place to put this, Dwell time
                    elif scheduled_arrival_time >= time_to_state_change:
                        dwell_time = scheduled_arrival_time - time_to_state_change
                        state.buses[bus_id].dwell_time += dwell_time.total_seconds()
                        time_to_state_change = time_to_state_change + dwell_time

                    # HACK: This shouldn't happen (where a new event is earlier than the current time)
                    time_to_state_change = max(time_to_state_change, new_time)

                    state.buses[bus_id].t_state_change = time_to_state_change

                    # TODO: I should add deadhead miles here too.
                    # For distance
                    state.buses[bus_id].total_servicekms_moved += state.buses[bus_id].distance_to_next_stop
                    state.buses[bus_id].distance_to_next_stop = distance

                    arrival_event = Event(
                        event_type=EventType.VEHICLE_ARRIVE_AT_STOP,
                        time=pd.Timestamp(time_to_state_change),
                        type_specific_information={
                            "bus_id": bus_id,
                            "current_block_trip": current_block_trip,
                            "stop": state.buses[bus_id].current_stop_number,
                            "stop_id": next_stop_id,
                            "route_id_direction": route_id_direction,
                        },
                    )
                    new_events.append(arrival_event)

            elif BusStatus.ALLOCATION == bus_state:
                pass

            elif BusStatus.BROKEN == bus_state:
                pass

        # Loop through all OVERLOAD buses
        for bus_id, bus_obj in state.buses.items():
            if BusType.REGULAR:
                continue
            if (BusType.OVERLOAD == bus_obj.type) and (BusStatus.ALLOCATION == bus_obj.status):
                time_at_last_stop = bus_obj.time_at_last_stop
                t_state_change = bus_obj.t_state_change

                if new_time >= t_state_change:
                    bus_obj.status = BusStatus.IDLE
                    # Missing the last fraction of a distance
                    # bus_obj.total_deadkms_moved += (bus_obj.distance_to_next_stop - distance_fraction)
                    bus_obj.percent_to_next_stop = 0.0
                    bus_obj.distance_to_next_stop = 0
                    bus_obj.time_at_last_stop = new_time
                    bus_obj.partial_deadkms_moved = 0.0
                elif new_time < t_state_change:
                    if bus_obj.time_at_last_stop:
                        journey_fraction = (new_time - time_at_last_stop) / (t_state_change - time_at_last_stop)
                    else:
                        journey_fraction = 0.0
                    bus_obj.percent_to_next_stop = journey_fraction
                    distance_fraction = journey_fraction * bus_obj.distance_to_next_stop
                    bus_obj.total_deadkms_moved += distance_fraction - bus_obj.partial_deadkms_moved
                    bus_obj.partial_deadkms_moved = distance_fraction
                    log(
                        self.csvlogger,
                        state.time,
                        f"Bus {bus_id} current total deadkms: {bus_obj.total_deadkms_moved:.2f}",
                        LogType.DEBUG,
                    )

                    log(
                        self.csvlogger,
                        new_time,
                        f"Bus {bus_id}: {journey_fraction * 100:.2f}% to {distance_fraction:.2f}/{bus_obj.distance_to_next_stop:.2f} kms to {bus_obj.current_stop_number}",
                    )

        state.time = new_time
        return new_events

    def handle_bus_arrival(self, curr_event, state, should_log=False):
        new_events = []
        passenger_time_to_leave = self.config.get("passenger_time_to_leave_min", 30)
        type_specific_information = curr_event.type_specific_information
        arrival_time = curr_event.time
        bus_id = type_specific_information["bus_id"]

        try:
            assert isinstance(passenger_time_to_leave, int)
        except:
            log(
                self.logger,
                curr_time=None,
                message=f"Passenger time type: {type(passenger_time_to_leave)}",
                type=LogType.ERROR,
            )
            # print(bus_id, type(passenger_time_to_leave))
        try:
            assert isinstance(arrival_time, dt.datetime)
        except:
            log(self.logger, curr_time=None, message=f"arrival_time type: {type(arrival_time)}", type=LogType.ERROR)
            # print(bus_id, type(arrival_time))

        curr_stop = type_specific_information["stop_id"]
        current_stop_number = type_specific_information["stop"]
        curr_route_id_dir = type_specific_information["route_id_direction"]
        curr_block_trip_id = type_specific_information["current_block_trip"]
        curr_block_id = type_specific_information["current_block_trip"][0]
        curr_trip_id = type_specific_information["current_block_trip"][1]

        try:
            assert isinstance(curr_block_id, str)
        except:
            curr_block_id = str(curr_block_id)

        bus_object = state.buses[bus_id]
        vehicle_capacity = bus_object.capacity
        stop_object = state.stops[curr_stop]
        passenger_set_counts = stop_object.passenger_waiting_dict_list

        # This should take care of the chains by itself by limiting the passengers you get to some window.
        picked_list = list(
            filter(
                lambda x: (
                    (x["route_id_dir"] == curr_route_id_dir)
                    & (x["arrival_time"] <= arrival_time)
                    & (x["arrival_time"] + pd.Timedelta(passenger_time_to_leave, unit="min") >= arrival_time)
                    & (x["block_id"] == int(curr_block_id))
                ),
                passenger_set_counts,
            )
        )

        pl, npl = self.get_remaining_passengers(
            state=state,
            arrival_time=arrival_time,
            curr_stop=curr_stop,
            curr_route_id_dir=curr_route_id_dir,
            curr_block_id=curr_block_id,
        )
        state.people_left_behind = npl

        picked_list.extend(pl)

        ons = 0
        offs = 0
        for waiting_passengers in picked_list:
            ons += waiting_passengers["ons"]
            offs += waiting_passengers["offs"]

        # print(curr_route_id_dir, curr_block_id, curr_stop, arrival_time, ons, offs)

        # Third condition will simulate people walking away after 30 minutes.
        not_picked_list = list(
            filter(
                lambda x: (
                    (x["route_id_dir"] != curr_route_id_dir)
                    | (x["arrival_time"] > arrival_time)
                    | (x["arrival_time"] + pd.Timedelta(passenger_time_to_leave, unit="min") < arrival_time)
                    | (x["block_id"] != int(curr_block_id))
                ),
                passenger_set_counts,
            )
        )
        stop_object.passenger_waiting_dict_list = not_picked_list
        # print(dt.datetime.now())
        # TODO: How to handle passenger leaving events? (don't handle, if they were there at the end of the simulation, then they were left behind.)
        # Clear them up at clear_remaining_passengers after some time elapsed (from their waiting time) check each stop with passenger_waiting_dict_list

        # Offs can only be as much as current load
        offs = min(offs, bus_object.current_load)
        if (bus_object.current_load + ons - offs) > vehicle_capacity:
            remaining = bus_object.current_load + ons - offs - vehicle_capacity
            got_on_bus = max(0, ons - remaining)

            scheduled_time = picked_list[0]["scheduled_time"]
            stop_sequence = picked_list[0]["stop_sequence"]
            trip_id = picked_list[0]["trip_id"]

            # Append back to the passenger_waiting_dict_list with same arrival_time as bus arrival and count == remaining
            remaining_dict = {
                "stop_id": curr_stop,
                "route_id_dir": curr_route_id_dir,
                "block_id": int(curr_block_id),
                "trip_id": trip_id,
                "stop_sequence": stop_sequence,
                "scheduled_time": scheduled_time,
                "arrival_time": arrival_time,
                "ons": remaining,
                "offs": 0,
                "left_behind": True,
            }
            state.people_left_behind.append(remaining_dict)
            event = Event(
                event_type=EventType.PASSENGER_LEFT_BEHIND,
                time=arrival_time,
                type_specific_information={"bus_id": bus_id, "trip_id": trip_id},
            )
            new_events.append(event)
        else:
            got_on_bus = ons
            remaining = 0

        if self.travel_model.is_bus_at_last_stop(curr_stop, curr_block_trip_id[1]):
            offs = bus_object.current_load
            got_on_bus = 0
            pass

        stop_object.total_passenger_ons += got_on_bus
        stop_object.total_passenger_offs += offs

        bus_object.current_load = bus_object.current_load + got_on_bus - offs
        bus_object.total_passengers_served += got_on_bus
        bus_object.total_stops += 1

        scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(curr_block_trip_id, current_stop_number)

        # if got_on_bus or offs:
        # if should_log and ((ons + offs + got_on_bus + remaining) > 0):
        if should_log:
            log_str = f"""Bus {bus_id} on trip: {curr_trip_id} scheduled for {scheduled_arrival_time.strftime('%H:%M:%S')} \
arrives at {curr_stop}: got_on:{got_on_bus:.0f}, on:{ons:.0f}, offs:{offs:.0f}, \
remain:{remaining:.0f}, bus_load:{bus_object.current_load:.0f}"""
            log(self.logger, state.time, log_str, LogType.INFO)

        # picked_list_str = ','.join([f"{p['arrival_time']}:{p['ons']}:{p['offs']}:{p['block_id']}" for p in passenger_df[mask]])
        # log(self.logger, state.time, f"Picked up {len(passenger_df[mask].index)} sets of passengers: {picked_list_str}.", LogType.DEBUG)

        if should_log and remaining:
            extra = ""
            if (int(curr_block_id), curr_trip_id) in state.served_trips:
                extra = ", already served."
            log(
                self.logger,
                state.time,
                f"Bus {bus_id} on trip: {curr_trip_id} left {remaining} people at stop {curr_stop},{curr_block_id}{extra}",
                LogType.ERROR,
            )

        return new_events

    # TODO: How do we treat passengers from say, stop A and stop Y, given that the bus breaks down at stop Z, traveling to stop ZZ. (AA -> ZZ)
    # People who got on at stop A would they just leave since they effectively have been waiting for Ty - Ta.
    # Transfer people from the bus to the nearest stop
    def handle_disruption_event(self, state, bus_id, stop_id, should_log=True):
        bus_obj = state.buses[bus_id]
        stop_obj = state.stops[stop_id]
        curr_time = state.time

        # BUG Bus breaksdown but is still arriving at stop (from rollout)
        current_block_trip = bus_obj.current_block_trip
        current_load = bus_obj.current_load
        current_stop_number = bus_obj.current_stop_number
        curr_route_id_direction = self.travel_model.get_route_id_direction(current_block_trip)
        if curr_route_id_direction:
            scheduled_arrival_time = self.travel_model.get_scheduled_arrival_time(
                current_block_trip, current_stop_number
            )
            remaining_dict = {
                "stop_id": stop_id,
                "route_id_dir": curr_route_id_direction,
                "block_id": int(current_block_trip[0]),
                "trip_id": int(current_block_trip[1]),
                "stop_sequence": current_stop_number,
                "scheduled_time": pd.Timestamp(scheduled_arrival_time),
                "arrival_time": curr_time,
                "ons": current_load,
                "offs": 0,
            }
            stop_obj.passenger_waiting_dict_list.append(remaining_dict)

            # TODO: I have to retroactively subtract ons from prior stops or hack, just subtract it from latest one. (total won't change)
            stop_obj.total_passenger_ons -= current_load
            return True
        else:
            log(
                self.logger,
                state.time,
                f"Bus {bus_id} on trip: {current_block_trip} {stop_id} has an error.",
                LogType.ERROR,
            )
            return False

    def get_remaining_passengers(self, state, arrival_time, curr_stop, curr_route_id_dir, curr_block_id):
        passenger_time_to_leave = self.config.get("passenger_time_to_leave_min", 30)
        passenger_set_counts = state.people_left_behind
        picked_list = list(
            filter(
                lambda x: (
                    (x["route_id_dir"] == curr_route_id_dir)
                    & (x["stop_id"] == curr_stop)
                    & (x["arrival_time"] <= arrival_time)
                    & (x["arrival_time"] + pd.Timedelta(passenger_time_to_leave, unit="min") >= arrival_time)
                    & (x["block_id"] == int(curr_block_id))
                    & (x["left_behind"] == True)
                ),
                passenger_set_counts,
            )
        )

        # Third condition will simulate people walking away after 30 minutes.
        not_picked_list = list(
            filter(
                lambda x: (
                    (x["route_id_dir"] != curr_route_id_dir)
                    | (x["stop_id"] != curr_stop)
                    | (x["arrival_time"] > arrival_time)
                    | (x["arrival_time"] + pd.Timedelta(passenger_time_to_leave, unit="min") < arrival_time)
                    | (x["block_id"] != int(curr_block_id))
                ),
                passenger_set_counts,
            )
        )

        return picked_list, not_picked_list

    def take_action(self, state, action):
        # print("take_action")
        action_type = action["type"]
        ofb_id = action["overload_bus"]
        # info: ('HILLOMNF', '7_TO DOWNTOWN', datetime.datetime(2021, 8, 23, 14, 13, 11), 8.0, ('5692', '246343'))
        # {'type': <ActionType.OVERLOAD_DISPATCH: 'overload_dispatch'>, 'overload_bus': '41', 'info': ('DELTAYSN', 25, Timestamp('2021-04-01 09:09:39.200000'), 3.0, (2200, '230186'), '22_TO DOWNTOWN')
        new_events = []

        if ActionType.OVERLOAD_DISPATCH == action_type:
            #### NEW
            info = action.get("info")

            if info:
                stop_id = info[0]
                stop_no = info[1]
                current_block_trip = info[4]

                # Don't dispatch to the same block trip.
                if current_block_trip in state.served_trips:
                    log(self.logger, state.time, f"Trip: {current_block_trip} already being served.", LogType.DEBUG)
                    return new_events, state.time

                # ofb_id = res["bus_id"]

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
                log(self.logger, state.time, f"Bus {ofb_id} moves {distance:.2f} deadkms.", LogType.DEBUG)

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

                log(
                    self.logger,
                    state.time,
                    f"Dispatching overflow bus {ofb_id} from {ofb_obj.current_stop} @ stop {stop_id}",
                    LogType.ERROR,
                )

                state.served_trips.append(current_block_trip)
            #### END NEW

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

            log(
                self.logger,
                state.time,
                f"Sending takeover overflow bus {ofb_id} from {ofb_obj.current_stop} to stop {broken_bus_obj.current_stop}",
                LogType.ERROR,
            )

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

            log(
                self.logger,
                state.time,
                f"Reallocating overflow bus {ofb_id} from {current_stop} to stop {reallocation_stop}",
                LogType.ERROR,
            )

        elif ActionType.NO_ACTION == action_type:
            # Do nothing
            pass

        return new_events, state.time
