import copy
import time

import pandas as pd
from Environment.enums import BusStatus, BusType, ActionType, EventType
from src.utils import *
import datetime as dt
import logging
import pickle


class Simulator:
    def __init__(
        self,
        starting_state,
        environment_model,
        event_processing_callback,
        starting_event_queue,
        valid_actions,
        config,
        travel_model,
    ) -> None:
        self.state = starting_state
        self.environment_model = environment_model
        self.event_processing_callback = event_processing_callback
        self.event_queue = starting_event_queue
        self.save_metrics = config.get("save_metrics", False)
        self.travel_model = travel_model

        self.start_sim_time = None
        self.starting_num_events = len(starting_event_queue)
        self.num_events_processed = 0
        self.decision_events = 0
        self.config = config
        self.reallocation = config.get("reallocation", False)
        self.log_name = config["mcts_log_name"]

        # TODO: Verify that input is int.
        self.decision_interval = int(config.get("decision_interval", 5))
        self.last_visual_log = None
        self.valid_actions = valid_actions
        self.action_timer = 0

        self.csvlogger = logging.getLogger("csvlogger")
        self.logger = logging.getLogger("debuglogger")

    # Other actors catch up to the event..?
    def run_simulation(self):
        log(self.logger, dt.datetime.now(), "Running simulation (real world time)", LogType.INFO)

        self.start_sim_time = time.time()

        # initialize state
        update_event = None

        while len(self.event_queue) > 0:
            self.update_sim_info()

            chosen_action = None
            if self.valid_actions is not None:
                _valid_actions = self.valid_actions.get_valid_actions(self.state)
            else:
                _valid_actions = None

            if self.config["method"].upper() == "MCTS":
                if self.config["scenario"].upper() == "1B":
                    self.decide_and_take_actions_1B(self.event_queue[0], _valid_actions)
            elif self.config["method"].upper() == "BASELINE":
                start_time = time.time()
                self.decide_and_take_actions_baseline(self.event_queue[0], _valid_actions)
                elapsed_time = time.time() - start_time
                self.action_timer += elapsed_time

            # update_event = self.event_queue.pop(0)
            self.event_queue.pop(0)
            if len(self.event_queue) > 0:
                start_time = time.time()
                new_events = self.environment_model.update(self.state, self.event_queue[0], should_log=True)
                elapsed_time = time.time() - start_time
                self.environment_model.update_timer += elapsed_time
                for event in new_events:
                    self.add_event(event)

            self.state.bus_events = copy.copy(self.event_queue)

            if self.save_metrics:
                self.log_metrics()

            early_end = self.config.get("early_end")
            if early_end and self.state.time >= str_timestamp_to_datetime(early_end):
                break
        # print(f"Done:{dt.datetime.now()}")
        # TODO: Calculate the deadmiles traveled to go back to the main depot (in nestor or MCC)

        self.return_sub_buses_to_garage()

        self.print_states(csv=True)
        # self.print_res()
        log(self.logger, dt.datetime.now(), "Finished simulation (real world time)", LogType.INFO)
        self.csvlogger.debug(f"{self.environment_model.update_timer:.2f} update time.")
        self.csvlogger.debug(f"{self.action_timer:.2f} action time.")
        return self.get_score()

    def return_sub_buses_to_garage(self):
        for bus_id, bus_obj in self.state.buses.items():
            if bus_obj.type == BusType.OVERLOAD:
                tt = 0
                dd = 0
                if "MTA" not in bus_obj.current_stop[0:3]:
                    tt, dd = self.travel_model.get_traveltime_distance_from_stops(
                        bus_obj.current_stop, "MTA", self.state.time
                    )
                bus_obj.total_deadkms_moved += dd
                bus_obj.total_deadsecs_moved += tt
                bus_obj.current_stop = "MTA"

    def get_score(self):
        a = 0
        b = 0
        for bus_id, bus_obj in self.state.buses.items():
            if bus_obj.type == BusType.OVERLOAD:
                a += bus_obj.total_deadkms_moved + bus_obj.partial_deadkms_moved
                b += bus_obj.total_deadsecs_moved

        c = 0
        for stop_id, stop_obj in self.state.stops.items():
            passenger_set_counts = stop_obj.passenger_waiting_dict_list
            for p_set in passenger_set_counts:
                c += p_set["ons"]
        # TODO: For baseline and for future return a,b,c (need to refactor the executors after)
        # return a + b + c
        self.csvlogger.debug(f"{a},{b},{c}")
        return a, b, c

    def decide_and_take_actions_baseline(self, update_event, _valid_actions):
        chosen_action = None
        if update_event is None:
            return

        bus_id = update_event.type_specific_information.get("bus_id")
        if bus_id and self.state.buses[bus_id].type == BusType.OVERLOAD:
            return

        if (update_event.event_type == EventType.DECISION_ALLOCATION_EVENT) and self.reallocation:
            chosen_action = self.event_processing_callback(
                _valid_actions, self.state, action_type=ActionType.OVERLOAD_ALLOCATE
            )
        elif (
            update_event.event_type == EventType.VEHICLE_ARRIVE_AT_STOP
            or update_event.event_type == EventType.PASSENGER_LEFT_BEHIND
            # update_event.event_type
            # == EventType.PASSENGER_LEFT_BEHIND
        ):
            chosen_action = self.event_processing_callback(
                _valid_actions, self.state, action_type=ActionType.OVERLOAD_DISPATCH
            )
        elif update_event.event_type == EventType.VEHICLE_BREAKDOWN:
            chosen_action = self.event_processing_callback(
                _valid_actions, self.state, action_type=ActionType.OVERLOAD_TO_BROKEN
            )

        if chosen_action is None:
            chosen_action = {"type": ActionType.NO_ACTION, "overload_bus": None, "info": None}

        if self.save_metrics:
            self.action_taken_log.debug(f"{self.num_events_processed},{self.state.time},{chosen_action}")

        if chosen_action["type"] != ActionType.NO_ACTION:
            log(self.logger, self.state.time, self.format_action_tuple(chosen_action), LogType.DEBUG)
        new_events, _ = self.environment_model.take_action(self.state, chosen_action)
        for event in new_events:
            self.add_event(event)

    # TODO: Check if this is trying to dispatch in the future? Since bus may not have reached the "current_stop" yet.
    # Need to check t_state_change
    def decide_and_take_actions_1B(self, update_event, _valid_actions):
        if update_event and (update_event.event_type == EventType.DECISION_ALLOCATION_EVENT) and self.reallocation:
            chosen_action = self.event_processing_callback(
                _valid_actions, self.state, action_type=ActionType.OVERLOAD_ALLOCATE
            )
            if chosen_action is None:
                chosen_action = {"type": ActionType.NO_ACTION, "overload_bus": None, "info": None}

            if self.save_metrics:
                self.action_taken_log.debug(f"{self.num_events_processed},{self.state.time},{chosen_action}")
            # log(self.csvlogger, self.state.time, f"Chosen action:{chosen_action}", LogType.DEBUG)
            new_events, _ = self.environment_model.take_action(self.state, chosen_action)
            for event in new_events:
                self.add_event(event)
            self.decision_events += 1

        if update_event and (
            (update_event.event_type == EventType.VEHICLE_ARRIVE_AT_STOP)
            or (update_event.event_type == EventType.VEHICLE_BREAKDOWN)
            or (update_event.event_type == EventType.PASSENGER_LEFT_BEHIND)
        ):
            bus_id = update_event.type_specific_information.get("bus_id")
            if bus_id and self.state.buses[bus_id].type == BusType.OVERLOAD:
                pass
            elif (
                bus_id
                and self.state.buses[bus_id].last_decision_epoch
                and (
                    (self.state.time - self.state.buses[bus_id].last_decision_epoch)
                    < dt.timedelta(minutes=self.decision_interval)
                )
                and (update_event.event_type != EventType.PASSENGER_LEFT_BEHIND)
                and (update_event.event_type != EventType.VEHICLE_BREAKDOWN)
            ):
                pass
            elif (
                (update_event.event_type == EventType.VEHICLE_BREAKDOWN)
                or (update_event.event_type == EventType.PASSENGER_LEFT_BEHIND)
                or (
                    self.state.buses[bus_id].last_decision_epoch
                    and (self.state.time - self.state.buses[bus_id].last_decision_epoch)
                    >= dt.timedelta(minutes=self.decision_interval)
                )
                or (self.state.buses[bus_id].last_decision_epoch is None)
            ):
                chosen_action = self.event_processing_callback(
                    _valid_actions, self.state, action_type=ActionType.OVERLOAD_DISPATCH
                )

                self.state.buses[bus_id].last_decision_epoch = self.state.time
                if chosen_action is None:
                    chosen_action = {"type": ActionType.NO_ACTION, "overload_bus": None, "info": None}

                if self.save_metrics:
                    self.action_taken_log.debug(f"{self.num_events_processed},{self.state.time},{chosen_action}")

                if chosen_action["type"] != ActionType.NO_ACTION:
                    log(self.logger, self.state.time, self.format_action_tuple(chosen_action), LogType.DEBUG)
                new_events, _ = self.environment_model.take_action(self.state, chosen_action)
                for event in new_events:
                    self.add_event(event)
                self.decision_events += 1

    def format_action_tuple(self, action_dict):
        # {'type': <ActionType.OVERLOAD_DISPATCH: 'overload_dispatch'>, 'overload_bus': '41', 'info': ('MXIDONEL', 17, Timestamp('2022-10-05 05:42:03.400000'), 6.0, (5504, '279150'), '55_TO DOWNTOWN')}
        if action_dict["type"] == ActionType.OVERLOAD_DISPATCH:
            log_str = f"{action_dict['type'].value} bus: {action_dict['overload_bus']} to {action_dict['info'][0]} with {action_dict['info'][3]} people waiting since {action_dict['info'][2].strftime('%H:%M:%S')} at trip: {action_dict['info'][4]}"
        else:
            log_str = f"{action_dict}"
        return log_str

    def update_sim_info(self):
        self.num_events_processed += 1

    def add_event(self, new_event):
        self.event_queue.append(new_event)
        self.event_queue.sort(key=lambda _: _.time, reverse=False)

    def log_metrics(self):
        log_time = self.state.time
        for stop_id, stop_obj in self.state.stops.items():
            if stop_obj.passenger_waiting:
                # self.stop_metrics_log.debug(f"{log_time},{stop_obj}")
                for route_id, v in stop_obj.passenger_waiting.items():
                    for arrival_time, w in v.items():
                        got_on_bus = w["got_on_bus"]
                        remaining = w["remaining"]
                        block = w["block_trip"][0]
                        trip = w["block_trip"][1]
                        ons = w["ons"]
                        offs = w["offs"]
                        output = f"{self.num_events_processed},{stop_id},{arrival_time},{got_on_bus},{remaining},{block},{trip},{ons},{offs},{stop_obj.total_passenger_ons},{stop_obj.total_passenger_offs},{stop_obj.total_passenger_walk_away}"
                        self.stop_metrics_log.debug(f"{log_time},{output}")

        for bus_id, bus_obj in self.state.buses.items():
            status = bus_obj.status
            bus_type = bus_obj.type
            current_load = bus_obj.current_load
            if bus_obj.current_block_trip:
                current_block = bus_obj.current_block_trip[0]
                current_trip = bus_obj.current_block_trip[1]
            else:
                current_block = None
                current_trip = None

            current_stop = bus_obj.current_stop
            time_at_last_stop = bus_obj.time_at_last_stop
            total_passengers_served = bus_obj.total_passengers_served
            capacity = bus_obj.capacity
            deadkms_moved = bus_obj.total_deadkms_moved
            servicekms_moved = bus_obj.total_servicekms_moved

            self.bus_metrics_log.debug(
                f"{self.num_events_processed},{log_time},{bus_id},{status},{bus_type},{capacity},{current_load},{current_block},{current_trip},{current_stop},{time_at_last_stop},{total_passengers_served},{deadkms_moved},{servicekms_moved}"
            )

    def print_states(self, csv=False):
        LOGTYPE = LogType.INFO
        log(self.logger, None, f"Total events processed: {self.num_events_processed}", LOGTYPE)
        log(self.logger, None, f"Total decision epochs: {self.decision_events}", LOGTYPE)
        if not csv:
            for bus_id, bus_obj in self.state.buses.items():
                log(self.csvlogger, None, f"--Bus ID: {bus_id}--", LOGTYPE)
                log(self.csvlogger, None, f"total dwell_time: {bus_obj.dwell_time:.2f} s", LOGTYPE)
                log(
                    self.csvlogger,
                    None,
                    f"aggregate delay_time: {(bus_obj.delay_time / bus_obj.total_stops):.2f} s",
                    LOGTYPE,
                )
                log(self.csvlogger, None, f"total_service_time: {bus_obj.total_service_time:.2f}", LOGTYPE)
                log(self.csvlogger, None, f"total_passengers_served: {bus_obj.total_passengers_served}", LOGTYPE)
                log(self.csvlogger, None, f"total_servicekms_moved: {bus_obj.total_servicekms_moved:.2f} km", LOGTYPE)
                log(self.csvlogger, None, f"total_deadkms_moved: {bus_obj.total_deadkms_moved:.2f} km", LOGTYPE)
                log(self.csvlogger, None, f"current_stop: {bus_obj.current_stop}", LOGTYPE)
                log(self.csvlogger, None, f"status: {bus_obj.status}", LOGTYPE)

            for stop_id, stop_obj in self.state.stops.items():
                # if stop_obj.total_passenger_walk_away > 0:
                log(self.csvlogger, None, f"--Stop ID: {stop_id}--", LOGTYPE)
                log(self.csvlogger, None, f"total_passenger_ons: {stop_obj.total_passenger_ons}", LOGTYPE)
                log(self.csvlogger, None, f"total_passenger_offs: {stop_obj.total_passenger_offs}", LOGTYPE)
                log(self.csvlogger, None, f"total_passenger_walk_away: {stop_obj.total_passenger_walk_away}", LOGTYPE)
        else:
            log(
                self.csvlogger,
                None,
                "bus_id,dwell_time,agg_delay,service_time,total_served,service_kms,current_stop,status,type,total_deadsecs,starting_stop",
                LOGTYPE,
            )
            for bus_id, bus_obj in self.state.buses.items():
                a = bus_id
                b = f"{bus_obj.dwell_time:.2f}"
                c = f"{(bus_obj.delay_time / bus_obj.total_stops):.2f}"
                d = f"{bus_obj.total_service_time:.2f}"
                e = f"{bus_obj.total_passengers_served}"
                f = f"{bus_obj.total_deadkms_moved + bus_obj.partial_deadkms_moved:.2f}"
                g = f"{bus_obj.current_stop}"
                h = f"{bus_obj.status}"
                i = f"{bus_obj.type}"
                j = f"{bus_obj.total_deadsecs_moved:.2f}"  # Deadseconds + trip back to depot(?)
                k = f"{bus_obj.starting_stop}"
                # csvlogger.info(f"{a},{b},{c},{d},{e},{f},{g},{h},{i},{j},{k}")
                log(self.csvlogger, None, f"{a},{b},{c},{d},{e},{f},{g},{h},{i},{j},{k}")

            log(
                self.csvlogger,
                None,
                "stop_id,total_passenger_ons,total_passenger_offs,total_passenger_walk_away",
                LOGTYPE,
            )
            for stop_id, stop_obj in self.state.stops.items():
                a = stop_id
                b = f"{stop_obj.total_passenger_ons}"
                c = f"{stop_obj.total_passenger_offs}"

                passenger_set_counts = self.state.people_left_behind
                picked_list = list(
                    filter(
                        lambda x: (x["stop_id"] == stop_id),
                        passenger_set_counts,
                    )
                )
                passenger_set_counts = stop_obj.passenger_waiting_dict_list
                d = 0
                for p_set in picked_list:
                    d += p_set["ons"]
                for p_set in passenger_set_counts:
                    d += p_set["ons"]
                if (float(b) + float(c) + float(d)) > 0:
                    log(self.csvlogger, None, f"{a},{b},{c},{d}")

        total_walk_aways = 0
        total_arrivals = 0
        total_boardings = 0
        for stop_id, stop_obj in self.state.stops.items():
            passenger_set_counts = stop_obj.passenger_waiting_dict_list
            stop_walk_aways = 0
            for p_set in passenger_set_counts:
                stop_walk_aways += p_set["ons"]
            total_walk_aways += stop_walk_aways
            total_boardings += stop_obj.total_passenger_ons

        log(self.csvlogger, dt.datetime.now(), f"passenger waiting dict list: {total_walk_aways}", LogType.INFO)
        for p_set in self.state.people_left_behind:
            if p_set.get("left_behind", False):
                remaining_passengers = p_set["ons"]
                total_walk_aways += remaining_passengers
        log(self.csvlogger, dt.datetime.now(), f"people left behind: {total_walk_aways}", LogType.INFO)

        total_arrivals += total_boardings + total_walk_aways
        log(self.csvlogger, dt.datetime.now(), f"Count of all passengers: {total_arrivals}", LogType.INFO)
        log(self.csvlogger, dt.datetime.now(), f"Count of all passengers who boarded: {total_boardings}", LogType.INFO)
        log(
            self.csvlogger,
            dt.datetime.now(),
            f"Count of all passengers who were left: {total_walk_aways}",
            LogType.INFO,
        )

    def save_state(self):
        current_time = self.state.time

    def save_visualization(self, event_time, granularity_s=None):
        if not self.last_visual_log:
            self.last_visual_log = event_time

        if granularity_s:
            if event_time - self.last_visual_log < dt.timedelta(seconds=granularity_s):
                return

                # print("Visualizer:", (event_time - self.last_visual_log))
        # self.visual_log.debug(f"time,id,trip_id,last_visited_stop,value,fraction,icon")
        for bus_id, bus_obj in self.state.buses.items():
            if bus_obj.status != BusStatus.IDLE:
                current_trip = ""
                if bus_obj.current_block_trip:
                    current_trip = bus_obj.current_block_trip[1]

                if bus_obj.type == BusType.REGULAR:
                    if bus_obj.status == BusStatus.BROKEN:
                        self.visual_log.debug(
                            f"{event_time},{bus_id},{current_trip},{bus_obj.current_stop},100,{bus_obj.percent_to_next_stop},alert,60"
                        )
                    else:
                        self.visual_log.debug(
                            f"{event_time},{bus_id},{current_trip},{bus_obj.current_stop},{bus_obj.current_load},{bus_obj.percent_to_next_stop},car,35"
                        )
                elif bus_obj.type == BusType.OVERLOAD:
                    self.visual_log.debug(
                        f"{event_time},{bus_id},{current_trip},{bus_obj.current_stop},{bus_obj.current_load},{bus_obj.percent_to_next_stop},car-suv,35"
                    )

        for stop_id, stop_obj in self.state.stops.items():
            pw = stop_obj.passenger_waiting
            ons = 0
            if pw:
                for k, v in pw.items():
                    if len(v) > 0:
                        for j, w in pw[k].items():
                            ons = w["load"]
                            remaining = w["remaining"]
                            if remaining == 0:
                                self.visual_log.debug(f"{event_time},{stop_id},,{stop_id},{ons},,add-person,25")
                            else:
                                self.visual_log.debug(f"{event_time},{stop_id},,{stop_id},{remaining},,add-person,25")

        if granularity_s:
            self.last_visual_log = event_time
