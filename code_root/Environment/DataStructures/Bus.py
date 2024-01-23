# Current_stop is the starting depot if the current_stop_number is 0
class Bus:
    def __init__(
        self,
        my_id,
        type,
        status,
        capacity,
        bus_block_trips=[],
        current_stop="",
        current_block_trip=None,
        current_stop_number=0,
        next_stop_number=0,
        t_state_change=0.0,
        percent_to_next_stop=0.0,
        distance_to_next_stop=0.0,
        total_servicekms_moved=0,
        total_deadkms_moved=0,
        total_deadsecs_moved=0,
        partial_deadkms_moved=0,
        total_service_time=0.0,
        total_passengers_served=0,
        available_time=0.0,
        dwell_time=0.0,
        delay_time=0.0,
        current_load=0,
        time_at_last_stop=None,
        total_stops=1,
        last_decision_epoch=None,
        starting_stop=None,
    ):
        self.my_id = my_id
        self.status = status
        self.type = type
        self.bus_block_trips = bus_block_trips
        self.current_block_trip = current_block_trip
        self.current_stop = current_stop
        self.current_stop_number = current_stop_number
        self.next_stop_number = next_stop_number
        self.t_state_change = t_state_change
        self.percent_to_next_stop = percent_to_next_stop
        self.distance_to_next_stop = distance_to_next_stop
        self.total_servicekms_moved = total_servicekms_moved
        self.total_deadkms_moved = total_deadkms_moved
        self.partial_deadkms_moved = partial_deadkms_moved
        self.total_passengers_served = total_passengers_served
        self.available_time = available_time
        self.dwell_time = dwell_time
        self.delay_time = delay_time
        self.capacity = capacity
        self.current_load = current_load
        self.time_at_last_stop = time_at_last_stop
        self.total_service_time = total_service_time
        self.total_stops = total_stops
        self.last_decision_epoch = last_decision_epoch
        self.total_deadsecs_moved = total_deadsecs_moved
        self.starting_stop = starting_stop

    def __str__(self):
        return f"{self.my_id},{self.status},{self.current_block_trip}"

    def __repr__(self):
        return f"{self.my_id},{self.status},{self.current_block_trip}"
