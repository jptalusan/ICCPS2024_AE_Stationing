from enum import Enum, IntEnum


class BusType(Enum):
    REGULAR = "regular"
    OVERLOAD = "overload"


class BusStatus(IntEnum):
    BROKEN = 0
    IDLE = 1
    IN_TRANSIT = 2
    ALLOCATION = 3


class EventType(Enum):
    VEHICLE_START_TRIP = "vehicle_start_trip"
    VEHICLE_ARRIVE_AT_STOP = "vehicle_arrive_at_stop"
    VEHICLE_ACCIDENT = "vehicle_accident"
    VEHICLE_BREAKDOWN = "vehicle_break_down"
    VEHICLE_REPAIRED = "vehicle_repaired"
    VEHICLE_FINISH_TRIP = "vehicle_finish_trip"
    VEHICLE_FINISH_BLOCK = "vehicle_finish_block"
    PASSENGER_ARRIVE_STOP = "passenger_arrive_stop"
    PASSENGER_LEAVE_STOP = "passenger_leave_stop"
    CONGESTION_LEVEL_CHANGE = "congestion_level_change"
    DECISION_ALLOCATION_EVENT = "decision_allocation_event"
    PASSENGER_LEFT_BEHIND = "passenger_left_behind"
    DECISION_DISPATCH_EVENT = "decision_dispatch_event"


class ActionType(Enum):
    OVERLOAD_DISPATCH = "overload_dispatch"
    OVERLOAD_ALLOCATE = "overload_allocate"
    OVERLOAD_TO_BROKEN = "overload_to_broken"
    OVERLOAD_ALL = "overload_all"
    NO_ACTION = "no_action"
    ROLLOUT = "rollout"


class LogType(Enum):
    ERROR = "error"
    INFO = "info"
    DEBUG = "debug"


class MCTSType(Enum):
    MODULAR_MCTS = "modular_mcts"
