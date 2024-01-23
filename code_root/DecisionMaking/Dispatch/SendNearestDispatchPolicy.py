from Environment.enums import BusStatus, BusType, ActionType
import random


class SendNearestDispatchPolicy:
    def __init__(self, travel_model) -> None:
        self.travel_model = travel_model
        pass

    # _valid_actions = [[ActionType.OVERLOAD_DISPATCH], idle_overload_buses, stops_with_left_behind_passengers]
    # _valid_actions = [[ActionType.OVERLOAD_TO_BROKEN], idle_overload_buses, broken_buses]

    # Right now only handling BROKEN and DISPATCH not ALLOC buses
    def select_overload_to_dispatch(self, state, actions):
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
            elif action_type == ActionType.OVERLOAD_DISPATCH:
                next_stop = info[0]
                pass
            elif action_type == ActionType.OVERLOAD_ALLOCATE:
                next_stop = info
                pass
            else:
                raise "Action not supported"

            distance = self.travel_model.get_distance_from_stop_to_stop(current_stop, next_stop, state.time)
            actions_with_distance.append((action, distance))

        if len(actions_with_distance) == 0:
            action_to_take = {"type": ActionType.NO_ACTION, "overload_bus": None, "info": None}
            return action_to_take

        actions_with_distance = sorted(actions_with_distance, key=lambda x: x[1], reverse=False)
        # print(actions_with_distance)
        actions_with_distance = actions_with_distance[0][0]
        return actions_with_distance
