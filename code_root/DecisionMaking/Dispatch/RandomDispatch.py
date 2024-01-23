from Environment.enums import BusStatus, BusType
import random


class RandomDispatch:
    def __init__(self, travel_model) -> None:
        self.travel_model = travel_model
        pass

    def select_overload_to_dispatch(self, state, actions):
        if actions is None:
            return None

        if len(actions) <= 0:
            return None
        random.seed(100)
        return random.choice(actions)
