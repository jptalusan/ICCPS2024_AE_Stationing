class DoNothing:
    def __init__(self, travel_model) -> None:
        self.travel_model = travel_model
        pass

    def select_overload_to_dispatch(self, state=None, actions=None):
        return []
