from src.utils import *
from Environment.enums import ActionType


# Limit dispatch to 1 overload bus per trip/route_id_dir
# Allocate to depot and activate buses
class RandomCoord:
    def __init__(self, environment_model, travel_model, dispatch_policy, logger) -> None:
        self.environment_model = environment_model
        self.travel_model = travel_model
        self.dispatch_policy = dispatch_policy
        self.event_counter = 0
        self.logger = logger

        self.metrics = dict()
        self.metrics["resp_times"] = dict()
        self.metrics["computation_times"] = dict()

        pass

    def event_processing_callback_funct(self, actions, state, action_type=ActionType.OVERLOAD_DISPATCH):
        """
        function that is called when each new event occurs in the underlying simulation.
        :param actions
        :param state
        :param action_type
        :return: action
        """
        if actions is None:
            return None

        if len(actions) == 0:
            return None

        # random_action = random.choice(actions)
        action = self.dispatch_policy.select_overload_to_dispatch(state, actions)

        return action

    def add_incident(self, state, incident_event):
        incident = incident_event.type_specific_information["incident_obj"]
        self.environment_model.add_incident(state, incident)
