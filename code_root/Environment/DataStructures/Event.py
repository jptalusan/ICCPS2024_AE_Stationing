from src.utils import *


# Time is datetime!
class Event:
    def __init__(self, event_type, time, type_specific_information=None):
        self.event_type = event_type
        self.time = time
        self.type_specific_information = type_specific_information

    def __str__(self):
        if self.type_specific_information:
            return f"{datetime_to_str(self.time)},{self.event_type.name},{self.type_specific_information}"
        return f"{datetime_to_str(self.time)},{self.event_type.name}"

    def __repr__(self):
        if self.type_specific_information:
            return f"{datetime_to_str(self.time)},{self.event_type.name},{self.type_specific_information}"
        return f"{datetime_to_str(self.time)},{self.event_type.name}"
