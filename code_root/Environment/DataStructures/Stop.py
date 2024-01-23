# Passenger waiting should be a list of tuples (route_id_dir, time, number of people)
class Stop:
    def __init__(
        self,
        stop_id,
        total_passenger_ons=0,
        total_passenger_offs=0,
        total_passenger_walk_away=0,
        passenger_waiting=None,
    ):
        self.stop_id = stop_id
        self.total_passenger_ons = total_passenger_ons
        self.total_passenger_offs = total_passenger_offs
        self.total_passenger_walk_away = total_passenger_walk_away
        self.passenger_waiting = passenger_waiting
        self.passenger_waiting_dict_list = []

    def __str__(self):
        return f"{self.stop_id},{self.passenger_waiting}"

    def __repr__(self):
        return f"{self.stop_id},{self.passenger_waiting}"
