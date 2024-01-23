import copy
import time
import random
import datetime as dt
from Environment.DataStructures.State import State
from DecisionMaking.CentralizedMCTS.DataStructures.TreeNode import TreeNode
from Environment.enums import EventType, ActionType, BusType, BusStatus
from src.utils import *
import itertools


class BareMinimumRollout:
    """
    Bare minimum rollout, send the nearest bus (if available) to cover for a broken down bus.
    """

    def __init__(self, rollout_horizon_delta_t, dispatch_policy):
        self.debug_rewards = None
        self.passenger_arrival_distribution = None
        self.deep_copy_time = 0
        # self.rollout_horizon_delta_t = rollout_horizon_delta_t  # 60*60*N for N hour horizon (0.6)
        if rollout_horizon_delta_t:
            self.rollout_horizon_delta_t = rollout_horizon_delta_t
        else:
            self.rollout_horizon_delta_t = None

        self.dispatch_policy = dispatch_policy
        self.horizon_time_limit = None

        self.total_walkaways = 0

    def rollout(self, node, environment_model, discount_factor, solve_start_time):
        s_copy_time = time.time()
        self.debug_rewards = []
        self.total_walkaways = 0

        truncated_events = copy.copy(node.future_events_queue)

        if self.rollout_horizon_delta_t is not None:
            self.horizon_time_limit = solve_start_time + dt.timedelta(seconds=self.rollout_horizon_delta_t)
            truncated_events = [
                event for event in truncated_events if solve_start_time <= event.time <= self.horizon_time_limit
            ]
        else:
            self.horizon_time_limit = solve_start_time + dt.timedelta(days=1)

        _state = State(
            stops=copy.deepcopy(node.state.stops),
            buses=copy.deepcopy(node.state.buses),
            bus_events=None,
            time=node.state.time,
        )
        _state.people_left_behind = copy.deepcopy(node.state.people_left_behind)

        # Why is possible_actions None?
        _node = TreeNode(
            state=_state,
            parent=None,
            depth=node.depth,
            is_terminal=node.is_terminal,
            possible_actions=None,
            action_to_get_here=None,
            score=node.score,
            num_visits=None,
            children=None,
            reward_to_here=node.reward_to_here,
            is_fully_expanded=False,
            actions_taken_tracker=None,
            action_sequence_to_here=None,
            event_at_node=node.event_at_node,
            # future_events_queue=copy.copy(node.future_events_queue)
            future_events_queue=truncated_events,
        )

        self.deep_copy_time += time.time() - s_copy_time

        # Run until all events finish

        while _node.future_events_queue:
            self.rollout_iter(_node, environment_model, discount_factor, solve_start_time)

        return _node.reward_to_here

    """
    SendNearestDispatch if a vehicle is broken, else do nothing.
    However, breakdown events are not generated in the event chains.
    I think, if i set this rollout to exactly the same as the Greedy, it might improve. (takes SO LONG) 
    """

    def rollout_iter(self, node, environment_model, discount_factor, solve_start_time):
        # rollout1
        valid_actions = []
        if node.event_at_node.event_type == EventType.DECISION_ALLOCATION_EVENT:
            valid_actions, _ = environment_model.generate_possible_actions(
                node.state, node.event_at_node, action_type=ActionType.OVERLOAD_ALLOCATE
            )
        elif (
            node.event_at_node.event_type == EventType.VEHICLE_ARRIVE_AT_STOP
            or node.event_at_node.event_type == EventType.PASSENGER_LEFT_BEHIND
        ):
            valid_actions, _ = environment_model.generate_possible_actions(
                node.state, node.event_at_node, action_type=ActionType.OVERLOAD_DISPATCH
            )
        elif node.event_at_node.event_type == EventType.VEHICLE_BREAKDOWN:
            valid_actions, _ = environment_model.generate_possible_actions(
                node.state, node.event_at_node, action_type=ActionType.OVERLOAD_TO_BROKEN
            )
        if len(valid_actions) > 0:
            # action_to_take = self.dispatch_policy.select_overload_to_dispatch(node.state, valid_actions)
            random.seed(100)
            action_to_take = random.choice(valid_actions)
        else:
            action_to_take = {"type": ActionType.NO_ACTION, "overload_bus": None, "info": "No actions RO."}

        immediate_reward, new_events, event_time = environment_model.take_action(node.state, action_to_take)

        # Limit the number of new events generated based on the time horizon!!!
        if len(new_events) > 0:
            if new_events[0].time <= self.horizon_time_limit:
                node.future_events_queue.append(new_events[0])
                node.future_events_queue.sort(key=lambda _: _.time)

        node.depth += 1
        node.event_at_node = node.future_events_queue.pop(0)
        new_events = self.process_event(node.state, node.event_at_node, environment_model)
        if len(new_events) > 0:
            if new_events[0].time <= self.horizon_time_limit:
                node.future_events_queue.append(new_events[0])
                node.future_events_queue.sort(key=lambda _: _.time)

        discounted_immediate_score = self.standard_discounted_score(
            immediate_reward, event_time - solve_start_time, discount_factor
        )

        node.reward_to_here = node.reward_to_here + discounted_immediate_score
        # print(len(node.future_events_queue))

    def standard_discounted_score(self, reward, time_since_start, discount_factor):
        discount = discount_factor ** time_since_start.total_seconds()
        discounted_reward = discount * reward
        return discounted_reward

    def process_event(self, state, event, environment_model):
        new_events = environment_model.update(state, event)
        return new_events
