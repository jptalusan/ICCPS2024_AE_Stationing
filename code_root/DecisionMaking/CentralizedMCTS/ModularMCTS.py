import copy
import math
import random
import time
import pandas as pd
from DecisionMaking.CentralizedMCTS.DataStructures.TreeNode import TreeNode
from Environment.DataStructures.State import State
from Environment.enums import ActionType, EventType

# import spdlog as spd
# from fastlogging import LogInit, Logger

# Logger.cbFormatter = lambda self, entry: f'{entry[3]}'


class ModularMCTS:
    def __init__(
        self,
        discount_factor,
        mdp_environment_model,
        rollout_policy,
        iter_limit,
        allowed_computation_time,
        exploit_explore_tradoff_param,
        action_type,
    ):
        self.allowed_computation_time = allowed_computation_time
        self.rollout_policy = rollout_policy
        self.iter_limit = iter_limit
        self.mdp_environment_model = mdp_environment_model
        self.discount_factor = discount_factor
        self.exploit_explore_tradoff_param = exploit_explore_tradoff_param

        self.solve_start_time = None
        self.number_of_nodes = None
        self.use_iter_lim = iter_limit is not None

        self.time_tracker = {"expand": 0, "select": 0, "rollout": 0}

        self.action_type = action_type

        self.debug_index = 0

        # spd.FileLogger(name=f'mcts', filename=f'logs/exploit_explore.log', truncate=False, multithreaded=True)
        # self.logger = spd.get('mcts')
        # self.logger.set_pattern("%v")
        # self.logger.set_level(spd.LogLevel.DEBUG)

        # self.logger = LogInit(pathName="logs/monday.log", console=False, colors=False)

    # TODO load the passengers to state either here or before i call solve
    def solve(self, state, starting_event_queue):
        _state = copy.deepcopy(state)
        self.solve_start_time = _state.time
        self.number_of_nodes = 0

        possible_actions, actions_taken_tracker = self.get_possible_actions(
            _state, starting_event_queue[0], self.action_type
        )

        # TODO: If there is only 1 possible action (NO_OP) don't do MCTS anymore.
        if len(possible_actions) == 1:
            scored_actions = []
            do_nothing_action = {"type": ActionType.NO_ACTION, "overload_bus": None, "info": None}
            scored_actions.append({"action": do_nothing_action, "score": 1, "num_visits": 1})

            return {
                "scored_actions": scored_actions,
                "number_nodes": 0,
                "time_taken": self.time_tracker,
                # 'tree': root
            }

        _root_is_terminal = len(starting_event_queue[1:]) <= 0

        # init tree
        root = TreeNode(
            state=_state,
            parent=None,
            depth=0,
            is_terminal=_root_is_terminal,
            possible_actions=possible_actions,
            action_to_get_here=None,
            score=0,
            num_visits=0,
            children=[],
            reward_to_here=0.0,
            actions_taken_tracker=actions_taken_tracker,
            is_fully_expanded=False,
            event_at_node=starting_event_queue[0],
            future_events_queue=starting_event_queue[1:],
        )

        if self.use_iter_lim:
            iter_count = 0

            while iter_count < self.iter_limit:
                iter_count += 1
                self.execute_iteration(root)
                self.debug_index += 1
                # print(f"MCTS {iter_count}")
        else:
            start_processing_time = time.time()
            curr_processing_time = 0
            iter_count = 0
            while curr_processing_time < self.allowed_computation_time:
                curr_processing_time = time.time() - start_processing_time
                iter_count += 1
                self.execute_iteration(root)
                # print(f"MCTS {iter_count}")

        # print(f"MCTS solve(), {len(possible_actions)} possible actions.")

        if len(root.children) == 0:
            root.is_terminal = False
            new_root_actions_taken_tracker = []

            for i, taken in root.actions_taken_tracker:
                new_root_actions_taken_tracker.append((i, True))

            root.actions_taken_tracker = new_root_actions_taken_tracker

            # Create a new state for every possible action
            for possible_action in possible_actions:
                _new_state = State(
                    stops=copy.deepcopy(root.state.stops),
                    buses=copy.deepcopy(root.state.buses),
                    bus_events=copy.copy(root.state.bus_events),
                    time=root.state.time,
                )
                _new_state.people_left_behind = copy.deepcopy(root.state.people_left_behind)

                actions_taken_to_new_node = copy.copy(root.action_sequence_to_here)
                actions_taken_to_new_node.append(possible_action)

                _new_node = TreeNode(
                    state=_new_state,
                    parent=root,
                    depth=root.depth + 1,
                    is_terminal=True,
                    possible_actions=[],
                    action_to_get_here=possible_action,
                    score=0,
                    num_visits=1,
                    children=[],
                    reward_to_here=0.0,
                    actions_taken_tracker=actions_taken_to_new_node,
                    is_fully_expanded=True,
                    event_at_node=None,
                    future_events_queue=[],
                )

                root.children.append(_new_node)

        actions_with_scores = self.get_scored_child_actions(root)

        return {
            "scored_actions": actions_with_scores,
            "number_nodes": self.number_of_nodes,
            "time_taken": self.time_tracker,
            # 'tree': root
        }

    def get_scored_child_actions(self, node):
        scored_actions = []
        for child in node.children:
            action = child.action_to_get_here
            score = child.score / child.num_visits
            num_visits = child.num_visits

            scored_actions.append({"action": action, "score": score, "num_visits": num_visits})

        return scored_actions

    def get_possible_actions(self, state, event, action_type):
        """
        Actions
        """
        possible_actions = self.mdp_environment_model.generate_possible_actions(state, event, action_type=action_type)
        # print(f"MCTS actions: {possible_actions}")
        return possible_actions

    def execute_iteration(self, node):
        select_start = time.time()
        selected_node = self.select_node(node)
        self.time_tracker["select"] += time.time() - select_start

        # Node selection
        if not selected_node.is_terminal:
            expand_start = time.time()
            # Node expansion
            new_node = self.expand_node(selected_node)
            self.time_tracker["expand"] += time.time() - expand_start
            self.number_of_nodes += 1

        else:
            new_node = selected_node

        # Simulation/Rollout
        rollout_start = time.time()
        score = self.rollout_policy.rollout(
            new_node, self.mdp_environment_model, self.discount_factor, self.solve_start_time
        )

        self.time_tracker["rollout"] += time.time() - rollout_start
        self.back_propagate(new_node, score)

    def back_propagate(self, node, score):
        while node is not None:
            node.num_visits += 1
            node.score += score
            node = node.parent

    def select_node(self, node):
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = self.get_best_child(node)
            else:
                return node
        # Returns the terminal node if it is best child
        return node

    def expand_node(self, node):
        action_to_take = self.pick_expand_action(node)

        _new_state = State(
            stops=copy.deepcopy(node.state.stops),
            buses=copy.deepcopy(node.state.buses),
            bus_events=copy.deepcopy(node.state.bus_events),
            time=node.state.time,
        )
        _new_state.people_left_behind = copy.deepcopy(node.state.people_left_behind)

        # Taking action generates new events
        immediate_reward, new_events, event_time = self.mdp_environment_model.take_action(_new_state, action_to_take)

        _new_node_future_event_queue = copy.deepcopy(node.future_events_queue)
        if new_events is not None:
            res = self.add_event_to_event_queue(_new_node_future_event_queue, new_events)

        _expand_node_depth = node.depth + 1
        _expand_node_event = _new_node_future_event_queue.pop(0)

        # Updating the state generates new events
        new_events2 = self.process_event(_new_state, _expand_node_event)
        res = self.add_event_to_event_queue(_new_node_future_event_queue, new_events2)

        new_possible_actions, actions_taken_tracker = self.get_possible_actions(
            _new_state, _expand_node_event, self.action_type
        )

        assert len(new_possible_actions) > 0
        is_new_node_fully_expanded = False

        actions_taken_to_new_node = copy.copy(node.action_sequence_to_here)
        actions_taken_to_new_node.append(action_to_take)

        discounted_immediate_score = self.standard_discounted_score(
            immediate_reward, event_time - self.solve_start_time, self.discount_factor
        )
        reward_to_here = node.reward_to_here + discounted_immediate_score

        _expand_node_is_terminal = len(_new_node_future_event_queue) <= 0

        _new_node = TreeNode(
            state=_new_state,
            parent=node,
            depth=_expand_node_depth,
            is_terminal=_expand_node_is_terminal,
            possible_actions=new_possible_actions,
            action_to_get_here=action_to_take,
            score=0,
            num_visits=0,
            children=[],
            reward_to_here=reward_to_here,
            actions_taken_tracker=actions_taken_tracker,
            action_sequence_to_here=actions_taken_to_new_node,
            is_fully_expanded=is_new_node_fully_expanded,
            event_at_node=_expand_node_event,
            future_events_queue=_new_node_future_event_queue,
        )

        node.children.append(_new_node)
        return _new_node

    def standard_discounted_score(self, reward, time_since_start, discount_factor):
        discount = discount_factor ** time_since_start.total_seconds()
        discounted_reward = discount * reward
        return discounted_reward

    def process_event(self, state, event):
        """
        Moves the state forward in time.
        """
        new_events = self.mdp_environment_model.update(state, event, should_log=False)
        return new_events

    def add_event_to_event_queue(self, queue, events):
        if len(events) == 0:
            return False

        # for event in events:
        #     queue.append(event)
        # queue.sort(key=lambda _: _.time, reverse=False)
        event = events[0]
        if event:
            queue.append(event)
        queue.sort(key=lambda _: _.time, reverse=False)
        return True

    def pick_expand_action(self, node):
        """
        Get unexplored actions
        param: node
        return: picked action
        """
        unexplored_actions = [(node.possible_actions[_[0]], _[0]) for _ in node.actions_taken_tracker if not _[1]]
        num_unexplored_actions = len(unexplored_actions)
        if num_unexplored_actions == 1:
            node.is_fully_expanded = True

        random.seed(100)
        action_index = random.choice(range(num_unexplored_actions))
        picked_action = unexplored_actions[action_index][0]
        node.actions_taken_tracker[unexplored_actions[action_index][1]] = (unexplored_actions[action_index][1], True)

        return picked_action

    def get_best_child(self, node):
        best_val = float("-inf")
        best_nodes = []

        for child in node.children:
            value = self.uct_score(child)
            if value > best_val:
                best_val = value
                best_nodes = [child]
            elif value == best_val:
                best_nodes.append(child)

        random.seed(100)
        return random.choice(best_nodes)

    def uct_score(self, node):
        exploit = node.score / node.num_visits
        explore = math.sqrt(math.log(node.parent.num_visits) / node.num_visits)

        # print(f"{exploit:.2f},{explore:.2f}")
        # self.logger.debug(f"{exploit:.2f},{explore:.2f}")

        # for positive params (reward, served)
        # scaled_explore_2 = -1 * self.exploit_explore_tradoff_param * explore
        # for negative params (cost, remain)
        scaled_explore_2 = abs(self.exploit_explore_tradoff_param) * explore

        score = exploit + scaled_explore_2
        # score = exploit + explore
        return score
