# Score is G
# Reward is r or gamma


class TreeNode:
    def __init__(
        self,
        state,
        parent,
        depth,
        is_terminal,
        possible_actions,
        action_to_get_here,
        score,
        num_visits,
        children,
        reward_to_here,
        is_fully_expanded,
        actions_taken_tracker,
        event_at_node,
        future_events_queue,
        action_sequence_to_here=[],
    ):
        self.state = state
        self.parent = parent
        self.depth = depth
        self.is_terminal = is_terminal
        self.possible_actions = possible_actions
        self.action_to_get_here = action_to_get_here
        self.score = score
        self.num_visits = num_visits
        self.children = children
        self.is_fully_expanded = is_fully_expanded
        self.reward_to_here = reward_to_here
        self.action_sequence_to_here = action_sequence_to_here
        self.event_at_node = event_at_node
        self.actions_taken_tracker = actions_taken_tracker
        self.future_events_queue = future_events_queue
