import numpy as np
from random import seed
import random

# TODO corriger le dictionnaire, il stock actuellement les états, il devrait stocker plutot le tuple etat transition
# TODO la lookup table devrait être sous la forme [7][7][4]
# TODO

ALPHA = 0
GAMMA = 0  # Discount Factor
STEP_PER_EPOCH = 10
EPOCH = 100
EXPLORATION_CONST = 0

state_action_dictionary = {}
action_list = []
reward = 0
current_position = (5, 5)
game_map = np.zeros((6, 6))

seed(1)

lookup_table = np.arange(7 * 7).reshape(7, 7)


def tuple_to_key(position_tuple):
    return lookup_table[position_tuple[0], position_tuple[1]]


def set_game_map():
    local_game_map = np.zeros((6, 6))
    local_game_map[0, 5] = 1000
    local_game_map[1, 0] = 50
    local_game_map[3, 2] = -10
    local_game_map[3, 3] = -10
    local_game_map[3, 4] = -10
    local_game_map[3, 5] = -10
    local_game_map[4, 2] = -10
    return local_game_map


def q_max_func(previous_state_action_value):
    return GAMMA * previous_state_action_value


def random_direction():
    global current_position
    direction = random.randint(0, 3)
    if direction == 0: return current_position[0], current_position[1] - 1
    if direction == 1: return current_position[0], current_position[1] + 1
    if direction == 2: return current_position[0] - 1, current_position[1]
    if direction == 3: return current_position[0] + 1, current_position[1]


# (state[0]-1, state[1])
def pick_direction(state):
    global state_action_dictionary
    left_val = state_action_dictionary[tuple_to_key((state[0], state[1] - 1))] if tuple_to_key(
        (state[0], state[1] - 1)) in state_action_dictionary else EXPLORATION_CONST
    right_val = state_action_dictionary[tuple_to_key((state[0], state[1] + 1))] if tuple_to_key(
        (state[0], state[1] + 1)) in state_action_dictionary else EXPLORATION_CONST
    upper_val = state_action_dictionary[tuple_to_key((state[0] - 1, state[1]))] if tuple_to_key(
        (state[0] - 1, state[1])) in state_action_dictionary else EXPLORATION_CONST
    lower_val = state_action_dictionary[tuple_to_key((state[0] + 1, state[1]))] if tuple_to_key(
        (state[0] + 1, state[1])) in state_action_dictionary else EXPLORATION_CONST
    if left_val > right_val and left_val > upper_val and left_val > lower_val: return (state[0], state[1] - 1)
    if left_val < right_val and right_val > upper_val and right_val > lower_val: return (state[0], state[1] + 1)
    if upper_val > left_val and right_val < upper_val and upper_val > lower_val: return (state[0] - 1, state[1])
    if lower_val > left_val and right_val < lower_val and lower_val > upper_val:
        return state[0] + 1, state[1]
    else:
        return random_direction()


def make_moove():
    global current_position
    global reward
    global game_map
    action_list.append(current_position)
    selected_position = pick_direction(current_position)
    if selected_position[0] > 5:
        # selected_position[0] = 5
        selected_position = (5, selected_position[1])
        reward = reward - 10
    elif selected_position[1] > 5:
        # selected_position[0] = 5
        selected_position = (selected_position[0], 5)
        reward = reward - 10
    elif selected_position[0] < 0:
        # selected_position[0] = 0
        selected_position = (0, selected_position[1])
        reward = reward - 10
    elif selected_position[1] < 0:
        # selected_position[0] = 0
        selected_position = (selected_position[0], 0)
        reward = reward - 10
    elif game_map[selected_position[0], selected_position[1]] == -10:
        selected_position = current_position
        reward = reward - 10
    else:
        current_position = selected_position
        reward = reward + game_map[selected_position[0], selected_position[1]]
    current_position = selected_position


def update_dict():
    global reward
    current_reward = reward
    action_list.reverse()
    for position in action_list:
        current_reward = current_reward * (1 - ALPHA)
        if tuple_to_key(position) in state_action_dictionary:
            state_action_dictionary[tuple_to_key(position)] = (
                    state_action_dictionary[tuple_to_key(position)] + current_reward)
        else:
            state_action_dictionary[tuple_to_key(position)] = current_reward
            # state_action_dictionary[tuple_to_key(position)] = (state_action_dictionary[tuple_to_key(position)] + current_reward) if tuple_to_key(position) in state_action_dictionary else state_action_dictionary[tuple_to_key(position)] = current_reward


def draw_world(actual_current_position, current_step):
    copy_map = game_map.copy()
    copy_map[actual_current_position[0], actual_current_position[1]] = 1
    print("current_step = ", current_step)
    print(copy_map)
    print("reward = ", reward)


def run_epoch():
    global action_list
    global reward
    global current_position
    action_list = []
    reward = 0
    current_position = (5, 5)
    for i in range(0, STEP_PER_EPOCH):
        make_moove()
        draw_world(current_position, i)
    update_dict()


def run_sim():
    global game_map
    game_map = set_game_map()
    for i in range(EPOCH):
        print("EPOCH = ", i)
        run_epoch()
    print(state_action_dictionary)

run_sim()